// SiteOne Crawler - observed crawler access on key pages
// (c) Jan Reges <jan.reges@siteone.cz>
//
// What SiteOne Crawler's own requests met on the key pages: refused or failed requests, redirect
// chains and loops, pages that look like a bot challenge or a "not found" page served with a
// 200, and unusually slow responses. AI vendors' crawlers may be treated differently by a CDN or
// WAF, so the report labels this "as seen by SiteOne Crawler". The other internal URLs are only
// counted, by their status codes.

use std::collections::{BTreeMap, HashMap, HashSet};

use once_cell::sync::Lazy;
use regex::Regex;
use scraper::{Html, Selector};

use crate::ai::geo::keys::{KeyPage, redirect_chain, visits_by_url};
use crate::result::status::Status;
use crate::result::visited_url::VisitedUrl;
use crate::types::ContentTypeId;
use crate::utils;

#[derive(Debug, Clone, PartialEq)]
pub struct AccessIssue {
    pub url: String,
    pub kind: AccessKind,
    /// "HTTP 403", "-2:TIMEOUT", the chain of a redirect, the title of a suspected page, …
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AccessKind {
    /// 401 or 403.
    Denied(i32),
    /// 429.
    RateLimited,
    /// 5xx.
    ServerError(i32),
    /// A failed request: connection error, timeout, reset or send error (negative codes).
    Transport(i32),
    /// More than 2 redirects in a row.
    RedirectChain(usize),
    RedirectLoop,
    /// A 200 that looks like a bot challenge or a block page.
    SuspectedChallenge,
    /// A 200 whose title or H1 says the page was not found.
    SuspectedSoft404,
    /// Slower than the crawl's p90 and than `SLOW_SECONDS`; the response time in seconds.
    Slow(f64),
}

impl AccessKind {
    /// A stable name of the kind, used for counts.
    pub fn key(&self) -> &'static str {
        match self {
            AccessKind::Denied(_) => "denied",
            AccessKind::RateLimited => "rate_limited",
            AccessKind::ServerError(_) => "server_error",
            AccessKind::Transport(_) => "transport",
            AccessKind::RedirectChain(_) => "redirect_chain",
            AccessKind::RedirectLoop => "redirect_loop",
            AccessKind::SuspectedChallenge => "suspected_challenge",
            AccessKind::SuspectedSoft404 => "suspected_soft_404",
            AccessKind::Slow(_) => "slow",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AccessStats {
    /// Key pages whose responses were checked.
    pub key_pages: usize,
    /// The other internal URLs, checked by their status codes and redirects only.
    pub other_urls: usize,
    /// The issues found on those other URLs, by `AccessKind::key`.
    pub other_issues: BTreeMap<&'static str, usize>,
    /// The crawl's 90th-percentile response time of HTML pages, as in the crawl summary
    /// (`BasicStats::total_requests_times_p90`); `None` without any crawled URL.
    pub p90_seconds: Option<f64>,
}

/// Title fragments (lowercase) of bot-challenge and block pages.
pub const CHALLENGE_TITLE_MARKERS: &[&str] = &["just a moment", "access denied", "captcha", "checking your browser"];
/// HTML fragments (lowercase) of bot-challenge pages. A "captcha" or "access denied" elsewhere
/// in a page is not enough: contact forms show reCAPTCHA.
pub const CHALLENGE_HTML_MARKERS: &[&str] = &["cf-chl", "cf_chl", "checking your browser"];
/// Title or H1 fragments (lowercase) of "not found" pages; "404" counts as a number of its own.
pub const SOFT_404_MARKERS: &[&str] = &["not found", "nenalezena", "nebyla nalezena"];
/// A key page slower than the crawl's p90 is reported only above this many seconds.
pub const SLOW_SECONDS: f64 = 3.0;

static TITLE_SELECTOR: Lazy<Selector> = Lazy::new(|| Selector::parse("title").unwrap());
static H1_SELECTOR: Lazy<Selector> = Lazy::new(|| Selector::parse("h1").unwrap());
static NUMBER_404: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b404\b").unwrap());

/// The access issues of the key pages, in key-page order, and the counts of the other internal
/// URLs. Challenge and soft-404 checks need the stored bodies of the key pages.
pub fn observed_access(status: &Status, key: &[KeyPage]) -> (Vec<AccessIssue>, AccessStats) {
    let visited = status.get_visited_urls();
    let by_url = visits_by_url(&visited);
    let by_uq_id: HashMap<&str, &VisitedUrl> = visited.iter().map(|visit| (visit.uq_id.as_str(), visit)).collect();
    let mut stats = AccessStats {
        p90_seconds: (!visited.is_empty()).then(|| status.get_basic_stats().total_requests_times_p90),
        ..AccessStats::default()
    };

    let mut issues = Vec::new();
    for page in key {
        let Some(visit) = by_uq_id.get(page.uq_id.as_str()) else {
            continue;
        };
        stats.key_pages += 1;
        if let Some(issue) = response_issue(visit, &by_url) {
            issues.push(issue);
            continue;
        }
        if visit.status_code != 200 || visit.content_type != ContentTypeId::Html {
            continue;
        }
        if let Some(body) = status.get_url_body_text(&visit.uq_id)
            && let Some(issue) = page_issue(&visit.url, &body)
        {
            issues.push(issue);
        }
        if let Some(p90) = stats.p90_seconds
            && visit.request_time > p90
            && visit.request_time > SLOW_SECONDS
        {
            issues.push(AccessIssue {
                url: visit.url.clone(),
                kind: AccessKind::Slow(visit.request_time),
                detail: format!("{:.1} s (p90 of the crawl {:.1} s)", visit.request_time, p90),
            });
        }
    }

    let key_ids: HashSet<&str> = key.iter().map(|page| page.uq_id.as_str()).collect();
    for visit in visited
        .iter()
        .filter(|visit| !visit.is_external && !key_ids.contains(visit.uq_id.as_str()))
    {
        stats.other_urls += 1;
        if let Some(issue) = response_issue(visit, &by_url) {
            *stats.other_issues.entry(issue.kind.key()).or_insert(0) += 1;
        }
    }
    (issues, stats)
}

/// An issue told by the status code alone, or by the redirects that start at the URL.
fn response_issue(visit: &VisitedUrl, by_url: &HashMap<String, &VisitedUrl>) -> Option<AccessIssue> {
    let code = visit.status_code;
    let (kind, detail) = match code {
        401 | 403 => (AccessKind::Denied(code), format!("HTTP {}", code)),
        429 => (AccessKind::RateLimited, format!("HTTP {}", code)),
        500..=599 => (AccessKind::ServerError(code), format!("HTTP {}", code)),
        -4..=-1 => (
            AccessKind::Transport(code),
            utils::get_http_client_code_with_error_description(code, false),
        ),
        301..=308 => {
            let chain = redirect_chain(visit, by_url);
            let mut urls: Vec<&str> = chain.visits.iter().map(|visit| visit.url.as_str()).collect();
            if let Some(target) = &chain.loops_to {
                urls.push(target);
                (AccessKind::RedirectLoop, urls.join(" → "))
            } else if chain.hops > 2 {
                (AccessKind::RedirectChain(chain.hops), urls.join(" → "))
            } else {
                return None;
            }
        }
        _ => return None,
    };
    Some(AccessIssue {
        url: visit.url.clone(),
        kind,
        detail,
    })
}

/// A 200 HTML page that looks like a bot challenge, or else like a "not found" page.
fn page_issue(url: &str, body: &str) -> Option<AccessIssue> {
    let document = Html::parse_document(body);
    let text_of = |selector: &Selector| {
        document
            .select(selector)
            .next()
            .map(|element| {
                element
                    .text()
                    .collect::<String>()
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_default()
    };
    let title = text_of(&TITLE_SELECTOR);
    let h1 = text_of(&H1_SELECTOR);
    let title_lower = title.to_lowercase();
    let html_lower = body.to_lowercase();

    let kind = if CHALLENGE_TITLE_MARKERS
        .iter()
        .any(|marker| title_lower.contains(marker))
        || CHALLENGE_HTML_MARKERS.iter().any(|marker| html_lower.contains(marker))
    {
        AccessKind::SuspectedChallenge
    } else if [&title, &h1].iter().any(|text| {
        let lower = text.to_lowercase();
        SOFT_404_MARKERS.iter().any(|marker| lower.contains(marker)) || NUMBER_404.is_match(&lower)
    }) {
        AccessKind::SuspectedSoft404
    } else {
        return None;
    };
    let detail = if h1.is_empty() || kind == AccessKind::SuspectedChallenge {
        format!("title: {}", title)
    } else {
        format!("title: {} · H1: {}", title, h1)
    };
    Some(AccessIssue {
        url: url.to_string(),
        kind,
        detail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::geo::keys::KeyPage;
    use crate::ai::geo::test_support::{add, new_status, page};
    use crate::result::visited_url::{SOURCE_A_HREF, SOURCE_IMG_SRC, SOURCE_INIT_URL, SOURCE_REDIRECT};

    fn key(status: &Status, uq_ids: &[&str]) -> Vec<KeyPage> {
        let visited = status.get_visited_urls();
        uq_ids
            .iter()
            .map(|uq_id| {
                let visit = visited
                    .iter()
                    .find(|visit| visit.uq_id == *uq_id)
                    .expect("a visited URL");
                KeyPage {
                    uq_id: visit.uq_id.clone(),
                    url: visit.url.clone(),
                    status_code: visit.status_code,
                    score: None,
                    is_homepage: visit.source_attr == SOURCE_INIT_URL,
                }
            })
            .collect()
    }

    fn kinds(issues: &[AccessIssue]) -> Vec<(&str, &AccessKind)> {
        issues.iter().map(|issue| (issue.url.as_str(), &issue.kind)).collect()
    }

    #[test]
    fn status_codes_are_classified() {
        let mut status = new_status();
        add(
            &mut status,
            page("home", "", SOURCE_INIT_URL, "https://example.com/", 200, None),
            Some("<title>Home</title>"),
        );
        let cases = [
            ("u401", 401),
            ("u403", 403),
            ("u429", 429),
            ("u500", 500),
            ("u503", 503),
            ("conn", -1),
            ("timeout", -2),
            ("reset", -3),
            ("send", -4),
            ("skipped", -6),
            ("u404", 404),
        ];
        for (uq_id, code) in cases {
            add(
                &mut status,
                page(
                    uq_id,
                    "home",
                    SOURCE_A_HREF,
                    &format!("https://example.com/{uq_id}"),
                    code,
                    None,
                ),
                None,
            );
        }
        let all: Vec<&str> = std::iter::once("home")
            .chain(cases.iter().map(|(uq_id, _)| *uq_id))
            .collect();
        let (issues, stats) = observed_access(&status, &key(&status, &all));
        assert_eq!(
            kinds(&issues),
            [
                ("https://example.com/u401", &AccessKind::Denied(401)),
                ("https://example.com/u403", &AccessKind::Denied(403)),
                ("https://example.com/u429", &AccessKind::RateLimited),
                ("https://example.com/u500", &AccessKind::ServerError(500)),
                ("https://example.com/u503", &AccessKind::ServerError(503)),
                ("https://example.com/conn", &AccessKind::Transport(-1)),
                ("https://example.com/timeout", &AccessKind::Transport(-2)),
                ("https://example.com/reset", &AccessKind::Transport(-3)),
                ("https://example.com/send", &AccessKind::Transport(-4)),
            ]
        );
        assert!(issues[6].detail.contains("TIMEOUT"), "{}", issues[6].detail);
        assert_eq!(stats.key_pages, all.len());
    }

    #[test]
    fn redirect_chains_longer_than_two_and_loops_are_reported() {
        let mut status = new_status();
        add(
            &mut status,
            page("home", "", SOURCE_INIT_URL, "https://example.com/", 200, None),
            None,
        );
        add(
            &mut status,
            page("a", "home", SOURCE_A_HREF, "https://example.com/a", 301, Some("/b")),
            None,
        );
        add(
            &mut status,
            page(
                "b",
                "a",
                SOURCE_REDIRECT,
                "https://example.com/b",
                302,
                Some("https://example.com/c"),
            ),
            None,
        );
        add(
            &mut status,
            page("c", "b", SOURCE_REDIRECT, "https://example.com/c", 301, Some("d")),
            None,
        );
        add(
            &mut status,
            page("d", "c", SOURCE_REDIRECT, "https://example.com/d", 200, None),
            None,
        );
        // A loop: the crawler does not fetch /e again, so the chain ends at /f.
        add(
            &mut status,
            page("e", "home", SOURCE_A_HREF, "https://example.com/e", 301, Some("/f")),
            None,
        );
        add(
            &mut status,
            page("f", "e", SOURCE_REDIRECT, "https://example.com/f", 301, Some("/e#top")),
            None,
        );
        // A single redirect to a page crawled from elsewhere is fine.
        add(
            &mut status,
            page("g", "home", SOURCE_A_HREF, "https://example.com/g", 301, Some("/d")),
            None,
        );

        let (issues, _) = observed_access(&status, &key(&status, &["home", "a", "b", "e", "g"]));
        assert_eq!(
            kinds(&issues),
            [
                ("https://example.com/a", &AccessKind::RedirectChain(3)),
                ("https://example.com/e", &AccessKind::RedirectLoop),
            ]
        );
        assert_eq!(
            issues[0].detail,
            "https://example.com/a → https://example.com/b → https://example.com/c → https://example.com/d"
        );
        assert_eq!(
            issues[1].detail,
            "https://example.com/e → https://example.com/f → https://example.com/e"
        );
    }

    #[test]
    fn challenge_pages_are_suspected_by_their_markers() {
        let mut status = new_status();
        add(
            &mut status,
            page("home", "", SOURCE_INIT_URL, "https://example.com/", 200, None),
            None,
        );
        let bodies = [
            (
                "cf",
                "<html><head><title>Just a moment...</title></head><body><script>window._cf_chl_opt={}</script></body></html>",
                true,
            ),
            (
                "chl",
                r#"<html><head><title>Example</title></head><body><div id="cf-chl-widget"></div></body></html>"#,
                true,
            ),
            (
                "denied",
                "<html><head><title>Access Denied</title></head><body>Reference #18</body></html>",
                true,
            ),
            (
                "check",
                "<html><head><title>Example</title></head><body><p>Checking your browser before accessing example.com.</p></body></html>",
                true,
            ),
            (
                "captcha",
                "<html><head><title>Please solve the CAPTCHA</title></head><body></body></html>",
                true,
            ),
            // A contact form with reCAPTCHA is a normal page.
            (
                "form",
                r#"<html><head><title>Contact</title></head><body><form><div class="g-recaptcha"></div><p>This site is protected by reCAPTCHA. Access denied pages are rare.</p></form></body></html>"#,
                false,
            ),
        ];
        for (uq_id, body, _) in bodies {
            add(
                &mut status,
                page(
                    uq_id,
                    "home",
                    SOURCE_A_HREF,
                    &format!("https://example.com/{uq_id}"),
                    200,
                    None,
                ),
                Some(body),
            );
        }
        let ids: Vec<&str> = bodies.iter().map(|(uq_id, _, _)| *uq_id).collect();
        let (issues, _) = observed_access(&status, &key(&status, &ids));
        let flagged: Vec<&str> = issues
            .iter()
            .filter(|issue| issue.kind == AccessKind::SuspectedChallenge)
            .map(|issue| issue.url.as_str())
            .collect();
        let expected: Vec<String> = bodies
            .iter()
            .filter(|(_, _, suspected)| *suspected)
            .map(|(uq_id, _, _)| format!("https://example.com/{uq_id}"))
            .collect();
        assert_eq!(flagged, expected);
    }

    #[test]
    fn soft_404_pages_are_suspected_by_their_title_or_h1() {
        let mut status = new_status();
        let bodies = [
            ("t", "<title>Page not found | Example</title><h1>Oops</h1>", true),
            ("h", "<title>Example</title><h1>Stránka nenalezena</h1>", true),
            ("n", "<title>Stránka nebyla nalezena</title>", true),
            ("code", "<title>404</title>", true),
            ("number", "<title>Top 4040 ideas</title><h1>Ideas</h1>", false),
            (
                "body",
                "<title>Blog</title><h1>Blog</h1><p>We fixed the 404 page not found error.</p>",
                false,
            ),
        ];
        for (uq_id, body, _) in bodies {
            add(
                &mut status,
                page(
                    uq_id,
                    "home",
                    SOURCE_A_HREF,
                    &format!("https://example.com/{uq_id}"),
                    200,
                    None,
                ),
                Some(body),
            );
        }
        let ids: Vec<&str> = bodies.iter().map(|(uq_id, _, _)| *uq_id).collect();
        let (issues, _) = observed_access(&status, &key(&status, &ids));
        let flagged: Vec<&str> = issues
            .iter()
            .filter(|issue| issue.kind == AccessKind::SuspectedSoft404)
            .map(|issue| issue.url.as_str())
            .collect();
        assert_eq!(
            flagged,
            [
                "https://example.com/t",
                "https://example.com/h",
                "https://example.com/n",
                "https://example.com/code"
            ]
        );
    }

    #[test]
    fn slow_key_pages_are_above_the_p90_and_three_seconds() {
        let mut status = new_status();
        let mut home = page("home", "", SOURCE_INIT_URL, "https://example.com/", 200, None);
        home.request_time = 0.2;
        add(&mut status, home, None);
        for i in 0..17 {
            let mut fast = page(
                &format!("f{i}"),
                "home",
                SOURCE_A_HREF,
                &format!("https://example.com/f{i}"),
                200,
                None,
            );
            fast.request_time = 0.2;
            add(&mut status, fast, None);
        }
        for (uq_id, seconds) in [("slow", 5.0), ("medium", 2.5)] {
            let mut visit = page(
                uq_id,
                "home",
                SOURCE_A_HREF,
                &format!("https://example.com/{uq_id}"),
                200,
                None,
            );
            visit.request_time = seconds;
            add(&mut status, visit, None);
        }
        let (issues, stats) = observed_access(&status, &key(&status, &["home", "slow", "medium"]));
        // 20 pages, 18 of them at 0.2 s: the p90 of the crawl is 0.2 s, and 2.5 s is above it
        // but not above 3 s.
        assert_eq!(stats.p90_seconds, Some(0.2));
        assert_eq!(kinds(&issues), [("https://example.com/slow", &AccessKind::Slow(5.0))]);

        // A slow site: 3 of 10 pages take 4 s, so the p90 is 4 s and only a slower page stands out.
        let mut slow_site = new_status();
        let timings = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 4.0, 4.0, 4.0, 6.0];
        for (i, seconds) in timings.iter().enumerate() {
            let attr = if i == 0 { SOURCE_INIT_URL } else { SOURCE_A_HREF };
            let mut visit = page(
                &format!("p{i}"),
                "p0",
                attr,
                &format!("https://example.com/p{i}"),
                200,
                None,
            );
            visit.request_time = *seconds;
            add(&mut slow_site, visit, None);
        }
        let (issues, stats) = observed_access(&slow_site, &key(&slow_site, &["p0", "p8", "p9"]));
        assert_eq!(stats.p90_seconds, Some(4.0));
        assert_eq!(kinds(&issues), [("https://example.com/p9", &AccessKind::Slow(6.0))]);
    }

    #[test]
    fn other_urls_are_only_counted() {
        let mut status = new_status();
        add(
            &mut status,
            page("home", "", SOURCE_INIT_URL, "https://example.com/", 200, None),
            None,
        );
        add(
            &mut status,
            page("img", "home", SOURCE_IMG_SRC, "https://example.com/a.png", 403, None),
            None,
        );
        add(
            &mut status,
            page("deep", "home", SOURCE_A_HREF, "https://example.com/deep", 503, None),
            None,
        );
        add(
            &mut status,
            page("deep2", "home", SOURCE_A_HREF, "https://example.com/deep2", -2, None),
            None,
        );
        let mut external = page("ext", "home", SOURCE_A_HREF, "https://other.example/", 500, None);
        external.is_external = true;
        add(&mut status, external, None);

        let (issues, stats) = observed_access(&status, &key(&status, &["home"]));
        assert!(issues.is_empty(), "{issues:?}");
        assert_eq!(stats.key_pages, 1);
        assert_eq!(stats.other_urls, 3, "external URLs are not the site's");
        assert_eq!(
            stats.other_issues,
            BTreeMap::from([("denied", 1), ("server_error", 1), ("transport", 1)])
        );
    }
}
