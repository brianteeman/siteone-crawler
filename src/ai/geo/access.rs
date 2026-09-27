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

use crate::ai::blocks::blocks_from_html;
use crate::ai::geo::keys::{KeyPage, redirect_chain, visits_by_url};
use crate::ai::geo::render::own_text_chars;
use crate::result::basic_stats::percentile;
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
    /// Any other 4xx: the homepage (or where its redirects lead) not found or gone, or a key page
    /// refused as a bad request, unavailable for legal reasons, …
    ClientError(i32),
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
            AccessKind::ClientError(_) => "client_error",
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
pub const CHALLENGE_TITLE_MARKERS: &[&str] = &[
    "just a moment",
    "attention required",
    "access denied",
    "captcha",
    "checking your browser",
];
/// HTML fragments (lowercase) of bot-challenge pages: Cloudflare's challenge script (also on a
/// localized page) and the old interstitial text. Not `cf-chl-`, which also names the Turnstile
/// widget of ordinary forms, and not a "captcha" or "access denied" in the page text.
pub const CHALLENGE_HTML_MARKERS: &[&str] = &["_cf_chl_opt", "checking your browser"];
/// A page is suspected of being a challenge only with less visible text than this: challenge and
/// block pages are tiny, an article about captchas is not.
pub const CHALLENGE_MAX_TEXT_CHARS: usize = 1_000;
/// Title or H1 fragments (lowercase) of "not found" pages; "404" counts as a number of its own.
pub const SOFT_404_MARKERS: &[&str] = &["not found", "nenalezena", "nebyla nalezena"];
/// A page is suspected of being a soft 404 only with less text of its own (Main region) than
/// this: a "not found" page is short whatever its site navigation, a page about 404 errors is not.
pub const SOFT_404_MAX_TEXT_CHARS: usize = 1_000;
/// A key page slower than the crawl's p90 is reported only above this many seconds.
pub const SLOW_SECONDS: f64 = 3.0;

static TITLE_SELECTOR: Lazy<Selector> = Lazy::new(|| Selector::parse("title").unwrap());
static H1_SELECTOR: Lazy<Selector> = Lazy::new(|| Selector::parse("h1").unwrap());
static NUMBER_404: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b404\b").unwrap());

/// The response time of a visit without the browser rendering that `--browser` adds to a rendered
/// page (a failed render adds none).
fn retrieval_time(status: &Status, visit: &VisitedUrl) -> f64 {
    match status.get_browser_diagnostics(&visit.uq_id) {
        Some(diagnostics) if diagnostics.render_error.is_none() => {
            (visit.request_time - diagnostics.render_total_ms as f64 / 1000.0).max(0.0)
        }
        _ => visit.request_time,
    }
}

/// The access issues of the key pages, in key-page order, and the counts of the other internal
/// URLs. Challenge and soft-404 checks need the stored bodies of the key pages.
pub fn observed_access(status: &Status, key: &[KeyPage]) -> (Vec<AccessIssue>, AccessStats) {
    let visited = status.get_visited_urls();
    let by_url = visits_by_url(&visited);
    let by_uq_id: HashMap<&str, &VisitedUrl> = visited.iter().map(|visit| (visit.uq_id.as_str(), visit)).collect();
    // With --browser a response time includes the rendering, which crawlers without a browser do
    // not wait for: they are compared by the HTTP time alone.
    let rendered = visited
        .iter()
        .any(|visit| status.get_browser_diagnostics(&visit.uq_id).is_some());
    let p90 = if rendered {
        let mut html_times: Vec<f64> = visited
            .iter()
            .filter(|visit| visit.content_type == ContentTypeId::Html && visit.status_code == 200)
            .map(|visit| retrieval_time(status, visit))
            .collect();
        if html_times.is_empty() {
            html_times = visited.iter().map(|visit| retrieval_time(status, visit)).collect();
        }
        percentile(&mut html_times, 90)
    } else {
        status.get_basic_stats().total_requests_times_p90
    };
    let mut stats = AccessStats {
        p90_seconds: (!visited.is_empty()).then_some(p90),
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
        let seconds = if rendered {
            retrieval_time(status, visit)
        } else {
            visit.request_time
        };
        if let Some(p90) = stats.p90_seconds
            && seconds > p90
            && seconds > SLOW_SECONDS
        {
            issues.push(AccessIssue {
                url: visit.url.clone(),
                kind: AccessKind::Slow(seconds),
                detail: format!("{:.1} s (p90 of the crawl {:.1} s)", seconds, p90),
            });
        }
    }

    let key_ids: HashSet<&str> = key.iter().map(|page| page.uq_id.as_str()).collect();
    for visit in visited
        .iter()
        .filter(|visit| !visit.is_external && !key_ids.contains(visit.uq_id.as_str()))
    {
        stats.other_urls += 1;
        // A broken link (4xx) is no access issue of a page that exists.
        if let Some(issue) = response_issue(visit, &by_url)
            && !matches!(issue.kind, AccessKind::ClientError(_))
        {
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
        // Only the homepage keeps a 404 or 410 among the key pages (see `key_pages`).
        400..=499 => (AccessKind::ClientError(code), format!("HTTP {}", code)),
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

/// A 200 HTML page that looks like a bot challenge (a marker on a short page), or else like a
/// "not found" page.
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

    let visible_chars = || {
        blocks_from_html(body)
            .iter()
            .map(|block| block.text.chars().count())
            .sum::<usize>()
    };
    let kind = if (CHALLENGE_TITLE_MARKERS
        .iter()
        .any(|marker| title_lower.contains(marker))
        || CHALLENGE_HTML_MARKERS.iter().any(|marker| html_lower.contains(marker)))
        && visible_chars() < CHALLENGE_MAX_TEXT_CHARS
    {
        AccessKind::SuspectedChallenge
    } else if [&title, &h1].iter().any(|text| {
        let lower = text.to_lowercase();
        SOFT_404_MARKERS.iter().any(|marker| lower.contains(marker)) || NUMBER_404.is_match(&lower)
    }) && own_text_chars(body) < SOFT_404_MAX_TEXT_CHARS
    {
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
            ("u400", 400),
            ("u451", 451),
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
                ("https://example.com/u404", &AccessKind::ClientError(404)),
                ("https://example.com/u400", &AccessKind::ClientError(400)),
                ("https://example.com/u451", &AccessKind::ClientError(451)),
            ]
        );
        assert!(issues[6].detail.contains("TIMEOUT"), "{}", issues[6].detail);
        assert_eq!(stats.key_pages, all.len());
    }

    #[test]
    fn a_homepage_that_is_not_found_is_an_access_issue() {
        // The initial URL answers 404.
        let mut status = new_status();
        add(
            &mut status,
            page("home", "", SOURCE_INIT_URL, "https://example.com/", 404, None),
            Some("<title>Page not found</title><h1>Not found</h1>"),
        );
        let key = crate::ai::geo::keys::key_pages(&status, &[], &[]);
        let (issues, _) = observed_access(&status, &key);
        assert_eq!(
            kinds(&issues),
            [("https://example.com/", &AccessKind::ClientError(404))]
        );
        assert_eq!(issues[0].detail, "HTTP 404");

        // The initial URL redirects to a page that is gone.
        let mut status = new_status();
        add(
            &mut status,
            page(
                "init",
                "",
                SOURCE_INIT_URL,
                "https://example.com/",
                301,
                Some("https://example.com/en/"),
            ),
            None,
        );
        add(
            &mut status,
            page("home", "init", SOURCE_REDIRECT, "https://example.com/en/", 410, None),
            None,
        );
        let key = crate::ai::geo::keys::key_pages(&status, &[], &[]);
        let (issues, _) = observed_access(&status, &key);
        assert_eq!(
            kinds(&issues),
            [("https://example.com/en/", &AccessKind::ClientError(410))]
        );
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
    fn challenge_pages_are_suspected_by_their_markers_on_short_pages() {
        let mut status = new_status();
        add(
            &mut status,
            page("home", "", SOURCE_INIT_URL, "https://example.com/", 200, None),
            None,
        );
        let article = format!(
            "<html><head><title>How to add a CAPTCHA to your contact form</title></head><body><h1>CAPTCHA</h1><p>{}</p></body></html>",
            "A long guide about forms. ".repeat(60)
        );
        let bodies = [
            ("cf", "<html><head><title>Just a moment...</title></head><body><script>window._cf_chl_opt={}</script></body></html>".to_string(), true),
            // A localized Cloudflare challenge: its script marker is the same.
            ("opt", "<html><head><title>Chvíli strpení…</title></head><body><script>window._cf_chl_opt={cvId:'3'}</script></body></html>".to_string(), true),
            ("attention", "<html><head><title>Attention Required! | Cloudflare</title></head><body><h1>Sorry, you have been blocked</h1></body></html>".to_string(), true),
            ("denied", "<html><head><title>Access Denied</title></head><body>Reference #18</body></html>".to_string(), true),
            ("check", "<html><head><title>Example</title></head><body><p>Checking your browser before accessing example.com.</p></body></html>".to_string(), true),
            ("captcha", "<html><head><title>Please solve the CAPTCHA</title></head><body></body></html>".to_string(), true),
            // A contact form with reCAPTCHA is a normal page.
            ("form", r#"<html><head><title>Contact</title></head><body><form><div class="g-recaptcha"></div><p>This site is protected by reCAPTCHA. Access denied pages are rare.</p></form></body></html>"#.to_string(), false),
            // So is a login form with Cloudflare Turnstile, whose widget ids start with "cf-chl-".
            ("turnstile", r#"<html><head><title>Log in</title></head><body><form><input type="hidden" id="cf-chl-widget-abc12_response"><button>Log in</button></form></body></html>"#.to_string(), false),
            // And a long page about captchas.
            ("article", article, false),
        ];
        for (uq_id, body, _) in &bodies {
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
        let article = "<p>Broken links and redirect chains waste crawl budget and frustrate visitors.</p>".repeat(20);
        let topic =
            format!("<title>Broken Link Checker</title><main><h1>Redirect and 404 Analysis</h1>{article}</main>");
        let guide = format!("<title>How to fix a page not found error</title><main><h1>Guide</h1>{article}</main>");
        let menu = format!(
            "<title>Page not found</title><nav>{}</nav><main><h1>Oops</h1><p>Sorry.</p></main>",
            "<a href=\"/x\">A long menu item of the site</a> ".repeat(60)
        );
        // In browser mode the rendered page holds the cookie dialog too.
        let consent = format!(
            "<title>Page not found</title><main><h1>Oops</h1></main><div role=\"dialog\">{}</div>",
            "<p>We use cookies to measure traffic and to personalise content and ads.</p>".repeat(30)
        );
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
            // A page about 404 errors has text of its own; a real "not found" page does not,
            // whatever the size of its site navigation.
            ("topic", topic.as_str(), false),
            ("guide", guide.as_str(), false),
            ("menu", menu.as_str(), true),
            ("consent", consent.as_str(), true),
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
                "https://example.com/code",
                "https://example.com/menu",
                "https://example.com/consent"
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
    fn in_browser_mode_the_rendering_time_is_no_retrieval_time() {
        use crate::browser::diagnostics::BrowserDiagnostics;
        let mut status = new_status();
        // Ten rendered pages: 50 ms over HTTP plus 3–4.5 s in the browser.
        for i in 0..10 {
            let attr = if i == 0 { SOURCE_INIT_URL } else { SOURCE_A_HREF };
            let render = 3.0 + i as f64 * 0.15;
            let mut visit = page(
                &format!("p{i}"),
                "p0",
                attr,
                &format!("https://example.com/p{i}"),
                200,
                None,
            );
            visit.request_time = 0.05 + render;
            add(&mut status, visit, None);
            status.add_browser_diagnostics(
                &format!("p{i}"),
                BrowserDiagnostics {
                    render_total_ms: (render * 1000.0) as u64,
                    ..Default::default()
                },
            );
        }
        // A failed render keeps the HTTP time alone, which is really slow.
        let mut failed = page("failed", "p0", SOURCE_A_HREF, "https://example.com/failed", 200, None);
        failed.request_time = 6.0;
        add(&mut status, failed, None);
        status.add_browser_diagnostics(
            "failed",
            BrowserDiagnostics {
                render_error: Some("navigation failed".to_string()),
                ..Default::default()
            },
        );
        let (issues, stats) = observed_access(&status, &key(&status, &["p0", "p8", "p9", "failed"]));
        assert_eq!(stats.p90_seconds, Some(0.05), "the p90 of the HTTP times");
        assert_eq!(kinds(&issues), [("https://example.com/failed", &AccessKind::Slow(6.0))]);
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
        // A broken link is no access issue.
        add(
            &mut status,
            page("gone", "deep", SOURCE_A_HREF, "https://example.com/gone", 404, None),
            None,
        );
        let mut external = page("ext", "home", SOURCE_A_HREF, "https://other.example/", 500, None);
        external.is_external = true;
        add(&mut status, external, None);

        let (issues, stats) = observed_access(&status, &key(&status, &["home"]));
        assert!(issues.is_empty(), "{issues:?}");
        assert_eq!(stats.key_pages, 1);
        assert_eq!(stats.other_urls, 4, "external URLs are not the site's");
        assert_eq!(
            stats.other_issues,
            BTreeMap::from([("denied", 1), ("server_error", 1), ("transport", 1)])
        );
    }
}
