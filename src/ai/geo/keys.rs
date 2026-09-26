// SiteOne Crawler - the key pages of the AI search readiness report
// (c) Jan Reges <jan.reges@siteone.cz>
//
// The pages whose checks decide the report's statuses: the homepage (with the pages its
// redirects lead to), the pages linked from it, the URLs listed in a crawled sitemap, and the
// pages their redirects lead to. The set does not depend on `--ai-max-pages`, so the deterministic
// verdicts stay the same whatever number of pages the LLM analyzes.

use std::collections::{HashMap, HashSet};

use crate::ai::selection::{build_candidates, url_passes_masks};
use crate::result::status::Status;
use crate::result::visited_url::{SOURCE_A_HREF, SOURCE_INIT_URL, SOURCE_SITEMAP, VisitedUrl};
use crate::types::ContentTypeId;

pub const KEY_PAGES_CAP: usize = 200;

/// A redirect chain longer than this is cut (the crawler's own chains end much sooner).
const MAX_REDIRECT_HOPS: usize = 20;

#[derive(Debug, Clone, PartialEq)]
pub struct KeyPage {
    pub uq_id: String,
    pub url: String,
    pub status_code: i32,
    /// The `build_candidates` score of an HTML 200 page; `None` for a failure or a redirect.
    pub score: Option<f64>,
    /// The initial URL or a page its redirects lead to.
    pub is_homepage: bool,
}

/// The key pages that pass the `--ai-include` / `--ai-exclude` masks, at most `KEY_PAGES_CAP`:
/// the homepage first, then the failed and redirected key URLs by URL, then the HTML 200 key
/// pages in `build_candidates` order (score, then URL). Internal pages only; a URL that answered
/// 404 or 410, and a non-HTML 200 response, is not a page. Empty without an initial URL.
pub fn key_pages(status: &Status, include: &[String], exclude: &[String]) -> Vec<KeyPage> {
    let visited = status.get_visited_urls();
    let by_url = visits_by_url(&visited);
    let candidates = build_candidates(status, include, exclude);
    let scores: HashMap<&str, f64> = candidates
        .candidates
        .iter()
        .map(|candidate| (candidate.uq_id.as_str(), candidate.score))
        .collect();

    let Some(initial) = visited.iter().find(|visit| visit.source_attr == SOURCE_INIT_URL) else {
        return Vec::new();
    };
    let homepage = redirect_chain(initial, &by_url).visits;
    let homepage_ids: HashSet<&str> = homepage.iter().map(|visit| visit.uq_id.as_str()).collect();

    let mut chosen: Vec<&VisitedUrl> = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();
    let linked_or_listed = visited.iter().filter(|visit| {
        (visit.source_attr == SOURCE_A_HREF && homepage_ids.contains(visit.source_uq_id.as_str()))
            || visit.source_attr == SOURCE_SITEMAP
    });
    for start in homepage.iter().copied().chain(linked_or_listed) {
        for visit in redirect_chain(start, &by_url).visits {
            if seen.insert(visit.uq_id.as_str()) {
                chosen.push(visit);
            }
        }
    }

    // An HTML 200 page passed the masks when `build_candidates` scored it; other responses are
    // checked here. The homepage stays a key page even when it answers 404.
    let passes = |visit: &VisitedUrl, is_homepage: bool| {
        let is_page = if visit.status_code == 200 {
            visit.content_type == ContentTypeId::Html
        } else {
            is_homepage || !matches!(visit.status_code, 404 | 410)
        };
        let unmasked = if visit.status_code == 200 {
            scores.contains_key(visit.uq_id.as_str())
        } else {
            url_passes_masks(&visit.url, include, exclude)
        };
        !visit.is_external && visit.is_allowed_for_crawling && is_page && unmasked
    };
    let key_page = |visit: &VisitedUrl| KeyPage {
        uq_id: visit.uq_id.clone(),
        url: visit.url.clone(),
        status_code: visit.status_code,
        score: scores.get(visit.uq_id.as_str()).copied(),
        is_homepage: homepage_ids.contains(visit.uq_id.as_str()),
    };

    let mut first: Vec<KeyPage> = Vec::new();
    let mut failed: Vec<KeyPage> = Vec::new();
    let mut pages: Vec<KeyPage> = Vec::new();
    for visit in chosen {
        let is_homepage = homepage_ids.contains(visit.uq_id.as_str());
        if !passes(visit, is_homepage) {
            continue;
        }
        let page = key_page(visit);
        if is_homepage {
            first.push(page);
        } else if page.score.is_none() {
            failed.push(page);
        } else {
            pages.push(page);
        }
    }
    failed.sort_by(|a, b| a.url.cmp(&b.url));
    pages.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.url.cmp(&b.url))
    });
    first.extend(failed);
    first.extend(pages);
    first.truncate(KEY_PAGES_CAP);
    first
}

/// The URLs a chain of redirects passes through, from `start`; `loops` when a redirect leads
/// back into the chain. The chain ends at a response that is not a redirect, or at a target
/// that was not crawled.
pub(crate) struct RedirectChain<'a> {
    pub visits: Vec<&'a VisitedUrl>,
    /// The target that closes a loop.
    pub loops_to: Option<String>,
    /// The redirects in the chain (a redirect to a URL that was not crawled counts too).
    pub hops: usize,
}

pub(crate) fn redirect_chain<'a>(start: &'a VisitedUrl, by_url: &HashMap<String, &'a VisitedUrl>) -> RedirectChain<'a> {
    let mut chain = RedirectChain {
        visits: vec![start],
        loops_to: None,
        hops: 0,
    };
    let mut current = start;
    while (301..=308).contains(&current.status_code) && chain.hops < MAX_REDIRECT_HOPS {
        let Some(target) = current
            .extras
            .as_ref()
            .and_then(|extras| extras.get("Location"))
            .and_then(|location| url::Url::parse(&current.url).ok()?.join(location).ok())
            .map(|mut target| {
                target.set_fragment(None);
                target.to_string()
            })
        else {
            break;
        };
        chain.hops += 1;
        if chain.visits.iter().any(|visit| normalized_url(&visit.url) == target) {
            chain.loops_to = Some(target);
            break;
        }
        let Some(next) = by_url.get(&target) else {
            break;
        };
        chain.visits.push(next);
        current = next;
    }
    chain
}

/// The visited URLs by their normalized URL (`normalized_url`).
pub(crate) fn visits_by_url(visited: &[VisitedUrl]) -> HashMap<String, &VisitedUrl> {
    visited
        .iter()
        .map(|visit| (normalized_url(&visit.url), visit))
        .collect()
}

/// A URL as the `url` crate serializes it, without a fragment; unparsable URLs stay as they are.
pub(crate) fn normalized_url(url: &str) -> String {
    match url::Url::parse(url) {
        Ok(mut parsed) => {
            parsed.set_fragment(None);
            parsed.to_string()
        }
        Err(_) => url.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::geo::test_support::{add, new_status, page};
    use crate::ai::selection::select_pages;
    use crate::result::visited_url::{SOURCE_A_HREF, SOURCE_IMG_SRC, SOURCE_INIT_URL, SOURCE_REDIRECT, SOURCE_SITEMAP};
    use crate::types::ContentTypeId;

    fn urls(pages: &[KeyPage]) -> Vec<&str> {
        pages.iter().map(|page| page.url.as_str()).collect()
    }

    /// Homepage → 30 linked pages; a sitemap listing 20 more; 30 deeper pages linked from
    /// `/l0`. Stored in the given order.
    fn site(reversed: bool) -> Status {
        let mut visits = vec![page("home", "", SOURCE_INIT_URL, "https://example.com/", 200, None)];
        for i in 0..30 {
            visits.push(page(
                &format!("l{i}"),
                "home",
                SOURCE_A_HREF,
                &format!("https://example.com/l{i}"),
                200,
                None,
            ));
        }
        for i in 0..20 {
            visits.push(page(
                &format!("s{i}"),
                "sitemap",
                SOURCE_SITEMAP,
                &format!("https://example.com/s{i}"),
                200,
                None,
            ));
        }
        for i in 0..30 {
            visits.push(page(
                &format!("d{i}"),
                "l0",
                SOURCE_A_HREF,
                &format!("https://example.com/d{i}"),
                200,
                None,
            ));
        }
        if reversed {
            visits[1..].reverse();
        }
        let mut status = new_status();
        for visited in visits {
            add(&mut status, visited, None);
        }
        status
    }

    #[test]
    fn key_pages_do_not_depend_on_the_ai_page_cap_or_the_crawl_order() {
        let status = site(false);
        assert_ne!(
            select_pages(&status, &[], &[], 10).selected.len(),
            select_pages(&status, &[], &[], 100).selected.len()
        );
        let key = key_pages(&status, &[], &[]);
        assert_eq!(key.len(), 1 + 30 + 20);
        assert_eq!(key[0].url, "https://example.com/");
        assert!(key[0].is_homepage && key[1..].iter().all(|page| !page.is_homepage));
        assert!(
            key.iter().all(|page| !page.url.contains("/d")),
            "deeper pages are not key"
        );
        assert_eq!(urls(&key), urls(&key_pages(&site(true), &[], &[])));
    }

    #[test]
    fn the_homepage_includes_its_redirects_and_links_count_from_the_final_page() {
        let mut status = new_status();
        add(
            &mut status,
            page(
                "init",
                "",
                SOURCE_INIT_URL,
                "http://example.com/",
                301,
                Some("https://example.com/"),
            ),
            None,
        );
        add(
            &mut status,
            page("home", "init", SOURCE_REDIRECT, "https://example.com/", 200, None),
            None,
        );
        add(
            &mut status,
            page("about", "home", SOURCE_A_HREF, "https://example.com/about", 200, None),
            None,
        );
        add(
            &mut status,
            page("deep", "about", SOURCE_A_HREF, "https://example.com/deep", 200, None),
            None,
        );

        let key = key_pages(&status, &[], &[]);
        assert_eq!(
            urls(&key),
            [
                "http://example.com/",
                "https://example.com/",
                "https://example.com/about"
            ]
        );
        assert!(key[0].is_homepage && key[1].is_homepage && !key[2].is_homepage);
        assert_eq!(key[0].status_code, 301);
        assert_eq!(key[0].score, None);
        assert!(key[1].score.is_some());
    }

    #[test]
    fn failures_and_redirect_targets_are_key_but_missing_pages_and_assets_are_not() {
        let mut status = new_status();
        add(
            &mut status,
            page("home", "", SOURCE_INIT_URL, "https://example.com/", 200, None),
            None,
        );
        let linked = [
            ("forbidden", "https://example.com/forbidden", 403),
            ("down", "https://example.com/down", 503),
            ("timeout", "https://example.com/timeout", -2),
            ("missing", "https://example.com/missing", 404),
            ("gone", "https://example.com/gone", 410),
            ("ok", "https://example.com/ok", 200),
        ];
        for (uq_id, url, code) in linked {
            add(&mut status, page(uq_id, "home", SOURCE_A_HREF, url, code, None), None);
        }
        add(
            &mut status,
            page(
                "moved",
                "home",
                SOURCE_A_HREF,
                "https://example.com/moved",
                301,
                Some("/moved/"),
            ),
            None,
        );
        add(
            &mut status,
            page(
                "moved2",
                "moved",
                SOURCE_REDIRECT,
                "https://example.com/moved/",
                200,
                None,
            ),
            None,
        );
        let mut pdf = page("pdf", "home", SOURCE_A_HREF, "https://example.com/doc.pdf", 200, None);
        pdf.content_type = ContentTypeId::Document;
        add(&mut status, pdf, None);
        let mut external = page("ext", "home", SOURCE_A_HREF, "https://other.example/", 403, None);
        external.is_external = true;
        add(&mut status, external, None);
        add(
            &mut status,
            page("img", "home", SOURCE_IMG_SRC, "https://example.com/a.png", 403, None),
            None,
        );

        assert_eq!(
            urls(&key_pages(&status, &[], &[])),
            [
                "https://example.com/",
                // Failures and redirects, by URL.
                "https://example.com/down",
                "https://example.com/forbidden",
                "https://example.com/moved",
                "https://example.com/timeout",
                // HTML 200 pages, by score and URL: the redirect target is one click deeper.
                "https://example.com/ok",
                "https://example.com/moved/",
            ]
        );
    }

    #[test]
    fn key_pages_respect_the_masks_and_the_cap() {
        let mut status = new_status();
        add(
            &mut status,
            page("home", "", SOURCE_INIT_URL, "https://example.com/", 200, None),
            None,
        );
        add(
            &mut status,
            page("p1", "home", SOURCE_A_HREF, "https://example.com/private/a", 200, None),
            None,
        );
        add(
            &mut status,
            page("p2", "home", SOURCE_A_HREF, "https://example.com/private/b", 503, None),
            None,
        );
        add(
            &mut status,
            page("pub", "home", SOURCE_A_HREF, "https://example.com/public", 200, None),
            None,
        );
        assert_eq!(
            urls(&key_pages(&status, &[], &["/private/".to_string()])),
            ["https://example.com/", "https://example.com/public"]
        );

        let mut big = new_status();
        add(
            &mut big,
            page("home", "", SOURCE_INIT_URL, "https://example.com/", 200, None),
            None,
        );
        for i in 0..250 {
            add(
                &mut big,
                page(
                    &format!("p{i}"),
                    "home",
                    SOURCE_A_HREF,
                    &format!("https://example.com/p{i:03}"),
                    200,
                    None,
                ),
                None,
            );
        }
        let key = key_pages(&big, &[], &[]);
        assert_eq!(key.len(), KEY_PAGES_CAP);
        assert_eq!(key[0].url, "https://example.com/");
        assert_eq!(key[1].url, "https://example.com/p000");
    }

    #[test]
    fn no_initial_url_means_no_key_pages() {
        let mut status = new_status();
        add(
            &mut status,
            page("s", "sitemap", SOURCE_SITEMAP, "https://example.com/s", 200, None),
            None,
        );
        assert!(key_pages(&status, &[], &[]).is_empty());
    }
}
