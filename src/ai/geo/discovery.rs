// SiteOne Crawler - discovery and freshness of the AI search readiness report
// (c) Jan Reges <jan.reges@siteone.cz>
//
// How engines find the pages and learn that they changed: the sitemaps declared in robots.txt or
// crawled (their `<loc>` and `<lastmod>` entries), key pages the sitemaps leave out, listed URLs
// that are no indexable pages (not 200, `noindex`, or canonicalized elsewhere), `lastmod` values
// that cannot be trusted (all identical, or in the future), the `Last-Modified` coverage of the
// key pages, and hreflang alternates of the key pages that lead to a URL that was not crawled or
// did not answer 200. Everything comes from the crawl stored in `Status`; nothing is fetched here.
// When the site has no sitemap and the crawl covered all of it, the kit proposes one (Phase 2).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use chrono::{DateTime, Duration, NaiveDate, Utc};
use once_cell::sync::Lazy;
use quick_xml::Reader;
use quick_xml::events::Event;
use scraper::{Html, Selector};

use crate::ai::geo::controls::{bing_policy, canonical_elsewhere, google_policy, sources};
use crate::ai::geo::keys::{KeyPage, normalized_url, redirect_chain, visits_by_url};
use crate::ai::geo::robots_ai::{normalize_path, robots_of, state_is_known};
use crate::ai::geo::signals::{LastModifiedCoverage, last_modified_coverage};
use crate::content_processor::xml_processor::XmlProcessor;
use crate::export::sitemap_exporter::lastmod_from_headers;
use crate::result::status::{RobotsFetchState, Status};
use crate::result::visited_url::{SOURCE_A_HREF, SOURCE_REDIRECT, SOURCE_SITEMAP, VisitedUrl};
use crate::types::ContentTypeId;

/// A `lastmod` more than this far after the crawl is in the future (the sitemap exporter's
/// tolerance for a `Last-Modified` without a response `Date`).
const LASTMOD_FUTURE_TOLERANCE_HOURS: i64 = 24;

/// Identical `lastmod` values are a pattern (the generation time) from this many on; two pages
/// may well have changed together.
const LASTMOD_IDENTICAL_MIN: usize = 3;

/// A sitemap file lists at most this many URLs (sitemaps.org).
const SITEMAP_MAX_URLS: usize = 50_000;

static ALTERNATE_SELECTOR: Lazy<Selector> = Lazy::new(|| Selector::parse("link[rel][hreflang][href]").unwrap());

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SitemapKind {
    /// `<urlset>`: page URLs.
    UrlSet,
    /// `<sitemapindex>`: the URLs of other sitemaps.
    Index,
}

/// One `<url>` or `<sitemap>` element.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SitemapEntry {
    pub loc: String,
    pub lastmod: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedSitemap {
    pub kind: SitemapKind,
    pub entries: Vec<SitemapEntry>,
}

/// The `lastmod` values of a sitemap.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LastmodStats {
    pub entries: usize,
    /// Entries with a `lastmod`.
    pub with_lastmod: usize,
    /// `lastmod` values that are no W3C datetime.
    pub invalid: usize,
    /// `lastmod` values more than a day after the crawl.
    pub future: usize,
    /// At least `LASTMOD_IDENTICAL_MIN` valid `lastmod` values, all the same moment: likely the
    /// generation time.
    pub all_identical: bool,
}

impl LastmodStats {
    /// Engines ignore a `lastmod` that is not accurate.
    pub fn suspicious(&self) -> bool {
        self.all_identical || self.future > 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SitemapState {
    Parsed {
        kind: SitemapKind,
        entries: usize,
        lastmod: LastmodStats,
    },
    /// Crawled, but not a valid sitemap: why.
    Malformed(String),
    /// Crawled with another status than 200.
    Failed(i32),
    /// Declared in robots.txt or listed in a sitemap index, but not crawled.
    NotCrawled,
    /// Declared in robots.txt (or listed in an index) with a URL that is not an absolute http(s)
    /// URL, which engines do not use (Google requires a fully-qualified sitemap URL).
    NotAbsolute,
    /// Redirects (through a crawled chain) to this URL.
    Redirected(String),
    /// A feed or text sitemap ("RSS", "Atom", "text"), which engines accept but this check does
    /// not read.
    Unsupported(&'static str),
}

/// A sitemap declared in robots.txt, crawled, or listed in a crawled sitemap index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SitemapFile {
    pub url: String,
    /// Declared in robots.txt.
    pub declared: bool,
    pub state: SitemapState,
}

/// A URL listed in a sitemap that is no indexable page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListedIssue {
    pub url: String,
    pub reason: ListedReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ListedReason {
    /// Answered another status than 200 (a redirect too: a sitemap should list final URLs).
    Status(i32),
    /// `noindex` for Google or Bing.
    NoIndex,
    /// The canonical URL named by the page.
    CanonicalElsewhere(String),
}

/// An hreflang alternate of a key page that leads nowhere useful.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HreflangIssue {
    pub page: String,
    pub lang: String,
    pub target: String,
    pub problem: HreflangProblem,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HreflangProblem {
    /// The target answered another status than 200.
    Status(i32),
    /// The target on the same site (with or without `www.`) was not crawled.
    NotCrawled,
    /// The target is on another site, which the crawl does not cover: not checked.
    OtherSite,
}

/// How the crawl ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrawlEnd {
    /// The queue ran empty.
    Complete,
    /// Stopped by Ctrl+C or a `stop` on stdin.
    Interrupted,
    /// Stopped taking new URLs at `--max-visited-urls`.
    UrlLimit,
    /// Dropped URLs because the queue was full (`--max-queue-length`).
    QueueLimit,
}

/// How much of the site the crawl covered: a sitemap is proposed only for a crawl of all of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrawlScope {
    pub start_url: String,
    pub single_page: bool,
    pub end: CrawlEnd,
    pub max_visited_urls: i64,
    /// `--max-depth` (0 = no limit).
    pub max_depth: i64,
    /// `--include-regex` or `--ignore-regex` limited the URLs crawled.
    pub url_filters: bool,
}

/// A page of the proposed sitemap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposedUrl {
    pub url: String,
    /// From a plausible `Last-Modified` only (the sitemap exporter's filter).
    pub lastmod: Option<String>,
}

/// The crawled HTML 200 pages the proposed sitemap leaves out, by reason.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LeftOut {
    /// `noindex` for Google or Bing (meta tags or X-Robots-Tag).
    pub noindex: usize,
    /// A canonical URL elsewhere.
    pub canonical_elsewhere: usize,
    /// Disallowed for Googlebot or Bingbot by the origin's robots.txt.
    pub blocked: usize,
    /// On another origin than the homepage (a sitemap lists the URLs of its own origin).
    pub other_origin: usize,
}

/// The sitemap the kit proposes: the canonical, indexable HTML 200 pages of the homepage's origin
/// that a crawl of the whole site found.
#[derive(Debug, Clone, PartialEq)]
pub struct SitemapProposal {
    /// `scheme://host[:port]` of the homepage, where the sitemap belongs.
    pub origin: String,
    /// In the sitemap exporter's order: fewer slashes first, then alphabetically.
    pub urls: Vec<ProposedUrl>,
    pub left_out: LeftOut,
    pub scope: CrawlScope,
}

/// The discovery and freshness checks.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Discovery {
    /// By URL.
    pub sitemaps: Vec<SitemapFile>,
    /// The distinct URLs listed by the parsed `<urlset>` sitemaps.
    pub listed_urls: usize,
    /// Listed URLs that were not crawled, so not checked.
    pub listed_not_crawled: usize,
    /// By URL.
    pub listed_issues: Vec<ListedIssue>,
    /// Indexable key pages (200 HTML, no `noindex`, no canonical elsewhere) compared with the
    /// sitemaps; 0 — not assessed — when no `<urlset>` was parsed or a known sitemap could not be
    /// read, since the comparison would then be incomplete.
    pub key_pages_compared: usize,
    /// Of those, the ones no sitemap lists, in key-page order.
    pub missing_from_sitemaps: Vec<String>,
    pub hreflang_issues: Vec<HreflangIssue>,
    pub last_modified: LastModifiedCoverage,
    /// The sitemap proposed for a site without one (see `sitemap_proposal`).
    pub sitemap_proposal: Option<SitemapProposal>,
    /// Why no sitemap is proposed although the site has none (a `sitemap_proposal` error other
    /// than `sitemap_exists`).
    pub proposal_withheld: Option<&'static str>,
}

/// Parses a sitemap: a `<urlset>` or a `<sitemapindex>` root (with any namespace prefix), the
/// `<loc>` and `<lastmod>` of each `<url>` / `<sitemap>` (elements of extensions, such as image
/// locations, are ignored). Invalid XML (an undefined entity, an invalid character reference, a
/// second root element, text outside the root), another root element and a truncated file are
/// errors.
pub fn parse_sitemap(xml: &str) -> Result<ParsedSitemap, String> {
    #[derive(Clone, Copy)]
    enum Field {
        Loc,
        Lastmod,
    }
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut kind: Option<SitemapKind> = None;
    let mut depth = 0usize;
    let mut entries = Vec::new();
    let mut entry: Option<SitemapEntry> = None;
    let mut field: Option<Field> = None;
    let mut text = String::new();
    let mut root_closed = false;
    loop {
        let event = reader
            .read_event_into(&mut buf)
            .map_err(|error| format!("invalid XML at byte {}: {}", reader.error_position(), error))?;
        match event {
            Event::Start(_) | Event::Empty(_) | Event::Text(_) | Event::CData(_) | Event::GeneralRef(_)
                if root_closed =>
            {
                return Err(format!(
                    "invalid XML at byte {}: content after the root element",
                    reader.buffer_position()
                ));
            }
            Event::Text(_) | Event::CData(_) | Event::GeneralRef(_) if depth == 0 => {
                return Err(format!(
                    "invalid XML at byte {}: text outside the root element",
                    reader.buffer_position()
                ));
            }
            Event::Start(ref element) | Event::Empty(ref element) => {
                let is_empty = matches!(event, Event::Empty(_));
                depth += 1;
                let name = element.local_name();
                match (kind, depth, name.as_ref()) {
                    (None, 1, b"urlset") => kind = Some(SitemapKind::UrlSet),
                    (None, 1, b"sitemapindex") => kind = Some(SitemapKind::Index),
                    (None, _, other) => {
                        return Err(format!(
                            "not a sitemap: the root element is <{}>",
                            String::from_utf8_lossy(other)
                        ));
                    }
                    (Some(SitemapKind::UrlSet), 2, b"url") | (Some(SitemapKind::Index), 2, b"sitemap") => {
                        entry = Some(SitemapEntry::default());
                    }
                    (Some(_), 3, b"loc") if entry.is_some() => field = Some(Field::Loc),
                    (Some(_), 3, b"lastmod") if entry.is_some() => field = Some(Field::Lastmod),
                    _ => {}
                }
                text.clear();
                if is_empty {
                    field = None;
                    depth -= 1;
                    root_closed = depth == 0;
                }
            }
            Event::Text(ref content) if field.is_some() => {
                text.push_str(&content.decode().map_err(|error| error.to_string())?);
            }
            Event::CData(ref content) if field.is_some() => text.push_str(&String::from_utf8_lossy(content)),
            Event::GeneralRef(ref reference) => {
                // Only the predefined entities and valid character references are text: a sitemap
                // declares no entities of its own.
                let resolved = match reference.resolve_char_ref() {
                    Ok(Some(ch)) if is_xml_char(ch) => Some(ch.to_string()),
                    Ok(None) => reference
                        .decode()
                        .ok()
                        .and_then(|name| quick_xml::escape::resolve_xml_entity(&name).map(str::to_string)),
                    _ => None,
                };
                let Some(value) = resolved else {
                    return Err(format!(
                        "invalid XML at byte {}: an undefined entity or an invalid character reference &{};",
                        reader.buffer_position(),
                        String::from_utf8_lossy(reference)
                    ));
                };
                if field.is_some() {
                    text.push_str(&value);
                }
            }
            Event::End(ref element) => {
                let name = element.local_name();
                match (depth, name.as_ref(), field.take(), entry.as_mut()) {
                    (3, b"loc", Some(Field::Loc), Some(entry)) => entry.loc = text.trim().to_string(),
                    (3, b"lastmod", Some(Field::Lastmod), Some(entry)) if !text.trim().is_empty() => {
                        entry.lastmod = Some(text.trim().to_string());
                    }
                    (2, b"url" | b"sitemap", _, _) => {
                        if let Some(done) = entry.take().filter(|done| !done.loc.is_empty()) {
                            entries.push(done);
                        }
                    }
                    _ => {}
                }
                depth = depth.saturating_sub(1);
                root_closed = depth == 0;
            }
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }
    if depth > 0 {
        return Err("the file ends inside an element".to_string());
    }
    let kind = kind.ok_or_else(|| "no <urlset> or <sitemapindex> root element".to_string())?;
    Ok(ParsedSitemap { kind, entries })
}

/// A character XML 1.0 allows in a document (`Char`).
fn is_xml_char(ch: char) -> bool {
    matches!(ch, '\u{9}' | '\u{a}' | '\u{d}' | '\u{20}'..='\u{d7ff}' | '\u{e000}'..='\u{fffd}' | '\u{10000}'..)
}

/// The `lastmod` statistics of a sitemap's entries; `now` is when the crawl ran.
pub fn lastmod_stats(entries: &[SitemapEntry], now: DateTime<Utc>) -> LastmodStats {
    let mut stats = LastmodStats {
        entries: entries.len(),
        ..LastmodStats::default()
    };
    let mut moments: Vec<DateTime<Utc>> = Vec::new();
    for lastmod in entries.iter().filter_map(|entry| entry.lastmod.as_deref()) {
        stats.with_lastmod += 1;
        match parse_w3c_datetime(lastmod) {
            Some(moment) => {
                if moment > now + Duration::hours(LASTMOD_FUTURE_TOLERANCE_HOURS) {
                    stats.future += 1;
                }
                moments.push(moment);
            }
            None => stats.invalid += 1,
        }
    }
    stats.all_identical = moments.len() >= LASTMOD_IDENTICAL_MIN && moments.iter().all(|moment| *moment == moments[0]);
    stats
}

/// A W3C datetime (`YYYY`, `YYYY-MM`, `YYYY-MM-DD`, `YYYY-MM-DDThh:mmTZD`, with seconds and
/// fractions too) as a moment in UTC; a date alone is its midnight in UTC.
fn parse_w3c_datetime(value: &str) -> Option<DateTime<Utc>> {
    let value = value.trim();
    let zoned = match value.strip_suffix(['Z', 'z']) {
        Some(rest) => format!("{rest}+00:00"),
        None => value.to_string(),
    };
    if let Ok(moment) = DateTime::parse_from_rfc3339(&zoned) {
        return Some(moment.with_timezone(&Utc));
    }
    if let Ok(moment) = DateTime::parse_from_str(&zoned, "%Y-%m-%dT%H:%M%:z") {
        return Some(moment.with_timezone(&Utc));
    }
    let day = match value.len() {
        4 => format!("{value}-01-01"),
        7 => format!("{value}-01"),
        _ => value.to_string(),
    };
    NaiveDate::parse_from_str(&day, "%Y-%m-%d")
        .ok()?
        .and_hms_opt(0, 0, 0)
        .map(|moment| moment.and_utc())
}

/// The discovery checks over the crawl: the sitemaps `declared` in robots.txt, crawled, or listed
/// in a crawled index; the URLs they list; the key pages they leave out; the hreflang alternates
/// and the `Last-Modified` coverage of the key pages. `now` is when the crawl ran.
pub fn discovery(status: &Status, key: &[KeyPage], declared: &[String], now: DateTime<Utc>) -> Discovery {
    let visited = status.get_visited_urls();
    let by_url = visits_by_url(&visited);
    let by_uq_id: HashMap<&str, &VisitedUrl> = visited.iter().map(|visit| (visit.uq_id.as_str(), visit)).collect();
    let declared: HashSet<String> = declared.iter().map(|url| normalized_url(url.trim())).collect();
    let listing: HashSet<&str> = visited
        .iter()
        .filter(|visit| visit.source_attr == SOURCE_SITEMAP)
        .map(|visit| visit.source_uq_id.as_str())
        .collect();

    // Sitemaps by normalized URL, so they come out sorted.
    let mut files: BTreeMap<String, SitemapFile> = BTreeMap::new();
    let mut children: Vec<String> = Vec::new();
    let mut listed: BTreeSet<String> = BTreeSet::new();
    for visit in &visited {
        let url = normalized_url(&visit.url);
        let is_declared = declared.contains(&url);
        // A file that should be a sitemap is reported even when it is none; other XML is not.
        let expected = is_declared || looks_like_sitemap(&visit.url) || listing.contains(visit.uq_id.as_str());
        if visit.content_type != ContentTypeId::Xml && !expected {
            continue;
        }
        let state = if (301..=308).contains(&visit.status_code) {
            match redirect_target(visit, &by_url) {
                Some(target) => SitemapState::Redirected(target),
                None => SitemapState::Failed(visit.status_code),
            }
        } else if visit.status_code != 200 {
            SitemapState::Failed(visit.status_code)
        } else {
            let body = status.get_url_body_text(&visit.uq_id).unwrap_or_default();
            let parsed = if body.trim().is_empty() {
                Err("empty file".to_string())
            } else {
                parse_sitemap(&body)
            };
            match parsed {
                Ok(parsed) => {
                    let locs = parsed.entries.iter().map(|entry| entry.loc.clone());
                    match parsed.kind {
                        SitemapKind::Index => children.extend(locs),
                        SitemapKind::UrlSet => listed.extend(locs.map(|loc| normalized_url(&loc))),
                    }
                    SitemapState::Parsed {
                        kind: parsed.kind,
                        entries: parsed.entries.len(),
                        lastmod: lastmod_stats(&parsed.entries, now),
                    }
                }
                Err(error) => match other_format(&body) {
                    Some(format) => SitemapState::Unsupported(format),
                    None => SitemapState::Malformed(error),
                },
            }
        };
        if !expected && !matches!(state, SitemapState::Parsed { .. }) {
            continue;
        }
        files.insert(
            url,
            SitemapFile {
                url: visit.url.clone(),
                declared: is_declared,
                state,
            },
        );
    }
    let mut known: Vec<String> = declared.iter().cloned().collect();
    known.extend(children);
    for url in known {
        let normalized = normalized_url(&url);
        if !files.contains_key(&normalized) {
            let absolute = url::Url::parse(&url).is_ok_and(|url| matches!(url.scheme(), "http" | "https"));
            files.insert(
                normalized.clone(),
                SitemapFile {
                    url,
                    declared: declared.contains(&normalized),
                    state: if absolute {
                        SitemapState::NotCrawled
                    } else {
                        SitemapState::NotAbsolute
                    },
                },
            );
        }
    }
    // Key pages are compared only with a complete set: every known sitemap was read (a relative
    // declaration, which engines ignore, does not count), or redirects to one that was.
    let has_url_set = files.values().any(|file| {
        matches!(
            file.state,
            SitemapState::Parsed {
                kind: SitemapKind::UrlSet,
                ..
            }
        )
    });
    let is_parsed = |url: &str| {
        files
            .get(&normalized_url(url))
            .is_some_and(|file| matches!(file.state, SitemapState::Parsed { .. }))
    };
    let complete = files.values().all(|file| match &file.state {
        SitemapState::Parsed { .. } | SitemapState::NotAbsolute => true,
        SitemapState::Redirected(target) => is_parsed(target),
        _ => false,
    });
    let compare = has_url_set && complete;

    let mut found = Discovery {
        listed_urls: listed.len(),
        last_modified: last_modified_coverage(status, key, now),
        ..Discovery::default()
    };
    for url in &listed {
        let Some(visit) = by_url.get(url) else {
            found.listed_not_crawled += 1;
            continue;
        };
        let reason = if visit.status_code != 200 {
            Some(ListedReason::Status(visit.status_code))
        } else if visit.content_type == ContentTypeId::Html {
            page_head(status, visit).and_then(|head| not_indexable(status, visit, &head, now))
        } else {
            None
        };
        if let Some(reason) = reason {
            found.listed_issues.push(ListedIssue {
                url: visit.url.clone(),
                reason,
            });
        }
    }

    let mut seen_alternates: HashSet<(String, String)> = HashSet::new();
    for page in key {
        let Some(visit) = by_uq_id.get(page.uq_id.as_str()) else {
            continue;
        };
        if visit.status_code != 200 || visit.content_type != ContentTypeId::Html {
            continue;
        }
        let Some(head) = page_head(status, visit) else {
            continue;
        };
        if compare && not_indexable(status, visit, &head, now).is_none() {
            found.key_pages_compared += 1;
            if !listed.contains(&normalized_url(&visit.url)) {
                found.missing_from_sitemaps.push(visit.url.clone());
            }
        }
        let Ok(base) = url::Url::parse(&visit.url) else {
            continue;
        };
        for link in head.select(&ALTERNATE_SELECTOR) {
            let el = link.value();
            let is_alternate = el.attr("rel").is_some_and(|rel| {
                rel.split_ascii_whitespace()
                    .any(|token| token.eq_ignore_ascii_case("alternate"))
            });
            let href = el.attr("href").unwrap_or_default().trim();
            let Some(mut target) = base.join(href).ok().filter(|_| is_alternate && !href.is_empty()) else {
                continue;
            };
            target.set_fragment(None);
            let target = target.to_string();
            if !seen_alternates.insert((visit.url.clone(), target.clone())) {
                continue;
            }
            let problem = match by_url.get(&target) {
                None if !same_site(&base, &target) => HreflangProblem::OtherSite,
                None => HreflangProblem::NotCrawled,
                Some(alternate) if alternate.status_code != 200 => HreflangProblem::Status(alternate.status_code),
                Some(_) => continue,
            };
            found.hreflang_issues.push(HreflangIssue {
                page: visit.url.clone(),
                lang: el.attr("hreflang").unwrap_or_default().trim().to_string(),
                target,
                problem,
            });
        }
    }

    found.sitemaps = files.into_values().collect();
    found
}

/// The sitemap proposed for a site that has none, from a crawl of the whole site: its canonical,
/// indexable HTML 200 pages on the homepage's origin that robots.txt leaves open to Googlebot and
/// Bingbot, each once, with a `lastmod` only from a plausible `Last-Modified`. The error names why
/// there is none: `sitemap_exists` (declared in robots.txt or crawled), `robots_unknown`
/// (robots.txt, which may declare one, was not read), `robots_not_utf8`, `single_page`,
/// `interrupted`, `url_limit`, `queue_limit`, `limited_scope` (`--max-depth`, `--include-regex`,
/// `--ignore-regex`), `failed_urls` (a linked URL of the site could not be fetched, so pages behind
/// it may be missing), `no_pages` or `too_many_urls`.
pub fn sitemap_proposal(
    status: &Status,
    discovery: &Discovery,
    homepage_url: &str,
    robots: &RobotsFetchState,
    scope: &CrawlScope,
    now: DateTime<Utc>,
) -> Result<SitemapProposal, &'static str> {
    if !discovery.sitemaps.is_empty() {
        return Err("sitemap_exists");
    }
    if !state_is_known(robots) {
        return Err("robots_unknown");
    }
    if let RobotsFetchState::Ok { valid_utf8: false, .. } = robots {
        return Err("robots_not_utf8");
    }
    let rules = robots_of(robots);
    if rules.as_ref().is_some_and(|rules| !rules.sitemaps().is_empty()) {
        return Err("sitemap_exists");
    }
    if scope.single_page {
        return Err("single_page");
    }
    match scope.end {
        CrawlEnd::Complete => {}
        CrawlEnd::Interrupted => return Err("interrupted"),
        CrawlEnd::UrlLimit => return Err("url_limit"),
        CrawlEnd::QueueLimit => return Err("queue_limit"),
    }
    if scope.max_depth > 0 || scope.url_filters {
        return Err("limited_scope");
    }
    let origin = url::Url::parse(homepage_url)
        .map(|url| url.origin())
        .map_err(|_| "no_pages")?;
    let visited = status.get_visited_urls();
    // A link (or a sitemap or redirect entry) of the site that failed hides what is behind it.
    let failed = visited.iter().any(|visit| {
        !visit.is_external
            && matches!(visit.source_attr, SOURCE_A_HREF | SOURCE_SITEMAP | SOURCE_REDIRECT)
            && (visit.status_code == 429
                || visit.status_code >= 500
                || (visit.status_code < 0 && visit.status_code != -6))
    });
    if failed {
        return Err("failed_urls");
    }
    let mut left_out = LeftOut::default();
    let mut urls = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for visit in visited {
        if visit.status_code != 200
            || visit.content_type != ContentTypeId::Html
            || visit.is_external
            || !visit.is_allowed_for_crawling
        {
            continue;
        }
        let Ok(parsed) = url::Url::parse(&visit.url) else {
            continue;
        };
        if parsed.origin() != origin {
            left_out.other_origin += 1;
            continue;
        }
        let Some(head) = page_head(status, &visit) else {
            continue;
        };
        match not_indexable(status, &visit, &head, now) {
            Some(ListedReason::CanonicalElsewhere(_)) => {
                left_out.canonical_elsewhere += 1;
                continue;
            }
            Some(_) => {
                left_out.noindex += 1;
                continue;
            }
            None => {}
        }
        let path = &parsed[url::Position::BeforePath..];
        if let Some(rules) = &rules
            && !(rules.is_allowed("Googlebot", path) && rules.is_allowed("Bingbot", path))
        {
            left_out.blocked += 1;
            continue;
        }
        // The same page crawled under two spellings (`/služby/`, `/slu%C5%BEby/`) is listed once.
        if !seen.insert(normalize_path(path)) {
            continue;
        }
        urls.push(ProposedUrl {
            url: visit.url.clone(),
            lastmod: status
                .get_url_headers(&visit.uq_id)
                .and_then(|headers| lastmod_from_headers(&headers, now)),
        });
    }
    if urls.is_empty() {
        return Err("no_pages");
    }
    if urls.len() > SITEMAP_MAX_URLS {
        return Err("too_many_urls");
    }
    urls.sort_by(|a, b| {
        let slashes = |url: &str| url.trim_end_matches('/').matches('/').count();
        slashes(&a.url).cmp(&slashes(&b.url)).then_with(|| a.url.cmp(&b.url))
    });
    Ok(SitemapProposal {
        origin: origin.ascii_serialization(),
        urls,
        left_out,
        scope: scope.clone(),
    })
}

/// The same host, with or without `www.`.
fn same_site(page: &url::Url, target: &str) -> bool {
    let site = |host: &str| host.trim_start_matches("www.").to_string();
    url::Url::parse(target)
        .ok()
        .and_then(|target| target.host_str().map(site))
        .is_some_and(|host| page.host_str().map(site) == Some(host))
}

/// Where a redirect leads: the end of its crawled chain, or else its `Location`.
fn redirect_target(visit: &VisitedUrl, by_url: &HashMap<String, &VisitedUrl>) -> Option<String> {
    let chain = redirect_chain(visit, by_url);
    if chain.visits.len() > 1 {
        return chain.visits.last().map(|last| last.url.clone());
    }
    let location = visit.extras.as_ref()?.get("Location")?;
    let target = url::Url::parse(&visit.url).ok()?.join(location).ok()?;
    Some(target.to_string())
}

/// The sitemap formats engines accept besides the sitemaps protocol: an RSS (or RDF) or Atom
/// feed, and a text file of URLs, one per line.
fn other_format(body: &str) -> Option<&'static str> {
    let body = body.trim_start_matches('\u{feff}').trim();
    if !body.starts_with('<') {
        let mut lines = body.lines().map(str::trim).filter(|line| !line.is_empty()).peekable();
        return (lines.peek().is_some()
            && lines.all(|line| line.starts_with("http://") || line.starts_with("https://")))
        .then_some("text");
    }
    let mut reader = Reader::from_str(body);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref element) | Event::Empty(ref element)) => {
                return match element.local_name().as_ref() {
                    b"rss" | b"RDF" => Some("RSS"),
                    b"feed" => Some("Atom"),
                    _ => None,
                };
            }
            Ok(Event::Decl(_) | Event::Comment(_) | Event::PI(_) | Event::DocType(_) | Event::Text(_)) => buf.clear(),
            _ => return None,
        }
    }
}

/// The crawler's own rule for a sitemap URL: `sitemap` in the path and a `.xml` or `.gz` file.
fn looks_like_sitemap(url: &str) -> bool {
    url::Url::parse(url).is_ok_and(|url| {
        url.path().to_ascii_lowercase().contains("sitemap") && XmlProcessor::has_sitemap_extension(url.path())
    })
}

/// The head of a stored HTML page, parsed alone: the robots meta tags, the canonical link and the
/// hreflang alternates belong there, and parsing only the head keeps the checks of large sitemaps
/// cheap. The whole page when a robots meta tag may also sit in its body, where engines read it
/// too (as `controls::sources` does for the snippet controls). `None` without a stored body.
fn page_head(status: &Status, visit: &VisitedUrl) -> Option<Html> {
    let body = status.get_url_body_text(&visit.uq_id)?;
    let end = body
        .as_bytes()
        .windows(6)
        .position(|window| window.eq_ignore_ascii_case(b"</head"))
        .unwrap_or(body.len());
    let rest = body.get(end..).unwrap_or_default().to_ascii_lowercase();
    let robots_meta_in_body = rest.contains("<meta")
        && ["robots", "googlebot", "bingbot"]
            .iter()
            .any(|name| rest.contains(name));
    if robots_meta_in_body {
        return Some(Html::parse_document(&body));
    }
    Some(Html::parse_document(body.get(..end).unwrap_or(&body)))
}

/// Why a 200 HTML page is no indexable page: `noindex` for Google or Bing (meta tags or
/// X-Robots-Tag), or a canonical URL elsewhere.
fn not_indexable(status: &Status, visit: &VisitedUrl, head: &Html, now: DateTime<Utc>) -> Option<ListedReason> {
    let headers = status.get_url_headers(&visit.uq_id);
    let found = sources(head, headers.as_ref());
    if google_policy(&found, now).noindex || bing_policy(&found, now).noindex {
        return Some(ListedReason::NoIndex);
    }
    canonical_elsewhere(head, &visit.url).map(ListedReason::CanonicalElsewhere)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::geo::keys::KeyPage;
    use crate::ai::geo::test_support::{add, add_with_headers, new_status, page};
    use crate::result::status::RobotsFetchState;
    use crate::result::visited_url::{SOURCE_A_HREF, SOURCE_IMG_SRC, SOURCE_INIT_URL, SOURCE_SITEMAP};
    use crate::types::ContentTypeId;
    use chrono::TimeZone;

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 26, 12, 0, 0).unwrap()
    }

    fn xml(uq_id: &str, source: &str, url: &str, code: i32) -> crate::result::visited_url::VisitedUrl {
        let mut visit = page(uq_id, source, SOURCE_A_HREF, url, code, None);
        visit.content_type = ContentTypeId::Xml;
        visit.content_type_header = Some("application/xml".to_string());
        visit
    }

    fn urlset(urls: &[(&str, Option<&str>)]) -> String {
        let entries: String = urls
            .iter()
            .map(|(loc, lastmod)| match lastmod {
                Some(lastmod) => format!("<url><loc>{loc}</loc><lastmod>{lastmod}</lastmod></url>"),
                None => format!("<url><loc>{loc}</loc></url>"),
            })
            .collect();
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{entries}</urlset>"#
        )
    }

    fn key_page(uq_id: &str, url: &str, code: i32) -> KeyPage {
        KeyPage {
            uq_id: uq_id.to_string(),
            url: url.to_string(),
            status_code: code,
            score: None,
            is_homepage: uq_id == "home",
        }
    }

    #[test]
    fn sitemaps_parse_as_url_sets_or_indexes_and_malformed_files_are_errors() {
        let parsed = parse_sitemap(
            r#"<?xml version="1.0"?>
            <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <url><loc> https://example.com/ </loc><lastmod>2026-09-01</lastmod></url>
              <url><loc>https://example.com/search?q=a&amp;page=2</loc></url>
            </urlset>"#,
        )
        .unwrap();
        assert_eq!(parsed.kind, SitemapKind::UrlSet);
        assert_eq!(
            parsed.entries,
            [
                SitemapEntry {
                    loc: "https://example.com/".to_string(),
                    lastmod: Some("2026-09-01".to_string()),
                },
                SitemapEntry {
                    loc: "https://example.com/search?q=a&page=2".to_string(),
                    lastmod: None,
                },
            ]
        );

        let index = parse_sitemap(
            r#"<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <sitemap><loc>https://example.com/sitemap-pages.xml</loc><lastmod>2026-09-20T10:00:00+02:00</lastmod></sitemap>
              <sitemap><loc>https://example.com/sitemap-posts.xml.gz</loc></sitemap>
            </sitemapindex>"#,
        )
        .unwrap();
        assert_eq!(index.kind, SitemapKind::Index);
        assert_eq!(
            index.entries.iter().map(|entry| entry.loc.as_str()).collect::<Vec<_>>(),
            [
                "https://example.com/sitemap-pages.xml",
                "https://example.com/sitemap-posts.xml.gz"
            ]
        );

        let prefixed = parse_sitemap(
            r#"<s:urlset xmlns:s="http://www.sitemaps.org/schemas/sitemap/0.9"><s:url><s:loc>https://example.com/a</s:loc></s:url></s:urlset>"#,
        )
        .unwrap();
        assert_eq!(prefixed.entries.len(), 1);

        for malformed in [
            "<urlset><url><loc>https://example.com/</loc></urlset>",
            "<urlset><url><loc>https://example.com/",
            "<rss><channel><title>Feed</title></channel></rss>",
            "Not XML at all",
            "",
            // An undeclared entity, and an invalid character reference, are no text.
            "<urlset><url><loc>https://example.com/a&bogus;b</loc></url></urlset>",
            "<urlset><url><loc>https://example.com/a&#0;b</loc></url></urlset>",
            "<urlset><!-- &bogus; is a comment --><url><loc>https://e.test/</loc><x>&bogus;</x></url></urlset>",
            // A second root element, and text outside the root.
            "<urlset></urlset><urlset><url><loc>https://example.com/second</loc></url></urlset>",
            "<urlset/><urlset><url><loc>https://example.com/second</loc></url></urlset>",
            "<urlset><url><loc>https://example.com/</loc></url></urlset>trailing text",
        ] {
            assert!(parse_sitemap(malformed).is_err(), "{malformed:?} parsed");
        }
        // Character references and CDATA are text.
        let parsed = parse_sitemap(
            "<?xml version=\"1.0\"?>\n<urlset><url><loc><![CDATA[https://e.test/a&b]]></loc></url>\
             <url><loc>https://e.test/&#x10D;&#269;</loc></url></urlset>\n<!-- the end -->\n",
        )
        .unwrap();
        assert_eq!(
            parsed
                .entries
                .iter()
                .map(|entry| entry.loc.as_str())
                .collect::<Vec<_>>(),
            ["https://e.test/a&b", "https://e.test/čč"]
        );
    }

    #[test]
    fn sitemap_files_are_listed_with_their_state() {
        let mut status = new_status();
        add(
            &mut status,
            page("home", "", SOURCE_INIT_URL, "https://example.com/", 200, None),
            Some("<html><body>Home</body></html>"),
        );
        let index = r#"<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <sitemap><loc>https://example.com/sitemap-pages.xml</loc></sitemap>
            <sitemap><loc>https://example.com/sitemap-posts.xml</loc></sitemap></sitemapindex>"#;
        add(
            &mut status,
            xml("index", "home", "https://example.com/sitemap.xml", 200),
            Some(index),
        );
        let mut pages = xml("pages", "index", "https://example.com/sitemap-pages.xml", 200);
        pages.source_attr = SOURCE_SITEMAP;
        add(&mut status, pages, Some(&urlset(&[("https://example.com/", None)])));
        add(
            &mut status,
            xml("missing", "home", "https://example.com/missing-sitemap.xml", 404),
            None,
        );
        add(
            &mut status,
            xml("broken", "home", "https://example.com/broken-sitemap.xml", 200),
            Some("<urlset><url><loc>https://example.com/</loc></urlset>"),
        );
        // XML that is no sitemap and is not named or declared as one is not listed.
        add(
            &mut status,
            xml("feed", "home", "https://example.com/feed.xml", 200),
            Some("<rss><channel><title>Feed</title></channel></rss>"),
        );

        let declared = [
            "https://example.com/sitemap.xml".to_string(),
            "https://example.com/missing-sitemap.xml".to_string(),
            // Engines need a full URL here; a relative one is not used.
            "/sitemap-relative.xml".to_string(),
        ];
        let found = discovery(&status, &[], &declared, now());
        let mut files: Vec<(&str, bool, &SitemapState)> = found
            .sitemaps
            .iter()
            .map(|file| (file.url.as_str(), file.declared, &file.state))
            .collect();
        assert_eq!(files.len(), 6, "{files:#?}");
        assert_eq!(
            files.remove(0),
            ("/sitemap-relative.xml", true, &SitemapState::NotAbsolute)
        );
        assert_eq!(files[0].0, "https://example.com/broken-sitemap.xml");
        assert!(matches!(files[0].2, SitemapState::Malformed(_)), "{:?}", files[0]);
        assert_eq!(
            files[1],
            (
                "https://example.com/missing-sitemap.xml",
                true,
                &SitemapState::Failed(404)
            )
        );
        assert_eq!(files[2].0, "https://example.com/sitemap-pages.xml");
        assert!(
            matches!(
                files[2].2,
                SitemapState::Parsed {
                    kind: SitemapKind::UrlSet,
                    entries: 1,
                    ..
                }
            ),
            "{:?}",
            files[2]
        );
        assert_eq!(
            files[3],
            (
                "https://example.com/sitemap-posts.xml",
                false,
                &SitemapState::NotCrawled
            )
        );
        assert_eq!((files[4].0, files[4].1), ("https://example.com/sitemap.xml", true));
        assert!(
            matches!(
                files[4].2,
                SitemapState::Parsed {
                    kind: SitemapKind::Index,
                    entries: 2,
                    ..
                }
            ),
            "{:?}",
            files[4]
        );
    }

    #[test]
    fn key_pages_missing_from_the_sitemaps_are_listed() {
        let mut status = new_status();
        let body = |head: &str| format!("<html><head>{head}</head><body><p>Text</p></body></html>");
        let pages = [
            ("home", "https://example.com/", 200, body("")),
            ("a", "https://example.com/a", 200, body("")),
            ("b", "https://example.com/b", 200, body("")),
            (
                "c",
                "https://example.com/c",
                200,
                body(r#"<meta name="robots" content="noindex">"#),
            ),
            (
                "d",
                "https://example.com/d",
                200,
                body(r#"<link rel="canonical" href="/a">"#),
            ),
            ("e", "https://example.com/e", 403, body("")),
        ];
        for (uq_id, url, code, html) in &pages {
            let attr = if *uq_id == "home" {
                SOURCE_INIT_URL
            } else {
                SOURCE_A_HREF
            };
            add(&mut status, page(uq_id, "home", attr, url, *code, None), Some(html));
        }
        add(
            &mut status,
            xml("sm", "home", "https://example.com/sitemap.xml", 200),
            Some(&urlset(&[
                ("https://example.com/", None),
                ("https://example.com/a#top", None),
                ("https://example.com/not-key", None),
            ])),
        );
        let key: Vec<KeyPage> = pages
            .iter()
            .map(|(uq_id, url, code, _)| key_page(uq_id, url, *code))
            .collect();

        let found = discovery(&status, &key, &[], now());
        // /c is noindex, /d points its canonical elsewhere and /e is no page: none belongs in a
        // sitemap.
        assert_eq!(found.key_pages_compared, 3);
        assert_eq!(found.missing_from_sitemaps, ["https://example.com/b"]);

        let without_sitemap = discovery(&new_status(), &key, &[], now());
        assert_eq!(without_sitemap.key_pages_compared, 0, "nothing to compare with");
        assert!(without_sitemap.missing_from_sitemaps.is_empty());
    }

    /// The homepage and a blog post, both key pages, and a sitemap index at /sitemap.xml whose
    /// pages sitemap lists only the homepage.
    fn site_with_an_index(posts: Option<&str>) -> (Status, Vec<KeyPage>) {
        let mut status = new_status();
        let html = "<html><head></head><body><p>Text</p></body></html>";
        add(
            &mut status,
            page("home", "", SOURCE_INIT_URL, "https://example.com/", 200, None),
            Some(html),
        );
        add(
            &mut status,
            page(
                "post",
                "home",
                SOURCE_A_HREF,
                "https://example.com/blog/post",
                200,
                None,
            ),
            Some(html),
        );
        let index = r#"<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <sitemap><loc>https://example.com/pages.xml</loc></sitemap>
            <sitemap><loc>https://example.com/posts.xml</loc></sitemap></sitemapindex>"#;
        add(
            &mut status,
            xml("index", "home", "https://example.com/sitemap.xml", 200),
            Some(index),
        );
        add(
            &mut status,
            xml("pages", "index", "https://example.com/pages.xml", 200),
            Some(&urlset(&[("https://example.com/", None)])),
        );
        if let Some(body) = posts {
            add(
                &mut status,
                xml("posts", "index", "https://example.com/posts.xml", 200),
                Some(body),
            );
        }
        let key = vec![
            key_page("home", "https://example.com/", 200),
            key_page("post", "https://example.com/blog/post", 200),
        ];
        (status, key)
    }

    #[test]
    fn key_pages_are_compared_only_with_a_complete_set_of_sitemaps() {
        // The posts sitemap was not crawled: the blog post may well be listed there.
        let (status, key) = site_with_an_index(None);
        let partial = discovery(&status, &key, &[], now());
        assert_eq!(partial.key_pages_compared, 0, "not assessed");
        assert!(partial.missing_from_sitemaps.is_empty());

        let (status, key) = site_with_an_index(Some(&urlset(&[("https://example.com/blog/post", None)])));
        let complete = discovery(&status, &key, &["/relative.xml".to_string()], now());
        assert_eq!(
            complete.key_pages_compared, 2,
            "a declaration engines ignore does not make the set incomplete"
        );
        assert!(complete.missing_from_sitemaps.is_empty());

        let (status, key) = site_with_an_index(Some(&urlset(&[])));
        assert_eq!(
            discovery(&status, &key, &[], now()).missing_from_sitemaps,
            ["https://example.com/blog/post"]
        );
    }

    #[test]
    fn a_redirected_sitemap_is_followed_and_feeds_are_not_read() {
        let (mut status, key) = site_with_an_index(Some(&urlset(&[])));
        add(
            &mut status,
            page(
                "old",
                "home",
                SOURCE_A_HREF,
                "http://example.com/sitemap.xml",
                301,
                Some("https://example.com/sitemap.xml"),
            ),
            None,
        );
        let declared = ["http://example.com/sitemap.xml".to_string()];
        let found = discovery(&status, &key, &declared, now());
        let old = found
            .sitemaps
            .iter()
            .find(|file| file.url == "http://example.com/sitemap.xml")
            .unwrap();
        assert_eq!(
            old.state,
            SitemapState::Redirected("https://example.com/sitemap.xml".to_string())
        );
        assert_eq!(found.key_pages_compared, 2, "the redirect leads to a parsed sitemap");

        // Engines accept RSS, Atom and text sitemaps; this check does not read them.
        let formats = [
            (
                "rss",
                "https://example.com/feed.xml",
                "<rss version=\"2.0\"><channel></channel></rss>",
                "RSS",
            ),
            (
                "atom",
                "https://example.com/atom.xml",
                "<feed xmlns=\"http://www.w3.org/2005/Atom\"></feed>",
                "Atom",
            ),
            (
                "text",
                "https://example.com/urls.txt",
                "https://example.com/\nhttps://example.com/blog/post\n",
                "text",
            ),
        ];
        let mut declared = Vec::new();
        for (uq_id, url, body, _) in formats {
            add(&mut status, xml(uq_id, "home", url, 200), Some(body));
            declared.push(url.to_string());
        }
        let found = discovery(&status, &key, &declared, now());
        for (_, url, _, format) in formats {
            let file = found.sitemaps.iter().find(|file| file.url == url).unwrap();
            assert_eq!(file.state, SitemapState::Unsupported(format), "{url}");
        }
        assert_eq!(found.key_pages_compared, 0, "their URLs are unknown");
    }

    #[test]
    fn listed_urls_that_are_no_indexable_pages_are_reported() {
        let mut status = new_status();
        add(
            &mut status,
            page("home", "", SOURCE_INIT_URL, "https://example.com/", 200, None),
            Some("<html><body>Home</body></html>"),
        );
        let listed = |uq_id: &str, url: &str, code: i32, location: Option<&str>| {
            page(uq_id, "sm", SOURCE_SITEMAP, url, code, location)
        };
        add(
            &mut status,
            listed("ok", "https://example.com/ok", 200, None),
            Some("<html><head></head><body>OK</body></html>"),
        );
        add(
            &mut status,
            listed("moved", "https://example.com/moved", 301, Some("/ok")),
            None,
        );
        add(&mut status, listed("gone", "https://example.com/gone", 404, None), None);
        add(
            &mut status,
            listed("noindex", "https://example.com/noindex", 200, None),
            Some(r#"<html><head><meta name="robots" content="noindex, follow"></head><body>x</body></html>"#),
        );
        add_with_headers(
            &mut status,
            listed("xrobots", "https://example.com/xrobots", 200, None),
            Some("<html><head></head><body>x</body></html>"),
            &[("x-robots-tag", "bingbot: noindex")],
        );
        add(
            &mut status,
            listed("canon", "https://example.com/canon", 200, None),
            Some(r#"<html><head><link rel="canonical" href="https://example.com/ok"></head><body>x</body></html>"#),
        );
        add(
            &mut status,
            xml("sm", "home", "https://example.com/sitemap.xml", 200),
            Some(&urlset(&[
                ("https://example.com/ok", None),
                ("https://example.com/moved", None),
                ("https://example.com/gone", None),
                ("https://example.com/noindex", None),
                ("https://example.com/xrobots", None),
                ("https://example.com/canon", None),
                ("https://example.com/never-crawled", None),
            ])),
        );

        let found = discovery(&status, &[], &[], now());
        assert_eq!(found.listed_urls, 7);
        assert_eq!(found.listed_not_crawled, 1);
        assert_eq!(
            found.listed_issues,
            [
                ListedIssue {
                    url: "https://example.com/canon".to_string(),
                    reason: ListedReason::CanonicalElsewhere("https://example.com/ok".to_string()),
                },
                ListedIssue {
                    url: "https://example.com/gone".to_string(),
                    reason: ListedReason::Status(404),
                },
                ListedIssue {
                    url: "https://example.com/moved".to_string(),
                    reason: ListedReason::Status(301),
                },
                ListedIssue {
                    url: "https://example.com/noindex".to_string(),
                    reason: ListedReason::NoIndex,
                },
                ListedIssue {
                    url: "https://example.com/xrobots".to_string(),
                    reason: ListedReason::NoIndex,
                },
            ]
        );
    }

    #[test]
    fn lastmod_values_all_identical_or_in_the_future_are_suspicious() {
        let entries = |lastmods: &[Option<&str>]| -> Vec<SitemapEntry> {
            lastmods
                .iter()
                .enumerate()
                .map(|(i, lastmod)| SitemapEntry {
                    loc: format!("https://example.com/{i}"),
                    lastmod: lastmod.map(str::to_string),
                })
                .collect()
        };
        let identical = lastmod_stats(
            &entries(&[
                Some("2026-09-26T10:00:00+00:00"),
                Some("2026-09-26T12:00:00+02:00"),
                Some("2026-09-26T10:00Z"),
            ]),
            now(),
        );
        assert!(identical.all_identical && identical.suspicious(), "{identical:?}");

        let future = lastmod_stats(&entries(&[Some("2026-09-01"), Some("2027-01-01")]), now());
        assert_eq!(future.future, 1);
        assert!(!future.all_identical && future.suspicious(), "{future:?}");

        let normal = lastmod_stats(
            &entries(&[
                Some("2026-09-01"),
                Some("2026-08"),
                None,
                Some("yesterday"),
                Some("2026-09-27"),
            ]),
            now(),
        );
        assert_eq!(
            normal,
            LastmodStats {
                entries: 5,
                with_lastmod: 4,
                invalid: 1,
                future: 0,
                all_identical: false,
            },
            "a day ahead is within the tolerance"
        );
        assert!(!normal.suspicious());

        let single = lastmod_stats(&entries(&[Some("2026-09-01")]), now());
        assert!(!single.all_identical, "one date is not a pattern");
        let two = lastmod_stats(&entries(&[Some("2026-09-01"), Some("2026-09-01")]), now());
        assert!(!two.all_identical, "two pages launched together are no pattern either");
    }

    #[test]
    fn hreflang_targets_that_are_not_200_or_not_crawled_are_reported() {
        let mut status = new_status();
        let head = r#"<link rel="alternate" hreflang="cs" href="https://example.com/">
            <link rel="alternate" hreflang="en" href="/en/">
            <link rel="alternate" hreflang="de" href="/de/">
            <link rel="alternate" hreflang="sk" href="/sk/">
            <link rel="alternate" hreflang="fr" href="https://example.fr/">
            <link rel="alternate" hreflang="it" href="https://www.example.com/it/">
            <link rel="alternate" hreflang="x-default" href="/">
            <link rel="alternate" type="application/rss+xml" href="/feed.xml">"#;
        add(
            &mut status,
            page("home", "", SOURCE_INIT_URL, "https://example.com/", 200, None),
            Some(&format!("<html><head>{head}</head><body>Home</body></html>")),
        );
        add(
            &mut status,
            page("en", "home", SOURCE_A_HREF, "https://example.com/en/", 200, None),
            Some("<html><body>EN</body></html>"),
        );
        add(
            &mut status,
            page("de", "home", SOURCE_A_HREF, "https://example.com/de/", 404, None),
            None,
        );
        add(
            &mut status,
            page("sk", "home", SOURCE_A_HREF, "https://example.com/sk/", 301, Some("/")),
            None,
        );
        let found = discovery(&status, &[key_page("home", "https://example.com/", 200)], &[], now());
        let issues: Vec<(&str, &str, &HreflangProblem)> = found
            .hreflang_issues
            .iter()
            .map(|issue| (issue.lang.as_str(), issue.target.as_str(), &issue.problem))
            .collect();
        assert_eq!(
            issues,
            [
                ("de", "https://example.com/de/", &HreflangProblem::Status(404)),
                ("sk", "https://example.com/sk/", &HreflangProblem::Status(301)),
                // Another site is outside the crawl: not checked rather than a problem.
                ("fr", "https://example.fr/", &HreflangProblem::OtherSite),
                ("it", "https://www.example.com/it/", &HreflangProblem::NotCrawled),
            ]
        );
        assert!(
            found
                .hreflang_issues
                .iter()
                .all(|issue| issue.page == "https://example.com/")
        );
        assert_eq!(
            found.last_modified.pages, 1,
            "the Last-Modified coverage of the key pages"
        );
    }

    fn complete_scope() -> CrawlScope {
        CrawlScope {
            start_url: "https://example.com/".to_string(),
            single_page: false,
            end: CrawlEnd::Complete,
            max_visited_urls: 10_000,
            max_depth: 0,
            url_filters: false,
        }
    }

    fn robots(content: &str) -> RobotsFetchState {
        RobotsFetchState::Ok {
            status: 200,
            content: content.to_string(),
            valid_utf8: true,
        }
    }

    /// A crawl without any sitemap: indexable pages, a `noindex` page, a canonicalized page, a
    /// 404, a page on another origin, a page blocked for Googlebot, a PDF and an external page.
    fn site_without_sitemap() -> Status {
        let mut status = new_status();
        let html =
            |extra_head: &str| format!("<html><head>{extra_head}</head><body><main><p>Text</p></main></body></html>");
        let pages: [(&str, &str, i32, String); 9] = [
            ("home", "https://example.com/", 200, html("")),
            // A robots meta tag in the body counts as well (as in `controls::sources`).
            (
                "body-noindex",
                "https://example.com/body-noindex",
                200,
                r#"<html><head></head><body><main><meta name="robots" content="noindex"><p>Text</p></main></body></html>"#
                    .to_string(),
            ),
            // Another meta tag and a mention of robots in the body do not.
            (
                "robots-text",
                "https://example.com/robots-text",
                200,
                r#"<html><head></head><body><main><meta itemprop="name" content="x"><p>We build robots.</p></main></body></html>"#
                    .to_string(),
            ),
            ("zeta", "https://example.com/zeta", 200, html("")),
            ("deep", "https://example.com/a/b?x=1&y=2", 200, html("")),
            (
                "hidden",
                "https://example.com/hidden",
                200,
                html(r#"<meta name="robots" content="noindex">"#),
            ),
            (
                "dup",
                "https://example.com/dup",
                200,
                html(r#"<link rel="canonical" href="https://example.com/zeta">"#),
            ),
            ("gone", "https://example.com/gone", 404, html("")),
            ("private", "https://example.com/private/page", 200, html("")),
        ];
        for (uq_id, url, code, body) in pages {
            let (source, attr) = if uq_id == "home" {
                ("", SOURCE_INIT_URL)
            } else {
                ("home", SOURCE_A_HREF)
            };
            let headers: &[(&str, &str)] = match uq_id {
                // A plausible Last-Modified, a day before the response.
                "zeta" => &[
                    ("last-modified", "Tue, 01 Sep 2026 10:00:00 GMT"),
                    ("date", "Wed, 02 Sep 2026 10:00:00 GMT"),
                ],
                // A dynamic page stamping "now": no lastmod.
                "deep" => &[
                    ("last-modified", "Wed, 02 Sep 2026 10:00:00 GMT"),
                    ("date", "Wed, 02 Sep 2026 10:00:00 GMT"),
                ],
                _ => &[],
            };
            add_with_headers(
                &mut status,
                page(uq_id, source, attr, url, code, None),
                Some(&body),
                headers,
            );
        }
        let mut www = page("www", "home", SOURCE_A_HREF, "https://www.example.com/other", 200, None);
        www.is_external = false;
        add(&mut status, www, Some(&html("")));
        let mut external = page("ext", "home", SOURCE_A_HREF, "https://partner.example.org/", 200, None);
        external.is_external = true;
        add(&mut status, external, Some(&html("")));
        let mut pdf = page("pdf", "home", SOURCE_A_HREF, "https://example.com/price.pdf", 200, None);
        pdf.content_type = ContentTypeId::Other;
        add(&mut status, pdf, None);
        status
    }

    #[test]
    fn a_sitemap_is_proposed_from_the_indexable_pages_of_a_complete_crawl() {
        let status = site_without_sitemap();
        let rules = robots("User-agent: *\nDisallow:\n\nUser-agent: Googlebot\nDisallow: /private/\n");
        let found = Discovery::default();
        let proposal = sitemap_proposal(
            &status,
            &found,
            "https://example.com/",
            &rules,
            &complete_scope(),
            now(),
        )
        .expect("a complete crawl without a sitemap");
        assert_eq!(proposal.origin, "https://example.com");
        // Canonical, indexable HTML 200 pages of the homepage's origin, in the sitemap exporter's
        // order (fewer slashes first, then alphabetically).
        assert_eq!(
            proposal.urls,
            [
                ProposedUrl {
                    url: "https://example.com/".to_string(),
                    lastmod: None,
                },
                ProposedUrl {
                    url: "https://example.com/robots-text".to_string(),
                    lastmod: None,
                },
                ProposedUrl {
                    url: "https://example.com/zeta".to_string(),
                    lastmod: Some("2026-09-01T10:00:00+00:00".to_string()),
                },
                ProposedUrl {
                    url: "https://example.com/a/b?x=1&y=2".to_string(),
                    lastmod: None,
                },
            ]
        );
        assert_eq!(
            (
                proposal.left_out.noindex,
                proposal.left_out.canonical_elsewhere,
                proposal.left_out.blocked,
                proposal.left_out.other_origin,
            ),
            (2, 1, 1, 1)
        );
        assert_eq!(proposal.scope, complete_scope());

        // Without robots.txt the pages are not blocked.
        let proposal = sitemap_proposal(
            &status,
            &found,
            "https://example.com/",
            &RobotsFetchState::NotFound { status: 404 },
            &complete_scope(),
            now(),
        )
        .unwrap();
        assert_eq!(proposal.urls.len(), 5);
        assert_eq!(proposal.left_out.blocked, 0);
    }

    #[test]
    fn no_sitemap_is_proposed_when_one_exists_or_the_crawl_did_not_cover_the_whole_site() {
        let status = site_without_sitemap();
        let rules = robots("User-agent: *\nDisallow:\n");
        let propose = |found: &Discovery, state: &RobotsFetchState, scope: &CrawlScope| {
            sitemap_proposal(&status, found, "https://example.com/", state, scope, now()).map(|_| ())
        };
        let none = Discovery::default();
        assert_eq!(propose(&none, &rules, &complete_scope()), Ok(()));

        let with_sitemap = Discovery {
            sitemaps: vec![SitemapFile {
                url: "https://example.com/sitemap.xml".to_string(),
                declared: true,
                state: SitemapState::NotCrawled,
            }],
            ..Discovery::default()
        };
        assert_eq!(propose(&with_sitemap, &rules, &complete_scope()), Err("sitemap_exists"));

        let scope = |change: fn(&mut CrawlScope)| {
            let mut scope = complete_scope();
            change(&mut scope);
            scope
        };
        let cases: [(CrawlScope, &str); 6] = [
            (scope(|scope| scope.single_page = true), "single_page"),
            (scope(|scope| scope.end = CrawlEnd::UrlLimit), "url_limit"),
            (scope(|scope| scope.end = CrawlEnd::QueueLimit), "queue_limit"),
            (scope(|scope| scope.end = CrawlEnd::Interrupted), "interrupted"),
            (scope(|scope| scope.max_depth = 2), "limited_scope"),
            (scope(|scope| scope.url_filters = true), "limited_scope"),
        ];
        for (scope, why) in cases {
            assert_eq!(propose(&none, &rules, &scope), Err(why), "{scope:?}");
        }
        // The robots.txt of the homepage's origin declares one, although no key page led to it.
        let declares = robots("User-agent: *\nDisallow:\n\nSitemap: https://example.com/sitemap.xml\n");
        assert_eq!(propose(&none, &declares, &complete_scope()), Err("sitemap_exists"));
        // Rules with undecodable bytes cannot be matched exactly.
        let lossy = RobotsFetchState::Ok {
            status: 200,
            content: "User-agent: *\nDisallow: /\u{fffd}\n".to_string(),
            valid_utf8: false,
        };
        assert_eq!(propose(&none, &lossy, &complete_scope()), Err("robots_not_utf8"));
        // Whether robots.txt declares a sitemap is not known.
        for state in [
            RobotsFetchState::Skipped,
            RobotsFetchState::NotAttempted,
            RobotsFetchState::Unavailable {
                status_or_error: "HTTP 503".to_string(),
            },
        ] {
            assert_eq!(
                propose(&none, &state, &complete_scope()),
                Err("robots_unknown"),
                "{state:?}"
            );
        }
        // No indexable page.
        let empty = new_status();
        assert_eq!(
            sitemap_proposal(&empty, &none, "https://example.com/", &rules, &complete_scope(), now()).map(|_| ()),
            Err("no_pages")
        );
    }

    #[test]
    fn no_sitemap_is_proposed_when_linked_pages_could_not_be_fetched() {
        let rules = robots("User-agent: *\nDisallow:\n");
        let propose = |status: &Status| {
            sitemap_proposal(
                status,
                &Discovery::default(),
                "https://example.com/",
                &rules,
                &complete_scope(),
                now(),
            )
            .map(|proposal| proposal.urls.len())
        };
        // A missing image does not hide pages.
        let mut status = site_without_sitemap();
        add(
            &mut status,
            page("img", "home", SOURCE_IMG_SRC, "https://example.com/a.png", 503, None),
            None,
        );
        assert_eq!(propose(&status), Ok(5));
        // A link that failed may hide the pages behind it.
        for (uq_id, code) in [("down", 503), ("busy", 429), ("slow", -2)] {
            let mut status = site_without_sitemap();
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
            assert_eq!(propose(&status), Err("failed_urls"), "HTTP {code}");
        }
    }

    #[test]
    fn the_proposed_sitemap_lists_each_page_once() {
        let mut status = site_without_sitemap();
        let body = "<html><head></head><body><main><p>Služby</p></main></body></html>";
        for (uq_id, url) in [
            ("raw", "https://example.com/služby/"),
            ("encoded", "https://example.com/slu%C5%BEby/"),
            ("lower", "https://example.com/slu%c5%bEby/"),
        ] {
            add(
                &mut status,
                page(uq_id, "home", SOURCE_A_HREF, url, 200, None),
                Some(body),
            );
        }
        let proposal = sitemap_proposal(
            &status,
            &Discovery::default(),
            "https://example.com/",
            &robots("User-agent: *\nDisallow:\n"),
            &complete_scope(),
            now(),
        )
        .unwrap();
        let services: Vec<&str> = proposal
            .urls
            .iter()
            .map(|url| url.url.as_str())
            .filter(|url| url.contains("slu"))
            .collect();
        assert_eq!(services.len(), 1, "{services:?}");
    }
}
