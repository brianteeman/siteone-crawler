// SiteOne Crawler - indexing and snippet directives per search engine
// (c) Jan Reges <jan.reges@siteone.cz>
//
// Reads the robots meta tags (`robots`, `googlebot`, `bingbot`) and every X-Robots-Tag header
// instance as separate sources with their engine scope: in a header instance, a `googlebot:` or
// `bingbot:` prefix scopes the directives after it, up to the next prefix. The effective policy
// of Google is Generic ∪ Googlebot, of Bing Generic ∪ Bingbot, the more restrictive value
// winning. Also the text hidden from snippets with `data-nosnippet`, and a canonical URL that
// points elsewhere.

use std::collections::HashMap;

use chrono::{DateTime, FixedOffset, NaiveDate, NaiveDateTime, Utc};
use ego_tree::NodeRef;
use once_cell::sync::Lazy;
use scraper::{ElementRef, Html, Node, Selector};

/// Which crawlers a directive source addresses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    /// `meta name="robots"` or an X-Robots-Tag without a user-agent prefix.
    Generic,
    Googlebot,
    Bingbot,
    /// Another user agent named in an X-Robots-Tag prefix.
    Other(String),
}

/// One meta tag, or one scope of one X-Robots-Tag header instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectiveSource {
    /// "meta robots", "meta googlebot", "X-Robots-Tag #2", …
    pub origin: String,
    pub scope: Scope,
    pub directives: Vec<Directive>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Directive {
    NoIndex,
    /// `none` = `noindex, nofollow`.
    None,
    NoSnippet,
    /// `max-snippet:N`; −1 means no limit.
    MaxSnippet(i64),
    NoArchive,
    NoCache,
    UnavailableAfter(DateTime<FixedOffset>),
    /// Any other directive (`nofollow`, `indexifembedded`, …) or one whose value cannot be read,
    /// as written.
    Other(String),
}

/// What one engine may do with a page, after combining its sources.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EnginePolicy {
    /// `noindex`, `none`, or an expired `unavailable_after`.
    pub noindex: bool,
    pub nosnippet: bool,
    /// The most restrictive `max-snippet`: the smallest limit, or −1 when every value is −1.
    pub max_snippet: Option<i64>,
    /// Bing: not used in Copilot answers. Bing treats `noarchive` together with `nocache` as
    /// `nocache`. Always false for Google, which uses neither directive.
    pub noarchive: bool,
    /// Bing: only the URL, title and snippet may be used in Copilot answers.
    pub nocache: bool,
    /// An `unavailable_after` date has passed.
    pub expired: bool,
    /// The origins of the sources that restrict the page, in document order.
    pub sources: Vec<String>,
}

/// Directive names, so that a date continuing after a comma is not taken for a user agent.
const KNOWN_DIRECTIVES: &[&str] = &[
    "all",
    "index",
    "noindex",
    "follow",
    "nofollow",
    "none",
    "nosnippet",
    "noarchive",
    "nocache",
    "notranslate",
    "noimageindex",
    "indexifembedded",
    "max-snippet",
    "max-image-preview",
    "max-video-preview",
    "unavailable_after",
];

static META_SELECTOR: Lazy<Selector> = Lazy::new(|| Selector::parse("meta[name][content]").unwrap());
static NOSNIPPET_SELECTOR: Lazy<Selector> =
    Lazy::new(|| Selector::parse("span[data-nosnippet], div[data-nosnippet], section[data-nosnippet]").unwrap());
static BASE_SELECTOR: Lazy<Selector> = Lazy::new(|| Selector::parse("base[href]").unwrap());
static LINK_SELECTOR: Lazy<Selector> = Lazy::new(|| Selector::parse("link[rel][href]").unwrap());

/// The directive sources of a page: its `robots`, `googlebot` and `bingbot` meta tags (other
/// names are ignored) and each X-Robots-Tag instance of `headers` (as flattened by
/// `get_flat_response_headers`, one instance per line), split by their `ua:` prefixes.
pub fn sources(document: &Html, headers: Option<&HashMap<String, String>>) -> Vec<DirectiveSource> {
    let mut found = Vec::new();
    for meta in document.select(&META_SELECTOR) {
        let name = meta
            .value()
            .attr("name")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        let scope = match name.as_str() {
            "robots" => Scope::Generic,
            "googlebot" => Scope::Googlebot,
            "bingbot" => Scope::Bingbot,
            _ => continue,
        };
        let content = meta.value().attr("content").unwrap_or_default();
        parse_source(content, false, scope, &format!("meta {name}"), &mut found);
    }
    let header = headers.and_then(|headers| {
        headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("x-robots-tag"))
            .map(|(_, value)| value)
    });
    if let Some(value) = header {
        for (index, instance) in value.split('\n').enumerate() {
            let origin = format!("X-Robots-Tag #{}", index + 1);
            parse_source(instance, true, Scope::Generic, &origin, &mut found);
        }
    }
    found
}

/// Parses one comma-separated directive list into `found`, one source per scope. A comma inside
/// an `unavailable_after` date (`Wednesday, 03-Nov-2027 …`) does not end the directive.
fn parse_source(text: &str, allow_prefixes: bool, scope: Scope, origin: &str, found: &mut Vec<DirectiveSource>) {
    let mut scope = scope;
    let mut pieces: Vec<String> = Vec::new();
    let mut flush = |scope: &Scope, pieces: &mut Vec<String>| {
        if !pieces.is_empty() {
            found.push(DirectiveSource {
                origin: origin.to_string(),
                scope: scope.clone(),
                directives: pieces.iter().map(|piece| parse_directive(piece)).collect(),
            });
            pieces.clear();
        }
    };
    for piece in text.split(',') {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        if allow_prefixes && let Some((agent, rest)) = user_agent_prefix(piece) {
            flush(&scope, &mut pieces);
            scope = match agent.to_ascii_lowercase().as_str() {
                "googlebot" => Scope::Googlebot,
                "bingbot" => Scope::Bingbot,
                _ => Scope::Other(agent.to_string()),
            };
            if !rest.is_empty() {
                pieces.push(rest.to_string());
            }
            continue;
        }
        let continues_a_date = !KNOWN_DIRECTIVES.contains(&directive_name(piece).as_str())
            && pieces
                .last()
                .is_some_and(|last| directive_name(last) == "unavailable_after");
        match pieces.last_mut() {
            Some(last) if continues_a_date => {
                last.push_str(", ");
                last.push_str(piece);
            }
            _ => pieces.push(piece.to_string()),
        }
    }
    flush(&scope, &mut pieces);
}

/// `googlebot: noindex` → `("googlebot", "noindex")`: a user-agent name (a letter, then letters,
/// digits, `-` or `_`) that is not a directive name, followed by a colon.
fn user_agent_prefix(piece: &str) -> Option<(&str, &str)> {
    let (name, rest) = piece.split_once(':')?;
    let name = name.trim();
    let valid = name.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    (valid && !KNOWN_DIRECTIVES.contains(&name.to_ascii_lowercase().as_str())).then(|| (name, rest.trim()))
}

fn directive_name(piece: &str) -> String {
    piece.split(':').next().unwrap_or_default().trim().to_ascii_lowercase()
}

fn parse_directive(piece: &str) -> Directive {
    let (name, value) = match piece.split_once(':') {
        Some((name, value)) => (name.trim().to_ascii_lowercase(), Some(value.trim())),
        None => (piece.trim().to_ascii_lowercase(), None),
    };
    let other = || Directive::Other(piece.trim().to_string());
    match (name.as_str(), value) {
        ("noindex", None) => Directive::NoIndex,
        ("none", None) => Directive::None,
        ("nosnippet", None) => Directive::NoSnippet,
        ("noarchive", None) => Directive::NoArchive,
        ("nocache", None) => Directive::NoCache,
        ("max-snippet", Some(value)) => value
            .parse::<i64>()
            .ok()
            .filter(|limit| *limit >= -1)
            .map_or_else(other, Directive::MaxSnippet),
        ("unavailable_after", Some(value)) => parse_date(value).map_or_else(other, Directive::UnavailableAfter),
        _ => other(),
    }
}

/// An `unavailable_after` date: RFC 3339 / ISO 8601, RFC 822 / 2822 (with or without a weekday,
/// with a zone name such as `GMT` or `PST`) or RFC 850 (`Wednesday, 03-Nov-2027 15:00:00 GMT`).
/// A date without a zone is UTC.
fn parse_date(value: &str) -> Option<DateTime<FixedOffset>> {
    let value = value.trim();
    if let Ok(date) = DateTime::parse_from_rfc3339(value) {
        return Some(date);
    }
    if let Ok(date) = DateTime::parse_from_rfc2822(value) {
        return Some(date);
    }
    let without_zone = ["GMT", "UTC", "Z"]
        .iter()
        .find_map(|zone| value.strip_suffix(zone))
        .map_or(value, str::trim_end);
    for format in [
        "%A, %d-%b-%Y %H:%M:%S",
        "%A, %d-%b-%y %H:%M:%S",
        "%d-%b-%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ] {
        if let Ok(date) = NaiveDateTime::parse_from_str(without_zone, format) {
            return Some(date.and_utc().fixed_offset());
        }
    }
    NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .ok()
        .and_then(|date| date.and_hms_opt(0, 0, 0))
        .map(|date| date.and_utc().fixed_offset())
}

/// Google's effective policy: the Generic and Googlebot sources. Google Search uses neither
/// `noarchive` nor `nocache`.
pub fn google_policy(src: &[DirectiveSource], now: DateTime<Utc>) -> EnginePolicy {
    policy(src, now, &Scope::Googlebot)
}

/// Bing's effective policy: the Generic and Bingbot sources; `noarchive` together with `nocache`
/// counts as `nocache` (Microsoft, "Announcing new options for webmasters to control usage of
/// their content in Bing Chat", 2023).
pub fn bing_policy(src: &[DirectiveSource], now: DateTime<Utc>) -> EnginePolicy {
    let mut policy = policy(src, now, &Scope::Bingbot);
    if policy.nocache {
        policy.noarchive = false;
    }
    policy
}

fn policy(src: &[DirectiveSource], now: DateTime<Utc>, engine: &Scope) -> EnginePolicy {
    let is_bing = *engine == Scope::Bingbot;
    let mut policy = EnginePolicy::default();
    let mut limits: Vec<i64> = Vec::new();
    for source in src
        .iter()
        .filter(|source| source.scope == Scope::Generic || source.scope == *engine)
    {
        let mut restricts = false;
        for directive in &source.directives {
            match directive {
                Directive::NoIndex | Directive::None => {
                    policy.noindex = true;
                    restricts = true;
                }
                Directive::NoSnippet => {
                    policy.nosnippet = true;
                    restricts = true;
                }
                Directive::MaxSnippet(limit) => {
                    limits.push(*limit);
                    restricts |= *limit >= 0;
                }
                Directive::NoArchive if is_bing => {
                    policy.noarchive = true;
                    restricts = true;
                }
                Directive::NoCache if is_bing => {
                    policy.nocache = true;
                    restricts = true;
                }
                Directive::UnavailableAfter(date) if date.with_timezone(&Utc) <= now => {
                    policy.expired = true;
                    policy.noindex = true;
                    restricts = true;
                }
                _ => {}
            }
        }
        if restricts && !policy.sources.contains(&source.origin) {
            policy.sources.push(source.origin.clone());
        }
    }
    policy.max_snippet = match limits.iter().filter(|limit| **limit >= 0).min() {
        Some(limit) => Some(*limit),
        None => limits.first().copied(),
    };
    policy
}

/// The visible text, in characters with whitespace collapsed, inside `data-nosnippet` elements
/// of the page's own content. Only `span`, `div` and `section` carry the attribute for Google;
/// nested ones count once; site chrome (`nav`, `aside`, a site `header` / `footer`, banner and
/// content-info roles) is left out, as are scripts, styles and other non-text elements.
pub fn data_nosnippet_chars(document: &Html) -> usize {
    document
        .select(&NOSNIPPET_SELECTOR)
        .filter(|element| !inside_nosnippet(*element) && !in_site_chrome(*element))
        .map(visible_chars)
        .sum()
}

fn inside_nosnippet(element: ElementRef) -> bool {
    element.ancestors().filter_map(ElementRef::wrap).any(|ancestor| {
        matches!(ancestor.value().name(), "span" | "div" | "section")
            && ancestor.value().attr("data-nosnippet").is_some()
    })
}

/// The same site-chrome rule as `crate::ai::blocks`: a `header` / `footer` inside `article` or
/// `main` belongs to the content.
fn in_site_chrome(element: ElementRef) -> bool {
    let mut chain: Vec<ElementRef> = element.ancestors().filter_map(ElementRef::wrap).collect();
    chain.reverse();
    chain.push(element);
    let mut in_content = false;
    for el in chain {
        let tag = el.value().name();
        let role = el.value().attr("role").map(|role| role.trim().to_ascii_lowercase());
        if matches!(tag, "nav" | "aside")
            || matches!(role.as_deref(), Some("banner" | "contentinfo"))
            || (matches!(tag, "header" | "footer") && !in_content)
        {
            return true;
        }
        in_content |= matches!(tag, "article" | "main") || role.as_deref() == Some("main");
    }
    false
}

fn visible_chars(element: ElementRef) -> usize {
    const NOT_TEXT: &[&str] = &["script", "style", "template", "noscript", "svg", "iframe"];
    let is_text_holder = |node: NodeRef<Node>| {
        !node
            .ancestors()
            .take_while(|ancestor| ancestor.id() != element.id())
            .filter_map(|ancestor| ancestor.value().as_element())
            .any(|ancestor| NOT_TEXT.contains(&ancestor.name()))
    };
    let text: String = element
        .descendants()
        .filter(|node| is_text_holder(*node))
        .filter_map(|node| node.value().as_text().map(|text| text.to_string()))
        .collect();
    text.split_whitespace().collect::<Vec<_>>().join(" ").chars().count()
}

/// The page's canonical URL when it names another URL: the first `link rel="canonical"`,
/// resolved against the page URL (and a `base href`), fragments ignored. `/a` and `/a/` are
/// different URLs.
pub fn canonical_elsewhere(document: &Html, page_url: &str) -> Option<String> {
    let mut page = url::Url::parse(page_url).ok()?;
    page.set_fragment(None);
    let base = document
        .select(&BASE_SELECTOR)
        .next()
        .and_then(|base| base.value().attr("href"))
        .and_then(|href| page.join(href.trim()).ok())
        .unwrap_or_else(|| page.clone());
    let href = document
        .select(&LINK_SELECTOR)
        .find(|link| {
            link.value().attr("rel").is_some_and(|rel| {
                rel.split_ascii_whitespace()
                    .any(|token| token.eq_ignore_ascii_case("canonical"))
            })
        })?
        .value()
        .attr("href")?
        .trim();
    if href.is_empty() {
        return None;
    }
    let mut canonical = base.join(href).ok()?;
    canonical.set_fragment(None);
    (canonical != page).then(|| canonical.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn html(head: &str, body: &str) -> Html {
        Html::parse_document(&format!("<html><head>{head}</head><body>{body}</body></html>"))
    }

    /// Response headers as the HTTP client flattens them (one value per header instance).
    fn flat_headers(name: &str, instances: &[&str]) -> HashMap<String, String> {
        let raw = HashMap::from([(
            name.to_string(),
            instances.iter().map(|value| value.to_string()).collect::<Vec<_>>(),
        )]);
        crate::utils::get_flat_response_headers(&raw)
    }

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 26, 12, 0, 0).unwrap()
    }

    fn header_sources(instances: &[&str]) -> Vec<DirectiveSource> {
        sources(&html("", ""), Some(&flat_headers("x-robots-tag", instances)))
    }

    #[test]
    fn two_header_instances_keep_their_scopes() {
        let found = header_sources(&["googlebot: noarchive", "nosnippet"]);
        assert_eq!(
            found,
            vec![
                DirectiveSource {
                    origin: "X-Robots-Tag #1".to_string(),
                    scope: Scope::Googlebot,
                    directives: vec![Directive::NoArchive],
                },
                DirectiveSource {
                    origin: "X-Robots-Tag #2".to_string(),
                    scope: Scope::Generic,
                    directives: vec![Directive::NoSnippet],
                },
            ]
        );
        let google = google_policy(&found, now());
        assert!(google.nosnippet && !google.noindex, "{google:?}");
        let bing = bing_policy(&found, now());
        assert!(bing.nosnippet && !bing.noarchive, "{bing:?}");
        assert_eq!(bing.sources, ["X-Robots-Tag #2"]);
    }

    #[test]
    fn a_ua_prefix_scopes_the_directives_that_follow_it() {
        let found =
            header_sources(&["googlebot: noindex, nofollow, bingbot: nosnippet, max-snippet: 20, otherbot: noindex"]);
        assert_eq!(
            found,
            vec![
                DirectiveSource {
                    origin: "X-Robots-Tag #1".to_string(),
                    scope: Scope::Googlebot,
                    directives: vec![Directive::NoIndex, Directive::Other("nofollow".to_string())],
                },
                DirectiveSource {
                    origin: "X-Robots-Tag #1".to_string(),
                    scope: Scope::Bingbot,
                    directives: vec![Directive::NoSnippet, Directive::MaxSnippet(20)],
                },
                DirectiveSource {
                    origin: "X-Robots-Tag #1".to_string(),
                    scope: Scope::Other("otherbot".to_string()),
                    directives: vec![Directive::NoIndex],
                },
            ]
        );
        let google = google_policy(&found, now());
        assert!(
            google.noindex && !google.nosnippet && google.max_snippet.is_none(),
            "{google:?}"
        );
        let bing = bing_policy(&found, now());
        assert!(
            !bing.noindex && bing.nosnippet && bing.max_snippet == Some(20),
            "{bing:?}"
        );
    }

    #[test]
    fn meta_googlebot_and_bingbot_apply_to_their_engine_only() {
        let document = html(
            r#"<meta name="googlebot" content="noindex"><meta name="BingBot" content="nosnippet">
<meta name="robots" content="max-snippet:50"><meta name="otherbot" content="noindex">
<meta name="description" content="noindex">"#,
            "",
        );
        let found = sources(&document, None);
        assert_eq!(
            found
                .iter()
                .map(|source| (source.origin.as_str(), &source.scope))
                .collect::<Vec<_>>(),
            [
                ("meta googlebot", &Scope::Googlebot),
                ("meta bingbot", &Scope::Bingbot),
                ("meta robots", &Scope::Generic),
            ]
        );
        let google = google_policy(&found, now());
        assert!(
            google.noindex && !google.nosnippet && google.max_snippet == Some(50),
            "{google:?}"
        );
        assert_eq!(google.sources, ["meta googlebot", "meta robots"]);
        let bing = bing_policy(&found, now());
        assert!(
            !bing.noindex && bing.nosnippet && bing.max_snippet == Some(50),
            "{bing:?}"
        );
        assert_eq!(bing.sources, ["meta bingbot", "meta robots"]);
    }

    #[test]
    fn a_ua_prefix_inside_a_meta_tag_scopes_nothing() {
        let found = sources(&html(r#"<meta name="robots" content="googlebot: noindex">"#, ""), None);
        assert_eq!(
            found[0].directives,
            [Directive::Other("googlebot: noindex".to_string())]
        );
        assert!(!google_policy(&found, now()).noindex);
    }

    #[test]
    fn none_means_noindex_for_both_engines() {
        let found = sources(&html(r#"<meta name="robots" content="NONE">"#, ""), None);
        assert_eq!(found[0].directives, [Directive::None]);
        assert!(google_policy(&found, now()).noindex);
        assert!(bing_policy(&found, now()).noindex);
    }

    #[test]
    fn max_snippet_takes_the_most_restrictive_value() {
        let found = header_sources(&["max-snippet:-1", "googlebot: max-snippet:120", "max-snippet:0"]);
        assert_eq!(google_policy(&found, now()).max_snippet, Some(0));

        let unlimited = header_sources(&["max-snippet:-1"]);
        assert_eq!(google_policy(&unlimited, now()).max_snippet, Some(-1));
        assert!(
            google_policy(&unlimited, now()).sources.is_empty(),
            "-1 restricts nothing"
        );

        let invalid = header_sources(&["max-snippet:abc", "max-snippet:-5"]);
        assert_eq!(invalid[0].directives, [Directive::Other("max-snippet:abc".to_string())]);
        assert_eq!(google_policy(&invalid, now()).max_snippet, None);
        assert_eq!(google_policy(&[], now()), EnginePolicy::default());
    }

    #[test]
    fn unavailable_after_in_the_past_means_noindex() {
        for past in [
            "unavailable_after: 2026-01-01",
            "unavailable_after: 2026-09-01T10:00:00+02:00",
            "unavailable_after: 25 Jun 2010 15:00:00 PST",
            "unavailable_after: Wed, 03 Nov 2021 15:00:00 GMT",
            "unavailable_after: Wednesday, 03-Nov-2021 15:00:00 GMT",
            "unavailable_after: Wednesday, 03-Nov-21 15:00:00 GMT",
        ] {
            let found = header_sources(&[&format!("{past}, nosnippet")]);
            assert!(
                matches!(found[0].directives[0], Directive::UnavailableAfter(_)),
                "{past}: {found:?}"
            );
            assert_eq!(found[0].directives[1], Directive::NoSnippet, "{past}");
            for policy in [google_policy(&found, now()), bing_policy(&found, now())] {
                assert!(policy.expired && policy.noindex, "{past}: {policy:?}");
            }
        }

        for future in [
            "unavailable_after: 2027-01-01",
            "unavailable_after: Fri, 03 Nov 2028 15:00:00 GMT",
        ] {
            let found = header_sources(&[future]);
            let google = google_policy(&found, now());
            assert!(!google.expired && !google.noindex, "{future}: {google:?}");
        }

        let unparseable = header_sources(&["unavailable_after: someday"]);
        assert_eq!(
            unparseable[0].directives,
            [Directive::Other("unavailable_after: someday".to_string())]
        );
        assert!(!google_policy(&unparseable, now()).expired);
    }

    #[test]
    fn bing_treats_nocache_together_with_noarchive_as_nocache() {
        let noarchive = bing_policy(&header_sources(&["noarchive"]), now());
        assert!(noarchive.noarchive && !noarchive.nocache, "{noarchive:?}");

        let nocache = bing_policy(&header_sources(&["nocache"]), now());
        assert!(nocache.nocache && !nocache.noarchive, "{nocache:?}");

        let both = sources(
            &html(
                r#"<meta name="robots" content="noarchive"><meta name="bingbot" content="nocache">"#,
                "",
            ),
            None,
        );
        let bing = bing_policy(&both, now());
        assert!(bing.nocache && !bing.noarchive, "{bing:?}");

        // Google Search uses neither.
        let google = google_policy(&both, now());
        assert!(
            !google.noarchive && !google.nocache && google.sources.is_empty(),
            "{google:?}"
        );
    }

    #[test]
    fn data_nosnippet_counts_supported_main_elements_once() {
        let document = html(
            "",
            r#"<main><p>Visible text here.</p>
<div data-nosnippet>Hidden <span data-nosnippet>twice</span>   text<script>var x = 1;</script></div>
<section data-nosnippet>Sec</section><span data-nosnippet="">Span</span>
<p data-nosnippet>Paragraph is not supported</p><ul><li data-nosnippet>Item</li></ul>
<article><footer><div data-nosnippet>Byline</div></footer></article></main>
<footer><div data-nosnippet>Site footer text</div></footer><nav><span data-nosnippet>Menu</span></nav>"#,
        );
        // "Hidden twice text" + "Sec" + "Span" + "Byline"
        assert_eq!(data_nosnippet_chars(&document), 17 + 3 + 4 + 6);
        assert_eq!(data_nosnippet_chars(&html("", "<p>No markers</p>")), 0);
    }

    #[test]
    fn canonical_is_resolved_without_folding_a_trailing_slash() {
        let canonical = |href: &str, page: &str| {
            canonical_elsewhere(&html(&format!(r#"<link rel="canonical" href="{href}">"#), ""), page)
        };
        assert_eq!(
            canonical("/a/", "https://example.com/a").as_deref(),
            Some("https://example.com/a/")
        );
        assert_eq!(
            canonical("/a", "https://example.com/a/").as_deref(),
            Some("https://example.com/a")
        );
        assert_eq!(canonical("/a/", "https://example.com/a/"), None);
        assert_eq!(canonical("a", "https://example.com/a"), None);
        assert_eq!(
            canonical("other", "https://example.com/dir/page").as_deref(),
            Some("https://example.com/dir/other")
        );
        assert_eq!(
            canonical("https://EXAMPLE.com:443/a#top", "https://example.com/a"),
            None
        );
        assert_eq!(canonical("", "https://example.com/a"), None);
        assert_eq!(canonical_elsewhere(&html("", ""), "https://example.com/a"), None);

        let with_base = html(
            r#"<base href="https://cdn.example.com/x/"><link rel="Canonical alternate" href="page">"#,
            "",
        );
        assert_eq!(
            canonical_elsewhere(&with_base, "https://example.com/a").as_deref(),
            Some("https://cdn.example.com/x/page")
        );
    }
}
