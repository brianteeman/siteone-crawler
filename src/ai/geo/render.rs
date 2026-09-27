// SiteOne Crawler - rendering risk of the key pages
// (c) Jan Reges <jan.reges@siteone.cz>
//
// Log studies found no JavaScript execution by the crawlers of OpenAI, Anthropic and Perplexity,
// while Google and Bing render pages. A key page whose raw HTML carries almost no text of its own
// and shows an app shell (an empty mount point of a JavaScript framework, or a `noscript` asking
// for JavaScript) is therefore "likely client-rendered": a rendering risk, not a proof. In browser
// mode the stored page is the rendered one, so the check compares its text with the text of the
// HTML as fetched, which the renderer recorded before replacing the body (no refetch).

use std::collections::HashMap;

use once_cell::sync::Lazy;
use scraper::{ElementRef, Html, Selector};

use crate::ai::blocks::blocks_from_html;
use crate::ai::geo::keys::KeyPage;
use crate::ai::geo::signals::main_text_chars;
use crate::browser::diagnostics::BrowserDiagnostics;
use crate::result::status::Status;
use crate::result::visited_url::VisitedUrl;
use crate::types::ContentTypeId;

/// A page with less text of its own (Main region, in characters) than this may be an app shell.
pub const APP_SHELL_MAX_TEXT_CHARS: usize = 250;

/// Browser mode: a rendered page with less text of its own than this is not compared.
pub const RENDERED_MIN_TEXT_CHARS: usize = 500;

/// Browser mode: a page whose HTML as fetched holds less than this share of its rendered text is
/// a rendering risk.
pub const RAW_TO_RENDERED_MAX_RATIO: f64 = 0.5;

/// With a JavaScript notice in `noscript` as its only sign, a page is an app shell only below
/// this much text of its own (a "Loading…" at most): server-rendering frameworks such as Gatsby
/// add the notice to every page.
pub const NOSCRIPT_ONLY_MAX_TEXT_CHARS: usize = 25;

/// The mount points of JavaScript frameworks (React, Vue, Next.js, Nuxt; Angular's `app-root`),
/// as CSS selectors, which are also the markers the report shows.
const MOUNT_POINTS: &[&str] = &["div#root", "div#app", "div#__next", "div#__nuxt", "app-root"];

/// Elements whose text is not visible content.
const NOT_TEXT: &[&str] = &["script", "style", "template", "noscript", "svg", "iframe"];

static MOUNT_SELECTORS: Lazy<Vec<(Selector, &'static str)>> = Lazy::new(|| {
    MOUNT_POINTS
        .iter()
        .map(|marker| (Selector::parse(marker).unwrap(), *marker))
        .collect()
});
static NOSCRIPT_SELECTOR: Lazy<Selector> = Lazy::new(|| Selector::parse("noscript").unwrap());

/// A key page that is likely client-rendered.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderRisk {
    pub url: String,
    /// The visible text of the page's own content (Main region) in the raw HTML, in characters.
    pub main_text_chars: usize,
    /// Browser mode: the same measure on the rendered page.
    pub rendered_text_chars: Option<usize>,
    /// Plain mode: the app-shell signs found: `div#root`, `div#app`, `div#__next`, `div#__nuxt`,
    /// `app-root` (each only when empty) or `noscript` (one that mentions JavaScript).
    pub markers: Vec<&'static str>,
}

/// The rendering check of the key pages.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RenderCheck {
    /// Whether the pages were rendered in a browser (`--browser`) and compared with their HTML.
    pub browser: bool,
    /// Key pages checked: HTML 200 responses with a stored body (browser mode: rendered, with the
    /// text size of their HTML recorded).
    pub checked: usize,
    pub risks: Vec<RenderRisk>,
    /// Browser mode: HTML 200 key pages that cannot be compared — the rendering failed, or the
    /// page was not rendered (added after the crawl).
    pub not_comparable: Vec<String>,
}

/// Overlays whose text is not the page's own content although it sits outside the site chrome:
/// dialogs, and the containers of common cookie-consent tools (as hidden before screenshots, see
/// `browser::cookie_consent`). A script adds them to the rendered page; they may be in the HTML too.
const OVERLAYS: &[&str] = &[
    "dialog",
    "[role=dialog]",
    "[role=alertdialog]",
    "[aria-modal=true]",
    "#onetrust-consent-sdk",
    "#onetrust-banner-sdk",
    "#CookieConsent",
    "#CybotCookiebotDialog",
    "#cookiescript_injected",
    "#didomi-host",
    "#usercentrics-root",
    "#qc-cmp2-container",
    "#truste-consent-track",
    "[id^=sp_message_container]",
    "#cmplz-cookiebanner-container",
    "#cookie-law-info-bar",
    "#cookie-notice",
    ".cc-window",
    ".iubenda-cs-container",
    ".osano-cm-window",
];

static OVERLAY_SELECTORS: Lazy<Vec<Selector>> = Lazy::new(|| {
    OVERLAYS
        .iter()
        .map(|selector| Selector::parse(selector).unwrap())
        .collect()
});

/// The text of a page's own content (Main region) without overlays, in characters: all of it, and
/// the part in collapsed blocks.
fn own_text(html: &str) -> (usize, usize) {
    let mut document = Html::parse_document(html);
    let overlays: Vec<_> = OVERLAY_SELECTORS
        .iter()
        .flat_map(|selector| {
            document
                .select(selector)
                .map(|element| element.id())
                .collect::<Vec<_>>()
        })
        .collect();
    if overlays.is_empty() {
        return main_text_chars(&blocks_from_html(html));
    }
    for id in overlays {
        if let Some(mut node) = document.tree.get_mut(id) {
            node.detach();
        }
    }
    main_text_chars(&blocks_from_html(&document.html()))
}

/// The text of a page's own content (Main region, collapsed blocks included, since they are in
/// the HTML, overlays left out), in characters: the measure of the HTML as fetched.
pub fn own_text_chars(html: &str) -> usize {
    own_text(html).0
}

/// The text a rendered page shows as its own content: `own_text_chars` without collapsed blocks
/// (closed tabs, `aria-hidden` carousel clones), in characters.
pub fn visible_own_text_chars(html: &str) -> usize {
    let (all, collapsed) = own_text(html);
    all - collapsed
}

/// The raw HTML of a page is an app shell when its own content (Main region, collapsed blocks
/// included, since they are in the HTML) has fewer than `APP_SHELL_MAX_TEXT_CHARS` characters
/// and it has an empty framework mount point, or a `noscript` that mentions JavaScript — the
/// latter alone only below `NOSCRIPT_ONLY_MAX_TEXT_CHARS`.
pub fn plain_render_risk(url: &str, html: &str) -> Option<RenderRisk> {
    let main_text_chars = own_text_chars(html);
    if main_text_chars >= APP_SHELL_MAX_TEXT_CHARS {
        return None;
    }
    let markers = app_shell_markers(&Html::parse_document(html));
    if markers == ["noscript"] && main_text_chars >= NOSCRIPT_ONLY_MAX_TEXT_CHARS {
        return None;
    }
    (!markers.is_empty()).then(|| RenderRisk {
        url: url.to_string(),
        main_text_chars,
        rendered_text_chars: None,
        markers,
    })
}

/// Browser mode: a rendered page with at least `RENDERED_MIN_TEXT_CHARS` of text of its own is a
/// rendering risk when its HTML as fetched held less than `RAW_TO_RENDERED_MAX_RATIO` of it.
pub fn browser_render_risk(url: &str, raw_text_chars: usize, rendered_text_chars: usize) -> Option<RenderRisk> {
    let risky = rendered_text_chars >= RENDERED_MIN_TEXT_CHARS
        && (raw_text_chars as f64) < RAW_TO_RENDERED_MAX_RATIO * rendered_text_chars as f64;
    risky.then(|| RenderRisk {
        url: url.to_string(),
        main_text_chars: raw_text_chars,
        rendered_text_chars: Some(rendered_text_chars),
        markers: Vec::new(),
    })
}

/// Checks the key pages that answered 200 with HTML and whose body was stored.
pub fn plain_render_risks(status: &Status, key: &[KeyPage]) -> RenderCheck {
    let mut check = RenderCheck::default();
    for (visit, body) in html_key_pages(status, key) {
        check.checked += 1;
        if let Some(risk) = plain_render_risk(&visit.url, &body) {
            check.risks.push(risk);
        }
    }
    check
}

/// Browser mode: compares each rendered HTML 200 key page with the text size of its HTML as
/// fetched, recorded by the renderer. A page whose rendering failed keeps the HTML as fetched and
/// gets the plain check; a page without a record (added after the crawl without rendering) is not
/// comparable.
pub fn browser_render_risks(status: &Status, key: &[KeyPage]) -> RenderCheck {
    let mut check = RenderCheck {
        browser: true,
        ..RenderCheck::default()
    };
    for (visit, body) in html_key_pages(status, key) {
        let diagnostics: Option<BrowserDiagnostics> = status.get_browser_diagnostics(&visit.uq_id);
        let risk = match diagnostics {
            Some(diagnostics) if diagnostics.render_error.is_some() => plain_render_risk(&visit.url, &body),
            Some(BrowserDiagnostics {
                raw_text_chars: Some(raw),
                ..
            }) => browser_render_risk(&visit.url, raw, visible_own_text_chars(&body)),
            _ => {
                check.not_comparable.push(visit.url.clone());
                continue;
            }
        };
        check.checked += 1;
        check.risks.extend(risk);
    }
    check
}

/// The key pages that answered 200 with HTML, with their stored body, in key-page order.
fn html_key_pages(status: &Status, key: &[KeyPage]) -> Vec<(VisitedUrl, String)> {
    let visited = status.get_visited_urls();
    let by_uq_id: HashMap<&str, &VisitedUrl> = visited.iter().map(|visit| (visit.uq_id.as_str(), visit)).collect();
    key.iter()
        .filter_map(|page| by_uq_id.get(page.uq_id.as_str()))
        .filter(|visit| visit.status_code == 200 && visit.content_type == ContentTypeId::Html)
        .filter_map(|visit| Some(((*visit).clone(), status.get_url_body_text(&visit.uq_id)?)))
        .collect()
}

/// The app-shell signs of a page, in `MOUNT_POINTS` order, then `noscript`.
fn app_shell_markers(document: &Html) -> Vec<&'static str> {
    let mut markers: Vec<&'static str> = MOUNT_SELECTORS
        .iter()
        .filter(|(selector, _)| document.select(selector).any(|element| !has_visible_text(element)))
        .map(|(_, marker)| *marker)
        .collect();
    if document.select(&NOSCRIPT_SELECTOR).any(|noscript| {
        noscript
            .text()
            .collect::<String>()
            .to_ascii_lowercase()
            .contains("javascript")
    }) {
        markers.push("noscript");
    }
    markers
}

/// True when the element holds text outside scripts, styles and other non-text elements.
fn has_visible_text(element: ElementRef) -> bool {
    element.descendants().any(|node| {
        node.value().as_text().is_some_and(|text| !text.trim().is_empty())
            && !node
                .ancestors()
                .take_while(|ancestor| ancestor.id() != element.id())
                .filter_map(|ancestor| ancestor.value().as_element())
                .any(|ancestor| NOT_TEXT.contains(&ancestor.name()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::geo::keys::KeyPage;
    use crate::ai::geo::test_support::{add, new_status, page};
    use crate::browser::diagnostics::BrowserDiagnostics;
    use crate::result::visited_url::{SOURCE_A_HREF, SOURCE_INIT_URL};

    const URL: &str = "https://example.com/app";

    fn long_text() -> String {
        "Server-rendered text that engines can read without running any script. ".repeat(5)
    }

    #[test]
    fn app_shells_with_almost_no_text_are_a_rendering_risk() {
        let cases = [
            (
                r#"<body><div id="root"></div><script src="/app.js"></script></body>"#,
                "div#root",
            ),
            (
                r#"<body><div id="app">  <!-- mounted by Vue --> </div></body>"#,
                "div#app",
            ),
            (
                r#"<body><div id="__next"><div class="spinner"></div><script>self.__next_f=[]</script></div></body>"#,
                "div#__next",
            ),
            (
                r#"<body><div id="__nuxt"><noscript>Loading</noscript></div></body>"#,
                "div#__nuxt",
            ),
            (r#"<body><app-root></app-root></body>"#, "app-root"),
            (
                r#"<body><noscript>You need to enable JavaScript to run this app.</noscript><div id="main"></div></body>"#,
                "noscript",
            ),
        ];
        for (html, marker) in cases {
            let risk = plain_render_risk(URL, html).unwrap_or_else(|| panic!("no risk found in {html}"));
            assert_eq!(risk.markers, [marker], "{html}");
            assert_eq!(risk.main_text_chars, 0, "{html}");
            assert_eq!(risk.url, URL);
        }

        // Site chrome rendered on the server does not make the page's own content readable.
        let footer = format!(
            r#"<body><div id="app"></div><footer><p>{}</p></footer></body>"#,
            long_text()
        );
        let risk = plain_render_risk(URL, &footer).expect("the content is still client-rendered");
        assert_eq!(risk.markers, ["div#app"]);
        assert_eq!(risk.main_text_chars, 0);

        // Short own text still counts as "almost no text".
        let short = r#"<body><h1>Dashboard</h1><div id="root"></div></body>"#;
        assert_eq!(plain_render_risk(URL, short).map(|risk| risk.main_text_chars), Some(9));
    }

    #[test]
    fn normal_pages_are_no_rendering_risk() {
        let long = long_text();
        let cases = [
            // Next.js rendered on the server: the mount point holds the content.
            format!(r#"<body><div id="__next"><main><h1>Pricing</h1><p>{long}</p></main></div></body>"#),
            // Enough text of its own, even with an empty widget mount point.
            format!(r#"<body><main><p>{long}</p></main><div id="root"></div></body>"#),
            // A short page without an app shell.
            r#"<body><h1>Contact</h1><p>Call us.</p></body>"#.to_string(),
            // A tracking pixel in `noscript` does not ask for JavaScript.
            r#"<body><h1>Contact</h1><noscript><img src="https://www.facebook.com/tr?id=1"></noscript></body>"#
                .to_string(),
            // A mount point with text of its own is not empty.
            r#"<body><div id="root"><p>Hello</p></div></body>"#.to_string(),
            // Gatsby renders on the server and still adds this notice to every page: a JavaScript
            // notice alone means a shell only when the page has next to no text of its own.
            r#"<body><noscript id="gatsby-noscript">This app works best with JavaScript enabled.</noscript>
            <div id="___gatsby"><main><h1>Contact</h1><p>Call us or write to us.</p></main></div></body>"#
                .to_string(),
        ];
        for html in &cases {
            assert_eq!(plain_render_risk(URL, html), None, "{html}");
        }
    }

    #[test]
    fn the_browser_comparison_flags_text_that_appears_only_after_rendering() {
        let risk = browser_render_risk(URL, 120, 2_000).expect("most of the text comes from JavaScript");
        assert_eq!(risk.url, URL);
        assert_eq!((risk.main_text_chars, risk.rendered_text_chars), (120, Some(2_000)));
        assert!(risk.markers.is_empty());
        // Half of the text or more is in the HTML as fetched.
        assert_eq!(browser_render_risk(URL, 1_000, 2_000), None);
        assert_eq!(browser_render_risk(URL, 1_900, 2_000), None);
        // A script may also remove text.
        assert_eq!(browser_render_risk(URL, 3_000, 2_000), None);
        // Too little rendered text to compare.
        assert_eq!(browser_render_risk(URL, 0, RENDERED_MIN_TEXT_CHARS - 1), None);
        assert!(browser_render_risk(URL, 0, RENDERED_MIN_TEXT_CHARS).is_some());
    }

    #[test]
    fn dialogs_consent_banners_and_hidden_clones_are_no_rendered_text_of_the_page() {
        let own = "<p>Our plans cover hosting, backups and support for every size of business.</p>".repeat(15);
        let raw = format!("<html><body><main><h1>Pricing</h1>{own}</main></body></html>");
        let raw_chars = own_text_chars(&raw);
        let dialog =
            "<p>We use cookies to measure traffic and to personalise content and ads. Details below.</p>".repeat(30);
        let rendered = format!(
            "<html><body><main><h1>Pricing</h1>{own}\
             <div class=\"slick-slide slick-cloned\" aria-hidden=\"true\">{own}</div></main>\
             <div id=\"CybotCookiebotDialog\">{dialog}</div>\
             <div role=\"dialog\" aria-modal=\"true\"><p>Subscribe to our newsletter for news and offers every week.</p></div>\
             </body></html>"
        );
        assert_eq!(visible_own_text_chars(&rendered), raw_chars);
        assert_eq!(
            browser_render_risk(URL, raw_chars, visible_own_text_chars(&rendered)),
            None
        );
        // A consent dialog in the HTML as fetched is no text of the page either.
        let with_banner =
            format!("<html><body><div id=\"onetrust-consent-sdk\">{dialog}</div><div id=\"root\"></div></body></html>");
        assert_eq!(own_text_chars(&with_banner), 0);
        assert!(plain_render_risk(URL, &with_banner).is_some(), "still an app shell");
    }

    #[test]
    fn rendered_key_pages_are_compared_with_their_html_and_the_others_are_not_comparable() {
        let mut status = new_status();
        let long = long_text();
        let rendered = format!("<html><body><main><h1>Pricing</h1><p>{long}{long}</p></main></body></html>");
        let rendered_chars = own_text_chars(&rendered);
        assert!(rendered_chars >= RENDERED_MIN_TEXT_CHARS, "{rendered_chars}");
        let diagnostics = |raw: Option<usize>, error: Option<&str>| BrowserDiagnostics {
            raw_text_chars: raw,
            render_error: error.map(str::to_string),
            ..Default::default()
        };
        let pages = [
            // Server-rendered: the same text before and after rendering.
            (
                "home",
                "https://example.com/",
                Some(diagnostics(Some(rendered_chars), None)),
            ),
            // A single-page app: next to no text in the HTML as fetched.
            ("spa", "https://example.com/app", Some(diagnostics(Some(9), None))),
            // The render failed; the stored body is the HTML as fetched, checked as in a plain crawl.
            (
                "failed",
                "https://example.com/failed",
                Some(diagnostics(None, Some("navigation failed"))),
            ),
            // Inserted after the crawl (gap-fill) without rendering.
            ("bare", "https://example.com/bare", None),
        ];
        let shell = "<html><body><div id=\"root\"></div></body></html>";
        for (uq_id, url, diagnostics) in pages {
            let body = if uq_id == "failed" { shell } else { rendered.as_str() };
            let source = if uq_id == "home" { "" } else { "home" };
            let attr = if uq_id == "home" {
                SOURCE_INIT_URL
            } else {
                SOURCE_A_HREF
            };
            add(&mut status, page(uq_id, source, attr, url, 200, None), Some(body));
            if let Some(diagnostics) = diagnostics {
                status.add_browser_diagnostics(uq_id, diagnostics);
            }
        }
        add(
            &mut status,
            page("denied", "home", SOURCE_A_HREF, "https://example.com/denied", 403, None),
            Some(&rendered),
        );
        let key: Vec<KeyPage> = ["home", "spa", "failed", "bare", "denied"]
            .iter()
            .map(|uq_id| {
                let visit = status
                    .get_visited_urls()
                    .into_iter()
                    .find(|visit| visit.uq_id == *uq_id)
                    .unwrap();
                KeyPage {
                    uq_id: uq_id.to_string(),
                    url: visit.url.clone(),
                    status_code: visit.status_code,
                    score: None,
                    is_homepage: *uq_id == "home",
                }
            })
            .collect();

        let check = browser_render_risks(&status, &key);
        assert!(check.browser);
        assert_eq!(
            check.checked, 3,
            "the rendered HTML 200 pages with a known raw size, and the failed render"
        );
        assert_eq!(
            check.risks,
            [
                RenderRisk {
                    url: "https://example.com/app".to_string(),
                    main_text_chars: 9,
                    rendered_text_chars: Some(rendered_chars),
                    markers: Vec::new(),
                },
                RenderRisk {
                    url: "https://example.com/failed".to_string(),
                    main_text_chars: 0,
                    rendered_text_chars: None,
                    markers: vec!["div#root"],
                },
            ]
        );
        assert_eq!(check.not_comparable, ["https://example.com/bare"]);
        // The plain check never marks its result as a browser comparison.
        assert!(!plain_render_risks(&status, &key).browser);
    }

    #[test]
    fn key_pages_with_a_stored_html_body_are_checked() {
        let mut status = new_status();
        let long = long_text();
        let home = format!("<html><body><main><h1>Home</h1><p>{long}</p></main></body></html>");
        add(
            &mut status,
            page("home", "", SOURCE_INIT_URL, "https://example.com/", 200, None),
            Some(&home),
        );
        add(
            &mut status,
            page("app", "home", SOURCE_A_HREF, "https://example.com/app", 200, None),
            Some(r#"<html><body><div id="root"></div></body></html>"#),
        );
        add(
            &mut status,
            page("denied", "home", SOURCE_A_HREF, "https://example.com/denied", 403, None),
            Some(r#"<html><body><div id="root"></div></body></html>"#),
        );
        add(
            &mut status,
            page("nobody", "home", SOURCE_A_HREF, "https://example.com/nobody", 200, None),
            None,
        );
        let key: Vec<KeyPage> = [
            ("home", "https://example.com/", 200),
            ("app", "https://example.com/app", 200),
            ("denied", "https://example.com/denied", 403),
            ("nobody", "https://example.com/nobody", 200),
        ]
        .iter()
        .map(|(uq_id, url, code)| KeyPage {
            uq_id: uq_id.to_string(),
            url: url.to_string(),
            status_code: *code,
            score: None,
            is_homepage: *uq_id == "home",
        })
        .collect();

        let check = plain_render_risks(&status, &key);
        assert_eq!(check.checked, 2, "only HTML 200 pages with a body are checked");
        assert_eq!(
            check.risks.iter().map(|risk| risk.url.as_str()).collect::<Vec<_>>(),
            ["https://example.com/app"]
        );
    }
}
