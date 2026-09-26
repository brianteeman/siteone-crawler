// SiteOne Crawler - rendering risk of the key pages
// (c) Jan Reges <jan.reges@siteone.cz>
//
// Log studies found no JavaScript execution by the crawlers of OpenAI, Anthropic and Perplexity,
// while Google and Bing render pages. A key page whose raw HTML carries almost no text of its own
// and shows an app shell (an empty mount point of a JavaScript framework, or a `noscript` asking
// for JavaScript) is therefore "likely client-rendered": a rendering risk, not a proof.

use std::collections::HashMap;

use once_cell::sync::Lazy;
use scraper::{ElementRef, Html, Selector};

use crate::ai::blocks::blocks_from_html;
use crate::ai::geo::keys::KeyPage;
use crate::ai::geo::signals::main_text_chars;
use crate::result::status::Status;
use crate::result::visited_url::VisitedUrl;
use crate::types::ContentTypeId;

/// A page with less text of its own (Main region, in characters) than this may be an app shell.
pub const APP_SHELL_MAX_TEXT_CHARS: usize = 250;

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
    /// The app-shell signs found: `div#root`, `div#app`, `div#__next`, `div#__nuxt`, `app-root`
    /// (each only when empty) or `noscript` (one that mentions JavaScript).
    pub markers: Vec<&'static str>,
}

/// The plain-mode rendering check of the key pages.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RenderCheck {
    /// Key pages checked: HTML 200 responses with a stored body.
    pub checked: usize,
    pub risks: Vec<RenderRisk>,
}

/// The raw HTML of a page is an app shell when its own content (Main region, collapsed blocks
/// included, since they are in the HTML) has fewer than `APP_SHELL_MAX_TEXT_CHARS` characters
/// and it has an empty framework mount point or a `noscript` that mentions JavaScript.
pub fn plain_render_risk(url: &str, html: &str) -> Option<RenderRisk> {
    let (main_text_chars, _) = main_text_chars(&blocks_from_html(html));
    if main_text_chars >= APP_SHELL_MAX_TEXT_CHARS {
        return None;
    }
    let markers = app_shell_markers(&Html::parse_document(html));
    (!markers.is_empty()).then(|| RenderRisk {
        url: url.to_string(),
        main_text_chars,
        markers,
    })
}

/// Checks the key pages that answered 200 with HTML and whose body was stored.
pub fn plain_render_risks(status: &Status, key: &[KeyPage]) -> RenderCheck {
    let visited = status.get_visited_urls();
    let by_uq_id: HashMap<&str, &VisitedUrl> = visited.iter().map(|visit| (visit.uq_id.as_str(), visit)).collect();
    let mut check = RenderCheck::default();
    for page in key {
        let Some(visit) = by_uq_id.get(page.uq_id.as_str()) else {
            continue;
        };
        if visit.status_code != 200 || visit.content_type != ContentTypeId::Html {
            continue;
        }
        let Some(body) = status.get_url_body_text(&visit.uq_id) else {
            continue;
        };
        check.checked += 1;
        if let Some(risk) = plain_render_risk(&visit.url, &body) {
            check.risks.push(risk);
        }
    }
    check
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
        ];
        for html in &cases {
            assert_eq!(plain_render_risk(URL, html), None, "{html}");
        }
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
