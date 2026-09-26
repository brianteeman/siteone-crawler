// SiteOne Crawler - supporting page signals of the AI search readiness report
// (c) Jan Reges <jan.reges@siteone.cz>
//
// Smaller signals around the main checks: how much of a page's own text stays collapsed (tabs,
// accordions, closed disclosures), how many contextual links a page's content has to other pages
// of the site, images without an `alt` attribute, robots restrictions sent with PDF files (whose
// text is not inspected), and how many key pages report a usable `Last-Modified`.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use scraper::{ElementRef, Html, Selector};

use crate::ai::blocks::{Block, Region, blocks_from_html};
use crate::ai::geo::controls::{EnginePolicy, bing_policy, google_policy, in_site_chrome, sources};
use crate::ai::geo::keys::KeyPage;
use crate::export::sitemap_exporter::lastmod_from_headers;
use crate::result::status::Status;
use crate::result::visited_url::VisitedUrl;
use crate::types::ContentTypeId;

/// Elements whose links are never shown: the same as those `blocks_from_html` skips.
const NOT_SHOWN: &[&str] = &["head", "script", "style", "template", "noscript", "svg", "iframe"];

static LINK_SELECTOR: Lazy<Selector> = Lazy::new(|| Selector::parse("a[href]").unwrap());
static IMG_SELECTOR: Lazy<Selector> = Lazy::new(|| Selector::parse("img").unwrap());

/// The supporting signals of one page.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PageSignals {
    /// The visible text of the page's own content (Main-region blocks), in characters.
    pub main_chars: usize,
    /// The part of `main_chars` in collapsed blocks (closed tabs, accordions, disclosures).
    pub collapsed_chars: usize,
    /// Links from the page's own content (Main region) to other pages of the same site.
    pub in_content_links: usize,
    /// `img` elements without an `alt` attribute; an empty `alt` marks a decorative image and is
    /// valid (the accessibility analyzer's rule).
    pub images_without_alt: usize,
}

impl PageSignals {
    /// The collapsed part of the page's own text; `None` for a page without such text.
    pub fn collapsed_share(&self) -> Option<f64> {
        (self.main_chars > 0).then(|| self.collapsed_chars as f64 / self.main_chars as f64)
    }
}

/// A PDF whose `X-Robots-Tag` restricts Google or Bing.
#[derive(Debug, Clone, PartialEq)]
pub struct PdfRestriction {
    pub url: String,
    pub google: EnginePolicy,
    pub bing: EnginePolicy,
}

/// How many key pages (HTML 200) send a `Last-Modified`, and how many of those dates are usable
/// as a `lastmod` (the sitemap exporter's filter: not a placeholder, not the response time).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LastModifiedCoverage {
    pub pages: usize,
    pub with_header: usize,
    pub plausible: usize,
}

/// The supporting signals of a page from its HTML.
pub fn page_signals(page_url: &str, html: &str) -> PageSignals {
    let (main_chars, collapsed_chars) = main_text_chars(&blocks_from_html(html));
    let document = Html::parse_document(html);
    PageSignals {
        main_chars,
        collapsed_chars,
        in_content_links: in_content_links(&document, page_url),
        images_without_alt: document
            .select(&IMG_SELECTOR)
            .filter(|img| img.value().attr("alt").is_none())
            .count(),
    }
}

/// The characters of the Main-region blocks, and of the collapsed ones among them. Each text node
/// is in exactly one block, so nested collapsed containers count once.
pub(crate) fn main_text_chars(blocks: &[Block]) -> (usize, usize) {
    blocks
        .iter()
        .filter(|block| block.region == Region::Main)
        .fold((0, 0), |(all, collapsed), block| {
            let chars = block.text.chars().count();
            (all + chars, collapsed + if block.collapsed { chars } else { 0 })
        })
}

/// Links of the Main region to another page of the site (the page's host with or without
/// `www.`); links to the page itself, to a fragment of it, and to other schemes do not count.
fn in_content_links(document: &Html, page_url: &str) -> usize {
    let Ok(mut page) = url::Url::parse(page_url) else {
        return 0;
    };
    page.set_fragment(None);
    let Some(site) = page.host_str().map(|host| host.trim_start_matches("www.").to_string()) else {
        return 0;
    };
    document
        .select(&LINK_SELECTOR)
        .filter(|link| !in_site_chrome(*link) && !is_not_shown(*link))
        .filter_map(|link| page.join(link.value().attr("href")?.trim()).ok())
        .filter(|target| {
            let mut without_fragment = target.clone();
            without_fragment.set_fragment(None);
            matches!(target.scheme(), "http" | "https")
                && target
                    .host_str()
                    .is_some_and(|host| host.trim_start_matches("www.") == site)
                && without_fragment != page
        })
        .count()
}

fn is_not_shown(element: ElementRef) -> bool {
    element
        .ancestors()
        .filter_map(|ancestor| ancestor.value().as_element())
        .any(|ancestor| NOT_SHOWN.contains(&ancestor.name()))
}

/// The internal PDFs (200) whose `X-Robots-Tag` restricts Google or Bing, by URL. Their text is
/// not inspected.
pub fn pdf_restrictions(status: &Status, now: DateTime<Utc>) -> Vec<PdfRestriction> {
    let no_document = Html::new_document();
    let mut restricted: Vec<PdfRestriction> = status
        .get_visited_urls()
        .iter()
        .filter(|visit| !visit.is_external && visit.status_code == 200 && is_pdf(visit))
        .filter_map(|visit| {
            let headers = status.get_url_headers(&visit.uq_id)?;
            let found = sources(&no_document, Some(&headers));
            let google = google_policy(&found, now);
            let bing = bing_policy(&found, now);
            (!google.sources.is_empty() || !bing.sources.is_empty()).then(|| PdfRestriction {
                url: visit.url.clone(),
                google,
                bing,
            })
        })
        .collect();
    restricted.sort_by(|a, b| a.url.cmp(&b.url));
    restricted
}

/// Served as `application/pdf`, or as a document whose path ends with `.pdf`.
fn is_pdf(visit: &VisitedUrl) -> bool {
    visit
        .content_type_header
        .as_deref()
        .is_some_and(|content_type| content_type.to_ascii_lowercase().contains("application/pdf"))
        || (visit.content_type == ContentTypeId::Document
            && url::Url::parse(&visit.url).is_ok_and(|url| url.path().to_ascii_lowercase().ends_with(".pdf")))
}

/// The `Last-Modified` coverage of the key pages that answered 200 with HTML; `now` is when the
/// crawl ran.
pub fn last_modified_coverage(status: &Status, key: &[KeyPage], now: DateTime<Utc>) -> LastModifiedCoverage {
    let visited = status.get_visited_urls();
    let by_uq_id: HashMap<&str, &VisitedUrl> = visited.iter().map(|visit| (visit.uq_id.as_str(), visit)).collect();
    let mut coverage = LastModifiedCoverage::default();
    for page in key {
        let Some(visit) = by_uq_id.get(page.uq_id.as_str()) else {
            continue;
        };
        if visit.status_code != 200 || visit.content_type != ContentTypeId::Html {
            continue;
        }
        coverage.pages += 1;
        let Some(headers) = status.get_url_headers(&visit.uq_id) else {
            continue;
        };
        if headers.contains_key("last-modified") {
            coverage.with_header += 1;
            if lastmod_from_headers(&headers, now).is_some() {
                coverage.plausible += 1;
            }
        }
    }
    coverage
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::geo::keys::KeyPage;
    use crate::ai::geo::test_support::{add_with_headers, new_status, page};
    use crate::result::visited_url::{SOURCE_A_HREF, SOURCE_INIT_URL};
    use crate::types::ContentTypeId;
    use chrono::TimeZone;

    const URL: &str = "https://example.com/page";

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 26, 12, 0, 0).unwrap()
    }

    #[test]
    fn the_collapsed_share_counts_each_text_once_within_the_main_region() {
        let signals = page_signals(
            URL,
            r#"<body>
            <nav><div hidden><p>A long hidden mobile menu item</p></div></nav>
            <main>
              <p>Visible text</p>
              <details><summary>More</summary><div hidden><p>Deep hidden</p></div><p>Hidden price</p></details>
            </main>
            <footer><div class="collapse"><p>Footer collapsed</p></div></footer>
            </body>"#,
        );
        // "Visible text" + "More" + "Deep hidden" + "Hidden price"; the nested hidden block counts
        // once, the chrome not at all.
        assert_eq!(signals.main_chars, 12 + 4 + 11 + 12);
        assert_eq!(signals.collapsed_chars, 11 + 12);
        assert_eq!(signals.collapsed_share(), Some(23.0 / 39.0));

        let empty = page_signals(URL, "<body><nav><p>Menu</p></nav></body>");
        assert_eq!((empty.main_chars, empty.collapsed_share()), (0, None));
    }

    #[test]
    fn in_content_links_are_internal_links_of_the_main_region() {
        let signals = page_signals(
            URL,
            r##"<body>
            <header><nav><a href="/menu">Menu</a></nav></header>
            <main>
              <p>See <a href="/pricing">pricing</a>, <a href="https://example.com/contact">contact</a>,
                 <a href="https://www.example.com/about">about</a> and <a href="https://other.example/">a partner</a>.</p>
              <p><a href="#top">top</a> <a href="/page#faq">faq</a> <a href="mailto:a@example.com">mail</a>
                 <a href="tel:+420800123456">call</a> <a href="javascript:void(0)">js</a> <a>no href</a></p>
              <template><a href="/template">template</a></template>
              <article><footer><a href="/author">Author</a></footer></article>
            </main>
            <aside><a href="/related">Related</a></aside>
            <footer><a href="/privacy">Privacy</a></footer>
            </body>"##,
        );
        // /pricing, /contact, www.example.com/about (the same site) and the article's /author.
        assert_eq!(signals.in_content_links, 4);
        assert_eq!(page_signals(URL, "<body><p>No links</p></body>").in_content_links, 0);
    }

    #[test]
    fn images_without_an_alt_attribute_are_counted() {
        let signals = page_signals(
            URL,
            r#"<body><header><img src="/logo.png"></header>
            <main><img src="/a.png"><img src="/b.png" alt=""><img src="/c.png" alt="Chart"></main></body>"#,
        );
        // An empty alt marks a decorative image, as in the accessibility analyzer.
        assert_eq!(signals.images_without_alt, 2);
    }

    #[test]
    fn robots_restrictions_of_pdf_files_are_listed() {
        let mut status = new_status();
        let pdf = |uq_id: &str, url: &str, code: i32| {
            let mut visit = page(uq_id, "home", SOURCE_A_HREF, url, code, None);
            visit.content_type = ContentTypeId::Document;
            visit.content_type_header = Some("application/pdf".to_string());
            visit
        };
        add_with_headers(
            &mut status,
            page("home", "", SOURCE_INIT_URL, "https://example.com/", 200, None),
            None,
            &[("x-robots-tag", "noindex")],
        );
        add_with_headers(
            &mut status,
            pdf("a", "https://example.com/a.pdf", 200),
            None,
            &[("x-robots-tag", "noindex")],
        );
        add_with_headers(
            &mut status,
            pdf("b", "https://example.com/b.pdf", 200),
            None,
            &[("x-robots-tag", "googlebot: nosnippet\nbingbot: noarchive")],
        );
        add_with_headers(&mut status, pdf("c", "https://example.com/c.pdf", 200), None, &[]);
        add_with_headers(
            &mut status,
            pdf("d", "https://example.com/d.pdf", 200),
            None,
            &[("x-robots-tag", "nofollow")],
        );
        let mut external = pdf("e", "https://other.example/e.pdf", 200);
        external.is_external = true;
        add_with_headers(&mut status, external, None, &[("x-robots-tag", "noindex")]);
        add_with_headers(
            &mut status,
            pdf("f", "https://example.com/f.pdf", 404),
            None,
            &[("x-robots-tag", "noindex")],
        );
        // A PDF served under a generic document type is recognized by its extension.
        let mut generic = pdf("g", "https://example.com/files/G.PDF?v=2", 200);
        generic.content_type_header = Some("application/octet-stream".to_string());
        add_with_headers(&mut status, generic, None, &[("x-robots-tag", "none")]);

        let listed = pdf_restrictions(&status, now());
        assert_eq!(
            listed.iter().map(|pdf| pdf.url.as_str()).collect::<Vec<_>>(),
            [
                "https://example.com/a.pdf",
                "https://example.com/b.pdf",
                "https://example.com/files/G.PDF?v=2"
            ]
        );
        assert!(listed[0].google.noindex && listed[0].bing.noindex);
        assert!(listed[1].google.nosnippet && !listed[1].google.noindex);
        assert!(listed[1].bing.noarchive && !listed[1].bing.nosnippet);
        assert!(listed[2].google.noindex && listed[2].bing.noindex);
    }

    #[test]
    fn last_modified_coverage_counts_usable_dates_on_key_pages() {
        let mut status = new_status();
        let visits = [
            (
                "home",
                "https://example.com/",
                vec![("last-modified", "Fri, 17 Jul 2026 17:29:24 GMT")],
            ),
            // Stamped with the response time: present but says nothing about the content.
            (
                "now",
                "https://example.com/now",
                vec![
                    ("last-modified", "Sat, 26 Sep 2026 11:59:00 GMT"),
                    ("date", "Sat, 26 Sep 2026 11:59:00 GMT"),
                ],
            ),
            ("none", "https://example.com/none", vec![]),
            (
                "other",
                "https://example.com/other",
                vec![("last-modified", "Fri, 17 Jul 2026 17:29:24 GMT")],
            ),
        ];
        for (uq_id, url, headers) in &visits {
            let attr = if *uq_id == "home" {
                SOURCE_INIT_URL
            } else {
                SOURCE_A_HREF
            };
            add_with_headers(&mut status, page(uq_id, "home", attr, url, 200, None), None, headers);
        }
        let key: Vec<KeyPage> = ["home", "now", "none"]
            .iter()
            .map(|uq_id| KeyPage {
                uq_id: uq_id.to_string(),
                url: String::new(),
                status_code: 200,
                score: None,
                is_homepage: *uq_id == "home",
            })
            .collect();
        assert_eq!(
            last_modified_coverage(&status, &key, now()),
            LastModifiedCoverage {
                pages: 3,
                with_header: 2,
                plausible: 1,
            }
        );
    }
}
