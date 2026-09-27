// SiteOne Crawler - structured data of the AI search readiness report
// (c) Jan Reges <jan.reges@siteone.cz>
//
// Reads the structured data a page already has (JSON-LD, with the types it declares, its parse
// errors and the key values that are not visible on the page; Microdata and RDFa presence), and
// builds the kit's JSON-LD with typed templates from the crawler's own copies of verified page
// blocks — never from values written by a model: WebSite, Organization (a logo, contacts and
// social profiles only from the site chrome, `sameAs` only for profiles whose handle is the
// brand's name),
// BreadcrumbList from crawled ancestors, FAQPage from visible questions with their answers, and
// Article / BlogPosting from the H1 and a verified byline. Install snippets escape `<`, `>` and
// `&`, so no value can close the script element.

use std::collections::HashMap;

use chrono::NaiveDate;
use once_cell::sync::Lazy;
use scraper::{ElementRef, Html, Selector};
use serde_json::{Map, Value, json};
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;

use crate::ai::blocks::{Block, Region, hidden_for_good};
use crate::ai::geo::analyze::{DateRole, shown_text};
use crate::ai::geo::controls::in_site_chrome;
use crate::ai::geo::keys::normalized_url;
use crate::ai::grounding::{find_token_bounded, numbers_in};

const SCHEMA_ORG: &str = "https://schema.org";

/// At most this many key values of a page are checked for visibility; the rest are counted in
/// `ExistingMarkup::values_not_checked`, so a huge catalog in JSON-LD cannot stall the report.
pub const MAX_CHECKED_VALUES: usize = 200;

/// Properties whose values must be visible on the page (Google: structured data must match the
/// visible content). `author` as a plain string counts as `author.name`, and `price` only inside
/// `offers`.
const KEY_PROPERTIES: &[&str] = &[
    "name",
    "headline",
    "telephone",
    "email",
    "streetAddress",
    "postalCode",
    "addressLocality",
    "sku",
];

/// Leading labels of a byline before the author's name (a longer label before its start).
const AUTHOR_LABELS: &[&str] = &[
    "written by",
    "posted by",
    "by",
    "author",
    "autor článku",
    "autorka článku",
    "autorka",
    "autor",
    "napsala",
    "napsal",
];

/// Where the name after a byline label ends: the delimiters before a role, a date or a reading
/// time (a digit ends it too, and the words `on` and `dne`).
const BYLINE_DELIMITERS: &[&str] = &[",", "|", "·", "•", ";", "(", "/", " – ", " — ", " - "];
const BYLINE_DATE_WORDS: &[&str] = &["on", "dne"];

/// A person's name has at most this many words.
const MAX_AUTHOR_WORDS: usize = 5;

/// Separators of a title's site suffix (`Pricing | Example`).
const TITLE_SEPARATORS: &[&str] = &[" | ", " - ", " – ", " — ", " · ", " :: ", " » "];

/// Single path segments that are not profiles on social platforms: share and intent endpoints,
/// posts, search and login pages.
const NOT_PROFILES: &[&str] = &[
    "sharer",
    "sharer.php",
    "share",
    "share.php",
    "dialog",
    "intent",
    "home",
    "login",
    "search",
    "explore",
    "hashtag",
    "i",
    "p",
    "reel",
    "reels",
    "tr",
    "plugins",
    "watch",
];

static SCRIPT_SELECTOR: Lazy<Selector> = Lazy::new(|| Selector::parse("script[type]").unwrap());
static MARKUP_SELECTOR: Lazy<Selector> =
    Lazy::new(|| Selector::parse("[itemscope], [itemtype], [typeof], [vocab]").unwrap());
static HTML_SELECTOR: Lazy<Selector> = Lazy::new(|| Selector::parse("html[lang]").unwrap());
static LINK_SELECTOR: Lazy<Selector> = Lazy::new(|| Selector::parse("a[href]").unwrap());
static IMG_SELECTOR: Lazy<Selector> = Lazy::new(|| Selector::parse("img[src]").unwrap());
static H1_SELECTOR: Lazy<Selector> = Lazy::new(|| Selector::parse("h1").unwrap());
static TITLE_SELECTOR: Lazy<Selector> = Lazy::new(|| Selector::parse("title").unwrap());
static SITE_NAME_SELECTOR: Lazy<Selector> =
    Lazy::new(|| Selector::parse(r#"meta[property="og:site_name"][content]"#).unwrap());

/// The structured data a page already has.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExistingMarkup {
    /// The types of the JSON-LD entities (top-level objects and `@graph` items) as short names
    /// (`Article` for `https://schema.org/Article` or `schema:Article`), each once.
    pub jsonld_types: Vec<String>,
    /// `JSON-LD block N: <error>` for every block that is not valid JSON (N counts from 1).
    pub parse_errors: Vec<String>,
    /// Key values not visible on the page: (`Type.property.path`, value), in block order.
    pub invisible_values: Vec<(String, String)>,
    /// `itemscope` / `itemtype` attributes.
    pub has_microdata: bool,
    /// `typeof` / `vocab` attributes (Open Graph's `property` alone is not RDFa here).
    pub has_rdfa: bool,
    /// The types declared by Microdata `itemtype` and RDFa `typeof`, as short names, each once.
    pub other_types: Vec<String>,
    /// The types of objects nested in the JSON-LD entities (a `publisher` Organization, an
    /// `offers` Offer, …) that are not entity types too, each once.
    pub nested_types: Vec<String>,
    /// Key values left unchecked above `MAX_CHECKED_VALUES`.
    pub values_not_checked: usize,
}

impl ExistingMarkup {
    /// The page already declares this type somewhere: as a JSON-LD entity, nested in one, or in
    /// Microdata or RDFa (for "merge with the existing markup" notes).
    pub fn declares(&self, kind: &str) -> bool {
        self.jsonld_types
            .iter()
            .chain(&self.nested_types)
            .chain(&self.other_types)
            .any(|known| known == kind)
    }
}

/// The structured data of a page: its JSON-LD blocks, checked against `visible_text` (the text of
/// the page's visible blocks, Main and Chrome) together with the page title and its declared
/// `og:site_name` (SEO plugins name the WebPage by its title and the WebSite by the site name),
/// and the presence of Microdata and RDFa. An empty JSON-LD block is ignored.
pub fn existing(document: &Html, visible_text: &str) -> ExistingMarkup {
    let lang = document
        .select(&HTML_SELECTOR)
        .next()
        .and_then(|html| html.value().attr("lang"))
        .unwrap_or_default();
    let title: String = document
        .select(&TITLE_SELECTOR)
        .next()
        .map(|title| title.text().collect())
        .unwrap_or_default();
    let site_name = document
        .select(&SITE_NAME_SELECTOR)
        .next()
        .and_then(|meta| meta.value().attr("content"))
        .unwrap_or_default();
    let shown = format!("{visible_text}\n{title}\n{site_name}");
    let mut check = ValueCheck {
        visible: Visible::new(&shown, lang),
        invisible: Vec::new(),
        checked: 0,
        not_checked: 0,
    };
    let mut markup = ExistingMarkup::default();
    let scripts = document.select(&SCRIPT_SELECTOR).filter(|script| {
        script
            .value()
            .attr("type")
            .is_some_and(|kind| kind.trim().to_ascii_lowercase().starts_with("application/ld+json"))
    });
    for (index, script) in scripts.enumerate() {
        let text: String = script.text().collect();
        if text.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(text.trim()) {
            Ok(value) => {
                for entity in entities(&value) {
                    let types = types_of(entity);
                    for kind in &types {
                        push_unique(&mut markup.jsonld_types, kind);
                    }
                    for nested in entity.values().flat_map(nested_objects) {
                        collect_types(nested, &mut markup.nested_types);
                    }
                    let label = types.first().map_or("entity", String::as_str);
                    check.values(entity, label, false);
                }
            }
            Err(error) => markup
                .parse_errors
                .push(format!("JSON-LD block {}: {}", index + 1, error)),
        }
    }
    let entity_types = markup.jsonld_types.clone();
    markup.nested_types.retain(|kind| !entity_types.contains(kind));
    markup.invisible_values = check.invisible;
    markup.values_not_checked = check.not_checked;
    for element in document.select(&MARKUP_SELECTOR) {
        let el = element.value();
        markup.has_microdata |= el.attr("itemscope").is_some() || el.attr("itemtype").is_some();
        markup.has_rdfa |= el.attr("typeof").is_some() || el.attr("vocab").is_some();
        for declared in [el.attr("itemtype"), el.attr("typeof")].into_iter().flatten() {
            for kind in declared.split_whitespace().filter_map(short_type) {
                push_unique(&mut markup.other_types, &kind);
            }
        }
    }
    markup
}

/// The entities of a JSON-LD value: its objects, and the items of an `@graph` (the wrapper
/// itself only when it has a type).
fn entities(value: &Value) -> Vec<&Map<String, Value>> {
    match value {
        Value::Array(items) => items.iter().flat_map(entities).collect(),
        Value::Object(object) => {
            let mut found = Vec::new();
            if object.contains_key("@type") || !object.contains_key("@graph") {
                found.push(object);
            }
            if let Some(graph) = object.get("@graph") {
                found.extend(entities(graph));
            }
            found
        }
        _ => Vec::new(),
    }
}

fn types_of(entity: &Map<String, Value>) -> Vec<String> {
    match entity.get("@type") {
        Some(Value::String(kind)) => short_type(kind).into_iter().collect(),
        Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).filter_map(short_type).collect(),
        _ => Vec::new(),
    }
}

/// `https://schema.org/Article`, `schema:Article` and `Article` are all `Article`.
fn short_type(kind: &str) -> Option<String> {
    let short = kind.trim().rsplit(['/', '#', ':']).next().unwrap_or_default();
    (!short.is_empty()).then(|| short.to_string())
}

fn push_unique(list: &mut Vec<String>, value: &str) {
    if !list.iter().any(|known| known == value) {
        list.push(value.to_string());
    }
}

/// The types of an object and of the objects nested in it.
fn collect_types(object: &Map<String, Value>, found: &mut Vec<String>) {
    for kind in types_of(object) {
        push_unique(found, &kind);
    }
    for nested in object.values().flat_map(nested_objects) {
        collect_types(nested, found);
    }
}

/// The visibility check of the key values of a page's JSON-LD.
struct ValueCheck<'a> {
    visible: Visible<'a>,
    /// (`Type.property.path`, value) of the values not visible.
    invisible: Vec<(String, String)>,
    checked: usize,
    not_checked: usize,
}

impl ValueCheck<'_> {
    /// Walks an entity (and the objects nested in it) and records the key values that are not
    /// visible, up to `MAX_CHECKED_VALUES`. `in_offers` is true below an `offers` property.
    fn values(&mut self, object: &Map<String, Value>, path: &str, in_offers: bool) {
        for (key, value) in object.iter().filter(|(key, _)| !key.starts_with('@')) {
            let key_path = format!("{path}.{key}");
            let checked = if KEY_PROPERTIES.contains(&key.as_str()) || (key == "price" && in_offers) {
                Some((key.as_str(), key_path.clone()))
            } else if key == "author" && value.is_string() {
                Some(("name", format!("{key_path}.name")))
            } else {
                None
            };
            if let Some((property, label)) = checked {
                for leaf in leaves(value) {
                    if self.checked >= MAX_CHECKED_VALUES {
                        self.not_checked += 1;
                        continue;
                    }
                    self.checked += 1;
                    if !self.visible.contains(property, &leaf) {
                        self.invisible.push((label.clone(), leaf));
                    }
                }
            }
            let nested_in_offers = in_offers || key == "offers";
            for nested in nested_objects(value) {
                self.values(nested, &key_path, nested_in_offers);
            }
        }
    }
}

/// The non-empty strings and numbers of a value (an array holds several).
fn leaves(value: &Value) -> Vec<String> {
    match value {
        Value::String(text) if !text.trim().is_empty() => vec![text.trim().to_string()],
        Value::Number(number) => vec![number.to_string()],
        Value::Array(items) => items.iter().flat_map(leaves).collect(),
        _ => Vec::new(),
    }
}

fn nested_objects(value: &Value) -> Vec<&Map<String, Value>> {
    match value {
        Value::Object(object) => vec![object],
        Value::Array(items) => items.iter().flat_map(nested_objects).collect(),
        _ => Vec::new(),
    }
}

/// The visible text of a page, prepared for the checks of its values.
struct Visible<'a> {
    text: &'a str,
    numbers: Vec<f64>,
    /// The digits of each run of digits, spaces and phone punctuation.
    digit_runs: Vec<String>,
}

impl<'a> Visible<'a> {
    fn new(text: &'a str, lang: &str) -> Self {
        let digit_runs = text
            .split(|c: char| !(c.is_ascii_digit() || c.is_whitespace() || "+()-/.".contains(c)))
            .map(digits)
            .filter(|run| !run.is_empty())
            .collect();
        Self {
            text,
            numbers: numbers_in(text, lang),
            digit_runs,
        }
    }

    /// Phone numbers and postal codes match by their digits (`+420800123456` = `800 123 456`,
    /// `11000` = `110 00`); a price as a number; other values as whole tokens.
    fn contains(&self, property: &str, value: &str) -> bool {
        match property {
            "telephone" | "postalCode" if !value.chars().any(char::is_alphabetic) => {
                let wanted = digits(value);
                !wanted.is_empty()
                    && self
                        .digit_runs
                        .iter()
                        .any(|run| run.ends_with(&wanted) || (run.len() >= 6 && wanted.ends_with(run.as_str())))
            }
            "price" => match value.trim().parse::<f64>() {
                Ok(price) => self.numbers.iter().any(|number| (number - price).abs() < 0.005),
                Err(_) => find_token_bounded(self.text, value).is_some(),
            },
            _ => find_token_bounded(self.text, value).is_some(),
        }
    }
}

fn digits(text: &str) -> String {
    text.chars().filter(char::is_ascii_digit).collect()
}

/// The kit's WebSite entity.
pub fn website(site_name: &str, origin: &str) -> Value {
    let origin = origin.trim_end_matches('/');
    json!({
        "@context": SCHEMA_ORG,
        "@type": "WebSite",
        "@id": format!("{origin}/#website"),
        "name": site_name,
        "url": format!("{origin}/"),
    })
}

/// The kit's Organization entity from what the homepage's site chrome shows (nothing hidden for
/// good, see `blocks::hides_for_good`): a `logo` (the first image of
/// the site header, or linking to the homepage, whose `src`, `alt` or `class` says "logo" — not a
/// partner's or a payment logo in the footer), a `contactPoint` from the first `tel:` and
/// `mailto:` links whose values the chrome blocks show, and `sameAs` with the organization
/// profiles whose handle is the site's name (`name_forms`: without case, diacritics, punctuation
/// or a trailing legal form — `@SiteOne-Crawler` for "SiteOne Crawler"). The other social profiles
/// of the chrome — a founder's LinkedIn, a product's or a country's account, a web agency or a
/// partner whose handle holds the brand word — are returned as possible profiles for the owner to
/// add by hand.
pub fn organization(
    site_name: &str,
    origin: &str,
    homepage: &Html,
    chrome_blocks: &[Block],
    base: &url::Url,
) -> (Value, Vec<String>) {
    let origin = origin.trim_end_matches('/');
    let mut entity = Map::new();
    entity.insert("@context".to_string(), json!(SCHEMA_ORG));
    entity.insert("@type".to_string(), json!("Organization"));
    entity.insert("@id".to_string(), json!(format!("{origin}/#organization")));
    entity.insert("name".to_string(), json!(site_name));
    entity.insert("url".to_string(), json!(format!("{origin}/")));

    let logo = homepage
        .select(&IMG_SELECTOR)
        .filter(|img| {
            in_site_chrome(*img) && !hidden_for_good(*img) && (in_site_header(*img) || links_home(*img, base))
        })
        .filter(|img| {
            ["src", "alt", "class"].iter().any(|attr| {
                img.value()
                    .attr(attr)
                    .is_some_and(|value| value.to_lowercase().contains("logo"))
            })
        })
        .find_map(|img| {
            base.join(img.value().attr("src")?.trim())
                .ok()
                .filter(|url| matches!(url.scheme(), "http" | "https"))
        });
    if let Some(logo) = logo {
        entity.insert("logo".to_string(), json!(logo.to_string()));
    }

    let chrome_text = chrome_blocks
        .iter()
        .filter(|block| block.region == Region::Chrome && !block.hidden)
        .map(|block| block.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let visible = Visible::new(&chrome_text, "");
    let links: Vec<ElementRef> = homepage
        .select(&LINK_SELECTOR)
        .filter(|link| in_site_chrome(*link) && !hidden_for_good(*link))
        .collect();
    let href = |link: &ElementRef| link.value().attr("href").unwrap_or_default().trim().to_string();

    let telephone = links.iter().find_map(|link| {
        let number = decoded(strip_scheme(&href(link), "tel:")?.split(';').next()?);
        (!number.is_empty() && visible.contains("telephone", &number)).then_some(number)
    });
    let email = links.iter().find_map(|link| {
        let address = decoded(strip_scheme(&href(link), "mailto:")?.split(['?', ',']).next()?);
        (address.contains('@') && visible.contains("email", &address)).then_some(address)
    });
    if telephone.is_some() || email.is_some() {
        let mut contact = Map::new();
        contact.insert("@type".to_string(), json!("ContactPoint"));
        if let Some(telephone) = telephone {
            contact.insert("telephone".to_string(), json!(telephone));
        }
        if let Some(email) = email {
            contact.insert("email".to_string(), json!(email));
        }
        entity.insert("contactPoint".to_string(), json!([contact]));
    }

    let brand = name_forms(site_name);
    let mut same_as: Vec<String> = Vec::new();
    let mut possible: Vec<String> = Vec::new();
    for link in &links {
        let Some(mut url) = base.join(&href(link)).ok() else {
            continue;
        };
        let Some(profile) = social_profile(&url) else {
            continue;
        };
        url.set_query(None);
        url.set_fragment(None);
        let url = url.to_string();
        // Only the brand's own name is the brand's account, wherever it stands on the page.
        if profile.organization
            && name_forms(&decoded(&profile.handle))
                .iter()
                .any(|form| brand.contains(form))
        {
            push_unique(&mut same_as, &url);
            possible.retain(|known| *known != url);
        } else if !same_as.contains(&url) {
            push_unique(&mut possible, &url);
        }
    }
    if !same_as.is_empty() {
        entity.insert("sameAs".to_string(), json!(same_as));
    }
    (Value::Object(entity), possible)
}

/// Inside the site header: a `header` outside `article` and `main`, or `[role=banner]`.
fn in_site_header(element: ElementRef) -> bool {
    let chain: Vec<ElementRef> = element.ancestors().filter_map(ElementRef::wrap).collect();
    let is_content = |el: &ElementRef| {
        matches!(el.value().name(), "article" | "main")
            || el
                .value()
                .attr("role")
                .is_some_and(|role| role.trim().eq_ignore_ascii_case("main"))
    };
    chain.iter().enumerate().any(|(at, el)| {
        let banner = el
            .value()
            .attr("role")
            .is_some_and(|role| role.trim().eq_ignore_ascii_case("banner"));
        banner || (el.value().name() == "header" && !chain[at + 1..].iter().any(is_content))
    })
}

/// Inside a link to the homepage (the site's root, with or without `www.`).
fn links_home(element: ElementRef, base: &url::Url) -> bool {
    let site = base.host_str().map(|host| host.trim_start_matches("www."));
    element
        .ancestors()
        .filter_map(ElementRef::wrap)
        .find(|el| el.value().name() == "a")
        .and_then(|link| base.join(link.value().attr("href")?.trim()).ok())
        .is_some_and(|target| {
            target.path() == "/"
                && target.query().is_none()
                && target.host_str().map(|host| host.trim_start_matches("www.")) == site
        })
}

/// `href` without a case-insensitive scheme prefix such as `tel:`.
fn strip_scheme<'a>(href: &'a str, scheme: &str) -> Option<&'a str> {
    href.get(..scheme.len())
        .filter(|prefix| prefix.eq_ignore_ascii_case(scheme))
        .and_then(|_| href.get(scheme.len()..))
}

fn decoded(value: &str) -> String {
    percent_encoding::percent_decode_str(value)
        .decode_utf8_lossy()
        .trim()
        .to_string()
}

/// A link to a profile on a social platform.
struct SocialProfile {
    handle: String,
    /// A profile an organization can have (not a personal LinkedIn or a showcase page).
    organization: bool,
}

/// The profile behind a link to Facebook, Instagram, LinkedIn, X / Twitter, YouTube, TikTok,
/// GitHub, Pinterest, Threads or Bluesky; `None` for other links and for share, intent and post
/// links.
fn social_profile(url: &url::Url) -> Option<SocialProfile> {
    let host = url.host_str()?.to_ascii_lowercase();
    let on = |domain: &str| host == domain || host.ends_with(&format!(".{domain}"));
    let segments: Vec<&str> = url.path_segments()?.filter(|segment| !segment.is_empty()).collect();
    let profile = |handle: &str, organization: bool| {
        Some(SocialProfile {
            handle: handle.to_string(),
            organization,
        })
    };
    match segments.as_slice() {
        [kind, handle, ..] if on("linkedin.com") => match kind.to_ascii_lowercase().as_str() {
            "company" | "school" => profile(handle, true),
            "in" | "pub" | "showcase" => profile(handle, false),
            _ => None,
        },
        [first, rest @ ..] if on("youtube.com") => match (first.strip_prefix('@'), rest) {
            (Some(handle), _) => profile(handle, true),
            (None, [handle, ..]) if matches!(first.to_ascii_lowercase().as_str(), "channel" | "c" | "user") => {
                profile(handle, true)
            }
            _ => None,
        },
        ["profile", handle] if on("bsky.app") => profile(handle, true),
        [handle] if on("tiktok.com") || on("threads.net") || on("threads.com") => {
            profile(handle.strip_prefix('@')?, true)
        }
        [handle]
            if (on("facebook.com")
                || on("instagram.com")
                || on("x.com")
                || on("twitter.com")
                || on("github.com")
                || on("pinterest.com"))
                && !NOT_PROFILES.contains(&handle.to_ascii_lowercase().as_str()) =>
        {
            profile(handle.trim_start_matches('@'), true)
        }
        _ => None,
    }
}

/// Legal forms that may follow a company's name, as `compact` writes them (`s.r.o.` → `sro`,
/// `spol. s r.o.` → `spolsro`, `Pty Ltd` → `ptyltd`).
const LEGAL_FORMS: &[&str] = &[
    "sro",
    "spolsro",
    "as",
    "ks",
    "vos",
    "zs",
    "ops",
    "se",
    "ltd",
    "limited",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "llc",
    "llp",
    "plc",
    "gmbh",
    "ag",
    "kg",
    "sa",
    "sas",
    "sarl",
    "srl",
    "spa",
    "bv",
    "nv",
    "oy",
    "ab",
    "aps",
    "ptyltd",
    "spzoo",
];

/// The forms of a name that identify it: the whole name `compact`, and the name without a trailing
/// legal form (`Kočka s.r.o.` → `kockasro`, `kocka`; `aster-gmbh` → `astergmbh`, `aster`). A
/// legal form counts only as whole words after the name, so `Atlas` stays `atlas`.
fn name_forms(name: &str) -> Vec<String> {
    let words: Vec<String> = name
        .split(|c: char| !c.is_alphanumeric())
        .map(compact)
        .filter(|word| !word.is_empty())
        .collect();
    let mut forms = vec![words.concat()];
    for suffix in 1..words.len() {
        let rest = words.len() - suffix;
        if LEGAL_FORMS.contains(&words[rest..].concat().as_str()) {
            forms.push(words[..rest].concat());
        }
    }
    forms.retain(|form| !form.is_empty());
    forms.dedup();
    forms
}

/// Lowercase letters and digits only, without diacritics: `Example s.r.o.` → `examplesro`.
fn compact(text: &str) -> String {
    text.nfd()
        .filter(|c| !is_combining_mark(*c))
        .flat_map(char::to_lowercase)
        .filter(|c| c.is_alphanumeric())
        .collect()
}

/// A page's name for a breadcrumb: its first H1 outside the site chrome, or else its title
/// without a site suffix (`Pricing | Example` → `Pricing`).
pub fn page_name(document: &Html, site_name: &str) -> Option<String> {
    let collapse = |text: &str| text.split_whitespace().collect::<Vec<_>>().join(" ");
    // The H1 as a visitor sees it: without hidden text, SVG titles or scripts.
    if let Some(h1) = document
        .select(&H1_SELECTOR)
        .filter(|h1| !in_site_chrome(*h1) && !hidden_for_good(*h1))
        .map(|h1| collapse(&shown_text(h1)))
        .find(|text| !text.is_empty())
    {
        return Some(h1);
    }
    let title = document
        .select(&TITLE_SELECTOR)
        .next()
        .map(|title| collapse(&title.text().collect::<String>()))?;
    if title.is_empty() {
        return None;
    }
    let site = compact(site_name);
    let without_suffix = TITLE_SEPARATORS.iter().find_map(|separator| {
        let (head, tail) = title.rsplit_once(separator)?;
        let tail = compact(tail);
        (!site.is_empty()
            && !tail.is_empty()
            && (tail.contains(&site) || site.contains(&tail))
            && !head.trim().is_empty())
        .then(|| head.trim().to_string())
    });
    Some(without_suffix.unwrap_or(title))
}

/// A BreadcrumbList for a page from its crawled ancestors (`crawled`: URL as crawled → name):
/// the homepage and every ancestor path that was crawled — with a trailing slash or without it,
/// preferring the page's own style — then the page itself. URLs are matched in their normalized
/// form (the crawler keeps `/služby/` as written, the `url` crate encodes it) and shown as
/// crawled. `None` for fewer than 2 elements or a page missing from `crawled`.
pub fn breadcrumb(page_url: &str, crawled: &HashMap<String, String>) -> Option<Value> {
    let mut known: Vec<(&String, &String)> = crawled.iter().filter(|(_, name)| !name.trim().is_empty()).collect();
    known.sort();
    let mut by_normalized: HashMap<String, (&String, &String)> = HashMap::new();
    for (url, name) in known {
        by_normalized.entry(normalized_url(url)).or_insert((url, name));
    }
    let page = url::Url::parse(page_url).ok()?;
    let page_key = normalized_url(page_url);
    let (page_crawled, page_name) = *by_normalized.get(&page_key)?;
    let origin = &page[..url::Position::BeforePath];
    let parts: Vec<&str> = page.path_segments()?.filter(|segment| !segment.is_empty()).collect();
    let slash_first = page.path().ends_with('/');
    let mut trail: Vec<(&String, &String)> = Vec::new();
    for depth in 0..parts.len() {
        let candidates = if depth == 0 {
            vec![format!("{origin}/")]
        } else {
            let path = format!("{origin}/{}", parts[..depth].join("/"));
            let with_slash = format!("{path}/");
            if slash_first {
                vec![with_slash, path]
            } else {
                vec![path, with_slash]
            }
        };
        if let Some(level) = candidates
            .iter()
            .map(|url| normalized_url(url))
            .filter(|key| *key != page_key)
            .find_map(|key| by_normalized.get(&key).copied())
        {
            trail.push(level);
        }
    }
    trail.push((page_crawled, page_name));
    if trail.len() < 2 {
        return None;
    }
    let items: Vec<Value> = trail
        .iter()
        .enumerate()
        .map(|(index, (url, name))| {
            json!({"@type": "ListItem", "position": index + 1, "name": name.trim(), "item": url})
        })
        .collect();
    Some(json!({
        "@context": SCHEMA_ORG,
        "@type": "BreadcrumbList",
        "@id": format!("{page_url}#breadcrumb"),
        "itemListElement": items,
    }))
}

/// A FAQPage from visible questions with their answers, in document order. A pair is kept only
/// when its question and answer blocks are the page's own content (Main region) that the page
/// shows (not hidden for good), it has an answer, and every answer block lies between its
/// question and the next pair's question — so
/// pairs can neither cross nor share an answer, and no question can be another's answer. `None`
/// when no pair is left.
pub fn faq(page_url: &str, pairs: &[(Block, Vec<Block>)]) -> Option<Value> {
    let mut ordered: Vec<&(Block, Vec<Block>)> = pairs.iter().collect();
    ordered.sort_by_key(|(question, _)| question.id);
    let questions: Vec<Value> = ordered
        .iter()
        .enumerate()
        .filter(|(at, (question, answers))| {
            let next_question = ordered.get(at + 1).map_or(usize::MAX, |(next, _)| next.id);
            !question.text.trim().is_empty()
                && question.region == Region::Main
                && !question.hidden
                && !answers.is_empty()
                && answers.iter().all(|answer| {
                    answer.region == Region::Main
                        && !answer.hidden
                        && answer.id > question.id
                        && answer.id < next_question
                })
        })
        .filter_map(|(_, (question, answers))| {
            let texts: Vec<&str> = answers
                .iter()
                .map(|answer| answer.text.trim())
                .filter(|text| !text.is_empty())
                .collect();
            (!texts.is_empty()).then(|| {
                json!({
                    "@type": "Question",
                    "name": question.text.trim(),
                    "acceptedAnswer": {"@type": "Answer", "text": texts.join("\n")},
                })
            })
        })
        .collect();
    (!questions.is_empty()).then(|| {
        json!({
            "@context": SCHEMA_ORG,
            "@type": "FAQPage",
            "@id": format!("{page_url}#faq"),
            "mainEntity": questions,
        })
    })
}

/// An Article (a BlogPosting for a blog) with the H1 as its headline, the verified author's name
/// (`Byline::author_name`) and the byline's date: `datePublished`, or `dateModified` for a date the
/// page labels as the last change.
pub fn article(
    page_url: &str,
    h1: &Block,
    author: Option<&str>,
    date: Option<(NaiveDate, DateRole)>,
    is_blog: bool,
) -> Value {
    let mut entity = Map::new();
    entity.insert("@context".to_string(), json!(SCHEMA_ORG));
    entity.insert(
        "@type".to_string(),
        json!(if is_blog { "BlogPosting" } else { "Article" }),
    );
    entity.insert("@id".to_string(), json!(format!("{page_url}#article")));
    entity.insert("headline".to_string(), json!(h1.text.trim()));
    if let Some(name) = author {
        entity.insert("author".to_string(), json!({"@type": "Person", "name": name}));
    }
    if let Some((date, role)) = date {
        let key = match role {
            DateRole::Published => "datePublished",
            DateRole::Modified => "dateModified",
        };
        entity.insert(key.to_string(), json!(date.format("%Y-%m-%d").to_string()));
    }
    entity.insert("mainEntityOfPage".to_string(), json!(page_url));
    Value::Object(entity)
}

/// The author's name a byline states after its leading label (`By`, `Autor článku:`, `Napsala`,
/// …): the text up to the first delimiter (`BYLINE_DELIMITERS`, a digit, or the word `on` or
/// `dne`), when it is a person's name (`person_name`) — `By Jane Smith, Editorial Director` names
/// Jane Smith. `None` without a label: a missing author is better than a wrong one.
pub(crate) fn byline_name(text: &str) -> Option<String> {
    let rest = without_author_label(text)?;
    // Where the first word `on` or `dne` starts.
    let mut date_word = None;
    let mut start = None;
    for (at, c) in rest.char_indices().chain(std::iter::once((rest.len(), ' '))) {
        if !c.is_whitespace() {
            start.get_or_insert(at);
        } else if let Some(from) = start.take()
            && BYLINE_DATE_WORDS
                .iter()
                .any(|word| rest[from..at].eq_ignore_ascii_case(word))
        {
            date_word = Some(from);
            break;
        }
    }
    let end = BYLINE_DELIMITERS
        .iter()
        .filter_map(|delimiter| rest.find(delimiter))
        .chain(rest.find(|c: char| c.is_ascii_digit()))
        .chain(date_word)
        .min()
        .unwrap_or(rest.len());
    person_name(&rest[..end])
}

/// The text, whitespace collapsed, when it is a bare person's name: 1–`MAX_AUTHOR_WORDS` words,
/// no digit or colon, written like a name (`is_person_name`).
pub(crate) fn person_name(text: &str) -> Option<String> {
    let name = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let words = name.split_whitespace().count();
    let bare = (1..=MAX_AUTHOR_WORDS).contains(&words) && !name.chars().any(|c| c.is_ascii_digit() || c == ':');
    (bare && is_person_name(&name)).then_some(name)
}

/// The rest of a byline after its leading author label (`By`, `Autor:`, `Napsala`, …, followed
/// by a colon or a space); `None` without one.
pub(crate) fn without_author_label(text: &str) -> Option<&str> {
    let text = text.trim();
    AUTHOR_LABELS.iter().find_map(|label| {
        text.get(..label.len())
            .filter(|prefix| prefix.eq_ignore_ascii_case(label))
            .and_then(|_| text.get(label.len()..))
            .filter(|rest| rest.starts_with(':') || rest.starts_with(char::is_whitespace))
            .map(|rest| rest.trim_start_matches(':').trim())
    })
}

/// Lowercase words of a person's name (`Ludwig van Beethoven`, `Leonardo da Vinci`).
const NAME_PARTICLES: &[&str] = &[
    "af", "al", "bin", "da", "das", "de", "del", "della", "den", "der", "di", "do", "dos", "du", "ibn", "la", "le",
    "st.", "ten", "ter", "van", "von", "y", "zu",
];

/// Written like a person's name: words of letters (with `-`, `'`, `’` and `.` for initials), each
/// starting with a capital (or a letter of a script without case) but the name particles, and
/// the first word never a particle. "Wear gloves" or "the editorial team" is no name.
fn is_person_name(name: &str) -> bool {
    name.split_whitespace().enumerate().all(|(at, word)| {
        let letters = word
            .chars()
            .all(|c| c.is_alphabetic() || matches!(c, '-' | '\'' | '’' | '.'));
        let capital = word
            .chars()
            .next()
            .is_some_and(|first| first.is_alphabetic() && !first.is_lowercase());
        letters && (capital || (at > 0 && NAME_PARTICLES.contains(&word.to_lowercase().as_str())))
    })
}

/// A ready-to-paste `<script type="application/ld+json">` element with the pretty JSON. `<`, `>`
/// and `&` are written as the JSON escapes `\u003c`, `\u003e` and `\u0026` (they can only occur
/// inside strings, and the escapes mean the same), so no value can end the script element.
pub fn install_snippet(json: &Value) -> String {
    let body = serde_json::to_string_pretty(json)
        .unwrap_or_default()
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026");
    format!("<script type=\"application/ld+json\">\n{body}\n</script>\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::blocks::{BlockKind, Region, blocks_from_html};
    use serde_json::json;

    fn block(id: usize, kind: BlockKind, text: &str) -> Block {
        Block {
            id,
            region: Region::Main,
            kind,
            heading_path: Vec::new(),
            text: text.to_string(),
            collapsed: false,
            hidden: false,
        }
    }

    fn jsonld_page(scripts: &[&str]) -> Html {
        let scripts: String = scripts
            .iter()
            .map(|script| format!(r#"<script type="application/ld+json">{script}</script>"#))
            .collect();
        Html::parse_document(&format!(
            r#"<html lang="cs"><head>{scripts}</head><body><p>Text</p></body></html>"#
        ))
    }

    #[test]
    fn existing_json_ld_types_graphs_and_parse_errors_are_read() {
        let document = Html::parse_document(
            r#"<html><head>
            <script type="application/ld+json">{"@context":"https://schema.org","@type":"Organization","name":"Example"}</script>
            <script type="application/ld+json">{"@context":"https://schema.org","@graph":[{"@type":"WebSite","name":"Example"},
              {"@type":["Product","schema:Thing"],"name":"Widget","offers":{"@type":"Offer","price":"1290.00"}}]}</script>
            <script type="application/ld+json">[{"@type":"BreadcrumbList"},{"@type":"Organization"}]</script>
            <script type="application/ld+json">{"@type": "FAQPage", </script>
            <script type="Application/LD+JSON; charset=utf-8">{"@type":"https://schema.org/Article"}</script>
            <script type="text/javascript">{"@type":"Ignored"}</script>
            </head><body></body></html>"#,
        );
        let markup = existing(&document, "Example Widget 1 290 Kč");
        assert_eq!(
            markup.jsonld_types,
            [
                "Organization",
                "WebSite",
                "Product",
                "Thing",
                "BreadcrumbList",
                "Article"
            ],
            "top-level and @graph entities only, each type once; nested ones (Offer) are not entities"
        );
        assert_eq!(markup.parse_errors.len(), 1, "{:?}", markup.parse_errors);
        assert!(
            markup.parse_errors[0].starts_with("JSON-LD block 4:"),
            "{:?}",
            markup.parse_errors
        );
        assert!(!markup.has_microdata && !markup.has_rdfa);
        assert!(markup.other_types.is_empty());
    }

    #[test]
    fn key_values_that_are_not_visible_are_listed() {
        let document = jsonld_page(&[
            r#"{"@type":"Organization","name":"Example s.r.o.","telephone":"+420800123456","email":"info@example.com",
                "address":{"@type":"PostalAddress","streetAddress":"Karlova 1","postalCode":"11000","addressLocality":"Praha"}}"#,
            r#"{"@type":"Product","name":"Widget","sku":"W-1","offers":[{"@type":"Offer","price":1290},{"@type":"Offer","price":"990"}]}"#,
            r#"{"@type":"Article","headline":"A headline nobody sees","author":{"@type":"Person","name":"Jan Novák"},"description":"Not checked"}"#,
            r#"{"@type":"BlogPosting","headline":"Hypotéky","author":"Petr Svoboda"}"#,
        ]);
        let visible = "Example s.r.o.\nZákaznická linka 800 123 456\nWidget\nCena 1 290 Kč\n\
                       Karlova 1, 110 00 Praha 1\nHypotéky\nAutor: Jan Novák";
        let markup = existing(&document, visible);
        assert_eq!(
            markup.invisible_values,
            [
                ("Organization.email".to_string(), "info@example.com".to_string()),
                ("Product.offers.price".to_string(), "990".to_string()),
                ("Product.sku".to_string(), "W-1".to_string()),
                ("Article.headline".to_string(), "A headline nobody sees".to_string()),
                ("BlogPosting.author.name".to_string(), "Petr Svoboda".to_string()),
            ],
            "the phone and postal code match by their digits, the price as a number; properties in key order"
        );
    }

    #[test]
    fn the_title_and_the_declared_site_name_count_as_shown() {
        // What SEO plugins emit on every page: the WebPage is named by the title, the WebSite by
        // the site name, and neither has to be in the body text.
        let document = Html::parse_document(
            r#"<html><head><title>Pricing - Example</title><meta property="og:site_name" content="Example Ltd">
            <script type="application/ld+json">{"@context":"https://schema.org","@graph":[
              {"@type":"WebPage","name":"Pricing - Example"},{"@type":"WebSite","name":"Example Ltd"},
              {"@type":"Product","name":"Hidden product"}]}</script></head><body><h1>Pricing</h1></body></html>"#,
        );
        assert_eq!(
            existing(&document, "Pricing").invisible_values,
            [("Product.name".to_string(), "Hidden product".to_string())]
        );
    }

    #[test]
    fn nested_types_count_for_duplicates_and_empty_blocks_are_no_errors() {
        let markup = existing(
            &jsonld_page(&[
                r#"{"@type":"WebSite","name":"Text","publisher":{"@type":"Organization","name":"Text","logo":{"@type":"ImageObject"}}}"#,
                "  ",
            ]),
            "Text",
        );
        assert_eq!(markup.jsonld_types, ["WebSite"]);
        assert_eq!(markup.nested_types, ["Organization", "ImageObject"]);
        assert!(markup.parse_errors.is_empty(), "{:?}", markup.parse_errors);
        assert!(markup.declares("Organization") && markup.declares("WebSite"));
        assert!(!markup.declares("Product"));
        let microdata = existing(
            &Html::parse_document(r#"<div itemscope itemtype="https://schema.org/Product"></div>"#),
            "",
        );
        assert!(microdata.declares("Product"));
    }

    #[test]
    fn the_number_of_checked_values_is_capped() {
        let names: Vec<String> = (0..MAX_CHECKED_VALUES + 50)
            .map(|i| format!(r#"{{"@type":"Product","name":"Hidden {i}"}}"#))
            .collect();
        let markup = existing(&jsonld_page(&[&format!("[{}]", names.join(","))]), "Nothing");
        assert_eq!(markup.invisible_values.len(), MAX_CHECKED_VALUES);
        assert_eq!(markup.values_not_checked, 50);
    }

    #[test]
    fn microdata_and_rdfa_are_detected_with_their_types() {
        let markup = existing(
            &Html::parse_document(
                r#"<html><head><meta property="og:title" content="Open Graph is not RDFa"></head><body>
                <div itemscope itemtype="https://schema.org/LocalBusiness"><span itemprop="name">Example</span></div>
                <div vocab="https://schema.org/" typeof="Person schema:Author"><span property="name">Jan</span></div>
                </body></html>"#,
            ),
            "",
        );
        assert!(markup.has_microdata && markup.has_rdfa);
        assert_eq!(markup.other_types, ["LocalBusiness", "Person", "Author"]);

        let open_graph_only = existing(
            &Html::parse_document(r#"<html><head><meta property="og:title" content="x"></head><body></body></html>"#),
            "",
        );
        assert!(!open_graph_only.has_microdata && !open_graph_only.has_rdfa);
    }

    #[test]
    fn website_has_an_id_a_name_and_the_homepage() {
        let expected = json!({
            "@context": "https://schema.org",
            "@type": "WebSite",
            "@id": "https://example.com/#website",
            "name": "Example",
            "url": "https://example.com/"
        });
        assert_eq!(website("Example", "https://example.com"), expected);
        assert_eq!(website("Example", "https://example.com/"), expected);
    }

    const HOMEPAGE: &str = r#"<html><body>
        <header><a href="/"><img src="/img/logo.svg" alt="Example"></a>
          <a href="tel:+420800123456">+420 800 123 456</a></header>
        <main>
          <p>Sales: <a href="tel:+420999888777">+420 999 888 777</a></p>
          <p><a href="https://www.linkedin.com/company/example-main">Our LinkedIn</a></p>
          <img src="/partner-logo.png" alt="Partner">
        </main>
        <footer>
          <p><a href="mailto:info@example.com">info@example.com</a> <a href="mailto:hidden@example.com"><svg></svg></a></p>
          <a href="https://www.linkedin.com/company/example-s-r-o/">LinkedIn</a>
          <a href="https://www.linkedin.com/in/jan-novak-founder">Founder</a>
          <a href="https://www.facebook.com/examplecz?ref=footer">Facebook</a>
          <a href="https://www.facebook.com/sharer/sharer.php?u=https://example.com/">Share</a>
          <a href="https://twitter.com/intent/tweet?text=Example">Tweet</a>
          <a href="https://www.instagram.com/someoneelse/">Instagram</a>
          <a href="https://www.youtube.com/@ExampleOfficial">YouTube</a>
          <a href="https://github.com/example">GitHub</a>
          <a href="https://bsky.app/profile/example.com">Bluesky</a>
        </footer></body></html>"#;

    #[test]
    fn organization_takes_brand_profiles_and_contacts_from_the_site_chrome_only() {
        let homepage = Html::parse_document(HOMEPAGE);
        let chrome: Vec<Block> = blocks_from_html(HOMEPAGE)
            .into_iter()
            .filter(|block| block.region == Region::Chrome)
            .collect();
        let base = url::Url::parse("https://example.com/").unwrap();
        let (organization, possible) = organization("Example", "https://example.com", &homepage, &chrome, &base);
        assert_eq!(
            organization,
            json!({
                "@context": "https://schema.org",
                "@type": "Organization",
                "@id": "https://example.com/#organization",
                "name": "Example",
                "url": "https://example.com/",
                "logo": "https://example.com/img/logo.svg",
                "contactPoint": [{"@type": "ContactPoint", "telephone": "+420800123456", "email": "info@example.com"}],
                "sameAs": [
                    "https://www.linkedin.com/company/example-s-r-o/",
                    "https://github.com/example"
                ]
            })
        );
        // A founder's profile and a profile whose handle is more than the brand's name are only
        // suggested; share and intent links are no profiles at all.
        assert_eq!(
            possible,
            [
                "https://www.linkedin.com/in/jan-novak-founder",
                "https://www.facebook.com/examplecz",
                "https://www.instagram.com/someoneelse/",
                "https://www.youtube.com/@ExampleOfficial",
                "https://bsky.app/profile/example.com"
            ]
        );
    }

    /// The `sameAs` and the possible profiles of a site named `site_name` whose footer is `footer`.
    fn profiles(site_name: &str, footer: &str) -> (Value, Vec<String>) {
        let html = format!("<body><main><h1>Aster tools</h1></main><footer>{footer}</footer></body>");
        let base = url::Url::parse("https://aster.example/").unwrap();
        let (organization, possible) = organization(
            site_name,
            "https://aster.example",
            &Html::parse_document(&html),
            &[],
            &base,
        );
        (organization.get("sameAs").cloned().unwrap_or(Value::Null), possible)
    }

    #[test]
    fn a_profile_is_the_brand_only_when_its_handle_is_the_brand_name() {
        let footer = r#"<a href="https://github.com/aster">GitHub</a>
            <a href="https://www.instagram.com/asteroid.photos/">Asteroid</a>
            <a href="https://www.youtube.com/@AsterTools">YouTube</a>
            <a href="https://www.linkedin.com/company/aster-garden-tools/">LinkedIn</a>
            <a href="https://x.com/aster_2026">X</a>
            <a href="https://www.facebook.com/astercz">Facebook</a>
            <a href="https://bsky.app/profile/aster.example">Bluesky</a>"#;
        let (same_as, possible) = profiles("Aster", footer);
        assert_eq!(same_as, json!(["https://github.com/aster"]));
        // A handle that holds the brand with more (a product, a country, a year) may be another
        // account: left for the owner to review.
        assert_eq!(
            possible,
            [
                "https://www.instagram.com/asteroid.photos/",
                "https://www.youtube.com/@AsterTools",
                "https://www.linkedin.com/company/aster-garden-tools/",
                "https://x.com/aster_2026",
                "https://www.facebook.com/astercz",
                "https://bsky.app/profile/aster.example"
            ]
        );
        // The whole brand name decides, compared without case, diacritics or punctuation.
        let (same_as, _) = profiles("Aster Tools", footer);
        assert_eq!(same_as, json!(["https://www.youtube.com/@AsterTools"]));
        // A legal form after the name (on either side) is no part of it.
        let (same_as, possible) = profiles(
            "Kočka s.r.o.",
            r#"<a href="https://www.facebook.com/kocka">Facebook</a>
               <a href="https://www.youtube.com/@Ko%C4%8Dka">YouTube</a>
               <a href="https://www.linkedin.com/company/kocka-s-r-o/">LinkedIn</a>
               <a href="https://x.com/kockasro">X</a>
               <a href="https://github.com/kocka-cz">GitHub</a>"#,
        );
        assert_eq!(
            same_as,
            json!([
                "https://www.facebook.com/kocka",
                "https://www.youtube.com/@Ko%C4%8Dka",
                "https://www.linkedin.com/company/kocka-s-r-o/",
                "https://x.com/kockasro"
            ])
        );
        assert_eq!(possible, ["https://github.com/kocka-cz"]);
        for (site_name, handle) in [
            ("Aster a.s.", "aster"),
            ("Aster Ltd", "aster"),
            ("Aster, Inc.", "aster"),
            ("Aster GmbH", "aster-gmbh"),
            ("Aster", "aster-llc"),
        ] {
            let (same_as, _) = profiles(
                site_name,
                &format!(r#"<a href="https://github.com/{handle}">GitHub</a>"#),
            );
            assert_eq!(same_as, json!([format!("https://github.com/{handle}")]), "{site_name}");
        }
        // SiteOne Crawler's own accounts; the parent company's account is another one.
        let (same_as, possible) = profiles(
            "SiteOne Crawler",
            r#"<a href="https://twitter.com/siteone_crawler">X</a>
               <a href="https://www.youtube.com/@SiteOne-Crawler">YouTube</a>
               <a href="https://www.linkedin.com/company/siteone/">SiteOne</a>"#,
        );
        assert_eq!(
            same_as,
            json!([
                "https://twitter.com/siteone_crawler",
                "https://www.youtube.com/@SiteOne-Crawler"
            ])
        );
        assert_eq!(possible, ["https://www.linkedin.com/company/siteone/"]);
        // A brand name that ends like a legal form keeps it ("Atlas" is no "Atl a.s.").
        let (same_as, _) = profiles("Atlas", r#"<a href="https://github.com/atl">GitHub</a>"#);
        assert_eq!(same_as, Value::Null);
    }

    #[test]
    fn a_credited_or_partner_profile_is_left_for_review_and_the_brand_s_own_is_kept() {
        // Credits written any way: an agency's handle holding the brand word is never the brand.
        for credit in [
            r#"<p>Website built by <a href="https://github.com/aster-web-agency">Blaster, an independent web agency</a>.</p>"#,
            r#"<div><p>Website by</p><a href="https://github.com/aster-digital">Aster Digital</a></div>"#,
            r#"<div>Website built by <div><a href="https://github.com/aster-digital">Aster Digital</a></div></div>"#,
            r#"<div><p>Partners</p><ul><li><a href="https://github.com/aster-labs">Aster Labs</a></li></ul></div>"#,
            r#"<div><h4>Partners</h4><ul><li><a href="https://github.com/aster-labs"><svg></svg></a></li></ul></div>"#,
            r#"<p>Design: <a href="https://www.instagram.com/aster.studio/">Aster Studio</a></p>"#,
            r#"<p><small>Powered by <a href="https://github.com/aster-cms">Aster CMS</a></small></p>"#,
        ] {
            let footer = format!(r#"{credit}<p><a href="https://github.com/aster">Follow Aster</a></p>"#);
            let (same_as, possible) = profiles("Aster", &footer);
            assert_eq!(same_as, json!(["https://github.com/aster"]), "{credit}");
            assert_eq!(
                possible.len(),
                1,
                "the credited profile is left for review: {credit} {possible:?}"
            );
        }
        // The brand's own account stays the brand's wherever it stands: next to a credit of the
        // web agency, inside a "made by" line, or in an icon list.
        for own in [
            r#"<p>© Aster <a href="https://github.com/aster">Aster on GitHub</a>. Website by <a href="https://webstudio.example">Webstudio</a>.</p>"#,
            r#"<p>© 2026 Made by Aster · <a href="https://github.com/aster">GitHub</a></p>"#,
            r#"<div><h4>Partners</h4><p>Friends of the garden.</p></div><ul class="social"><li><a href="https://github.com/aster"><svg></svg></a></li></ul>"#,
        ] {
            let (same_as, possible) = profiles("Aster", own);
            assert_eq!(same_as, json!(["https://github.com/aster"]), "{own}");
            assert!(possible.is_empty(), "{own}: {possible:?}");
        }
    }

    #[test]
    fn the_logo_is_the_site_header_image_or_one_linking_home() {
        let base = url::Url::parse("https://example.com/").unwrap();
        let logo = |html: &str| {
            let (organization, _) = organization(
                "Example",
                "https://example.com",
                &Html::parse_document(html),
                &[],
                &base,
            );
            organization.get("logo").and_then(Value::as_str).map(str::to_string)
        };
        // A payment or partner logo in the footer is not the organization's logo.
        assert_eq!(
            logo(
                r#"<body><header><a href="/"><img src="/img/brand.svg" alt="Example"></a></header>
                <footer><img src="/img/logos/visa.svg" alt="Visa"><img class="partner-logo" src="/p.png"></footer></body>"#
            ),
            None
        );
        // A logo image that links to the homepage is, wherever it sits in the chrome.
        assert_eq!(
            logo(
                r#"<body><footer><img src="/img/logos/visa.svg" alt="Visa">
                <a href="https://example.com/"><img src="/img/logo-white.svg" alt="Example"></a></footer></body>"#
            )
            .as_deref(),
            Some("https://example.com/img/logo-white.svg")
        );
        // An article header is content, not the site header.
        assert_eq!(
            logo(r#"<body><main><article><header><img src="/logo-of-a-client.png"></header></article></main></body>"#),
            None
        );
        assert_eq!(
            logo(r#"<body><div role="banner"><img src="/assets/logo.png"></div></body>"#).as_deref(),
            Some("https://example.com/assets/logo.png")
        );
    }

    #[test]
    fn organization_ignores_logos_contacts_and_profiles_hidden_for_good() {
        let html = r#"<html><body><header>
              <div hidden>
                <img src="/retired-logo.svg" alt="Old logo">
                <a href="mailto:retired@example.com">retired@example.com</a>
                <a href="https://www.linkedin.com/company/example/posts/">Old social feed</a>
              </div>
              <img src="/current-logo.svg" alt="Example logo">
              <a href="mailto:support@example.com">support@example.com</a>
            </header></html>"#;
        let base = url::Url::parse("https://example.com/").unwrap();
        let chrome: Vec<Block> = blocks_from_html(html)
            .into_iter()
            .filter(|block| block.region == Region::Chrome)
            .collect();
        let (organization, possible) = organization(
            "Example",
            "https://example.com",
            &Html::parse_document(html),
            &chrome,
            &base,
        );
        assert_eq!(organization["logo"], "https://example.com/current-logo.svg");
        assert_eq!(organization["contactPoint"][0]["email"], "support@example.com");
        assert!(organization.get("sameAs").is_none(), "{organization}");
        assert!(possible.is_empty(), "{possible:?}");
    }

    #[test]
    fn organization_without_chrome_signals_has_no_optional_properties() {
        let html = r#"<html><body><main><img src="/logo.png"><a href="tel:+420800123456">+420 800 123 456</a>
            <a href="https://www.linkedin.com/company/example">LinkedIn</a></main></body></html>"#;
        let base = url::Url::parse("https://example.com/").unwrap();
        let (organization, possible) = organization(
            "Example",
            "https://example.com",
            &Html::parse_document(html),
            &[],
            &base,
        );
        assert_eq!(
            organization,
            json!({
                "@context": "https://schema.org",
                "@type": "Organization",
                "@id": "https://example.com/#organization",
                "name": "Example",
                "url": "https://example.com/"
            })
        );
        assert!(possible.is_empty());
    }

    #[test]
    fn breadcrumbs_use_the_crawled_ancestors_as_crawled() {
        let crawled: HashMap<String, String> = [
            ("https://example.com/", "Example"),
            ("https://example.com/blog/", "Blog"),
            ("https://example.com/blog/2026/post", "Post"),
            ("https://example.com/docs", "Docs"),
            ("https://example.com/docs/intro", "Intro"),
            ("https://example.com/about", "About"),
        ]
        .iter()
        .map(|(url, name)| (url.to_string(), name.to_string()))
        .collect();
        let items = |page: &str| -> Vec<(u64, String, String)> {
            let list = breadcrumb(page, &crawled).unwrap_or_else(|| panic!("no breadcrumb for {page}"));
            assert_eq!(list["@type"], "BreadcrumbList");
            assert_eq!(list["@id"], format!("{page}#breadcrumb"));
            list["itemListElement"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| {
                    assert_eq!(item["@type"], "ListItem");
                    (
                        item["position"].as_u64().unwrap(),
                        item["name"].as_str().unwrap().to_string(),
                        item["item"].as_str().unwrap().to_string(),
                    )
                })
                .collect()
        };
        let entry = |position: u64, name: &str, url: &str| (position, name.to_string(), url.to_string());
        // /blog/2026 was not crawled, so it is left out.
        assert_eq!(
            items("https://example.com/blog/2026/post"),
            [
                entry(1, "Example", "https://example.com/"),
                entry(2, "Blog", "https://example.com/blog/"),
                entry(3, "Post", "https://example.com/blog/2026/post"),
            ]
        );
        assert_eq!(
            items("https://example.com/docs/intro"),
            [
                entry(1, "Example", "https://example.com/"),
                entry(2, "Docs", "https://example.com/docs"),
                entry(3, "Intro", "https://example.com/docs/intro"),
            ]
        );
        assert_eq!(items("https://example.com/about").len(), 2);
        assert_eq!(breadcrumb("https://example.com/", &crawled), None, "a single element");
        assert_eq!(breadcrumb("https://example.com/unknown", &crawled), None);
    }

    #[test]
    fn page_names_come_from_the_h1_or_the_title_without_the_site_suffix() {
        let name = |html: &str| page_name(&Html::parse_document(html), "Example");
        assert_eq!(
            name("<title>Pricing | Example</title><header><h1>Logo</h1></header><main><h1> Our   pricing </h1></main>")
                .as_deref(),
            Some("Our pricing")
        );
        assert_eq!(name("<title>Pricing | Example</title>").as_deref(), Some("Pricing"));
        assert_eq!(name("<title>Pricing – EXAMPLE</title>").as_deref(), Some("Pricing"));
        assert_eq!(
            name("<title>Questions - answers</title>").as_deref(),
            Some("Questions - answers")
        );
        assert_eq!(name("<title> </title>"), None);
        assert_eq!(
            name("<main><h1>Garden service<span hidden>INTERNAL_ONLY</span><svg><title>ICON_ONLY</title></svg></h1></main>")
                .as_deref(),
            Some("Garden service"),
            "only the text a visitor sees"
        );
        // Blocks nested in the heading are set apart, inline elements are not.
        assert_eq!(
            name("<main><h1><div>Pricing</div><div>for <b>small</b> teams</div></h1></main>").as_deref(),
            Some("Pricing for small teams")
        );
        assert_eq!(
            name("<main><h1><span>Pri</span><em>cing</em><br>plans</h1></main>").as_deref(),
            Some("Pricing plans")
        );
    }

    #[test]
    fn faq_is_built_from_questions_followed_by_their_answers() {
        let url = "https://example.com/faq";
        let question = block(3, BlockKind::Heading, "Kolik to stojí?");
        let answers = vec![
            block(4, BlockKind::Paragraph, "290 Kč měsíčně."),
            block(5, BlockKind::ListItem, "Bez závazků."),
        ];
        let unanswered = block(6, BlockKind::Heading, "Jak zrušit smlouvu?");
        let answered_before = block(8, BlockKind::Heading, "Kde vás najdu?");
        let earlier = block(7, BlockKind::Paragraph, "V Praze.");
        let pairs = vec![
            (question, answers),
            (unanswered, Vec::new()),
            (answered_before, vec![earlier]),
        ];
        assert_eq!(
            faq(url, &pairs),
            Some(json!({
                "@context": "https://schema.org",
                "@type": "FAQPage",
                "@id": "https://example.com/faq#faq",
                "mainEntity": [{
                    "@type": "Question",
                    "name": "Kolik to stojí?",
                    "acceptedAnswer": {"@type": "Answer", "text": "290 Kč měsíčně.\nBez závazků."}
                }]
            }))
        );
        assert_eq!(faq(url, &pairs[1..]), None, "no question keeps an answer");
    }

    #[test]
    fn faq_pairs_may_not_cross_or_leave_the_page_content() {
        let question = |id: usize, text: &str| block(id, BlockKind::Heading, text);
        let answer = |id: usize, text: &str| block(id, BlockKind::Paragraph, text);
        let mut in_chrome = question(20, "Newsletter?");
        in_chrome.region = Region::Chrome;
        let pairs = vec![
            // Given out of document order: the markup follows the page.
            (question(10, "Kde vás najdu?"), vec![answer(11, "V Praze.")]),
            // An answer after the next question: an off-by-one pairing.
            (
                question(3, "Kolik to stojí?"),
                vec![answer(4, "290 Kč."), answer(8, "Zdarma.")],
            ),
            (question(7, "Je to zdarma?"), vec![answer(8, "Zdarma.")]),
            // An answer that is the next question's block.
            (
                question(12, "Mohu platit kartou?"),
                vec![answer(13, "Ano."), question(14, "A hotově?")],
            ),
            (question(14, "A hotově?"), vec![answer(15, "Také.")]),
            (in_chrome, vec![answer(21, "Ano.")]),
        ];
        let built = faq("https://example.com/faq", &pairs).expect("some pairs are valid");
        let names: Vec<&str> = built["mainEntity"]
            .as_array()
            .unwrap()
            .iter()
            .map(|question| question["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["Je to zdarma?", "Kde vás najdu?", "A hotově?"]);
    }

    #[test]
    fn breadcrumb_levels_match_crawled_urls_in_any_encoding() {
        // The crawler keeps URLs as the pages wrote them, without percent-encoding.
        let crawled: HashMap<String, String> = [
            ("https://example.com/", "Example"),
            ("https://example.com/služby/", "Služby"),
            ("https://example.com/služby/ceník", "Ceník"),
        ]
        .iter()
        .map(|(url, name)| (url.to_string(), name.to_string()))
        .collect();
        let urls = |page: &str| -> Vec<String> {
            breadcrumb(page, &crawled).unwrap_or_else(|| panic!("no breadcrumb for {page}"))["itemListElement"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item["item"].as_str().unwrap().to_string())
                .collect()
        };
        let expected = [
            "https://example.com/",
            "https://example.com/služby/",
            "https://example.com/služby/ceník",
        ];
        assert_eq!(urls("https://example.com/služby/ceník"), expected);
        assert_eq!(urls("https://example.com/slu%C5%BEby/cen%C3%ADk"), expected);
    }

    #[test]
    fn article_uses_the_h1_and_the_verified_byline() {
        let url = "https://example.com/blog/hypoteky";
        let h1 = block(0, BlockKind::Heading, "Jak vybrat hypotéku");
        let date = chrono::NaiveDate::from_ymd_opt(2026, 9, 25).map(|date| (date, DateRole::Published));
        assert_eq!(
            article(url, &h1, Some("Jan Novák"), date, false),
            json!({
                "@context": "https://schema.org",
                "@type": "Article",
                "@id": "https://example.com/blog/hypoteky#article",
                "headline": "Jak vybrat hypotéku",
                "author": {"@type": "Person", "name": "Jan Novák"},
                "datePublished": "2026-09-25",
                "mainEntityOfPage": "https://example.com/blog/hypoteky"
            })
        );
        assert_eq!(
            article(url, &h1, None, None, true),
            json!({
                "@context": "https://schema.org",
                "@type": "BlogPosting",
                "@id": "https://example.com/blog/hypoteky#article",
                "headline": "Jak vybrat hypotéku",
                "mainEntityOfPage": "https://example.com/blog/hypoteky"
            })
        );
        let changed = chrono::NaiveDate::from_ymd_opt(2026, 9, 25).map(|date| (date, DateRole::Modified));
        let entity = article(url, &h1, None, changed, false);
        assert_eq!(entity["dateModified"], "2026-09-25");
        assert!(entity.get("datePublished").is_none(), "{entity}");
    }

    #[test]
    fn a_byline_label_names_the_author_up_to_the_first_delimiter() {
        for (text, name) in [
            ("By Jane Doe | 5 min read", "Jane Doe"),
            ("Written by: Jane Doe", "Jane Doe"),
            ("Napsala Eva Malá · 25. září 2026", "Eva Malá"),
            ("Autorka: Věra O'Neill", "Věra O'Neill"),
            ("By Jane Smith, Editorial Director", "Jane Smith"),
            ("By Jan Novák on September 25, 2026", "Jan Novák"),
            ("Napsal Jan Novák dne 1. 9. 2026", "Jan Novák"),
            ("Autor článku: Jan Novák", "Jan Novák"),
            ("By Jane Doe 25 September 2026", "Jane Doe"),
            ("By Jane Doe (Editor)", "Jane Doe"),
            ("By Ludwig van Beethoven – composer", "Ludwig van Beethoven"),
        ] {
            assert_eq!(byline_name(text).as_deref(), Some(name), "{text}");
        }
        // Without a label, or when what the label introduces is no bare name, no author is named:
        // a missing author is better than a wrong one.
        for text in [
            "Jan Novák, 25. 9. 2026",
            "Jane Smith",
            "Posted on September 25, 2026 by Jan Novák",
            "25. 9. 2026 | Jan Novák",
            "By the editorial team",
            "By Jan Novák and the whole editorial team of Example",
            "Essential Safety Precautions",
        ] {
            assert_eq!(byline_name(text), None, "{text}");
        }
    }

    #[test]
    fn person_names_are_bare_words_written_like_a_name() {
        for name in [
            "Jane Doe-Smith",
            "Ludwig van Beethoven",
            "J. R. R. Tolkien",
            "Věra O'Neill",
            "  Jan   Novák ",
        ] {
            assert!(person_name(name).is_some(), "{name}");
        }
        assert_eq!(person_name("  Jan   Novák ").as_deref(), Some("Jan Novák"));
        for text in [
            "Wear gloves",
            "the editorial team",
            "Jane Smith 2",
            "Autor: Jan",
            "",
            "A B C D E F",
        ] {
            assert_eq!(person_name(text), None, "{text:?}");
        }
    }

    #[test]
    fn install_snippets_cannot_break_out_of_the_script_element() {
        let value = json!({
            "@context": "https://schema.org",
            "@type": "Organization",
            "name": "</script><script>alert(1)</script> <!-- Tom & Jerry"
        });
        let snippet = install_snippet(&value);
        let body = snippet
            .strip_prefix("<script type=\"application/ld+json\">\n")
            .and_then(|rest| rest.strip_suffix("\n</script>\n"))
            .unwrap_or_else(|| panic!("unexpected snippet frame: {snippet}"));
        assert!(
            !body.contains('<') && !body.contains('>') && !body.contains('&'),
            "{body}"
        );
        assert!(body.contains(r"\u003c/script\u003e"), "{body}");
        let parsed: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(parsed, value, "the escapes keep the meaning");
    }
}
