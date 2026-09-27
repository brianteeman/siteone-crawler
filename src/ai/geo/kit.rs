// SiteOne Crawler - the deployable kit of the AI search readiness report
// (c) Jan Reges <jan.reges@siteone.cz>
//
// Builds the files a site owner can deploy: an optional robots.txt block for AI-training crawlers
// (as a full proposed file only when appending it provably changes nothing else), JSON-LD built by
// the crawler from verified page blocks with a manifest of its evidence, answer-first lead drafts,
// entity review drafts, an llms.txt and a README with installation and verification steps. Nothing
// here is written by a model except the lead drafts, which stay labeled as drafts.

use std::collections::{HashMap, HashSet};

use percent_encoding::percent_decode_str;
use serde::Serialize;
use serde_json::{Value, json};
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;

use crate::ai::blocks::Block;
use crate::ai::geo::agents::{AI_AGENTS, AiAgent, Purpose};
use crate::ai::geo::analyze::{Answered, PageAnalysis, PageType, block_ref, is_question};
use crate::ai::geo::discovery::{CrawlEnd, SitemapProposal};
use crate::ai::geo::findings::ORGANIZATION_TYPES;
use crate::ai::geo::jsonld::{self, ExistingMarkup};
use crate::ai::geo::robots_ai::{AgentAccess, AiRobots, policy_equivalent};
use crate::ai::report::locale::ReportLocale;
use crate::result::status::RobotsFetchState;

pub const README_PATH: &str = "README.md";
pub const SNIPPET_PATH: &str = "robots/block-ai-training.snippet.txt";
pub const PROPOSED_PATH: &str = "robots/robots.proposed.txt";
pub const MANIFEST_PATH: &str = "jsonld/_manifest.json";
pub const ENTITY_DRAFTS_PATH: &str = "drafts/entity-drafts.md";
pub const LEADS_PATH: &str = "leads.md";
pub const LLMS_TXT_PATH: &str = "llms.txt";
pub const SITEMAP_PATH: &str = "sitemap/sitemap.proposed.xml";
pub const SITEMAP_COVERAGE_PATH: &str = "sitemap/coverage.json";
/// File stems of `jsonld/` that no page may take.
pub const RESERVED_NAMES: &[&str] = &["_manifest", "site-website", "site-organization"];
/// A page slug is cut to this many characters.
const MAX_SLUG_CHARS: usize = 60;
/// FAQPage markup needs a list: a single question heading with its paragraph (such as "What would
/// you improve?") is no FAQ.
const FAQ_MIN_QUESTIONS: usize = 2;
const FAQ_NOTE: &str = "FAQ rich results no longer appear in Google (since 7 May 2026): keep this markup only for real, visible questions and answers.";

/// One file of the kit, relative to the kit directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KitFile {
    pub relative_path: String,
    pub bytes: Vec<u8>,
}

impl KitFile {
    fn text(relative_path: &str, text: &str) -> Self {
        KitFile {
            relative_path: relative_path.to_string(),
            bytes: text.as_bytes().to_vec(),
        }
    }
}

/// A JSON-LD object of the kit, built by the crawler, with the page text it comes from.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KitEntry {
    /// The page to install it on (the homepage for the site-wide objects).
    pub page_url: String,
    /// The schema.org type.
    pub kind: String,
    /// The file name without its extension, in `jsonld/`.
    pub file_stem: String,
    pub json: Value,
    /// Where each value comes from: `B12: block text`, a crawled URL, a site-chrome link.
    pub evidence: Vec<String>,
    pub notes: Vec<String>,
}

impl KitEntry {
    pub fn json_path(&self) -> String {
        format!("jsonld/{}.json", self.file_stem)
    }

    pub fn html_path(&self) -> String {
        format!("jsonld/{}.html", self.file_stem)
    }
}

/// The site-wide objects: the site's name and origin, and the Organization built by
/// `jsonld::organization`.
pub struct SiteMarkup<'a> {
    pub site_name: &'a str,
    pub origin: &'a str,
    pub homepage_url: &'a str,
    pub organization: Value,
}

/// A page whose markup the kit builds: its breadcrumb from the crawled ancestors, and — with a
/// verified analysis — its FAQ and its Article.
pub struct MarkupPage<'a> {
    pub url: &'a str,
    pub analysis: Option<&'a PageAnalysis>,
    pub existing: &'a ExistingMarkup,
    /// No `noindex` for Google or Bing and no canonical URL elsewhere.
    pub indexable: bool,
}

/// What the kit is built from.
pub struct KitInput<'a> {
    pub locale: &'a ReportLocale,
    pub site_name: &'a str,
    pub markup: &'a [KitEntry],
    /// Social profiles of the site chrome not named after the brand or credited to someone else
    /// (for the manifest).
    pub possible_profiles: &'a [String],
    /// The verified analyses, in rank order.
    pub analyses: &'a [PageAnalysis],
    /// Page titles by URL.
    pub titles: &'a HashMap<String, String>,
    /// The robots.txt verdicts of the table agents for the homepage's origin.
    pub agents: &'a [(&'static AiAgent, AgentAccess)],
    /// The key-page paths of the homepage's origin.
    pub key_paths: &'a [String],
    /// Absolute sitemap URLs to declare in a new robots.txt.
    pub sitemaps: &'a [String],
    /// The sitemap proposed for a site without one.
    pub sitemap: Option<&'a SitemapProposal>,
    /// Why no sitemap is proposed although the site has none, in the report's language.
    pub sitemap_withheld: Option<&'a str>,
}

/// The optional robots.txt block for AI-training crawlers: `Disallow: /` for every training-only
/// agent without a group of its own in `robots`, never an `Allow` line. Google-Extended is only a
/// commented-out block, because it also controls Gemini grounding. A closing comment lists the
/// current verdicts of the search crawlers, which the block does not change.
pub fn training_snippet(robots: Option<&AiRobots>, access: &[(&'static AiAgent, AgentAccess)], today: &str) -> String {
    let named = |token: &str| robots.is_some_and(|robots| robots.has_named_group(token));
    let mut out = String::from(
        "# OPTIONAL — stops crawlers that collect AI training data. It does not affect visibility in AI\n\
         # search and answer engines. Review before use.\n",
    );
    out.push_str(&format!(
        "# Generated by SiteOne Crawler on {today}. Paste it as a whole, separated by empty lines:\n\
         # at the end of your robots.txt when no User-agent line follows its last Allow or Disallow rule,\n\
         # otherwise directly before the first User-agent line after that rule. Never inside a group\n\
         # (between its rules), never right after User-agent lines without rules: they would take these\n\
         # rules too.\n"
    ));
    let training: Vec<&AiAgent> = AI_AGENTS
        .iter()
        .filter(|agent| agent.is_training_only() && !named(agent.token))
        .collect();
    if training.is_empty() {
        out.push_str(
            "#\n# Every AI-training crawler listed here already has its own group in your robots.txt:\n\
             # nothing to add.\n",
        );
    }
    for agent in training {
        out.push_str(&format!("\nUser-agent: {}\nDisallow: /\n", agent.token));
    }
    if !named("Google-Extended") {
        out.push_str(
            "\n# Google-Extended also decides whether Gemini may use your pages for grounding (answers based\n\
             # on Google Search in Gemini apps and on Vertex AI), not only for training. It does not affect\n\
             # Google Search or AI Overviews. Uncomment the two lines below only if you accept that:\n\
             # User-agent: Google-Extended\n\
             # Disallow: /\n",
        );
    }
    out.push_str("\n# Search and answer crawlers, not changed by this block (current verdicts on the key pages):\n");
    let mut listed = false;
    for (agent, verdict) in access
        .iter()
        .filter(|(agent, _)| agent.purposes.contains(&Purpose::Search))
    {
        let verdict = match verdict {
            AgentAccess::Allowed => "allowed",
            AgentAccess::Blocked => "blocked",
            AgentAccess::Partly(_) => "partly blocked",
        };
        out.push_str(&format!("# {} ({}): {}\n", agent.token, agent.vendor, verdict));
        listed = true;
    }
    if !listed {
        out.push_str("# (not assessed: the rules of your robots.txt are not known)\n");
    }
    out
}

/// The full robots.txt with `snippet` appended, only when that provably changes nothing else:
/// - a fetched file (`Ok`) that is valid UTF-8, is no HTML page and does not end with a group
///   without rules gets `\n` (after a final newline) and the snippet;
/// - no file (a 404 or 410) becomes `User-agent: *` / `Allow: /`, the snippet and `sitemaps`;
/// - the snippet's groups must take effect in the parsed proposal, and every other crawler of the
///   table, `*` and every agent named in the file must keep its allow/deny answer on `/`, the
///   `key_paths` and a literal path of every rule (`policy_equivalent`).
///
/// Otherwise `Err` says why the proposal is withheld.
pub fn proposed_robots(
    state: &RobotsFetchState,
    robots: Option<&AiRobots>,
    snippet: &str,
    key_paths: &[String],
    sitemaps: &[String],
) -> Result<String, String> {
    let added = AiRobots::parse(snippet).named_tokens();
    if added.is_empty() {
        return Err("nothing to add: every AI-training crawler already has its own group".to_string());
    }
    let (before, proposed) = match state {
        RobotsFetchState::Ok {
            content, valid_utf8, ..
        } => {
            if !valid_utf8 {
                return Err("robots.txt is not valid UTF-8, so it cannot be reproduced byte for byte".to_string());
            }
            if content.trim_start_matches('\u{feff}').trim_start().starts_with('<') {
                return Err("robots.txt answered with an HTML page instead of robots.txt rules".to_string());
            }
            let before = robots.cloned().unwrap_or_else(|| AiRobots::parse(content));
            if before.ends_with_ruleless_group() {
                return Err(
                    "robots.txt ends with a group without rules (User-agent lines after the last Allow or \
                     Disallow line; a Crawl-delay line is no such rule, nor is a rule line that starts with a \
                     no-break space or another non-ASCII space, which Google does not read), so appended groups \
                     would merge into it"
                        .to_string(),
                );
            }
            let mut proposed = content.clone();
            if !proposed.ends_with('\n') {
                proposed.push('\n');
            }
            proposed.push('\n');
            proposed.push_str(snippet);
            (Some(before), proposed)
        }
        RobotsFetchState::NotFound { status: 404 | 410 } => {
            let mut proposed = format!("User-agent: *\nAllow: /\n\n{snippet}");
            if !sitemaps.is_empty() {
                if !proposed.ends_with('\n') {
                    proposed.push('\n');
                }
                proposed.push('\n');
                for sitemap in sitemaps {
                    proposed.push_str(&format!("Sitemap: {sitemap}\n"));
                }
            }
            (None, proposed)
        }
        RobotsFetchState::NotFound { status } => {
            return Err(format!(
                "robots.txt answered HTTP {status}: a file may exist but be hidden from the crawler"
            ));
        }
        RobotsFetchState::Unavailable { status_or_error } => {
            return Err(format!("robots.txt could not be read ({status_or_error})"));
        }
        RobotsFetchState::Skipped => return Err("robots.txt was not fetched (--ignore-robots-txt)".to_string()),
        RobotsFetchState::NotAttempted => return Err("robots.txt was not fetched".to_string()),
    };
    let after = AiRobots::parse(&proposed);
    // Each added group must be read (within the first 500 KiB) and deny `/`; a `/` that `*`
    // already denies proves nothing about the group itself.
    if let Some(token) = added
        .iter()
        .find(|token| !after.has_named_group(token) || after.is_allowed(token, "/"))
    {
        return Err(format!(
            "the added group for {token} would not take effect (engines read only the first 500 KiB of robots.txt)"
        ));
    }
    let others: Vec<String> = AI_AGENTS
        .iter()
        .map(|agent| agent.token.to_string())
        .filter(|token| !added.iter().any(|added| added.eq_ignore_ascii_case(token)))
        .collect();
    policy_equivalent(before.as_ref(), &after, &others, key_paths)
        .map_err(|why| format!("appending would change the rules of other crawlers: {why}"))?;
    Ok(proposed)
}

/// A file stem per page URL: `page-<slug>` from the URL path (decoded, lowercase ASCII, diacritics
/// folded, other characters as `-`, at most 60 characters; `home` for the root), deduplicated with
/// `-2`, `-3`, … in input order. A stem never equals a `RESERVED_NAMES` entry.
pub fn allocate_names(pages: &[String]) -> HashMap<String, String> {
    let mut used: HashSet<String> = RESERVED_NAMES.iter().map(|name| name.to_string()).collect();
    let mut names: HashMap<String, String> = HashMap::new();
    for url in pages {
        if names.contains_key(url) {
            continue;
        }
        let base = format!("page-{}", slug(url));
        let mut name = base.clone();
        let mut next = 2;
        while !used.insert(name.clone()) {
            name = format!("{base}-{next}");
            next += 1;
        }
        names.insert(url.clone(), name);
    }
    names
}

fn slug(url: &str) -> String {
    let path = url::Url::parse(url).map_or_else(|_| url.to_string(), |parsed| parsed.path().to_string());
    let decoded = percent_decode_str(&path).decode_utf8_lossy().to_string();
    let mut slug = String::new();
    for c in decoded
        .nfd()
        .filter(|c| !is_combining_mark(*c))
        .flat_map(char::to_lowercase)
    {
        if c.is_ascii_alphanumeric() {
            slug.push(c);
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
    }
    let cut: String = slug.trim_matches('-').chars().take(MAX_SLUG_CHARS).collect();
    let cut = cut.trim_end_matches('-');
    if cut.is_empty() {
        "home".to_string()
    } else {
        cut.to_string()
    }
}

fn merge_notes(existing: &ExistingMarkup, kinds: &[&str]) -> Vec<String> {
    kinds
        .iter()
        .find(|kind| existing.declares(kind))
        .map(|kind| {
            vec![format!(
                "The page already has {kind} markup: merge with the existing markup; do not add a duplicate."
            )]
        })
        .unwrap_or_default()
}

fn block_evidence(block: &Block) -> String {
    format!("{}: {}", block_ref(block.id), block.text)
}

/// The kit's JSON-LD: WebSite and Organization for the site, then per indexable page its
/// BreadcrumbList (from `crawled`: URL as crawled → page name), its FAQPage from at least
/// `FAQ_MIN_QUESTIONS` verified question and answer blocks, and for an article its Article (a
/// BlogPosting under `/blog`) from the H1 and a verified byline. A page with `noindex` or a
/// canonical URL elsewhere gets none: engines would not show it. A page that already declares a
/// type gets a note to merge instead of adding a duplicate.
pub fn markup_entries(
    site: &SiteMarkup,
    homepage_existing: &ExistingMarkup,
    pages: &[MarkupPage],
    crawled: &HashMap<String, String>,
) -> Vec<KitEntry> {
    let mut entries = vec![KitEntry {
        page_url: site.homepage_url.to_string(),
        kind: "WebSite".to_string(),
        file_stem: "site-website".to_string(),
        json: jsonld::website(site.site_name, site.origin),
        evidence: vec![format!("name: {}", site.site_name)],
        notes: merge_notes(homepage_existing, &["WebSite"]),
    }];
    let organization = &site.organization;
    let mut evidence = vec![format!("name: {}", site.site_name)];
    if let Some(logo) = organization.get("logo").and_then(Value::as_str) {
        evidence.push(format!("logo (site header): {logo}"));
    }
    for point in organization
        .get("contactPoint")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for field in ["telephone", "email"] {
            if let Some(value) = point.get(field).and_then(Value::as_str) {
                evidence.push(format!("{field} (site header/footer): {value}"));
            }
        }
    }
    for profile in organization
        .get("sameAs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        evidence.push(format!("sameAs (site header/footer): {profile}"));
    }
    entries.push(KitEntry {
        page_url: site.homepage_url.to_string(),
        kind: "Organization".to_string(),
        file_stem: "site-organization".to_string(),
        json: organization.clone(),
        evidence,
        notes: merge_notes(homepage_existing, ORGANIZATION_TYPES),
    });

    let urls: Vec<String> = pages.iter().map(|page| page.url.to_string()).collect();
    let names = allocate_names(&urls);
    for page in pages.iter().filter(|page| page.indexable) {
        let stem = &names[page.url];
        if let Some(json) = jsonld::breadcrumb(page.url, crawled) {
            let evidence = json["itemListElement"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|item| {
                    format!(
                        "{} — {}",
                        item["name"].as_str().unwrap_or_default(),
                        item["item"].as_str().unwrap_or_default()
                    )
                })
                .collect();
            entries.push(KitEntry {
                page_url: page.url.to_string(),
                kind: "BreadcrumbList".to_string(),
                file_stem: format!("{stem}-breadcrumb"),
                json,
                evidence,
                notes: merge_notes(page.existing, &["BreadcrumbList"]),
            });
        }
        let Some(analysis) = page.analysis else {
            continue;
        };
        // A verified pair may start at any heading; the markup takes only real questions.
        let pairs: Vec<(Block, Vec<Block>)> = analysis
            .faq_pairs
            .iter()
            .filter(|pair| is_question(&pair.question.text))
            .map(|pair| (pair.question.clone(), pair.answers.clone()))
            .collect();
        if let Some(json) = jsonld::faq(page.url, &pairs).filter(|json| {
            json["mainEntity"]
                .as_array()
                .is_some_and(|questions| questions.len() >= FAQ_MIN_QUESTIONS)
        }) {
            let evidence = pairs
                .iter()
                .flat_map(|(question, answers)| std::iter::once(question).chain(answers))
                .map(block_evidence)
                .collect();
            let mut notes = merge_notes(page.existing, &["FAQPage"]);
            notes.push(FAQ_NOTE.to_string());
            entries.push(KitEntry {
                page_url: page.url.to_string(),
                kind: "FAQPage".to_string(),
                file_stem: format!("{stem}-faq"),
                json,
                evidence,
                notes,
            });
        }
        if analysis.page_type == PageType::Article
            && let Some(h1) = &analysis.h1
        {
            let is_blog = url::Url::parse(page.url).is_ok_and(|parsed| {
                parsed
                    .path_segments()
                    .into_iter()
                    .flatten()
                    .any(|segment| segment.eq_ignore_ascii_case("blog") || segment.eq_ignore_ascii_case("blogs"))
            });
            let byline = &analysis.byline;
            let date = byline.date.zip(byline.date_role);
            let json = jsonld::article(page.url, h1, byline.author.as_ref(), date, is_blog);
            let mut evidence = vec![block_evidence(h1)];
            if json.get("author").is_some()
                && let Some(author) = &byline.author
            {
                evidence.push(block_evidence(author));
            }
            if let Some(date) = &byline.date_block {
                evidence.push(block_evidence(date));
            }
            let kind = json["@type"].as_str().unwrap_or("Article").to_string();
            entries.push(KitEntry {
                page_url: page.url.to_string(),
                kind,
                file_stem: format!("{stem}-article"),
                json,
                evidence,
                notes: merge_notes(page.existing, &["Article", "BlogPosting", "NewsArticle"]),
            });
        }
    }
    entries
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Page or model text for Markdown: one line, and nothing in it can open a link, an image or raw
/// HTML (`\`, `[` and `]` are escaped, `<` is written as `&lt;`).
fn md_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in one_line(text).chars() {
        match c {
            '\\' | '[' | ']' => {
                out.push('\\');
                out.push(c);
            }
            '<' => out.push_str("&lt;"),
            c => out.push(c),
        }
    }
    out
}

fn section_name(page_type: PageType) -> &'static str {
    match page_type {
        PageType::Homepage => "Home",
        PageType::Product => "Products",
        PageType::Service => "Services",
        PageType::Category => "Categories",
        PageType::Article => "Articles",
        PageType::Faq => "FAQ",
        PageType::Contact => "Contact",
        PageType::About => "About",
        PageType::Pricing => "Pricing",
        PageType::Legal => "Legal",
        PageType::Career => "Careers",
        PageType::Directory => "Directories",
        PageType::Other => "Other pages",
    }
}

/// An llms.txt (llmstxt.org) of the analyzed pages: the site name, the homepage's own meta
/// description, and one section per page type in the order the types first occur, each listing
/// its pages in rank order as `- [title](url): meta description`. The file goes onto the site, so
/// it holds only the site's own text: never a model-written summary, which the crawler cannot
/// verify.
pub fn llms_txt(site_name: &str, analyses: &[PageAnalysis], titles: &HashMap<String, String>) -> String {
    let topic_of = |analysis: &PageAnalysis| analysis.description.trim().to_string();
    let mut out = format!("# {}\n\n", md_escape(site_name));
    let homepage = analyses
        .iter()
        .find(|analysis| url::Url::parse(&analysis.url).is_ok_and(|url| url.path() == "/"))
        .or_else(|| {
            analyses
                .iter()
                .find(|analysis| analysis.page_type == PageType::Homepage)
        });
    if let Some(topic) = homepage.map(topic_of).filter(|topic| !topic.is_empty()) {
        out.push_str(&format!("> {}\n\n", md_escape(&topic)));
    }
    let mut sections: Vec<(PageType, Vec<String>)> = Vec::new();
    for analysis in analyses {
        let title = titles
            .get(&analysis.url)
            .filter(|title| !title.trim().is_empty())
            .unwrap_or(&analysis.title);
        let title = if title.trim().is_empty() {
            analysis.url.clone()
        } else {
            one_line(title)
                .replace('\\', "\\\\")
                .replace('[', "(")
                .replace(']', ")")
                .replace('<', "&lt;")
        };
        let target = analysis.url.replace(' ', "%20").replace('(', "%28").replace(')', "%29");
        let topic = md_escape(&topic_of(analysis));
        let line = if topic.is_empty() {
            format!("- [{title}]({target})")
        } else {
            format!("- [{title}]({target}): {topic}")
        };
        match sections
            .iter_mut()
            .find(|(page_type, _)| *page_type == analysis.page_type)
        {
            Some((_, lines)) => lines.push(line),
            None => sections.push((analysis.page_type, vec![line])),
        }
    }
    let sections: Vec<String> = sections
        .iter()
        .map(|(page_type, lines)| format!("## {}\n\n{}\n", section_name(*page_type), lines.join("\n")))
        .collect();
    out.push_str(&sections.join("\n"));
    out
}

/// The lead drafts and the unanswered questions of the analyzed pages; empty when no page has
/// either.
pub fn leads_md(locale: &ReportLocale, analyses: &[PageAnalysis]) -> String {
    let cs = locale.is_czech();
    let mut sections: Vec<String> = Vec::new();
    for analysis in analyses {
        let unanswered: Vec<&str> = analysis
            .questions
            .iter()
            .filter(|question| question.answered == Answered::No)
            .map(|question| question.question.as_str())
            .collect();
        if analysis.lead.is_none() && unanswered.is_empty() {
            continue;
        }
        let mut section = format!("## {}\n\n", md_escape(&analysis.url));
        if !analysis.title.trim().is_empty() {
            section.push_str(&format!("*{}*\n\n", md_escape(&analysis.title)));
        }
        if let Some(lead) = &analysis.lead {
            section.push_str(if cs {
                "**Návrh — před zveřejněním zkontrolujte**\n\n"
            } else {
                "**Draft — review before publishing**\n\n"
            });
            section.push_str(&format!("> {}\n\n", md_escape(&lead.text)));
            section.push_str(if cs {
                "Podpůrné výňatky ze stránky:\n\n"
            } else {
                "Supporting excerpts from the page:\n\n"
            });
            for excerpt in &lead.excerpts {
                section.push_str(&format!("- {}: {}\n", excerpt.block, md_escape(&excerpt.text)));
            }
            section.push('\n');
        }
        if !unanswered.is_empty() {
            section.push_str(if cs {
                "Otázky, na které stránka neodpovídá (v prověřených blocích):\n\n"
            } else {
                "Questions not found on this page (in the inspected blocks):\n\n"
            });
            for question in unanswered {
                section.push_str(&format!("- {}\n", md_escape(question)));
            }
            section.push('\n');
        }
        sections.push(section);
    }
    if sections.is_empty() {
        return String::new();
    }
    let header = if cs {
        "# Návrhy úvodních odstavců (odpověď hned na začátku)\n\n\
         Redakční NÁVRHY, které napsala AI výhradně z textu dané stránky. Čísla v nich crawler ověřil \
         v citovaných blocích, formulace ověřené nejsou. Před zveřejněním je zkontrolujte a přepište \
         vlastními slovy.\n\n"
    } else {
        "# Answer-first lead drafts\n\n\
         Editorial DRAFTS written by AI only from the text of each page. The crawler checked that every \
         number occurs in the cited blocks; the wording was not verified. Review and rewrite them in your \
         own voice before publishing.\n\n"
    };
    format!("{header}{}", sections.join(""))
}

fn table_cell(text: &str) -> String {
    md_escape(text).replace('|', "\\|")
}

/// The entity drafts of the analyzed pages as review tables; empty when there are none.
pub fn entity_drafts_md(locale: &ReportLocale, analyses: &[PageAnalysis]) -> String {
    let cs = locale.is_czech();
    let mut sections: Vec<String> = Vec::new();
    for analysis in analyses.iter().filter(|analysis| !analysis.entity_drafts.is_empty()) {
        let mut section = format!("## {}\n\n", md_escape(&analysis.url));
        for draft in &analysis.entity_drafts {
            section.push_str(&format!("### {}\n\n", draft.kind));
            section.push_str(if cs {
                "| Vlastnost | Hodnota | Doklad na stránce |\n|---|---|---|\n"
            } else {
                "| Property | Value | Evidence on the page |\n|---|---|---|\n"
            });
            for property in &draft.properties {
                section.push_str(&format!(
                    "| {} | {} | {}: {} |\n",
                    table_cell(&property.name),
                    table_cell(&property.value),
                    property.excerpt.block,
                    table_cell(&property.excerpt.text)
                ));
            }
            section.push_str(if cs {
                "\n*Před zveřejněním zkontrolujte.*\n\n"
            } else {
                "\n*Review before publishing.*\n\n"
            });
        }
        sections.push(section);
    }
    if sections.is_empty() {
        return String::new();
    }
    let header = if cs {
        "# Návrhy entit — NENASAZOVAT\n\n\
         Popisy hlavních entit stránek (produkty, služby, akce, provozovny, osoby), které navrhla AI. \
         Každou hodnotu crawler našel v citovaném bloku stránky, typ ani význam ale ověřené nejsou. \
         Slouží jako podklad pro vaše vlastní strukturovaná data — před zveřejněním je zkontrolujte.\n\n"
    } else {
        "# Entity drafts — NOT for deployment\n\n\
         Descriptions of the pages' main entities (products, services, events, businesses, people) \
         proposed by AI. The crawler found every value in the cited block of the page, but neither the \
         type nor the meaning was verified. Use them as a starting point for your own structured data — \
         review before publishing.\n\n"
    };
    format!("{header}{}", sections.join(""))
}

/// The kit's README: what each file of `files` does and how to install it, with the strength of
/// the evidence behind it, why a proposed robots.txt was withheld, the manual checks and how to
/// verify the deployment.
pub fn readme(locale: &ReportLocale, files: &[String], withheld: Option<&str>, today: &str) -> String {
    let cs = locale.is_czech();
    let has = |path: &str| files.iter().any(|file| file == path);
    let has_jsonld = files.iter().any(|file| file.starts_with("jsonld/"));
    let mut out = String::new();
    if cs {
        out.push_str(&format!(
            "# Sada pro připravenost na AI vyhledávání\n\n\
             Vytvořil SiteOne Crawler {today}. Všechny soubory jsou volitelné. Nic z toho neslibuje citace, \
             pozice ani zařazení do odpovědí AI — o tom rozhodují vyhledávače samy. Každý soubor před \
             nasazením zkontrolujte.\n\n## Soubory\n\n"
        ));
    } else {
        out.push_str(&format!(
            "# AI search readiness kit\n\n\
             Generated by SiteOne Crawler on {today}. Every file is optional. Nothing here promises citations, \
             rankings or inclusion in AI answers — the engines decide that themselves. Review every file \
             before you deploy it.\n\n## Files\n\n"
        ));
    }
    if has(SNIPPET_PATH) {
        out.push_str(&format!("### {SNIPPET_PATH}\n\n"));
        out.push_str(if cs {
            "Volitelné skupiny pro robots.txt, které zastaví crawlery sbírající data pro trénování AI \
             (GPTBot, ClaudeBot, CCBot, …) a které ve vašem robots.txt ještě nemají vlastní skupinu. Na \
             viditelnost v AI vyhledávačích a odpovědních systémech nemá vliv (evidence: strong — tyto \
             crawlery dokumentují jejich provozovatelé). Instalace: vložte blok celý, oddělený prázdnými \
             řádky, na konec robots.txt, pokud za jeho posledním pravidlem Allow nebo Disallow už nenásleduje \
             žádný řádek User-agent — jinak přímo před první řádek User-agent za tímto pravidlem. Nikdy ho \
             nevkládejte doprostřed skupiny (mezi její pravidla) ani hned za řádky User-agent bez pravidel \
             Allow nebo Disallow, které by si pravidla bloku přivlastnily.\n\n"
        } else {
            "Optional robots.txt groups that stop the crawlers collecting AI training data (GPTBot, \
             ClaudeBot, CCBot, …) that have no group of their own in your robots.txt yet. It does not \
             affect visibility in AI search and answer engines (evidence: strong — the vendors document \
             these crawlers). To install it, paste it as a whole, separated by empty lines, at the end of \
             your robots.txt when no User-agent line follows its last Allow or Disallow rule — otherwise \
             directly before the first User-agent line after that rule. Never paste it inside a group \
             (between its rules), and never right after User-agent lines without Allow or Disallow rules, \
             which would take its rules too.\n\n"
        });
    }
    if has(PROPOSED_PATH) {
        out.push_str(&format!("### {PROPOSED_PATH}\n\n"));
        out.push_str(if cs {
            "Váš současný robots.txt s připojeným blokem. Crawler ověřil, že všechny ostatní crawlery si na \
             klíčových stránkách zachovají přesně současné povolení i zákazy. Současný soubor jím nahraďte až \
             po kontrole.\n\n"
        } else {
            "Your current robots.txt with the block appended. The crawler checked that every other crawler \
             keeps exactly its current allow/deny answers on your key pages. Replace your robots.txt with it \
             only after review.\n\n"
        });
    }
    if let Some(why) = withheld {
        if cs {
            out.push_str(&format!(
                "**robots.proposed.txt nebyl vytvořen: {why}.** Blok vložte ručně, oddělený prázdnými řádky, \
                 na konec robots.txt, pokud za jeho posledním pravidlem Allow nebo Disallow už nenásleduje \
                 žádný řádek User-agent — jinak přímo před první řádek User-agent za tímto pravidlem. Nikdy \
                 ho nevkládejte doprostřed skupiny (mezi její pravidla) ani hned za řádky User-agent bez \
                 pravidel Allow nebo Disallow (Crawl-delay takovým pravidlem není): skupina by si pravidla \
                 bloku přivlastnila. Pak zkontrolujte, že ostatní crawlery mají stále stejná pravidla.\n\n"
            ));
        } else {
            out.push_str(&format!(
                "**robots.proposed.txt was not generated: {why}.** Paste the block by hand, separated by \
                 empty lines, at the end of your robots.txt when no User-agent line follows its last Allow or \
                 Disallow rule — otherwise directly before the first User-agent line after that rule. Never \
                 paste it inside a group (between its rules), and never right after User-agent lines without \
                 Allow or Disallow rules (a Crawl-delay line is no such rule): that group would take the \
                 block's rules too. Then check that the other crawlers keep their rules.\n\n"
            ));
        }
    }
    if has_jsonld {
        out.push_str("### jsonld/\n\n");
        out.push_str(if cs {
            "Strukturovaná data (JSON-LD), která crawler sestavil z viditelného obsahu vašich stránek — nikdy \
             z textu napsaného AI. Každý soubor .html je hotový element <script type=\"application/ld+json\"> \
             pro stránku uvedenou v jsonld/_manifest.json; vložte ho do <head> nebo <body> té stránky. Soubor \
             .json obsahuje totéž pro pole v CMS. _manifest.json uvádí ke každé stránce soubory, doklady (text \
             stránky, ze kterého hodnota pochází) a poznámky, např. že stránka už taková data má a máte je \
             sloučit místo přidání duplicity. Hodnoty musí zůstat shodné s viditelným textem stránky. Může \
             pomoci s rozšířenými výsledky, kde se používají, a se srozumitelností pro Bing (evidence: weak — \
             pozorovací studie neprokázala, že by JSON-LD zvyšoval citace v AI). O způsobilosti \
             pro zobrazení ve vyhledávání rozhoduje vyhledávač. Rozšířené výsledky FAQ Google od května 2026 \
             nezobrazuje.\n\n"
        } else {
            "Structured data (JSON-LD) built by the crawler from the visible content of your pages — never \
             from text written by AI. Each .html file is a ready-to-paste <script type=\"application/ld+json\"> \
             element for the page named in jsonld/_manifest.json; put it into the <head> or <body> of that \
             page. Each .json file holds the same object for a CMS field. _manifest.json lists every page with \
             its files, the evidence (the page text each value comes from) and notes, e.g. when the page \
             already has such markup and you should merge instead of adding a duplicate. Keep every value \
             identical to the visible text of the page. It can help rich results where they apply and clarity \
             for Bing (evidence: weak — an observational study showed no demonstrated AI-citation uplift from \
             JSON-LD). Eligibility for a search feature is decided by the engine. FAQ rich results no \
             longer appear in Google (since May 2026).\n\n"
        });
    }
    if has(ENTITY_DRAFTS_PATH) {
        out.push_str(&format!("### {ENTITY_DRAFTS_PATH}\n\n"));
        out.push_str(if cs {
            "NENASAZOVAT: popisy produktů, služeb, akcí, provozoven nebo osob, které navrhla AI, s textem \
             stránky, kde se každá hodnota našla. Po kontrole poslouží jako podklad pro vaše vlastní \
             strukturovaná data.\n\n"
        } else {
            "NOT deployable: descriptions of products, services, events, businesses or people proposed by \
             AI, with the page text each value was found in. After review, use them as a starting point for \
             your own structured data.\n\n"
        });
    }
    if has(LEADS_PATH) {
        out.push_str(&format!("### {LEADS_PATH}\n\n"));
        out.push_str(if cs {
            "Redakční návrhy úvodních vět, které hned odpovídají na hlavní otázku stránky, napsané AI z textu \
             stránky, s výňatky, o které se opírají, a otázky návštěvníků, na které stránka neodpovídá. Jen \
             návrhy — před zveřejněním je přepište vlastními slovy (evidence: moderate).\n\n"
        } else {
            "Editorial drafts of answer-first opening sentences, written by AI from the page's own text, with \
             the excerpts they rely on, and the visitor questions the page does not answer. Drafts only — \
             rewrite them in your own voice before publishing (evidence: moderate).\n\n"
        });
    }
    if has(LLMS_TXT_PATH) {
        out.push_str(&format!("### {LLMS_TXT_PATH}\n\n"));
        out.push_str(if cs {
            "Přehled analyzovaných stránek ve formátu llmstxt.org s vlastními meta popisy stránek (žádný text \
             napsaný AI). Volitelný; umístěte ho do kořene webu \
             (/llms.txt) (evidence: weak — žádá se o něj zřídka a žádný velký poskytovatel AI se nezavázal ho \
             používat).\n\n"
        } else {
            "An index of the analyzed pages in the llmstxt.org format, with the pages' own meta descriptions (no \
             AI-written text). Optional; place it at the root of your \
             site (/llms.txt) (evidence: weak — it is rarely requested and no major AI provider has committed \
             to using it).\n\n"
        });
    }
    out.push_str(if cs {
        "## Ruční kontroly (z webu nejsou vidět)\n\n\
         - Google Search Console → Nastavení → Search generative AI: Include / Exclude / Inherit (zkontrolujte \
           i subdomény a dědění mezi službami).\n\
         - Přehled výkonu generativní AI v Search Console.\n\
         - Přehled AI Performance v Bing Webmaster Tools.\n\
         - Nastavení CDN/WAF typu „blokovat AI boty“ a ověřování botů (Cloudflare aj.); při použití WAF \
           povolte zveřejněné rozsahy IP adres OpenAI.\n\n\
         ## Jak nasazení ověřit\n\n\
         - robots.txt: přehled robots.txt v Google Search Console, tester robots.txt v Bing Webmaster Tools a \
           nastavení vašeho CDN/WAF.\n\
         - JSON-LD: Rich Results Test od Googlu (https://search.google.com/test/rich-results) a Schema Markup \
           Validator (https://validator.schema.org/). Data musí zůstat shodná s viditelným textem.\n\
         - Aktuálnost pro Bing: IndexNow (https://www.indexnow.org/) a přesné lastmod v sitemapě.\n\
         - Markdown verze pro AI agenty: SiteOne Crawler je umí exportovat (--markdown-export-dir); při \
           vyjednávání obsahu je servírujte s Content-Type: text/markdown a Vary: Accept. Jde o pohodlí pro \
           agenty, ne o faktor citací.\n\
         - llms.txt: volitelný.\n"
    } else {
        "## Manual checks (not visible from the website)\n\n\
         - Google Search Console → Settings → Search generative AI: Include / Exclude / Inherit (check \
           subdomains and property inheritance too).\n\
         - The generative-AI performance report in Search Console.\n\
         - The AI Performance report in Bing Webmaster Tools.\n\
         - CDN/WAF \"block AI bots\" settings and bot verification (Cloudflare and others); when you use a \
           WAF, allow OpenAI's published IP ranges.\n\n\
         ## How to verify\n\n\
         - robots.txt: the robots.txt report in Google Search Console, the robots.txt tester in Bing \
           Webmaster Tools, and your CDN/WAF settings.\n\
         - JSON-LD: Google's Rich Results Test (https://search.google.com/test/rich-results) and the Schema \
           Markup Validator (https://validator.schema.org/). Keep the markup identical to the visible text.\n\
         - Freshness for Bing: IndexNow (https://www.indexnow.org/) and an accurate lastmod in your \
           sitemap.\n\
         - Markdown versions for AI agents: SiteOne Crawler can export them (--markdown-export-dir); when \
           you use content negotiation, serve them with Content-Type: text/markdown and Vary: Accept. This \
           is a convenience for agents, not a citation factor.\n\
         - llms.txt: optional.\n"
    });
    out
}

/// The proposed sitemap: a `<urlset>` of `<loc>` (URL-escaped, then entity-escaped) and, when
/// known, `<lastmod>`; no `<priority>` or `<changefreq>`, which Google ignores.
pub fn sitemap_xml(proposal: &SitemapProposal) -> String {
    let mut xml = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n",
    );
    for entry in &proposal.urls {
        let loc = url::Url::parse(&entry.url).map_or_else(|_| entry.url.clone(), |url| url.to_string());
        xml.push_str(&format!(
            "  <url>\n    <loc>{}</loc>\n",
            crate::export::sitemap_exporter::escape_xml(&loc)
        ));
        if let Some(lastmod) = &entry.lastmod {
            xml.push_str(&format!("    <lastmod>{lastmod}</lastmod>\n"));
        }
        xml.push_str("  </url>\n");
    }
    xml.push_str("</urlset>\n");
    xml
}

/// What the proposed sitemap covers: the crawl's scope, the pages left out and why, and where the
/// `lastmod` values come from.
pub fn coverage_json(proposal: &SitemapProposal, today: &str) -> String {
    let scope = &proposal.scope;
    let coverage = json!({
        "generator": "SiteOne Crawler",
        "generated": today,
        "origin": proposal.origin,
        "urls": proposal.urls.len(),
        "withLastmod": proposal.urls.iter().filter(|url| url.lastmod.is_some()).count(),
        "leftOut": {
            "noindex": proposal.left_out.noindex,
            "canonicalElsewhere": proposal.left_out.canonical_elsewhere,
            "blockedByRobotsTxt": proposal.left_out.blocked,
            "otherOrigin": proposal.left_out.other_origin,
        },
        "scope": {
            "startUrl": scope.start_url,
            "crawlComplete": scope.end == CrawlEnd::Complete,
            "singlePage": scope.single_page,
            "maxVisitedUrls": scope.max_visited_urls,
            "maxDepth": scope.max_depth,
            "urlFilters": scope.url_filters,
        },
        "notes": [
            "Lists the canonical, indexable HTML pages that answered 200 on this origin, as the crawl found them by following links from the start URL.",
            "Pages that no crawled page links to are not in it; add them by hand.",
            "Pages with noindex, a canonical URL elsewhere, or blocked for Googlebot or Bingbot by robots.txt are left out.",
            "lastmod comes only from a plausible Last-Modified response header; the other URLs have none rather than a wrong one.",
        ],
    });
    format!("{}\n", serde_json::to_string_pretty(&coverage).unwrap_or_default())
}

/// The manifest of the kit's JSON-LD.
fn manifest(markup: &[KitEntry], possible_profiles: &[String], today: &str) -> String {
    let entries: Vec<Value> = markup
        .iter()
        .map(|entry| {
            json!({
                "url": entry.page_url,
                "type": entry.kind,
                "files": [entry.json_path(), entry.html_path()],
                "evidence": entry.evidence,
                "notes": entry.notes,
            })
        })
        .collect();
    let manifest = json!({
        "generator": "SiteOne Crawler",
        "generated": today,
        "notes": [
            "Built by the crawler from the visible content of the pages, never from text written by AI.",
            "Keep every value identical to the visible text of the page.",
            "Eligibility for a search feature is decided by the engine.",
        ],
        "entries": entries,
        "possibleProfiles": possible_profiles,
        "possibleProfilesNote": "Social profiles in the site header or footer that are not named after the brand, or that the text around them credits to someone else (a web agency, a partner): add them to sameAs by hand if they are yours.",
    });
    format!("{}\n", serde_json::to_string_pretty(&manifest).unwrap_or_default())
}

/// Every file of the kit, README first: the training snippet when it adds a group, the proposed
/// robots.txt when it is safe (otherwise the README says why not), the JSON-LD with its manifest, the entity drafts, the
/// lead drafts and the llms.txt — each only when it has content.
pub fn build(
    input: &KitInput,
    robots_state: &RobotsFetchState,
    robots: Option<&AiRobots>,
    today: &str,
) -> Vec<KitFile> {
    let mut files: Vec<KitFile> = Vec::new();
    let snippet = training_snippet(robots, input.agents, today);
    // A block that adds no group (every training crawler has one already) is left out.
    let withheld = if AiRobots::parse(&snippet).named_tokens().is_empty() {
        None
    } else {
        files.push(KitFile::text(SNIPPET_PATH, &snippet));
        match proposed_robots(robots_state, robots, &snippet, input.key_paths, input.sitemaps) {
            Ok(proposed) => {
                files.push(KitFile::text(PROPOSED_PATH, &proposed));
                None
            }
            Err(why) => Some(why),
        }
    };
    if !input.markup.is_empty() {
        files.push(KitFile::text(
            MANIFEST_PATH,
            &manifest(input.markup, input.possible_profiles, today),
        ));
        for entry in input.markup {
            let pretty = serde_json::to_string_pretty(&entry.json).unwrap_or_default();
            files.push(KitFile::text(&entry.json_path(), &format!("{pretty}\n")));
            files.push(KitFile::text(&entry.html_path(), &jsonld::install_snippet(&entry.json)));
        }
    }
    let drafts = entity_drafts_md(input.locale, input.analyses);
    if !drafts.is_empty() {
        files.push(KitFile::text(ENTITY_DRAFTS_PATH, &drafts));
    }
    let leads = leads_md(input.locale, input.analyses);
    if !leads.is_empty() {
        files.push(KitFile::text(LEADS_PATH, &leads));
    }
    if !input.analyses.is_empty() {
        files.push(KitFile::text(
            LLMS_TXT_PATH,
            &llms_txt(input.site_name, input.analyses, input.titles),
        ));
    }
    if let Some(sitemap) = input.sitemap {
        files.push(KitFile::text(SITEMAP_PATH, &sitemap_xml(sitemap)));
        files.push(KitFile::text(SITEMAP_COVERAGE_PATH, &coverage_json(sitemap, today)));
    }
    let mut names: Vec<String> = vec![README_PATH.to_string()];
    names.extend(files.iter().map(|file| file.relative_path.clone()));
    let mut text = readme(input.locale, &names, withheld.as_deref(), today);
    text.push_str(&sitemap_readme(input.locale, input.sitemap, input.sitemap_withheld));
    files.insert(0, KitFile::text(README_PATH, &text));
    files
}

/// The README part on the proposed sitemap, or on why there is none; empty when the site has one.
fn sitemap_readme(locale: &ReportLocale, sitemap: Option<&SitemapProposal>, withheld: Option<&str>) -> String {
    let cs = locale.is_czech();
    if let Some(sitemap) = sitemap {
        let address = format!("{}/sitemap.xml", sitemap.origin);
        return if cs {
            format!(
                "\n## Navržená sitemapa\n\n\
                 Web nemá sitemapu uvedenou v robots.txt ani nalezenou při procházení a procházení prošlo \
                 každý nalezený odkaz (nezkrátil ho žádný limit, filtr ani nenačtený odkaz). {SITEMAP_PATH} proto uvádí jeho kanonické indexovatelné stránky, které vrátily \
                 200 (počet: {}); {SITEMAP_COVERAGE_PATH} popisuje rozsah procházení a vynechané stránky. Pokud už \
                 sitemapu máte (např. odeslanou v Search Console), porovnejte ji s tímto souborem místo \
                 nahrazení. Instalace: zkontrolujte seznam, nejdřív ověřte, že {address} ještě neexistuje \
                 (crawler tuto adresu nežádal), uložte soubor jako {address}, uveďte ho v \
                 robots.txt řádkem „Sitemap: {address}“ a odešlete ho v Google Search Console a Bing \
                 Webmaster Tools (evidence: moderate — sitemapa pomáhá vyhledávačům stránky najít, \
                 zařazení nezaručuje). Stránky, na které nevede žádný odkaz, v ní chybí.\n",
                sitemap.urls.len()
            )
        } else {
            format!(
                "\n## Proposed sitemap\n\n\
                 The site has no sitemap declared in robots.txt or found by the crawl, and the crawl \
                 followed every link it found (no limit, filter or failed link cut it short). {SITEMAP_PATH} therefore lists its {} canonical, indexable pages \
                 that answered 200; {SITEMAP_COVERAGE_PATH} states the crawl's scope and the pages left \
                 out. If you already have a sitemap (for example one submitted in Search Console), compare \
                 it with this file instead of replacing it. To install it: review the list, first check that {address} \
                 does not exist yet (the crawler did not request it), save the file as {address}, declare it in robots.txt with the line \"Sitemap: {address}\", and submit \
                 it in Google Search Console and Bing Webmaster Tools (evidence: moderate — a sitemap \
                 helps engines find pages; it does not make them index them). Pages that no page links \
                 to are not in it.\n",
                sitemap.urls.len()
            )
        };
    }
    match withheld {
        Some(why) if cs => format!(
            "\n**sitemap.proposed.xml nebyl vytvořen: {why}.** Procházení žádnou sitemapu nenašlo; SiteOne \
             Crawler ji umí vytvořit z úplného procházení (--sitemap-xml-file).\n"
        ),
        Some(why) => format!(
            "\n**sitemap.proposed.xml was not generated: {why}.** The crawl found no sitemap; SiteOne \
             Crawler can write one from a complete crawl (--sitemap-xml-file).\n"
        ),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::blocks::blocks_from_html;
    use crate::ai::geo::analyze::{analyzed_page, build_page_request, parse_analysis, verify_analysis};
    use crate::ai::geo::discovery::{CrawlScope, LeftOut, ProposedUrl};
    use crate::ai::geo::robots_ai::{evaluate, robots_of};

    const TODAY: &str = "2026-09-26";

    fn ok(content: &str) -> RobotsFetchState {
        RobotsFetchState::Ok {
            status: 200,
            content: content.to_string(),
            valid_utf8: true,
        }
    }

    fn paths(list: &[&str]) -> Vec<String> {
        list.iter().map(|path| path.to_string()).collect()
    }

    /// The verdicts of every table agent under `robots`.
    fn verdicts(robots: Option<&AiRobots>) -> Vec<(&'static AiAgent, AgentAccess)> {
        AI_AGENTS
            .iter()
            .map(|agent| (agent, evaluate(robots, agent, &paths(&["/", "/sluzby"]))))
            .collect()
    }

    /// A verified analysis of `html` with the model answer `extra` (see `analyze.rs`).
    fn analysis_of(url: &str, html: &str, extra: impl Fn(&[Block]) -> String) -> PageAnalysis {
        let page = analyzed_page(url, html);
        let blocks = blocks_from_html(html);
        let (_, coverage) = build_page_request(&page, &blocks, "", false, "cs", 1_000_000, 4_000, 0.0);
        let json = format!(
            r#"{{"page_type":"service","main_topic":"Hypotéky","states_offer_early":false,"questions":[]{}}}"#,
            extra(&blocks)
        );
        verify_analysis(parse_analysis(&json).unwrap(), &page, &blocks, &coverage)
    }

    fn id(blocks: &[Block], text: &str) -> String {
        crate::ai::geo::analyze::block_ref(blocks.iter().find(|block| block.text == text).unwrap().id)
    }

    fn uncommented(snippet: &str) -> Vec<&str> {
        snippet
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect()
    }

    #[test]
    fn the_training_snippet_blocks_only_training_crawlers_without_a_group_and_never_allows() {
        let robots = AiRobots::parse("User-agent: GPTBot\nDisallow: /private\n\nUser-agent: *\nDisallow: /admin\n");
        let snippet = training_snippet(Some(&robots), &verdicts(Some(&robots)), TODAY);
        assert!(snippet.starts_with(
            "# OPTIONAL — stops crawlers that collect AI training data. It does not affect visibility in AI\n\
             # search and answer engines. Review before use."
        ));
        assert!(snippet.contains(TODAY));
        assert_eq!(
            uncommented(&snippet),
            [
                "User-agent: ClaudeBot",
                "Disallow: /",
                "User-agent: Applebot-Extended",
                "Disallow: /",
                "User-agent: meta-externalagent",
                "Disallow: /",
                "User-agent: CCBot",
                "Disallow: /",
            ],
            "GPTBot already has a group of its own"
        );
        for line in snippet.lines() {
            let line = line.trim_start_matches(['#', ' ']).to_ascii_lowercase();
            assert!(!line.starts_with("allow"), "{line}");
        }
    }

    #[test]
    fn google_extended_is_only_a_commented_out_block() {
        let snippet = training_snippet(None, &verdicts(None), TODAY);
        let mentions: Vec<&str> = snippet
            .lines()
            .filter(|line| line.contains("Google-Extended"))
            .collect();
        assert!(!mentions.is_empty());
        assert!(mentions.iter().all(|line| line.starts_with('#')), "{mentions:?}");
        assert!(snippet.contains("# User-agent: Google-Extended\n# Disallow: /"));
        assert!(snippet.to_lowercase().contains("grounding"));

        let named = AiRobots::parse("User-agent: Google-Extended\nDisallow:\n");
        let snippet = training_snippet(Some(&named), &verdicts(Some(&named)), TODAY);
        assert!(
            !snippet.contains("User-agent: Google-Extended"),
            "it has a group already"
        );
    }

    #[test]
    fn the_snippet_closes_with_the_current_verdicts_of_the_search_crawlers() {
        let robots =
            AiRobots::parse("User-agent: OAI-SearchBot\nDisallow: /\n\nUser-agent: PerplexityBot\nDisallow: /sluzby\n");
        let snippet = training_snippet(Some(&robots), &verdicts(Some(&robots)), TODAY);
        let tail = &snippet[snippet.find("# Search and answer crawlers").expect("a closing comment")..];
        assert!(tail.contains("# Googlebot (Google): allowed"));
        assert!(tail.contains("# OAI-SearchBot (OpenAI): blocked"));
        assert!(tail.contains("# PerplexityBot (Perplexity): partly blocked"));
        assert!(!tail.contains("GPTBot"), "only search crawlers");
        let everyone = AiRobots::parse(
            "User-agent: GPTBot\nUser-agent: ClaudeBot\nUser-agent: Applebot-Extended\nUser-agent: meta-externalagent\nUser-agent: CCBot\nDisallow: /\n",
        );
        let nothing = training_snippet(Some(&everyone), &verdicts(Some(&everyone)), TODAY);
        assert!(uncommented(&nothing).is_empty());
        assert!(nothing.contains("nothing to add"));
    }

    #[test]
    fn a_fetched_robots_txt_gets_the_snippet_appended() {
        let snippet = training_snippet(None, &verdicts(None), TODAY);
        let key = paths(&["/", "/sluzby"]);
        for content in ["User-agent: *\nDisallow: /admin\n", "User-agent: *\nDisallow: /admin"] {
            let state = ok(content);
            let robots = AiRobots::parse(content);
            let proposed = proposed_robots(&state, Some(&robots), &snippet, &key, &[]).expect("safe to append");
            assert_eq!(proposed, format!("User-agent: *\nDisallow: /admin\n\n{snippet}"));
            let after = AiRobots::parse(&proposed);
            assert!(!after.is_allowed("GPTBot", "/"));
            assert!(after.is_allowed("OAI-SearchBot", "/sluzby"));
            assert!(
                !after.is_allowed("OAI-SearchBot", "/admin"),
                "named groups replace * only for the added agents"
            );
        }
    }

    #[test]
    fn a_missing_robots_txt_gets_an_allow_all_group_the_snippet_and_the_sitemaps() {
        let snippet = training_snippet(None, &verdicts(None), TODAY);
        let sitemaps = paths(&["https://example.com/sitemap.xml"]);
        for status in [404, 410] {
            let proposed = proposed_robots(
                &RobotsFetchState::NotFound { status },
                None,
                &snippet,
                &paths(&["/"]),
                &sitemaps,
            )
            .expect("no file: nothing to break");
            assert_eq!(
                proposed,
                format!("User-agent: *\nAllow: /\n\n{snippet}\nSitemap: https://example.com/sitemap.xml\n")
            );
        }
        for status in [401, 403] {
            let why = proposed_robots(
                &RobotsFetchState::NotFound { status },
                None,
                &snippet,
                &paths(&["/"]),
                &[],
            )
            .expect_err("the file may exist but be hidden from the crawler");
            assert!(why.contains(&status.to_string()), "{why}");
        }
    }

    #[test]
    fn no_proposal_when_the_rules_are_unknown() {
        let snippet = training_snippet(None, &verdicts(None), TODAY);
        for state in [
            RobotsFetchState::Unavailable {
                status_or_error: "HTTP 503".to_string(),
            },
            RobotsFetchState::Unavailable {
                status_or_error: "-2:TIMEOUT".to_string(),
            },
            RobotsFetchState::Skipped,
            RobotsFetchState::NotAttempted,
        ] {
            assert!(
                proposed_robots(&state, None, &snippet, &paths(&["/"]), &[]).is_err(),
                "{state:?}"
            );
        }
    }

    #[test]
    fn no_proposal_when_appending_could_change_anything_else() {
        let snippet = training_snippet(None, &verdicts(None), TODAY);
        let key = paths(&["/", "/sluzby"]);
        let refuse = |state: RobotsFetchState, snippet: &str| {
            let robots = robots_of(&state);
            proposed_robots(&state, robots.as_ref(), snippet, &key, &[]).expect_err("withheld")
        };

        let ruleless = "User-agent: *\nDisallow: /admin\n\nUser-agent: Example\n";
        assert!(refuse(ok(ruleless), &snippet).contains("group without rules"));
        // For Google a rule behind a no-break space is no rule: appended groups would merge into
        // the Googlebot group and block Googlebot.
        let nbsp = "User-agent: Googlebot\n\u{a0}Disallow: /private\n";
        let why = refuse(ok(nbsp), &snippet);
        assert!(
            why.contains("group without rules") && why.contains("no-break space"),
            "{why}"
        );

        // A snippet that would also change `*` fails the policy-equivalence check.
        let why = refuse(
            ok("User-agent: *\nDisallow: /admin\n"),
            "User-agent: GPTBot\nUser-agent: *\nDisallow: /\n",
        );
        assert!(why.contains("would change"), "{why}");

        let latin1 = RobotsFetchState::Ok {
            status: 200,
            content: "User-agent: *\nDisallow: /caf\u{fffd}\n".to_string(),
            valid_utf8: false,
        };
        assert!(refuse(latin1, &snippet).contains("UTF-8"));
        assert!(refuse(ok("<!DOCTYPE html><html><body>App</body></html>"), &snippet).contains("HTML"));
        assert!(
            refuse(ok("\u{feff}<!DOCTYPE html><html><body>App</body></html>"), &snippet).contains("HTML"),
            "an HTML page behind a byte-order mark"
        );

        let huge = format!("User-agent: *\nDisallow: /admin\n#{}\n", "x".repeat(600 * 1024));
        assert!(refuse(ok(&huge), &snippet).contains("would not take effect"));
        // `/` is denied before and after, but past 500 KiB the added groups are not read, so the
        // training crawlers keep the `Allow: /public/` of `*`.
        let huge_deny = format!(
            "User-agent: *\nDisallow: /\nAllow: /public/\n#{}\n",
            "x".repeat(510 * 1024)
        );
        assert!(refuse(ok(&huge_deny), &snippet).contains("would not take effect"));

        let everyone = "User-agent: GPTBot\nUser-agent: ClaudeBot\nUser-agent: Applebot-Extended\n\
                        User-agent: meta-externalagent\nUser-agent: CCBot\nDisallow: /\n";
        let robots = AiRobots::parse(everyone);
        let nothing = training_snippet(Some(&robots), &verdicts(Some(&robots)), TODAY);
        assert!(refuse(ok(everyone), &nothing).contains("nothing to add"));
    }

    #[test]
    fn page_names_never_take_the_reserved_names() {
        let urls = paths(&[
            "https://example.com/",
            "https://example.com/index",
            "https://example.com/_manifest",
            "https://example.com/site-website",
            "https://example.com/home",
            "https://example.com/Služby/",
            "https://example.com/sluzby",
            "https://example.com/blog/2026/článek?page=2",
        ]);
        let names = allocate_names(&urls);
        let name = |url: &str| names[url].as_str();
        assert_eq!(name("https://example.com/"), "page-home");
        assert_eq!(name("https://example.com/index"), "page-index");
        assert_eq!(name("https://example.com/_manifest"), "page-manifest");
        assert_eq!(name("https://example.com/site-website"), "page-site-website");
        assert_eq!(name("https://example.com/home"), "page-home-2");
        assert_eq!(name("https://example.com/Služby/"), "page-sluzby");
        assert_eq!(name("https://example.com/sluzby"), "page-sluzby-2");
        assert_eq!(
            name("https://example.com/blog/2026/článek?page=2"),
            "page-blog-2026-clanek"
        );
        let mut unique: Vec<&String> = names.values().collect();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), urls.len());
        for stem in names.values() {
            assert!(stem.starts_with("page-") && !RESERVED_NAMES.contains(&stem.as_str()));
        }
        assert!(RESERVED_NAMES.contains(&"_manifest") && RESERVED_NAMES.contains(&"site-organization"));
    }

    const FAQ_PAGE: &str = "<html lang=\"cs\"><head><title>Časté dotazy | Example</title></head><body><main>\
        <h1>Časté dotazy</h1><h2>Jak dlouho trvá schválení?</h2><p>Obvykle do 5 dnů.</p>\
        <h2>Lze splatit předčasně?</h2><p>Ano, bez poplatku &lt;/script&gt;.</p></main></body></html>";

    const ARTICLE_PAGE: &str = "<html lang=\"cs\"><body><main><article><h1>Jak vybrat hypotéku</h1>\
        <p class=\"author\">Jan Novák</p><p>25. září 2026</p><p>Text článku.</p></article></main></body></html>";

    fn faq_analysis() -> PageAnalysis {
        analysis_of("https://example.com/faq", FAQ_PAGE, |blocks| {
            format!(
                r#","faq_pairs":[{{"question":"{}","answer":["{}"]}},{{"question":"{}","answer":["{}"]}}]"#,
                id(blocks, "Jak dlouho trvá schválení?"),
                id(blocks, "Obvykle do 5 dnů."),
                id(blocks, "Lze splatit předčasně?"),
                id(blocks, "Ano, bez poplatku </script>."),
            )
        })
    }

    fn article_analysis() -> PageAnalysis {
        let mut analysis = analysis_of("https://example.com/blog/hypoteka", ARTICLE_PAGE, |blocks| {
            format!(
                r#","byline":{{"author":"{}","date":"{}"}}"#,
                id(blocks, "Jan Novák"),
                id(blocks, "25. září 2026")
            )
        });
        analysis.page_type = PageType::Article;
        analysis
    }

    fn site() -> (SiteMarkup<'static>, ExistingMarkup) {
        let organization = serde_json::json!({
            "@context": "https://schema.org",
            "@type": "Organization",
            "@id": "https://example.com/#organization",
            "name": "Example",
            "url": "https://example.com/",
            "sameAs": ["https://www.linkedin.com/company/example"],
        });
        (
            SiteMarkup {
                site_name: "Example",
                origin: "https://example.com",
                homepage_url: "https://example.com/",
                organization,
            },
            ExistingMarkup {
                jsonld_types: vec!["Organization".to_string()],
                ..ExistingMarkup::default()
            },
        )
    }

    #[test]
    fn markup_entries_are_built_from_verified_blocks_with_merge_notes() {
        let (site, home_markup) = site();
        let faq = faq_analysis();
        let article = article_analysis();
        let faq_markup = ExistingMarkup {
            jsonld_types: vec!["FAQPage".to_string()],
            ..ExistingMarkup::default()
        };
        let none = ExistingMarkup::default();
        let crawled: HashMap<String, String> = [
            ("https://example.com/", "Example"),
            ("https://example.com/faq", "Časté dotazy"),
            ("https://example.com/blog", "Blog"),
            ("https://example.com/blog/hypoteka", "Jak vybrat hypotéku"),
        ]
        .into_iter()
        .map(|(url, name)| (url.to_string(), name.to_string()))
        .collect();
        let entries = markup_entries(
            &site,
            &home_markup,
            &[
                MarkupPage {
                    url: "https://example.com/",
                    analysis: None,
                    existing: &home_markup,
                    indexable: true,
                },
                MarkupPage {
                    url: "https://example.com/faq",
                    analysis: Some(&faq),
                    existing: &faq_markup,
                    indexable: true,
                },
                MarkupPage {
                    url: "https://example.com/blog/hypoteka",
                    analysis: Some(&article),
                    existing: &none,
                    indexable: true,
                },
            ],
            &crawled,
        );
        let stems: Vec<(&str, &str)> = entries
            .iter()
            .map(|entry| (entry.file_stem.as_str(), entry.kind.as_str()))
            .collect();
        assert_eq!(
            stems,
            [
                ("site-website", "WebSite"),
                ("site-organization", "Organization"),
                ("page-faq-breadcrumb", "BreadcrumbList"),
                ("page-faq-faq", "FAQPage"),
                ("page-blog-hypoteka-breadcrumb", "BreadcrumbList"),
                ("page-blog-hypoteka-article", "BlogPosting"),
            ]
        );
        let organization = &entries[1];
        assert!(
            organization
                .notes
                .iter()
                .any(|note| note.contains("merge with the existing markup"))
        );
        let local = ExistingMarkup {
            jsonld_types: vec!["LocalBusiness".to_string()],
            ..ExistingMarkup::default()
        };
        let with_local = markup_entries(&site, &local, &[], &crawled);
        assert!(
            with_local[1].notes.iter().any(|note| note.contains("LocalBusiness")),
            "a subtype of Organization is existing Organization markup: {:?}",
            with_local[1].notes
        );
        assert!(
            organization
                .evidence
                .iter()
                .any(|line| line.contains("linkedin.com/company/example"))
        );
        let faq_entry = &entries[3];
        assert_eq!(
            faq_entry.json["mainEntity"][1]["acceptedAnswer"]["text"],
            "Ano, bez poplatku </script>."
        );
        assert!(
            faq_entry
                .evidence
                .iter()
                .any(|line| line.ends_with("Jak dlouho trvá schválení?"))
        );
        assert!(faq_entry.notes.iter().any(|note| note.contains("merge")));
        assert!(faq_entry.notes.iter().any(|note| note.contains("FAQ rich results")));
        let article_entry = &entries[5];
        assert_eq!(article_entry.json["author"]["name"], "Jan Novák");
        assert_eq!(article_entry.json["datePublished"], "2026-09-25");
        assert_eq!(article_entry.json["headline"], "Jak vybrat hypotéku");
        assert!(article_entry.notes.is_empty());
    }

    #[test]
    fn no_markup_is_built_from_content_hidden_for_good() {
        let (site, home_markup) = site();
        let none = ExistingMarkup::default();
        let faq_of = |questions: &str| -> Option<serde_json::Value> {
            let html = format!(
                "<html lang=\"en\"><body><main><h1>Garden FAQ</h1>\
                 <p>Current terms are available by contacting support.</p>{questions}</main></body></html>"
            );
            let mut analysis = analysis_of("https://example.com/faq", &html, |blocks| {
                format!(
                    r#","faq_pairs":[{{"question":"{}","answer":["{}"]}},{{"question":"{}","answer":["{}"]}}]"#,
                    id(blocks, "Is every product free?"),
                    id(blocks, "Every product is free of charge."),
                    id(blocks, "Is shipping always free?"),
                    id(blocks, "All shipping is free.")
                )
            });
            analysis.page_type = PageType::Faq;
            let pages = [MarkupPage {
                url: "https://example.com/faq",
                analysis: Some(&analysis),
                existing: &none,
                indexable: true,
            }];
            markup_entries(&site, &home_markup, &pages, &HashMap::new())
                .into_iter()
                .find(|entry| entry.kind == "FAQPage")
                .map(|entry| entry.json)
        };
        let pairs = "<h2>Is every product free?</h2><p>Every product is free of charge.</p>\
                     <h2>Is shipping always free?</h2><p>All shipping is free.</p>";
        // Answers in table rows or cells the page hides, whatever element hides them.
        let table = |row: &str, cell: &str| {
            format!(
                "<h2>Is every product free?</h2><table><tr{row}><td{cell}>Every product is free of charge.</td></tr></table>\
                 <h2>Is shipping always free?</h2><table><tr{row}><td{cell}>All shipping is free.</td></tr></table>"
            )
        };
        for hidden in [
            format!("<div hidden>{pairs}</div>"),
            format!("<div style=\"display: none\">{pairs}</div>"),
            format!("<div style=\"visibility:hidden\">{pairs}</div>"),
            table(" hidden", ""),
            table(" style=\"display:none\"", ""),
            table(" style=\"visibility: hidden\"", ""),
            table("", " hidden"),
            table("", " style=\"display:none\""),
            "<h2>Is every product free?</h2><table><tr><td><span hidden>Every product is free of charge.</span></td></tr></table>\
             <h2>Is shipping always free?</h2><table><tbody hidden><tr><td>All shipping is free.</td></tr></tbody></table>"
                .to_string(),
        ] {
            assert_eq!(faq_of(&hidden), None, "{hidden}");
        }
        assert!(faq_of(&table("", "")).is_some(), "shown table rows answer");
        // A visitor can open a disclosure, a tab panel or a `hidden="until-found"` section.
        for shown in [
            "<details><summary>Is every product free?</summary><p>Every product is free of charge.</p></details>\
             <details><summary>Is shipping always free?</summary><p>All shipping is free.</p></details>"
                .to_string(),
            format!("<div role=\"tabpanel\" hidden>{pairs}</div>"),
            format!("<div hidden=\"until-found\">{pairs}</div>"),
        ] {
            assert!(faq_of(&shown).is_some(), "{shown}");
        }
    }

    #[test]
    fn an_article_names_only_an_author_the_page_marks_as_its_author() {
        let (site, home_markup) = site();
        let none = ExistingMarkup::default();
        let entry = |byline: &str, author: &str| -> KitEntry {
            let html = format!(
                "<html lang=\"en\"><body><main><article><h1>How to sharpen a spade</h1>{byline}\
                 <p>Published 1 September 2026</p><p>Wear gloves and file the edge.</p></article></main></body></html>"
            );
            let mut analysis = analysis_of("https://example.com/guides/spade", &html, |blocks| {
                format!(
                    r#","byline":{{"author":"{}","date":"{}"}}"#,
                    id(blocks, author),
                    id(blocks, "Published 1 September 2026")
                )
            });
            analysis.page_type = PageType::Article;
            let pages = [MarkupPage {
                url: "https://example.com/guides/spade",
                analysis: Some(&analysis),
                existing: &none,
                indexable: true,
            }];
            markup_entries(&site, &home_markup, &pages, &HashMap::new())
                .into_iter()
                .find(|entry| entry.kind == "Article")
                .expect("the article")
        };
        for (byline, author) in [
            ("<p>Essential Safety Precautions</p>", "Essential Safety Precautions"),
            ("<p>Jane Smith</p>", "Jane Smith"),
        ] {
            let article = entry(byline, author);
            assert_eq!(article.json.get("author"), None, "{byline}");
            assert!(
                !article.evidence.iter().any(|evidence| evidence.contains(author)),
                "{:?}",
                article.evidence
            );
            assert_eq!(article.json["datePublished"], "2026-09-01", "the date is kept");
        }
        for (byline, author) in [
            ("<p>By Jane Smith</p>", "By Jane Smith"),
            (
                "<p><a rel=\"author\" href=\"/team/jane\">Jane Smith</a></p>",
                "Jane Smith",
            ),
        ] {
            assert_eq!(entry(byline, author).json["author"]["name"], "Jane Smith", "{byline}");
        }
    }

    #[test]
    fn an_article_date_is_published_or_modified_as_the_page_says() {
        let (site, home_markup) = site();
        let none = ExistingMarkup::default();
        let article = |date_text: &str| -> serde_json::Value {
            let html = format!(
                "<html lang=\"en\"><body><main><article><h1>Maintaining your garden tools</h1>\
                 <p>By Jane Smith</p><p>{date_text}</p><p>Clean and dry your tools after each use.</p>\
                 </article></main></body></html>"
            );
            let mut analysis = analysis_of("https://example.com/blog/tools", &html, |blocks| {
                format!(
                    r#","byline":{{"author":"{}","date":"{}"}}"#,
                    id(blocks, "By Jane Smith"),
                    id(blocks, date_text)
                )
            });
            analysis.page_type = PageType::Article;
            let pages = [MarkupPage {
                url: "https://example.com/blog/tools",
                analysis: Some(&analysis),
                existing: &none,
                indexable: true,
            }];
            markup_entries(&site, &home_markup, &pages, &HashMap::new())
                .into_iter()
                .find(|entry| entry.kind == "BlogPosting")
                .expect("the article")
                .json
        };
        for (text, published, modified) in [
            ("Last updated September 20, 2026", None, Some("2026-09-20")),
            ("Aktualizováno 20. září 2026", None, Some("2026-09-20")),
            ("Published September 20, 2026", Some("2026-09-20"), None),
            ("September 20, 2026", Some("2026-09-20"), None),
            // Both a publication and a change: which one the date is cannot be told.
            ("Published and updated September 20, 2026", None, None),
        ] {
            let json = article(text);
            assert_eq!(
                json.get("datePublished").and_then(|date| date.as_str()),
                published,
                "{text}"
            );
            assert_eq!(
                json.get("dateModified").and_then(|date| date.as_str()),
                modified,
                "{text}"
            );
        }
    }

    #[test]
    fn llms_txt_lists_the_analyzed_pages_by_type_in_rank_order() {
        let mut home = analysis_of(
            "https://example.com/",
            "<html><body><main><h1>Example</h1></main></body></html>",
            |_| String::new(),
        );
        home.page_type = PageType::Homepage;
        home.description = "Hypotéky a úvěry pro domácnosti".to_string();
        let faq = faq_analysis();
        let article = article_analysis();
        let mut second_faq = analysis_of("https://example.com/faq-2", FAQ_PAGE, |_| String::new());
        second_faq.page_type = PageType::Faq;
        let mut faq = faq;
        faq.page_type = PageType::Faq;
        faq.description = "Odpovědi na časté dotazy k hypotékám".to_string();
        let titles: HashMap<String, String> = [
            ("https://example.com/", "Example"),
            ("https://example.com/faq", "Časté [dotazy]"),
            ("https://example.com/blog/hypoteka", "Jak vybrat hypotéku"),
        ]
        .into_iter()
        .map(|(url, title)| (url.to_string(), title.to_string()))
        .collect();
        let text = llms_txt("Example", &[home, faq, article, second_faq], &titles);
        assert_eq!(
            text,
            "# Example\n\n> Hypotéky a úvěry pro domácnosti\n\n\
             ## Home\n\n- [Example](https://example.com/): Hypotéky a úvěry pro domácnosti\n\n\
             ## FAQ\n\n- [Časté (dotazy)](https://example.com/faq): Odpovědi na časté dotazy k hypotékám\n\
             - [Časté dotazy | Example](https://example.com/faq-2)\n\n\
             ## Articles\n\n- [Jak vybrat hypotéku](https://example.com/blog/hypoteka)\n"
        );
    }

    #[test]
    fn page_and_model_text_cannot_add_links_to_llms_txt_leads_or_drafts() {
        let page = "<html lang=\"cs\"><head><title>Ceník \\</title></head><body><main><h1>Ceník</h1>\
            <p>Tarif [Mini] stojí málo.</p></main></body></html>";
        let mut analysis = analysis_of("https://example.com/cenik", page, |blocks| {
            format!(
                r#","lead":"Viz [odkaz](https://evil.example) a <b>tučně</b>.","lead_blocks":["{}"],
                   "entity_drafts":[{{"type":"Service","properties":{{"[x](https://evil.example)":{{"value":"Mini","block":"{}"}}}}}}]"#,
                id(blocks, "Tarif [Mini] stojí málo."),
                id(blocks, "Tarif [Mini] stojí málo.")
            )
        });
        analysis.description = "Ceník [klikni](https://evil.example)".to_string();
        let llms = llms_txt(
            "Example [s.r.o.](https://evil.example)",
            std::slice::from_ref(&analysis),
            &HashMap::new(),
        );
        let live_link = |text: &str| text.replace("\\]", "").contains("](https://evil.example)");
        assert!(!live_link(&llms), "{llms}");
        assert!(llms.contains("\\[klikni\\]"), "{llms}");
        assert!(
            llms.contains("- [Ceník \\\\](https://example.com/cenik)"),
            "a backslash cannot open the link text: {llms}"
        );
        let leads = leads_md(&ReportLocale::new("en"), std::slice::from_ref(&analysis));
        assert!(!live_link(&leads) && !leads.contains("<b>"), "{leads}");
        assert!(leads.contains("Tarif \\[Mini\\] stojí málo."), "{leads}");
        let drafts = entity_drafts_md(&ReportLocale::new("en"), &[analysis]);
        assert!(!live_link(&drafts), "{drafts}");
    }

    #[test]
    fn leads_md_shows_drafts_with_their_excerpts_and_the_unanswered_questions() {
        let lead_page = "<html lang=\"cs\"><body><main><h1>Hypotéky</h1>\
            <p>Hypotéka s úrokovou sazbou od 4,59 % ročně.</p></main></body></html>";
        let with_lead = analysis_of("https://example.com/hypoteky", lead_page, |blocks| {
            format!(
                r#","lead":"Hypotéka od 4,59 % ročně.","lead_blocks":["{}"],
                   "questions":[{{"question":"Jak dlouho trvá schválení?","answered":"no","blocks":[]}}]"#,
                id(blocks, "Hypotéka s úrokovou sazbou od 4,59 % ročně.")
            )
        });
        let plain = analysis_of("https://example.com/o-nas", lead_page, |_| String::new());
        let en = leads_md(&ReportLocale::new("en"), &[with_lead.clone(), plain.clone()]);
        assert!(en.contains("## https://example.com/hypoteky"));
        assert!(en.contains("**Draft — review before publishing**\n\n> Hypotéka od 4,59 % ročně."));
        assert!(en.contains("- B2: Hypotéka s úrokovou sazbou od 4,59 % ročně."));
        assert!(
            en.contains("Questions not found on this page (in the inspected blocks):\n\n- Jak dlouho trvá schválení?")
        );
        assert!(
            !en.contains("o-nas"),
            "a page without a draft or an open question is left out"
        );
        let cs = leads_md(&ReportLocale::new("cs"), &[with_lead]);
        assert!(cs.contains("**Návrh — před zveřejněním zkontrolujte**"));
        assert_eq!(leads_md(&ReportLocale::new("en"), &[plain]), "");
    }

    #[test]
    fn entity_drafts_md_is_marked_as_not_deployable() {
        let page = "<html lang=\"cs\"><body><main><h1>Hypotéky</h1>\
            <p>Hypotéka | od 4,59 % ročně.</p></main></body></html>";
        let analysis = analysis_of("https://example.com/hypoteky", page, |blocks| {
            format!(
                r#","entity_drafts":[{{"type":"Service","properties":{{"name":{{"value":"Hypotéka","block":"{}"}}}}}}]"#,
                id(blocks, "Hypotéka | od 4,59 % ročně.")
            )
        });
        let md = entity_drafts_md(&ReportLocale::new("en"), &[analysis.clone()]);
        assert!(md.starts_with("# Entity drafts — NOT for deployment"));
        assert!(md.contains("### Service"));
        assert!(md.contains("| name | Hypotéka | B2: Hypotéka \\| od 4,59 % ročně. |"));
        assert!(md.contains("Review before publishing"));
        assert!(entity_drafts_md(&ReportLocale::new("cs"), &[analysis]).contains("NENASAZOVAT"));
        assert_eq!(entity_drafts_md(&ReportLocale::new("en"), &[]), "");
    }

    #[test]
    fn the_readme_explains_each_file_and_a_withheld_proposal() {
        let files = paths(&[
            "README.md",
            "robots/block-ai-training.snippet.txt",
            "jsonld/_manifest.json",
            "jsonld/site-website.html",
            "leads.md",
            "llms.txt",
        ]);
        let en = readme(
            &ReportLocale::new("en"),
            &files,
            Some("robots.txt ends with a group without rules"),
            TODAY,
        );
        assert!(en.contains("robots.proposed.txt was not generated: robots.txt ends with a group without rules"));
        for part in [
            "robots/block-ai-training.snippet.txt",
            "jsonld/",
            "leads.md",
            "llms.txt",
            "Rich Results Test",
            "Schema Markup Validator",
            "IndexNow",
            "text/markdown",
            "Vary: Accept",
            "Search generative AI",
            "evidence: weak",
            TODAY,
        ] {
            assert!(en.contains(part), "{part}");
        }
        assert!(!en.contains("drafts/entity-drafts.md"), "only files the kit has");
        let cs = readme(&ReportLocale::new("cs"), &files, None, TODAY);
        assert!(cs.contains("Sada pro připravenost na AI vyhledávání"));
        assert!(!cs.contains("nebyl vytvořen"));
        for text in [&en, &cs] {
            let lower = text.to_lowercase();
            assert!(!lower.contains("guarantee") && !lower.contains("garant"));
        }
    }

    #[test]
    fn build_leaves_out_a_robots_block_that_adds_nothing() {
        let everyone = "User-agent: GPTBot\nUser-agent: ClaudeBot\nUser-agent: Applebot-Extended\n\
                        User-agent: meta-externalagent\nUser-agent: CCBot\nDisallow: /\n";
        let state = ok(everyone);
        let robots = robots_of(&state);
        let agents = verdicts(robots.as_ref());
        let locale = ReportLocale::new("en");
        let titles = HashMap::new();
        let input = KitInput {
            locale: &locale,
            site_name: "Example",
            markup: &[],
            possible_profiles: &[],
            analyses: &[],
            titles: &titles,
            agents: &agents,
            key_paths: &paths(&["/"]),
            sitemaps: &[],
            sitemap: None,
            sitemap_withheld: None,
        };
        let files = build(&input, &state, robots.as_ref(), TODAY);
        let names: Vec<&str> = files.iter().map(|file| file.relative_path.as_str()).collect();
        assert_eq!(names, ["README.md"], "every training crawler already has a group");
        let readme = String::from_utf8(files[0].bytes.clone()).unwrap();
        assert!(!readme.contains("was not generated"), "nothing was withheld: {readme}");
        assert!(!readme.contains(SNIPPET_PATH));
    }

    #[test]
    fn build_writes_the_kit_files() {
        let (site, home_markup) = site();
        let faq = faq_analysis();
        let crawled: HashMap<String, String> = HashMap::new();
        let markup = markup_entries(
            &site,
            &home_markup,
            &[MarkupPage {
                url: "https://example.com/faq",
                analysis: Some(&faq),
                existing: &ExistingMarkup::default(),
                indexable: true,
            }],
            &crawled,
        );
        let robots = AiRobots::parse("User-agent: *\nDisallow: /admin\n\nUser-agent: Example\n");
        let agents = verdicts(Some(&robots));
        let titles = HashMap::new();
        let locale = ReportLocale::new("en");
        let possible = paths(&["https://www.linkedin.com/in/jan-novak"]);
        let input = KitInput {
            locale: &locale,
            site_name: "Example",
            markup: &markup,
            possible_profiles: &possible,
            analyses: std::slice::from_ref(&faq),
            titles: &titles,
            agents: &agents,
            key_paths: &paths(&["/", "/faq"]),
            sitemaps: &[],
            sitemap: None,
            sitemap_withheld: None,
        };
        let files = build(
            &input,
            &ok("User-agent: *\nDisallow: /admin\n\nUser-agent: Example\n"),
            Some(&robots),
            TODAY,
        );
        let names: Vec<&str> = files.iter().map(|file| file.relative_path.as_str()).collect();
        assert_eq!(
            names,
            [
                "README.md",
                "robots/block-ai-training.snippet.txt",
                "jsonld/_manifest.json",
                "jsonld/site-website.json",
                "jsonld/site-website.html",
                "jsonld/site-organization.json",
                "jsonld/site-organization.html",
                "jsonld/page-faq-faq.json",
                "jsonld/page-faq-faq.html",
                "llms.txt",
            ],
            "no proposed robots.txt after a trailing group without rules"
        );
        let text = |path: &str| {
            String::from_utf8(
                files
                    .iter()
                    .find(|file| file.relative_path == path)
                    .unwrap()
                    .bytes
                    .clone(),
            )
            .unwrap()
        };
        assert!(text("README.md").contains("robots.proposed.txt was not generated"));
        assert!(text("README.md").contains("group without rules"));
        let html = text("jsonld/page-faq-faq.html");
        assert!(html.starts_with("<script type=\"application/ld+json\">"));
        assert!(!html.contains("</script>.") && html.contains("\\u003c/script\\u003e"));
        let json: serde_json::Value = serde_json::from_str(&text("jsonld/page-faq-faq.json")).unwrap();
        assert_eq!(json["@type"], "FAQPage");
        let manifest: serde_json::Value = serde_json::from_str(&text("jsonld/_manifest.json")).unwrap();
        assert_eq!(manifest["entries"][2]["url"], "https://example.com/faq");
        assert_eq!(
            manifest["entries"][2]["files"],
            serde_json::json!(["jsonld/page-faq-faq.json", "jsonld/page-faq-faq.html"])
        );
        assert_eq!(manifest["possibleProfiles"][0], "https://www.linkedin.com/in/jan-novak");

        let clean = ok("User-agent: *\nDisallow: /admin\n");
        let clean_robots = robots_of(&clean);
        let files = build(&input, &clean, clean_robots.as_ref(), TODAY);
        assert!(
            files
                .iter()
                .any(|file| file.relative_path == "robots/robots.proposed.txt")
        );
    }

    fn proposal() -> SitemapProposal {
        SitemapProposal {
            origin: "https://example.com".to_string(),
            urls: vec![
                ProposedUrl {
                    url: "https://example.com/".to_string(),
                    lastmod: None,
                },
                ProposedUrl {
                    url: "https://example.com/a?x=1&y=<2>".to_string(),
                    lastmod: Some("2026-09-01T10:00:00+00:00".to_string()),
                },
                ProposedUrl {
                    url: "https://example.com/služby/".to_string(),
                    lastmod: None,
                },
            ],
            left_out: LeftOut {
                noindex: 2,
                canonical_elsewhere: 1,
                blocked: 0,
                other_origin: 0,
            },
            scope: CrawlScope {
                start_url: "https://example.com/".to_string(),
                single_page: false,
                end: CrawlEnd::Complete,
                max_visited_urls: 10_000,
                max_depth: 0,
                url_filters: false,
                robots_skipped: Vec::new(),
            },
        }
    }

    #[test]
    fn a_proposed_sitemap_is_a_valid_url_set_with_its_coverage() {
        let proposal = proposal();
        let xml = sitemap_xml(&proposal);
        assert!(xml.starts_with(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n"
        ));
        // URL-escaped, then entity-escaped, as the protocol asks.
        assert!(
            xml.contains("<loc>https://example.com/a?x=1&amp;y=%3C2%3E</loc>"),
            "{xml}"
        );
        assert!(xml.contains("<loc>https://example.com/slu%C5%BEby/</loc>"), "{xml}");
        assert!(!xml.contains("<priority>") && !xml.contains("<changefreq>"), "{xml}");
        let parsed = crate::ai::geo::discovery::parse_sitemap(&xml).expect("a valid sitemap");
        assert_eq!(parsed.entries.len(), 3);
        assert_eq!(parsed.entries[1].lastmod.as_deref(), Some("2026-09-01T10:00:00+00:00"));
        assert_eq!(
            parsed.entries[0].lastmod, None,
            "no lastmod without a plausible Last-Modified"
        );

        let coverage: Value = serde_json::from_str(&coverage_json(&proposal, TODAY)).expect("JSON");
        assert_eq!(coverage["generated"], TODAY);
        assert_eq!(coverage["urls"], 3);
        assert_eq!(coverage["withLastmod"], 1);
        assert_eq!(coverage["origin"], "https://example.com");
        assert_eq!(coverage["leftOut"]["noindex"], 2);
        assert_eq!(coverage["leftOut"]["canonicalElsewhere"], 1);
        let scope = &coverage["scope"];
        assert_eq!(scope["startUrl"], "https://example.com/");
        assert_eq!(scope["crawlComplete"], true);
        assert_eq!(scope["maxVisitedUrls"], 10_000);
        assert_eq!(scope["maxDepth"], 0);
        assert_eq!(scope["urlFilters"], false);
        assert!(
            coverage["notes"].as_array().is_some_and(|notes| notes
                .iter()
                .any(|note| note.as_str().unwrap_or_default().contains("links"))),
            "pages without links to them are not in it: {coverage}"
        );
    }

    #[test]
    fn the_kit_carries_the_proposed_sitemap_or_says_why_there_is_none() {
        let state = RobotsFetchState::NotFound { status: 404 };
        let agents = verdicts(None);
        let titles = HashMap::new();
        let proposal = proposal();
        for language in ["en", "cs"] {
            let locale = ReportLocale::new(language);
            let mut input = KitInput {
                locale: &locale,
                site_name: "Example",
                markup: &[],
                possible_profiles: &[],
                analyses: &[],
                titles: &titles,
                agents: &agents,
                key_paths: &paths(&["/"]),
                sitemaps: &[],
                sitemap: Some(&proposal),
                sitemap_withheld: None,
            };
            let files = build(&input, &state, None, TODAY);
            let names: Vec<&str> = files.iter().map(|file| file.relative_path.as_str()).collect();
            assert!(
                names.contains(&SITEMAP_PATH) && names.contains(&SITEMAP_COVERAGE_PATH),
                "{names:?}"
            );
            let readme = String::from_utf8(files[0].bytes.clone()).unwrap();
            let check_first = if language == "cs" {
                "nejdřív ověřte, že https://example.com/sitemap.xml ještě neexistuje"
            } else {
                "first check that https://example.com/sitemap.xml does not exist yet"
            };
            for part in [
                SITEMAP_PATH,
                SITEMAP_COVERAGE_PATH,
                check_first,
                "https://example.com/sitemap.xml",
                "Sitemap: https://example.com/sitemap.xml",
                "Search Console",
                "Bing Webmaster Tools",
            ] {
                assert!(readme.contains(part), "{language}: {part}");
            }

            input.sitemap = None;
            input.sitemap_withheld = Some("the crawl stopped at --max-visited-urls");
            let files = build(&input, &state, None, TODAY);
            assert!(!files.iter().any(|file| file.relative_path.starts_with("sitemap/")));
            let readme = String::from_utf8(files[0].bytes.clone()).unwrap();
            let withheld = if language == "cs" {
                "sitemap.proposed.xml nebyl vytvořen: the crawl stopped at --max-visited-urls"
            } else {
                "sitemap.proposed.xml was not generated: the crawl stopped at --max-visited-urls"
            };
            assert!(readme.contains(withheld), "{language}: {readme}");
            assert!(!readme.contains(SITEMAP_COVERAGE_PATH), "{language}");
            // True whatever the reason (robots.txt may not have been read).
            assert!(
                !readme.contains("no sitemap declared in") && !readme.contains("uvedenou v robots.txt ani"),
                "{language}: {readme}"
            );
        }
    }

    #[test]
    fn page_markup_needs_an_indexable_page_and_a_list_of_questions() {
        let site = SiteMarkup {
            site_name: "Example",
            origin: "https://example.com",
            homepage_url: "https://example.com/",
            organization: jsonld::website("Example", "https://example.com"),
        };
        let crawled: HashMap<String, String> = [
            ("https://example.com/", "Example"),
            ("https://example.com/faq", "Časté dotazy"),
            ("https://example.com/napsat", "Napište nám"),
            ("https://example.com/sekce", "Sekce"),
        ]
        .into_iter()
        .map(|(url, name)| (url.to_string(), name.to_string()))
        .collect();
        let faq = faq_analysis();
        // A single question heading with its paragraph ("What would you improve?") is no list of
        // questions and answers.
        let one = analysis_of("https://example.com/napsat", FAQ_PAGE, |blocks| {
            format!(
                r#","faq_pairs":[{{"question":"{}","answer":["{}"]}}]"#,
                id(blocks, "Jak dlouho trvá schválení?"),
                id(blocks, "Obvykle do 5 dnů."),
            )
        });
        assert_eq!(one.faq_pairs.len(), 1, "the pair itself is verified");
        // Section headings pass the pair check (a heading may head an answer) but are no questions.
        let sections_page = "<html lang=\"en\"><body><main><h1>Sitemap generator</h1>\
            <h2>Key Features</h2><p>XML and TXT output.</p>\
            <h2>How It Works</h2><p>Only HTML pages that answered 200.</p></main></body></html>";
        let sections = analysis_of("https://example.com/sekce", sections_page, |blocks| {
            format!(
                r#","faq_pairs":[{{"question":"{}","answer":["{}"]}},{{"question":"{}","answer":["{}"]}}]"#,
                id(blocks, "Key Features"),
                id(blocks, "XML and TXT output."),
                id(blocks, "How It Works"),
                id(blocks, "Only HTML pages that answered 200."),
            )
        });
        assert_eq!(sections.faq_pairs.len(), 2, "both pairs are verified");
        let none = ExistingMarkup::default();
        let stems = |faq_indexable: bool| {
            markup_entries(
                &site,
                &none,
                &[
                    MarkupPage {
                        url: "https://example.com/faq",
                        analysis: Some(&faq),
                        existing: &none,
                        indexable: faq_indexable,
                    },
                    MarkupPage {
                        url: "https://example.com/napsat",
                        analysis: Some(&one),
                        existing: &none,
                        indexable: true,
                    },
                    MarkupPage {
                        url: "https://example.com/sekce",
                        analysis: Some(&sections),
                        existing: &none,
                        indexable: true,
                    },
                ],
                &crawled,
            )
            .into_iter()
            .map(|entry| entry.file_stem)
            .collect::<Vec<_>>()
        };
        assert_eq!(
            stems(true),
            [
                "site-website",
                "site-organization",
                "page-faq-breadcrumb",
                "page-faq-faq",
                "page-napsat-breadcrumb",
                "page-sekce-breadcrumb",
            ]
        );
        // A page with noindex or a canonical URL elsewhere gets no markup of its own.
        assert_eq!(
            stems(false),
            [
                "site-website",
                "site-organization",
                "page-napsat-breadcrumb",
                "page-sekce-breadcrumb"
            ]
        );
    }

    #[test]
    fn llms_txt_carries_the_pages_own_descriptions_and_no_model_text() {
        let page = "<html lang=\"en\"><head><title>Acme Tools</title>\
            <meta name=\"description\" content=\" Garden spades with steel blades and wooden handles. \"></head>\
            <body><main><h1>Acme Tools</h1><p>Acme Tools sells garden spades.</p></main></body></html>";
        let mut home = analysis_of("https://example.com/", page, |_| String::new());
        home.page_type = PageType::Homepage;
        home.main_topic =
            "Acme gives every customer free lifetime replacements and a 100% satisfaction guarantee.".to_string();
        let mut bare = analysis_of(
            "https://example.com/about",
            "<html lang=\"en\"><head><title>About</title></head><body><main><h1>About</h1></main></body></html>",
            |_| String::new(),
        );
        bare.page_type = PageType::About;
        bare.main_topic = "The best garden tools in the world".to_string();
        let text = llms_txt("Acme Tools", &[home, bare], &HashMap::new());
        assert_eq!(
            text,
            "# Acme Tools\n\n> Garden spades with steel blades and wooden handles.\n\n\
             ## Home\n\n- [Acme Tools](https://example.com/): Garden spades with steel blades and wooden handles.\n\n\
             ## About\n\n- [About](https://example.com/about)\n"
        );
    }

    #[test]
    fn llms_txt_leaves_out_topics_written_in_any_language() {
        let english =
            "<html lang=\"en-GB\"><head><title>Pricing</title></head><body><main><h1>Pricing</h1></main></body></html>";
        let mut home = analysis_of("https://example.com/", english, |_| String::new());
        home.page_type = PageType::Homepage;
        home.main_topic = "Nástroje pro zahradu".to_string();
        let mut pricing = analysis_of("https://example.com/pricing", english, |_| String::new());
        pricing.main_topic = "Garden tool prices".to_string();
        // Neither a Czech topic on an English site nor an English one belongs on the site.
        assert_eq!(
            llms_txt("Example", &[home, pricing], &HashMap::new()),
            "# Example\n\n## Home\n\n- [Pricing](https://example.com/)\n\n\
             ## Services\n\n- [Pricing](https://example.com/pricing)\n"
        );
    }

    #[test]
    fn the_robots_block_says_where_it_may_be_pasted() {
        let snippet = training_snippet(None, &verdicts(None), TODAY);
        let header = snippet.replace("\n# ", " ");
        for part in [
            "at the end of your robots.txt when no User-agent line follows its last Allow or Disallow rule",
            "otherwise directly before the first User-agent line after that rule",
            // Pasted after the first Disallow of `User-agent: *`, the block would take the rules
            // that follow it away from every other crawler.
            "Never inside a group (between its rules)",
            "never right after User-agent lines without rules",
        ] {
            assert!(header.contains(part), "{part}: {snippet}");
        }
        let files = vec![README_PATH.to_string(), SNIPPET_PATH.to_string()];
        let en = readme(
            &ReportLocale::new("en"),
            &files,
            Some("robots.txt ends with a group without rules"),
            TODAY,
        );
        assert!(
            en.contains("never right after User-agent lines without Allow or Disallow rules"),
            "{en}"
        );
        assert!(en.contains("Never paste it inside a group (between its rules)"), "{en}");
        assert!(
            !en.contains("after a line with an Allow or Disallow rule"),
            "a mid-group paste: {en}"
        );
        assert!(
            !en.contains("at the end of your robots.txt, after the last group"),
            "{en}"
        );
        let cs = readme(
            &ReportLocale::new("cs"),
            &files,
            Some("robots.txt ends with a group without rules"),
            TODAY,
        );
        assert!(
            cs.contains("hned za řádky User-agent bez pravidel Allow nebo Disallow"),
            "{cs}"
        );
        assert!(cs.contains("Nikdy ho nevkládejte doprostřed skupiny"), "{cs}");
        let reason = proposed_robots(
            &ok("User-agent: *\nDisallow: /tmp/\n\nUser-agent: AwarioRssBot\nCrawl-delay: 100\n"),
            None,
            &snippet,
            &paths(&["/"]),
            &[],
        )
        .expect_err("a trailing group without Allow or Disallow rules");
        assert!(
            reason.contains("Crawl-delay"),
            "the owner sees why a Crawl-delay group counts: {reason}"
        );
    }

    #[test]
    fn the_czech_sitemap_note_avoids_number_agreement() {
        let proposal = proposal();
        let cs = sitemap_readme(&ReportLocale::new("cs"), Some(&proposal), None);
        assert!(cs.contains("(počet: 3)"), "{cs}");
        assert!(!cs.contains("3 kanonických"), "{cs}");
    }
}
