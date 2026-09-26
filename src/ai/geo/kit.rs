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
use crate::ai::geo::analyze::{Answered, PageAnalysis, PageType, block_ref};
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
/// File stems of `jsonld/` that no page may take.
pub const RESERVED_NAMES: &[&str] = &["_manifest", "site-website", "site-organization"];
/// A page slug is cut to this many characters.
const MAX_SLUG_CHARS: usize = 60;
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
}

/// What the kit is built from.
pub struct KitInput<'a> {
    pub locale: &'a ReportLocale,
    pub site_name: &'a str,
    pub markup: &'a [KitEntry],
    /// Social profiles of the site chrome not named after the brand (for the manifest).
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
        "# Generated by SiteOne Crawler on {today}. Append it after the last group of your robots.txt,\n\
         # separated by an empty line.\n"
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
                    "robots.txt ends with a group without rules (a User-agent line after the last rule), \
                     so appended groups would merge into it"
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
    if let Some(token) = added.iter().find(|token| after.is_allowed(token, "/")) {
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

/// The kit's JSON-LD: WebSite and Organization for the site, then per page its BreadcrumbList
/// (from `crawled`: URL as crawled → page name), its FAQPage from verified question and answer
/// blocks, and for an article its Article (a BlogPosting under `/blog`) from the H1 and a verified
/// byline. A page that already declares a type gets a note to merge instead of adding a duplicate.
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
        notes: merge_notes(homepage_existing, &["Organization"]),
    });

    let urls: Vec<String> = pages.iter().map(|page| page.url.to_string()).collect();
    let names = allocate_names(&urls);
    for page in pages {
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
        let pairs: Vec<(Block, Vec<Block>)> = analysis
            .faq_pairs
            .iter()
            .map(|pair| (pair.question.clone(), pair.answers.clone()))
            .collect();
        if let Some(json) = jsonld::faq(page.url, &pairs) {
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
            let json = jsonld::article(page.url, h1, byline.author.as_ref(), byline.date, is_blog);
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

/// An llms.txt (llmstxt.org) of the analyzed pages: the site name, the homepage's main topic, and
/// one section per page type in the order the types first occur, each listing its pages in rank
/// order as `- [title](url): main topic`.
pub fn llms_txt(site_name: &str, analyses: &[PageAnalysis], titles: &HashMap<String, String>) -> String {
    let mut out = format!("# {}\n\n", one_line(site_name));
    let homepage = analyses
        .iter()
        .find(|analysis| url::Url::parse(&analysis.url).is_ok_and(|url| url.path() == "/"))
        .or_else(|| {
            analyses
                .iter()
                .find(|analysis| analysis.page_type == PageType::Homepage)
        });
    if let Some(homepage) = homepage
        && !homepage.main_topic.trim().is_empty()
    {
        out.push_str(&format!("> {}\n\n", one_line(&homepage.main_topic)));
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
            one_line(title).replace('[', "(").replace(']', ")")
        };
        let target = analysis.url.replace(' ', "%20").replace('(', "%28").replace(')', "%29");
        let topic = one_line(&analysis.main_topic);
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
        let mut section = format!("## {}\n\n", analysis.url);
        if !analysis.title.trim().is_empty() {
            section.push_str(&format!("*{}*\n\n", one_line(&analysis.title)));
        }
        if let Some(lead) = &analysis.lead {
            section.push_str(if cs {
                "**Návrh — před zveřejněním zkontrolujte**\n\n"
            } else {
                "**Draft — review before publishing**\n\n"
            });
            section.push_str(&format!("> {}\n\n", one_line(&lead.text)));
            section.push_str(if cs {
                "Podpůrné výňatky ze stránky:\n\n"
            } else {
                "Supporting excerpts from the page:\n\n"
            });
            for excerpt in &lead.excerpts {
                section.push_str(&format!("- {}: {}\n", excerpt.block, one_line(&excerpt.text)));
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
                section.push_str(&format!("- {}\n", one_line(question)));
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
    one_line(text).replace('|', "\\|")
}

/// The entity drafts of the analyzed pages as review tables; empty when there are none.
pub fn entity_drafts_md(locale: &ReportLocale, analyses: &[PageAnalysis]) -> String {
    let cs = locale.is_czech();
    let mut sections: Vec<String> = Vec::new();
    for analysis in analyses.iter().filter(|analysis| !analysis.entity_drafts.is_empty()) {
        let mut section = format!("## {}\n\n", analysis.url);
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
             crawlery dokumentují jejich provozovatelé). Instalace: vložte blok na konec robots.txt, za \
             poslední skupinu, oddělený prázdným řádkem.\n\n"
        } else {
            "Optional robots.txt groups that stop the crawlers collecting AI training data (GPTBot, \
             ClaudeBot, CCBot, …) that have no group of their own in your robots.txt yet. It does not \
             affect visibility in AI search and answer engines (evidence: strong — the vendors document \
             these crawlers). To install it, paste it at the end of your robots.txt, after the last group, \
             separated by an empty line.\n\n"
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
                "**robots.proposed.txt nebyl vytvořen: {why}.** Blok vložte ručně na konec robots.txt, za \
                 poslední skupinu a oddělený prázdným řádkem, a zkontrolujte, že žádná skupina nekončí bez \
                 pravidel.\n\n"
            ));
        } else {
            out.push_str(&format!(
                "**robots.proposed.txt was not generated: {why}.** Paste the block by hand at the end of your \
                 robots.txt, after the last group and separated by an empty line, and check that no group \
                 ends without rules.\n\n"
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
            "Přehled analyzovaných stránek ve formátu llmstxt.org. Volitelný; umístěte ho do kořene webu \
             (/llms.txt) (evidence: weak — žádá se o něj zřídka a žádný velký poskytovatel AI se nezavázal ho \
             používat).\n\n"
        } else {
            "An index of the analyzed pages in the llmstxt.org format. Optional; place it at the root of your \
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
        "possibleProfilesNote": "Social profiles in the site header or footer that are not named after the brand: add them to sameAs by hand if they are yours.",
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
    let mut names: Vec<String> = vec![README_PATH.to_string()];
    names.extend(files.iter().map(|file| file.relative_path.clone()));
    files.insert(
        0,
        KitFile::text(README_PATH, &readme(input.locale, &names, withheld.as_deref(), today)),
    );
    files
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::blocks::blocks_from_html;
    use crate::ai::geo::analyze::{analyzed_page, build_page_request, parse_analysis, verify_analysis};
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
            r#"{{"page_type":"service","main_topic":"Hypotéky","states_offer_early":true,"questions":[]{}}}"#,
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
        <p>Jan Novák</p><p>25. září 2026</p><p>Text článku.</p></article></main></body></html>";

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
                },
                MarkupPage {
                    url: "https://example.com/faq",
                    analysis: Some(&faq),
                    existing: &faq_markup,
                },
                MarkupPage {
                    url: "https://example.com/blog/hypoteka",
                    analysis: Some(&article),
                    existing: &none,
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
    fn llms_txt_lists_the_analyzed_pages_by_type_in_rank_order() {
        let mut home = analysis_of(
            "https://example.com/",
            "<html><body><main><h1>Example</h1></main></body></html>",
            |_| String::new(),
        );
        home.page_type = PageType::Homepage;
        home.main_topic = "Hypotéky a úvěry pro domácnosti".to_string();
        let faq = faq_analysis();
        let article = article_analysis();
        let mut second_faq = analysis_of("https://example.com/faq-2", FAQ_PAGE, |_| String::new());
        second_faq.page_type = PageType::Faq;
        let mut faq = faq;
        faq.page_type = PageType::Faq;
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
             ## FAQ\n\n- [Časté (dotazy)](https://example.com/faq): Hypotéky\n\
             - [Časté dotazy | Example](https://example.com/faq-2): Hypotéky\n\n\
             ## Articles\n\n- [Jak vybrat hypotéku](https://example.com/blog/hypoteka): Hypotéky\n"
        );
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
}
