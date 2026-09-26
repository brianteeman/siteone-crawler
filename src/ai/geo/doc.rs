// SiteOne Crawler - the AI search readiness report
// (c) Jan Reges <jan.reges@siteone.cz>
//
// The stored result of `--ai-geo` and its renderings: Markdown, JSON and a self-contained HTML
// page. The report opens with what it can and cannot tell, shows every category with its status
// and coverage ("not assessed" when its checks did not run), the priorities with their evidence
// strength, scope and dated source, the details of every check, the manual checks and the kit.
// All fixed texts are English or Czech (`text`); every value from the site is escaped.

use std::collections::HashMap;

use serde_json::{Value, json};

use crate::ai::blocks::Block;
use crate::ai::geo::access::{AccessIssue, AccessKind, AccessStats};
use crate::ai::geo::agents::{AiAgent, Compliance, Purpose};
use crate::ai::geo::analyze::{Answered, OfferEarly, PageAnalysis, Priority, answered_label, block_ref};
use crate::ai::geo::controls::EnginePolicy;
use crate::ai::geo::discovery::{Discovery, HreflangProblem, ListedReason, SitemapKind, SitemapState};
use crate::ai::geo::findings::{
    self, CategoryId, CategoryState, CheckStatus, Checks, Evidence, Issue, OriginPolicy, PageControls, PageMarkup,
    SourceRef,
};
use crate::ai::geo::jsonld::install_snippet;
use crate::ai::geo::kit::{self, KitEntry, KitFile, KitInput};
use crate::ai::geo::render::RenderCheck;
use crate::ai::geo::robots_ai::{AgentAccess, robots_of};
use crate::ai::geo::signals::PdfRestriction;
use crate::ai::report::locale::ReportLocale;
use crate::result::status::RobotsFetchState;

/// The version of the report's JSON.
pub const SCHEMA: &str = "siteone-crawler/ai-geo/2";
/// When the evidence behind the checks was last researched.
pub const RESEARCH_DATE: &str = "2026-09-26";
/// At most this many affected pages are shown as examples of an issue.
const MAX_EXAMPLES: usize = 3;

/// The run behind a report.
#[derive(Debug, Clone, Default)]
pub struct GeoMeta {
    pub host: String,
    /// The homepage.
    pub url: String,
    pub site_name: String,
    pub crawled_at: String,
    pub report_language: String,
    pub provider: String,
    pub model: String,
    pub context_window: i64,
    pub crawled_pages: usize,
    pub key_pages: usize,
    /// Pages chosen for the per-page analysis.
    pub analyzed_pages: usize,
    /// Analyzed pages whose blocks did not all fit the input budget.
    pub reduced_input: usize,
    pub calls: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Why the per-page analysis did not run at all.
    pub analysis_unavailable: Option<String>,
}

/// A check the owner has to make by hand, with where to make it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManualCheck {
    /// The text key.
    pub key: &'static str,
    pub url: &'static str,
}

/// The fact-consistency report of the same run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsistencyLink {
    pub file: String,
    pub possible_inconsistencies: usize,
}

/// The AI search readiness report.
#[derive(Debug, Clone)]
pub struct GeoDoc {
    pub schema: &'static str,
    pub meta: GeoMeta,
    pub evidence_note: String,
    pub summary: String,
    pub categories: Vec<(CategoryId, CategoryState)>,
    pub priorities: Vec<Issue>,
    pub issues: Vec<Issue>,
    pub policy: Vec<OriginPolicy>,
    pub access: Vec<AccessIssue>,
    pub access_stats: AccessStats,
    pub controls: Vec<PageControls>,
    pub pdfs: Vec<PdfRestriction>,
    pub render: RenderCheck,
    pub pages: Vec<PageAnalysis>,
    pub structured: Vec<PageMarkup>,
    pub discovery: Discovery,
    pub manual_checks: Vec<ManualCheck>,
    /// The kit's JSON-LD.
    pub kit: Vec<KitEntry>,
    /// Every file of the kit, relative to the kit directory.
    pub kit_files: Vec<String>,
    /// Social profiles of the site chrome that the kit's Organization does not claim.
    pub possible_profiles: Vec<String>,
    pub consistency: Option<ConsistencyLink>,
    /// (URL, redacted error) of the pages whose analysis failed.
    pub failed_pages: Vec<(String, String)>,
    pub sources: Vec<SourceRef>,
}

impl Default for GeoDoc {
    fn default() -> Self {
        GeoDoc::new(GeoMeta::default(), Checks::default(), Vec::new(), Vec::new())
    }
}

/// The manual checks, in report order.
pub fn manual_checks() -> Vec<ManualCheck> {
    vec![
        ManualCheck {
            key: "manual.search_generative_ai",
            url: findings::SRC_GOOGLE_AI_CONTROL.0,
        },
        ManualCheck {
            key: "manual.gsc_report",
            url: findings::SRC_GOOGLE_AI_REPORT.0,
        },
        ManualCheck {
            key: "manual.bing_report",
            url: findings::SRC_BING_AI_PERFORMANCE.0,
        },
        ManualCheck {
            key: "manual.cdn_waf",
            url: findings::SRC_OPENAI_BOTS.0,
        },
    ]
}

impl GeoDoc {
    /// The report of what the checks found: category states, issues, priorities and the summary
    /// are derived here, deterministically. `meta`'s page counts are taken from `checks`.
    pub fn new(mut meta: GeoMeta, checks: Checks, kit: Vec<KitEntry>, possible_profiles: Vec<String>) -> Self {
        let (categories, issues) = findings::assess(&checks);
        let priorities: Vec<Issue> = findings::priorities(&issues).into_iter().cloned().collect();
        let locale = ReportLocale::new(&meta.report_language);
        let summary = summary(&locale, &categories, &priorities.iter().collect::<Vec<_>>());
        meta.key_pages = checks.key_pages.len();
        meta.analyzed_pages = checks.analysis.selected;
        meta.reduced_input = checks
            .analysis
            .pages
            .iter()
            .filter(|page| !page.coverage.is_complete())
            .count();
        meta.analysis_unavailable = checks.analysis.unavailable.clone();
        GeoDoc {
            schema: SCHEMA,
            meta,
            evidence_note: text(&locale, "evidence.note").to_string(),
            summary,
            categories,
            priorities,
            issues,
            policy: checks.policy,
            access: checks.access,
            access_stats: checks.access_stats,
            controls: checks.controls,
            pdfs: checks.pdfs,
            render: checks.render,
            pages: checks.analysis.pages,
            structured: checks.markup,
            discovery: checks.discovery,
            manual_checks: manual_checks(),
            kit,
            kit_files: Vec::new(),
            possible_profiles,
            consistency: None,
            failed_pages: checks.analysis.failed,
            sources: findings::SOURCES.to_vec(),
        }
    }

    fn locale(&self) -> ReportLocale {
        ReportLocale::new(&self.meta.report_language)
    }

    /// The kit files for this report (`kit::build`): the robots.txt block and proposal for the
    /// homepage's origin, the JSON-LD, the drafts and the llms.txt.
    pub fn build_kit(&self, today: &str) -> Vec<KitFile> {
        let locale = self.locale();
        let origin = url::Url::parse(&self.meta.url)
            .map(|url| url.origin().ascii_serialization())
            .unwrap_or_default();
        let policy = self
            .policy
            .iter()
            .find(|policy| policy.origin.trim_end_matches('/') == origin);
        let state = policy.map_or(RobotsFetchState::NotAttempted, |policy| policy.state.clone());
        let robots = robots_of(&state);
        let agents: Vec<(&'static AiAgent, AgentAccess)> = policy
            .map(|policy| {
                policy
                    .agents
                    .iter()
                    .map(|verdict| (verdict.agent, verdict.access.clone()))
                    .collect()
            })
            .unwrap_or_default();
        let key_paths = policy.map(|policy| policy.paths.clone()).unwrap_or_default();
        let sitemaps: Vec<String> = self
            .discovery
            .sitemaps
            .iter()
            .filter(|sitemap| {
                matches!(sitemap.state, SitemapState::Parsed { .. })
                    && !origin.is_empty()
                    && sitemap.url.starts_with(&format!("{origin}/"))
            })
            .map(|sitemap| sitemap.url.clone())
            .collect();
        let titles: HashMap<String, String> = HashMap::new();
        let input = KitInput {
            locale: &locale,
            site_name: &self.meta.site_name,
            markup: &self.kit,
            possible_profiles: &self.possible_profiles,
            analyses: &self.pages,
            titles: &titles,
            agents: &agents,
            key_paths: &key_paths,
            sitemaps: &sitemaps,
        };
        kit::build(&input, &state, robots.as_ref(), today)
    }
}

// ---- texts ----

/// A fixed text of the report in English or Czech; empty for an unknown key.
pub fn text(locale: &ReportLocale, key: &str) -> &'static str {
    let table = if locale.is_czech() { CS } else { EN };
    table
        .iter()
        .find(|(known, _)| *known == key)
        .map_or("", |(_, text)| text)
}

/// A template with `{0}`, `{1}`, … replaced by `args`, `{*}` by all of them joined with `; ` and
/// `{n}` by `count`.
fn fill(template: &str, args: &[String], count: usize) -> String {
    let mut out = template
        .replace("{n}", &count.to_string())
        .replace("{*}", &args.join("; "));
    for (index, arg) in args.iter().enumerate() {
        out = out.replace(&format!("{{{index}}}"), arg);
    }
    out
}

fn issue_title(locale: &ReportLocale, issue: &Issue) -> String {
    fill(
        text(locale, &format!("issue.{}", issue.title_key)),
        &issue.args,
        issue.pages.len(),
    )
}

fn issue_fix(locale: &ReportLocale, issue: &Issue) -> String {
    fill(
        text(locale, &format!("fix.{}", issue.title_key)),
        &issue.args,
        issue.pages.len(),
    )
}

/// The report's opening lines: how many categories have problems and points to review on how
/// many key pages, what could not be assessed, and the biggest lever (the first priority).
pub fn summary(locale: &ReportLocale, cats: &[(CategoryId, CategoryState)], priorities: &[&Issue]) -> String {
    let count = |status: CheckStatus| cats.iter().filter(|(_, state)| state.status == status).count();
    let key_pages = cats
        .iter()
        .find(|(id, _)| *id == CategoryId::ObservedAccess)
        .map_or(0, |(_, state)| state.attempted);
    let mut out = fill(
        text(locale, "summary.template"),
        &[
            count(CheckStatus::Problem).to_string(),
            count(CheckStatus::Attention).to_string(),
            key_pages.to_string(),
        ],
        0,
    );
    let not_assessed = |category: CategoryId| {
        cats.iter()
            .any(|(id, state)| *id == category && state.status == CheckStatus::NotAssessed)
    };
    if not_assessed(CategoryId::CrawlerPolicy) {
        out.push(' ');
        out.push_str(text(locale, "summary.policy_not_assessed"));
    }
    if not_assessed(CategoryId::ObservedAccess) {
        out.push(' ');
        out.push_str(text(locale, "summary.access_not_assessed"));
    }
    out.push(' ');
    match priorities.first() {
        Some(first) => out.push_str(&fill(text(locale, "summary.lever"), &[issue_title(locale, first)], 0)),
        None => out.push_str(text(locale, "summary.nothing_found")),
    }
    out
}

fn status_label(locale: &ReportLocale, status: CheckStatus) -> &'static str {
    text(locale, &format!("status.{}", status.key()))
}

fn evidence_label(locale: &ReportLocale, evidence: Evidence) -> &'static str {
    text(locale, &format!("evidence.{}", evidence.key()))
}

// ---- document model shared by the Markdown and HTML renderings ----

/// Inline content: plain text (escaped by each renderer), emphasis, code, a link (only safe URLs
/// become links) or a status badge.
enum Inline {
    Text(String),
    Strong(String),
    Code(String),
    Link(String, String),
    Status(CheckStatus, String),
}

type Line = Vec<Inline>;

enum Node {
    Para(Line),
    List(Vec<Line>),
    Table(Vec<String>, Vec<Vec<Line>>),
    /// Preformatted text with its language (`html`, `json`).
    Code(String, &'static str),
    Note(Line),
    Heading(String),
}

struct Section {
    anchor: &'static str,
    title: String,
    nodes: Vec<Node>,
}

fn t(text: impl Into<String>) -> Inline {
    Inline::Text(text.into())
}

fn line(text: impl Into<String>) -> Line {
    vec![t(text)]
}

fn link(text: impl Into<String>, href: impl Into<String>) -> Inline {
    Inline::Link(text.into(), href.into())
}

/// `**label**: value`.
fn field(label: &str, value: impl Into<String>) -> Line {
    vec![Inline::Strong(label.to_string()), t(": "), t(value)]
}

/// The engines of an issue, with "All" in the report's language.
fn scope_text(locale: &ReportLocale, scope: &str) -> String {
    if scope == "All" {
        text(locale, "label.all_engines").to_string()
    } else {
        scope.to_string()
    }
}

/// Whether a URL may become a link: http(s) or relative, never `javascript:` and the like. Control
/// characters and whitespace are removed first, as browsers do.
fn safe_href(url: &str) -> bool {
    let normalized: String = url
        .chars()
        .filter(|c| !c.is_ascii_whitespace() && !c.is_control())
        .collect();
    for (at, c) in normalized.char_indices() {
        match c {
            ':' => {
                let scheme = normalized[..at].to_ascii_lowercase();
                return matches!(scheme.as_str(), "http" | "https");
            }
            '/' | '?' | '#' => return true,
            c if c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.') => continue,
            _ => return true,
        }
    }
    true
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Text for HTML.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

/// Whitespace runs as single spaces, keeping a space at either end.
fn collapse(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = false;
    for c in text.chars() {
        if c.is_whitespace() {
            space = true;
        } else {
            if space {
                out.push(' ');
                space = false;
            }
            out.push(c);
        }
    }
    if space {
        out.push(' ');
    }
    out
}

/// Text for Markdown: one line, no raw HTML, and no cell break inside a table.
fn md_text(text: &str, in_table: bool) -> String {
    let out = collapse(text).replace('<', "&lt;");
    if in_table { out.replace('|', "\\|") } else { out }
}

fn md_inline(inline: &Inline, in_table: bool) -> String {
    match inline {
        Inline::Text(text) => md_text(text, in_table),
        Inline::Strong(text) | Inline::Status(_, text) => format!("**{}**", md_text(text, in_table)),
        Inline::Code(text) => {
            let code = one_line(text).replace('`', "'");
            let code = if in_table { code.replace('|', "\\|") } else { code };
            format!("`{code}`")
        }
        Inline::Link(text, href) if safe_href(href) => {
            let label = md_text(text, in_table).replace('[', "(").replace(']', ")");
            let target = href.replace(' ', "%20").replace('(', "%28").replace(')', "%29");
            format!("[{label}]({target})")
        }
        Inline::Link(text, href) => {
            if text == href {
                md_text(text, in_table)
            } else {
                md_text(&format!("{text} ({href})"), in_table)
            }
        }
    }
}

fn md_line(line: &Line, in_table: bool) -> String {
    line.iter().map(|inline| md_inline(inline, in_table)).collect()
}

fn html_inline(inline: &Inline) -> String {
    match inline {
        Inline::Text(text) => escape(text),
        Inline::Strong(text) => format!("<strong>{}</strong>", escape(text)),
        Inline::Code(text) => format!("<code>{}</code>", escape(text)),
        Inline::Link(text, href) if safe_href(href) => {
            format!("<a href=\"{}\">{}</a>", escape(href), escape(text))
        }
        Inline::Link(text, _) => escape(text),
        Inline::Status(status, text) => format!("<span class=\"st st-{}\">{}</span>", status.key(), escape(text)),
    }
}

fn html_line(line: &Line) -> String {
    line.iter().map(html_inline).collect()
}

fn md_nodes(nodes: &[Node], out: &mut String) {
    for node in nodes {
        match node {
            Node::Para(line) => out.push_str(&format!("{}\n\n", md_line(line, false))),
            Node::List(items) => {
                for item in items {
                    out.push_str(&format!("- {}\n", md_line(item, false)));
                }
                out.push('\n');
            }
            Node::Table(head, rows) => {
                let head: Vec<String> = head.iter().map(|cell| md_text(cell, true)).collect();
                out.push_str(&format!("| {} |\n", head.join(" | ")));
                out.push_str(&format!("|{}\n", "---|".repeat(head.len())));
                for row in rows {
                    let cells: Vec<String> = row.iter().map(|cell| md_line(cell, true)).collect();
                    out.push_str(&format!("| {} |\n", cells.join(" | ")));
                }
                out.push('\n');
            }
            Node::Code(code, language) => {
                let longest = code.split(|c| c != '`').map(str::len).max().unwrap_or(0);
                let fence = "`".repeat(longest.max(2) + 1);
                out.push_str(&format!("{fence}{language}\n{}\n{fence}\n\n", code.trim_end()));
            }
            Node::Note(line) => out.push_str(&format!("> {}\n\n", md_line(line, false))),
            Node::Heading(title) => out.push_str(&format!("### {}\n\n", md_text(title, false))),
        }
    }
}

fn html_nodes(nodes: &[Node], out: &mut String) {
    for node in nodes {
        match node {
            Node::Para(line) => out.push_str(&format!("<p>{}</p>\n", html_line(line))),
            Node::List(items) => {
                out.push_str("<ul>\n");
                for item in items {
                    out.push_str(&format!("<li>{}</li>\n", html_line(item)));
                }
                out.push_str("</ul>\n");
            }
            Node::Table(head, rows) => {
                out.push_str("<div class=\"tw\"><table><thead><tr>");
                for cell in head {
                    out.push_str(&format!("<th>{}</th>", escape(cell)));
                }
                out.push_str("</tr></thead><tbody>\n");
                for row in rows {
                    out.push_str("<tr>");
                    for cell in row {
                        out.push_str(&format!("<td>{}</td>", html_line(cell)));
                    }
                    out.push_str("</tr>\n");
                }
                out.push_str("</tbody></table></div>\n");
            }
            Node::Code(code, _) => out.push_str(&format!("<pre><code>{}</code></pre>\n", escape(code.trim_end()))),
            Node::Note(line) => out.push_str(&format!("<p class=\"note\">{}</p>\n", html_line(line))),
            Node::Heading(title) => out.push_str(&format!("<h3>{}</h3>\n", escape(title))),
        }
    }
}

// ---- sections ----

impl GeoDoc {
    fn title(&self, locale: &ReportLocale) -> String {
        let title = text(locale, "label.title");
        if self.meta.site_name.trim().is_empty() {
            title.to_string()
        } else {
            format!("{title}: {}", one_line(&self.meta.site_name))
        }
    }

    fn state(&self, category: CategoryId) -> Option<&CategoryState> {
        self.categories
            .iter()
            .find(|(id, _)| *id == category)
            .map(|(_, state)| state)
    }

    fn sections(&self, locale: &ReportLocale, kit_dir: &str) -> Vec<Section> {
        vec![
            self.evidence_section(locale),
            self.overview_section(locale),
            self.priorities_section(locale),
            self.policy_section(locale),
            self.access_section(locale),
            self.controls_section(locale),
            self.render_section(locale),
            self.pages_section(locale),
            self.structured_section(locale, kit_dir),
            self.discovery_section(locale),
            self.manual_section(locale),
            self.kit_section(locale, kit_dir),
            self.method_section(locale),
        ]
    }

    fn evidence_section(&self, locale: &ReportLocale) -> Section {
        let evidence_rows: Vec<Vec<Line>> = [Evidence::Strong, Evidence::Moderate, Evidence::Weak]
            .iter()
            .map(|evidence| {
                vec![
                    vec![Inline::Strong(evidence_label(locale, *evidence).to_string())],
                    line(text(locale, &format!("evidence.{}.meaning", evidence.key()))),
                ]
            })
            .collect();
        let sources: Vec<Line> = self
            .sources
            .iter()
            .map(|(url, date)| vec![link(*url, *url), t(format!(" ({date})"))])
            .collect();
        Section {
            anchor: "s-evidence",
            title: text(locale, "sec.evidence").to_string(),
            nodes: vec![
                Node::Para(line(&self.evidence_note)),
                Node::Note(line(text(locale, "evidence.cannot_see"))),
                Node::List(
                    [
                        "evidence.fact.google",
                        "evidence.fact.bing",
                        "evidence.fact.js",
                        "evidence.fact.jsonld",
                    ]
                    .iter()
                    .map(|key| line(text(locale, key)))
                    .collect(),
                ),
                Node::Heading(text(locale, "label.evidence_strength").to_string()),
                Node::Table(
                    vec![
                        text(locale, "label.evidence").to_string(),
                        text(locale, "label.meaning").to_string(),
                    ],
                    evidence_rows,
                ),
                Node::Heading(text(locale, "label.sources").to_string()),
                Node::List(sources),
            ],
        }
    }

    fn coverage_text(&self, locale: &ReportLocale, category: CategoryId, state: &CategoryState) -> String {
        if state.status == CheckStatus::NotAssessed {
            let mut reason = text(locale, &format!("reason.{}", state.reason)).to_string();
            if state.reason == "ai_unavailable"
                && let Some(why) = &self.meta.analysis_unavailable
            {
                reason = format!("{reason}: {why}");
            }
            return reason;
        }
        let unit = match category {
            CategoryId::CrawlerPolicy => "unit.origins",
            CategoryId::AnswerExtractability | CategoryId::EntityClarity => "unit.analyzed_pages",
            _ => "unit.key_pages",
        };
        let mut out = fill(
            text(locale, "label.checked"),
            &[
                state.succeeded.to_string(),
                state.attempted.to_string(),
                text(locale, unit).to_string(),
            ],
            0,
        );
        if state.failed > 0 {
            out.push_str(&format!(
                " · {}",
                fill(text(locale, "label.failed_count"), &[state.failed.to_string()], 0)
            ));
        }
        if state.skipped > 0 {
            out.push_str(&format!(
                " · {}",
                fill(text(locale, "label.skipped_count"), &[state.skipped.to_string()], 0)
            ));
        }
        out
    }

    fn overview_section(&self, locale: &ReportLocale) -> Section {
        let rows: Vec<Vec<Line>> = self
            .categories
            .iter()
            .map(|(id, state)| {
                vec![
                    line(text(locale, &format!("cat.{}", id.key()))),
                    vec![Inline::Status(
                        state.status,
                        status_label(locale, state.status).to_string(),
                    )],
                    line(self.coverage_text(locale, *id, state)),
                ]
            })
            .collect();
        Section {
            anchor: "s-overview",
            title: text(locale, "sec.overview").to_string(),
            nodes: vec![Node::Table(
                vec![
                    text(locale, "label.category").to_string(),
                    text(locale, "label.status").to_string(),
                    text(locale, "label.coverage").to_string(),
                ],
                rows,
            )],
        }
    }

    fn pages_cell(pages: &[String]) -> Line {
        let mut cell: Line = vec![t(format!("{}", pages.len()))];
        for (index, page) in pages.iter().take(MAX_EXAMPLES).enumerate() {
            cell.push(t(if index == 0 { ": " } else { ", " }));
            cell.push(link(page.clone(), page.clone()));
        }
        if pages.len() > MAX_EXAMPLES {
            cell.push(t(", …"));
        }
        cell
    }

    fn priorities_section(&self, locale: &ReportLocale) -> Section {
        let nodes = if self.priorities.is_empty() {
            vec![Node::Para(line(text(locale, "label.no_priorities")))]
        } else {
            let rows = self
                .priorities
                .iter()
                .enumerate()
                .map(|(index, issue)| {
                    vec![
                        line((index + 1).to_string()),
                        vec![Inline::Status(
                            issue.status,
                            status_label(locale, issue.status).to_string(),
                        )],
                        line(evidence_label(locale, issue.evidence)),
                        line(scope_text(locale, issue.scope)),
                        vec![
                            t(format!("{} ", text(locale, &format!("cat.{}", issue.category.key())))),
                            t("— "),
                            Inline::Strong(issue_title(locale, issue)),
                        ],
                        line(issue_fix(locale, issue)),
                        Self::pages_cell(&issue.pages),
                        vec![link(issue.source.1, issue.source.0)],
                    ]
                })
                .collect();
            vec![Node::Table(
                [
                    "#",
                    text(locale, "label.status"),
                    text(locale, "label.evidence"),
                    text(locale, "label.scope"),
                    text(locale, "label.finding"),
                    text(locale, "label.fix"),
                    text(locale, "label.pages"),
                    text(locale, "label.source"),
                ]
                .iter()
                .map(|head| head.to_string())
                .collect(),
                rows,
            )]
        };
        Section {
            anchor: "s-priorities",
            title: text(locale, "sec.priorities").to_string(),
            nodes,
        }
    }

    /// The issues of a category as a list: status, evidence, title, fix and examples.
    fn issue_list(&self, locale: &ReportLocale, category: CategoryId) -> Node {
        self.issues_of(locale, &[category])
    }

    /// The issues of these categories as one list; "not assessed" when none of them was assessed,
    /// otherwise "nothing found" when there are no issues.
    fn issues_of(&self, locale: &ReportLocale, categories: &[CategoryId]) -> Node {
        let items: Vec<Line> = self
            .issues
            .iter()
            .filter(|issue| categories.contains(&issue.category))
            .map(|issue| {
                let mut item = vec![
                    Inline::Status(issue.status, status_label(locale, issue.status).to_string()),
                    t(format!(
                        " {} ({}: {}; {}) — ",
                        issue_title(locale, issue),
                        text(locale, "label.evidence"),
                        evidence_label(locale, issue.evidence),
                        scope_text(locale, issue.scope)
                    )),
                    t(issue_fix(locale, issue)),
                    t(" ("),
                    link(
                        format!("{} {}", text(locale, "label.source"), issue.source.1),
                        issue.source.0,
                    ),
                    t(")"),
                ];
                if !issue.pages.is_empty() {
                    item.push(t(format!(" {} ", text(locale, "label.examples"))));
                    for (index, page) in issue.pages.iter().take(MAX_EXAMPLES).enumerate() {
                        if index > 0 {
                            item.push(t(", "));
                        }
                        item.push(link(page.clone(), page.clone()));
                    }
                }
                item
            })
            .collect();
        if items.is_empty() {
            let not_assessed = categories.iter().all(|category| {
                self.state(*category)
                    .is_none_or(|state| state.status == CheckStatus::NotAssessed)
            });
            Node::Para(line(text(
                locale,
                if not_assessed {
                    "status.not_assessed"
                } else {
                    "label.no_issues"
                },
            )))
        } else {
            Node::List(items)
        }
    }

    fn policy_section(&self, locale: &ReportLocale) -> Section {
        let mut nodes = vec![self.issue_list(locale, CategoryId::CrawlerPolicy)];
        for origin in &self.policy {
            nodes.push(Node::Heading(origin.origin.clone()));
            let state = match &origin.state {
                RobotsFetchState::Ok { status, .. } => fill(text(locale, "state.ok"), &[status.to_string()], 0),
                RobotsFetchState::NotFound { status } => {
                    fill(text(locale, "state.not_found"), &[status.to_string()], 0)
                }
                RobotsFetchState::Unavailable { status_or_error } => fill(
                    text(locale, "state.unavailable"),
                    std::slice::from_ref(status_or_error),
                    0,
                ),
                RobotsFetchState::Skipped => text(locale, "state.skipped").to_string(),
                RobotsFetchState::NotAttempted => text(locale, "state.not_attempted").to_string(),
            };
            nodes.push(Node::Para(field(text(locale, "label.robots_state"), state)));
            if !origin.agents.is_empty() {
                let rows = origin
                    .agents
                    .iter()
                    .map(|verdict| {
                        let agent = verdict.agent;
                        let purposes: Vec<&str> = agent
                            .purposes
                            .iter()
                            .map(|purpose| {
                                text(
                                    locale,
                                    match purpose {
                                        Purpose::Search => "purpose.search",
                                        Purpose::UserFetch => "purpose.user_fetch",
                                        Purpose::Training => "purpose.training",
                                        Purpose::Grounding => "purpose.grounding",
                                    },
                                )
                            })
                            .collect();
                        let compliance = text(
                            locale,
                            match agent.compliance {
                                Compliance::Honors => "compliance.honors",
                                Compliance::MayNotHonor => "compliance.may_not",
                                Compliance::ControlToken => "compliance.control",
                            },
                        );
                        let verdict_text = match &verdict.access {
                            AgentAccess::Allowed => text(locale, "verdict.allowed").to_string(),
                            AgentAccess::Blocked => text(locale, "verdict.blocked").to_string(),
                            AgentAccess::Partly(blocked) => fill(
                                text(locale, "verdict.partly"),
                                &[blocked.len().to_string(), origin.paths.len().to_string()],
                                0,
                            ),
                        };
                        vec![
                            vec![link(agent.token, agent.doc_url)],
                            line(agent.vendor),
                            line(purposes.join(", ")),
                            line(compliance),
                            vec![Inline::Strong(verdict_text)],
                            verdict
                                .rule
                                .as_ref()
                                .map_or_else(Vec::new, |rule| vec![Inline::Code(rule.clone())]),
                            line(text(locale, agent.note_key)),
                        ]
                    })
                    .collect();
                nodes.push(Node::Table(
                    [
                        "label.agent",
                        "label.vendor",
                        "label.purpose",
                        "label.compliance",
                        "label.verdict",
                        "label.rule",
                        "label.note",
                    ]
                    .iter()
                    .map(|key| text(locale, key).to_string())
                    .collect(),
                    rows,
                ));
            }
            if !origin.content_signals.is_empty() {
                nodes.push(Node::Para(line(text(locale, "label.content_signals"))));
                nodes.push(Node::List(
                    origin
                        .content_signals
                        .iter()
                        .map(|signal| vec![Inline::Code(signal.clone())])
                        .collect(),
                ));
            }
            if !origin.sitemaps.is_empty() {
                nodes.push(Node::Para(line(text(locale, "label.declared_sitemaps"))));
                nodes.push(Node::List(
                    origin
                        .sitemaps
                        .iter()
                        .map(|sitemap| vec![link(sitemap.clone(), sitemap.clone())])
                        .collect(),
                ));
            }
        }
        Section {
            anchor: "s-policy",
            title: text(locale, "sec.policy").to_string(),
            nodes,
        }
    }

    fn access_section(&self, locale: &ReportLocale) -> Section {
        let mut nodes = vec![
            Node::Note(line(text(locale, "access.label"))),
            self.issue_list(locale, CategoryId::ObservedAccess),
        ];
        if !self.access.is_empty() {
            nodes.push(Node::Table(
                vec![
                    text(locale, "label.url").to_string(),
                    text(locale, "label.finding").to_string(),
                    text(locale, "label.detail").to_string(),
                ],
                self.access
                    .iter()
                    .map(|issue| {
                        vec![
                            vec![link(issue.url.clone(), issue.url.clone())],
                            line(kind_label(locale, &issue.kind)),
                            line(issue.detail.clone()),
                        ]
                    })
                    .collect(),
            ));
        }
        let stats = &self.access_stats;
        let mut facts: Vec<Line> = vec![line(fill(
            text(locale, "label.other_urls"),
            &[stats.other_urls.to_string()],
            0,
        ))];
        for (kind, count) in &stats.other_issues {
            facts.push(line(format!("{}: {count}", text(locale, &format!("kind.{kind}")))));
        }
        if let Some(p90) = stats.p90_seconds {
            facts.push(line(fill(text(locale, "label.p90"), &[format!("{p90:.2}")], 0)));
        }
        nodes.push(Node::List(facts));
        Section {
            anchor: "s-access",
            title: text(locale, "sec.access").to_string(),
            nodes,
        }
    }

    fn controls_section(&self, locale: &ReportLocale) -> Section {
        let mut nodes = vec![self.issue_list(locale, CategoryId::IndexingControls)];
        let restricted: Vec<&PageControls> = self
            .controls
            .iter()
            .filter(|page| {
                policy_text(locale, &page.google).is_some()
                    || policy_text(locale, &page.bing).is_some()
                    || page.nosnippet_share.is_some_and(|share| share > 0.0)
                    || page.canonical_elsewhere.is_some()
            })
            .collect();
        if !restricted.is_empty() {
            let none = text(locale, "label.none");
            nodes.push(Node::Table(
                vec![
                    text(locale, "label.url").to_string(),
                    "Google".to_string(),
                    "Bing".to_string(),
                    "data-nosnippet".to_string(),
                    text(locale, "label.canonical").to_string(),
                ],
                restricted
                    .iter()
                    .map(|page| {
                        vec![
                            vec![link(page.url.clone(), page.url.clone())],
                            line(policy_text(locale, &page.google).unwrap_or_else(|| none.to_string())),
                            line(policy_text(locale, &page.bing).unwrap_or_else(|| none.to_string())),
                            line(
                                page.nosnippet_share
                                    .map_or_else(|| none.to_string(), |share| format!("{} %", (share * 100.0).round())),
                            ),
                            page.canonical_elsewhere.as_ref().map_or_else(
                                || line(none),
                                |canonical| vec![link(canonical.clone(), canonical.clone())],
                            ),
                        ]
                    })
                    .collect(),
            ));
        }
        if !self.pdfs.is_empty() {
            nodes.push(Node::Para(line(text(locale, "label.pdf_note"))));
            nodes.push(Node::List(
                self.pdfs
                    .iter()
                    .map(|pdf| {
                        let mut item = vec![link(pdf.url.clone(), pdf.url.clone())];
                        if let Some(google) = policy_text(locale, &pdf.google) {
                            item.push(t(format!(" · Google: {google}")));
                        }
                        if let Some(bing) = policy_text(locale, &pdf.bing) {
                            item.push(t(format!(" · Bing: {bing}")));
                        }
                        item
                    })
                    .collect(),
            ));
        }
        Section {
            anchor: "s-controls",
            title: text(locale, "sec.controls").to_string(),
            nodes,
        }
    }

    fn render_section(&self, locale: &ReportLocale) -> Section {
        let mut nodes = vec![
            Node::Para(line(fill(
                text(locale, "label.render_checked"),
                &[self.render.checked.to_string()],
                0,
            ))),
            self.issue_list(locale, CategoryId::Rendering),
        ];
        if !self.render.risks.is_empty() {
            nodes.push(Node::Table(
                vec![
                    text(locale, "label.url").to_string(),
                    text(locale, "label.main_chars").to_string(),
                    text(locale, "label.markers").to_string(),
                ],
                self.render
                    .risks
                    .iter()
                    .map(|risk| {
                        vec![
                            vec![link(risk.url.clone(), risk.url.clone())],
                            line(risk.main_text_chars.to_string()),
                            line(risk.markers.join(", ")),
                        ]
                    })
                    .collect(),
            ));
        }
        Section {
            anchor: "s-render",
            title: text(locale, "sec.render").to_string(),
            nodes,
        }
    }

    fn pages_section(&self, locale: &ReportLocale) -> Section {
        let mut nodes = vec![self.issues_of(locale, &[CategoryId::AnswerExtractability, CategoryId::EntityClarity])];
        if let Some(why) = &self.meta.analysis_unavailable {
            nodes.push(Node::Note(line(fill(
                text(locale, "label.ai_unavailable"),
                std::slice::from_ref(why),
                0,
            ))));
        } else if self.pages.is_empty() && self.failed_pages.is_empty() {
            nodes.push(Node::Para(line(text(locale, "label.no_pages"))));
        }
        for page in &self.pages {
            nodes.push(Node::Heading(if page.title.trim().is_empty() {
                page.url.clone()
            } else {
                one_line(&page.title)
            }));
            let offer = match page.states_offer_early {
                OfferEarly::Yes => "offer.yes",
                OfferEarly::No => "offer.no",
                OfferEarly::NotApplicable => "offer.not_applicable",
            };
            let coverage = if page.coverage.is_complete() {
                text(locale, "label.coverage_complete").to_string()
            } else {
                fill(
                    text(locale, "label.coverage_reduced"),
                    &[
                        page.coverage.included.len().to_string(),
                        page.coverage.total.to_string(),
                    ],
                    0,
                )
            };
            nodes.push(Node::List(vec![
                vec![link(page.url.clone(), page.url.clone())],
                field(text(locale, "label.page_type"), page.page_type.key()),
                field(text(locale, "label.main_topic"), page.main_topic.clone()),
                field(text(locale, "label.offer_early"), text(locale, offer)),
                field(text(locale, "label.inspected"), coverage),
            ]));
            if !page.questions.is_empty() {
                nodes.push(Node::Table(
                    vec![
                        text(locale, "label.question").to_string(),
                        text(locale, "label.answer").to_string(),
                        text(locale, "label.excerpts").to_string(),
                    ],
                    page.questions
                        .iter()
                        .map(|question| {
                            vec![
                                line(question.question.clone()),
                                line(answered_label(locale, question.answered, &page.coverage)),
                                line(
                                    question
                                        .excerpts
                                        .iter()
                                        .map(|excerpt| format!("{}: {}", excerpt.block, excerpt.text))
                                        .collect::<Vec<_>>()
                                        .join(" / "),
                                ),
                            ]
                        })
                        .collect(),
                ));
            }
            if !page.vague_references.is_empty() {
                nodes.push(Node::Para(line(text(locale, "label.vague"))));
                nodes.push(Node::List(
                    page.vague_references
                        .iter()
                        .map(|excerpt| line(format!("{}: {}", excerpt.block, excerpt.text)))
                        .collect(),
                ));
            }
            if !page.improvements.is_empty() {
                nodes.push(Node::Table(
                    vec![
                        text(locale, "label.priority").to_string(),
                        text(locale, "label.improvement").to_string(),
                        text(locale, "label.fix").to_string(),
                        text(locale, "label.blocks").to_string(),
                    ],
                    page.improvements
                        .iter()
                        .map(|improvement| {
                            let priority = match improvement.priority {
                                Priority::High => "priority.high",
                                Priority::Medium => "priority.medium",
                                Priority::Low => "priority.low",
                            };
                            vec![
                                line(text(locale, priority)),
                                line(improvement.issue.clone()),
                                line(improvement.fix.clone()),
                                line(
                                    improvement
                                        .excerpts
                                        .iter()
                                        .map(|excerpt| excerpt.block.clone())
                                        .collect::<Vec<_>>()
                                        .join(", "),
                                ),
                            ]
                        })
                        .collect(),
                ));
            }
            if page.page_type == crate::ai::geo::analyze::PageType::Article {
                let missing = text(locale, "label.byline_missing").to_string();
                let author = page
                    .byline
                    .author
                    .as_ref()
                    .map_or_else(|| missing.clone(), |block| block.text.clone());
                let date = page
                    .byline
                    .date
                    .map_or(missing, |date| date.format("%Y-%m-%d").to_string());
                nodes.push(Node::Para(field(
                    text(locale, "label.byline"),
                    format!("{author} · {date}"),
                )));
            }
            if let Some(lead) = &page.lead {
                nodes.push(Node::Note(field(text(locale, "label.lead"), lead.text.clone())));
            }
        }
        if !self.failed_pages.is_empty() {
            nodes.push(Node::Heading(text(locale, "label.failed_pages").to_string()));
            nodes.push(Node::List(
                self.failed_pages
                    .iter()
                    .map(|(url, error)| vec![link(url.clone(), url.clone()), t(format!(": {error}"))])
                    .collect(),
            ));
        }
        Section {
            anchor: "s-pages",
            title: text(locale, "sec.pages").to_string(),
            nodes,
        }
    }

    fn structured_section(&self, locale: &ReportLocale, kit_dir: &str) -> Section {
        let mut nodes = vec![self.issue_list(locale, CategoryId::StructuredData)];
        if !self.structured.is_empty() {
            nodes.push(Node::Heading(text(locale, "label.existing").to_string()));
            nodes.push(Node::Table(
                vec![
                    text(locale, "label.url").to_string(),
                    text(locale, "label.types").to_string(),
                    text(locale, "label.parse_errors").to_string(),
                    text(locale, "label.invisible").to_string(),
                    "Microdata/RDFa".to_string(),
                ],
                self.structured
                    .iter()
                    .map(|page| {
                        let markup = &page.markup;
                        let other = if markup.has_microdata || markup.has_rdfa {
                            let mut kinds = Vec::new();
                            if markup.has_microdata {
                                kinds.push("Microdata");
                            }
                            if markup.has_rdfa {
                                kinds.push("RDFa");
                            }
                            format!("{} {}", kinds.join(" + "), markup.other_types.join(", "))
                        } else {
                            text(locale, "label.none").to_string()
                        };
                        vec![
                            vec![link(page.url.clone(), page.url.clone())],
                            line(markup.jsonld_types.join(", ")),
                            line(markup.parse_errors.join("; ")),
                            line(
                                markup
                                    .invisible_values
                                    .iter()
                                    .map(|(path, value)| format!("{path} = {value}"))
                                    .collect::<Vec<_>>()
                                    .join("; "),
                            ),
                            line(other.trim().to_string()),
                        ]
                    })
                    .collect(),
            ));
            if !self
                .structured
                .iter()
                .any(|page| page.markup.has_microdata || page.markup.has_rdfa)
            {
                nodes.push(Node::Para(line(text(locale, "label.no_microdata"))));
            }
        }
        nodes.push(Node::Heading(text(locale, "label.kit_markup").to_string()));
        nodes.push(Node::Note(vec![
            t(text(locale, "label.eligibility")),
            t(" "),
            t(text(locale, "label.faq_gone")),
        ]));
        if self.kit.is_empty() {
            nodes.push(Node::Para(line(text(locale, "label.no_kit_markup"))));
        }
        for entry in &self.kit {
            nodes.push(Node::Heading(format!("{} — {}", entry.kind, entry.page_url)));
            let html_path = format!("{kit_dir}/{}", entry.html_path());
            let json_path = format!("{kit_dir}/{}", entry.json_path());
            nodes.push(Node::Para(vec![
                t(format!("{} ", text(locale, "label.install_html"))),
                link(entry.html_path(), html_path),
                t(format!(" {} ", text(locale, "label.install_page"))),
                link(entry.page_url.clone(), entry.page_url.clone()),
                t(format!("; {} ", text(locale, "label.install_json"))),
                link(entry.json_path(), json_path),
                t("."),
            ]));
            if !entry.evidence.is_empty() {
                nodes.push(Node::Para(line(text(locale, "label.evidence_blocks"))));
                nodes.push(Node::List(
                    entry.evidence.iter().map(|evidence| line(evidence.clone())).collect(),
                ));
            }
            if !entry.notes.is_empty() {
                nodes.push(Node::List(entry.notes.iter().map(|note| line(note.clone())).collect()));
            }
            nodes.push(Node::Code(install_snippet(&entry.json), "html"));
        }
        Section {
            anchor: "s-structured",
            title: text(locale, "sec.structured").to_string(),
            nodes,
        }
    }

    fn discovery_section(&self, locale: &ReportLocale) -> Section {
        let discovery = &self.discovery;
        let mut nodes = vec![self.issue_list(locale, CategoryId::Discovery)];
        if !discovery.sitemaps.is_empty() {
            nodes.push(Node::Heading(text(locale, "label.sitemaps").to_string()));
            nodes.push(Node::Table(
                vec![
                    text(locale, "label.url").to_string(),
                    text(locale, "label.declared").to_string(),
                    text(locale, "label.state").to_string(),
                ],
                discovery
                    .sitemaps
                    .iter()
                    .map(|sitemap| {
                        vec![
                            vec![link(sitemap.url.clone(), sitemap.url.clone())],
                            line(text(locale, if sitemap.declared { "offer.yes" } else { "offer.no" })),
                            line(sitemap_state(locale, &sitemap.state)),
                        ]
                    })
                    .collect(),
            ));
        }
        if !discovery.missing_from_sitemaps.is_empty() {
            nodes.push(Node::Para(line(text(locale, "label.missing_from_sitemaps"))));
            nodes.push(Node::List(
                discovery
                    .missing_from_sitemaps
                    .iter()
                    .map(|page| vec![link(page.clone(), page.clone())])
                    .collect(),
            ));
        }
        if !discovery.listed_issues.is_empty() {
            nodes.push(Node::Para(line(text(locale, "label.listed_issues"))));
            nodes.push(Node::List(
                discovery
                    .listed_issues
                    .iter()
                    .map(|listed| {
                        let reason = match &listed.reason {
                            ListedReason::Status(code) => format!("HTTP {code}"),
                            ListedReason::NoIndex => "noindex".to_string(),
                            ListedReason::CanonicalElsewhere(target) => format!("canonical → {target}"),
                        };
                        vec![link(listed.url.clone(), listed.url.clone()), t(format!(" ({reason})"))]
                    })
                    .collect(),
            ));
        }
        if !discovery.hreflang_issues.is_empty() {
            nodes.push(Node::Para(line(text(locale, "label.hreflang"))));
            nodes.push(Node::List(
                discovery
                    .hreflang_issues
                    .iter()
                    .map(|issue| {
                        let problem = match issue.problem {
                            HreflangProblem::Status(code) => format!("HTTP {code}"),
                            HreflangProblem::NotCrawled => text(locale, "sitemap.not_crawled").to_string(),
                            HreflangProblem::OtherSite => text(locale, "label.not_checked").to_string(),
                        };
                        vec![
                            link(issue.page.clone(), issue.page.clone()),
                            t(format!(" [{}] → ", issue.lang)),
                            link(issue.target.clone(), issue.target.clone()),
                            t(format!(" ({problem})")),
                        ]
                    })
                    .collect(),
            ));
        }
        let modified = discovery.last_modified;
        if modified.pages > 0 {
            nodes.push(Node::Para(line(fill(
                text(locale, "label.last_modified"),
                &[
                    modified.with_header.to_string(),
                    modified.pages.to_string(),
                    modified.plausible.to_string(),
                ],
                0,
            ))));
        }
        Section {
            anchor: "s-discovery",
            title: text(locale, "sec.discovery").to_string(),
            nodes,
        }
    }

    fn manual_section(&self, locale: &ReportLocale) -> Section {
        Section {
            anchor: "s-manual",
            title: text(locale, "sec.manual").to_string(),
            nodes: vec![
                Node::Para(line(text(locale, "manual.label"))),
                Node::List(
                    self.manual_checks
                        .iter()
                        .map(|check| vec![t(format!("{} ", text(locale, check.key))), link(check.url, check.url)])
                        .collect(),
                ),
            ],
        }
    }

    fn kit_section(&self, locale: &ReportLocale, kit_dir: &str) -> Section {
        let mut nodes = Vec::new();
        if self.kit_files.is_empty() {
            nodes.push(Node::Para(line(text(locale, "label.kit_empty"))));
        } else {
            nodes.push(Node::Para(vec![
                t(format!("{} ", text(locale, "label.kit_dir"))),
                link(format!("{kit_dir}/README.md"), format!("{kit_dir}/README.md")),
            ]));
            nodes.push(Node::List(
                self.kit_files
                    .iter()
                    .map(|file| vec![link(file.clone(), format!("{kit_dir}/{file}"))])
                    .collect(),
            ));
        }
        if !self.possible_profiles.is_empty() {
            nodes.push(Node::Para(line(text(locale, "label.possible_profiles"))));
            nodes.push(Node::List(
                self.possible_profiles
                    .iter()
                    .map(|profile| vec![link(profile.clone(), profile.clone())])
                    .collect(),
            ));
        }
        Section {
            anchor: "s-kit",
            title: text(locale, "sec.kit").to_string(),
            nodes,
        }
    }

    fn method_section(&self, locale: &ReportLocale) -> Section {
        let meta = &self.meta;
        let analyzed_ok = self.pages.len();
        let mut facts: Vec<Line> = vec![
            line(format!(
                "{}: {}",
                text(locale, "label.pages_crawled"),
                meta.crawled_pages
            )),
            line(format!("{}: {}", text(locale, "label.key_pages"), meta.key_pages)),
            line(format!(
                "{}: {} ({} {}, {} {}, {} {})",
                text(locale, "label.analyzed"),
                meta.analyzed_pages,
                analyzed_ok,
                text(locale, "label.analyzed_ok"),
                self.failed_pages.len(),
                text(locale, "label.analyzed_failed"),
                meta.reduced_input,
                text(locale, "label.analyzed_reduced"),
            )),
        ];
        if !meta.model.is_empty() {
            facts.push(line(format!(
                "{}: {} / {} · {} {}k · {} {}",
                text(locale, "label.model"),
                meta.provider,
                meta.model,
                text(locale, "label.context"),
                meta.context_window / 1000,
                text(locale, "label.language"),
                meta.report_language
            )));
        }
        if meta.calls > 0 {
            facts.push(line(fill(
                text(locale, "label.calls"),
                &[
                    meta.calls.to_string(),
                    meta.input_tokens.to_string(),
                    meta.output_tokens.to_string(),
                ],
                0,
            )));
        }
        facts.push(line(format!(
            "{}: {RESEARCH_DATE}",
            text(locale, "label.research_date")
        )));
        let rows = self
            .categories
            .iter()
            .map(|(id, state)| {
                vec![
                    line(text(locale, &format!("cat.{}", id.key()))),
                    line(state.attempted.to_string()),
                    line(state.succeeded.to_string()),
                    line(state.failed.to_string()),
                    line(state.skipped.to_string()),
                ]
            })
            .collect();
        Section {
            anchor: "s-method",
            title: text(locale, "sec.method").to_string(),
            nodes: vec![
                Node::List(facts),
                Node::Table(
                    [
                        "label.category",
                        "label.attempted",
                        "label.succeeded",
                        "label.failed",
                        "label.skipped",
                    ]
                    .iter()
                    .map(|key| text(locale, key).to_string())
                    .collect(),
                    rows,
                ),
            ],
        }
    }
}

fn kind_label(locale: &ReportLocale, kind: &AccessKind) -> String {
    let label = text(locale, &format!("kind.{}", kind.key()));
    match kind {
        AccessKind::Slow(seconds) => format!("{label} ({seconds:.1} s)"),
        AccessKind::RedirectChain(hops) => format!("{label} ({hops})"),
        _ => label.to_string(),
    }
}

/// The restricting directives of an engine policy with where they come from; `None` without any.
fn policy_text(locale: &ReportLocale, policy: &EnginePolicy) -> Option<String> {
    let mut list: Vec<String> = Vec::new();
    if policy.expired {
        list.push(text(locale, "label.expired").to_string());
    } else if policy.noindex {
        list.push("noindex".to_string());
    }
    if policy.nosnippet {
        list.push("nosnippet".to_string());
    }
    if let Some(limit) = policy.max_snippet.filter(|limit| *limit >= 0) {
        list.push(format!("max-snippet:{limit}"));
    }
    if policy.noarchive {
        list.push("noarchive".to_string());
    }
    if policy.nocache {
        list.push("nocache".to_string());
    }
    if list.is_empty() {
        return None;
    }
    let mut out = list.join(", ");
    if !policy.sources.is_empty() {
        out.push_str(&format!(" ({})", policy.sources.join(", ")));
    }
    Some(out)
}

fn sitemap_state(locale: &ReportLocale, state: &SitemapState) -> String {
    match state {
        SitemapState::Parsed { kind, entries, lastmod } => {
            let mut out = fill(
                text(
                    locale,
                    match kind {
                        SitemapKind::UrlSet => "sitemap.urlset",
                        SitemapKind::Index => "sitemap.index",
                    },
                ),
                &[entries.to_string()],
                0,
            );
            if lastmod.suspicious() {
                out.push_str(&format!(" · {}", text(locale, "sitemap.suspicious_lastmod")));
            }
            out
        }
        SitemapState::Malformed(why) => fill(text(locale, "sitemap.malformed"), std::slice::from_ref(why), 0),
        SitemapState::Failed(code) => format!("HTTP {code}"),
        SitemapState::NotCrawled => text(locale, "sitemap.not_crawled").to_string(),
        SitemapState::NotAbsolute => text(locale, "sitemap.not_absolute").to_string(),
        SitemapState::Redirected(target) => fill(text(locale, "sitemap.redirected"), std::slice::from_ref(target), 0),
        SitemapState::Unsupported(format) => fill(text(locale, "sitemap.unsupported"), &[format.to_string()], 0),
    }
}

// ---- renderings ----

impl GeoDoc {
    pub fn to_markdown(&self, kit_dir: &str) -> String {
        let locale = self.locale();
        let mut out = format!("# {}\n\n", md_text(&self.title(&locale), false));
        let mut header = vec![md_text(&self.meta.host, false)];
        if !self.meta.crawled_at.is_empty() {
            header.push(md_text(&self.meta.crawled_at, false));
        }
        out.push_str(&format!("> {}\n\n", header.join(" · ")));
        out.push_str(&format!(
            "**{}:** {}\n\n",
            text(&locale, "label.summary"),
            md_text(&self.summary, false)
        ));
        for section in self.sections(&locale, kit_dir) {
            out.push_str(&format!("## {}\n\n", md_text(&section.title, false)));
            md_nodes(&section.nodes, &mut out);
        }
        out.push_str(&format!(
            "---\n\n{} [SiteOne Crawler](https://crawler.siteone.io/) ([GitHub](https://github.com/janreges/siteone-crawler)).\n",
            text(&locale, "label.generated_by")
        ));
        out
    }

    pub fn to_html(&self, kit_dir: &str) -> String {
        let locale = self.locale();
        let title = self.title(&locale);
        let sections = self.sections(&locale, kit_dir);
        let mut out = String::with_capacity(64 * 1024);
        out.push_str(&format!(
            "<!DOCTYPE html>\n<html lang=\"{}\" data-theme=\"light\">\n<head>\n<meta charset=\"utf-8\">\n\
             <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>{}</title>\n\
             <style>\n{CSS}\n</style>\n</head>\n<body>\n",
            escape(if locale.code().is_empty() { "en" } else { locale.code() }),
            escape(&title)
        ));
        out.push_str(&format!(
            "<header class=\"bar\"><a class=\"brand\" href=\"https://crawler.siteone.io/\" target=\"_blank\" rel=\"noopener\">SiteOne Crawler <span>{}</span></a>\
             <button id=\"themeBtn\" class=\"btn\" aria-label=\"{}\">◐</button></header>\n",
            escape(text(&locale, "label.title")),
            escape(text(&locale, "label.toggle_theme"))
        ));
        out.push_str(&format!(
            "<section class=\"hero\"><h1>{}</h1><div class=\"meta\">",
            escape(&title)
        ));
        if safe_href(&self.meta.url) && !self.meta.url.is_empty() {
            out.push_str(&format!(
                "<a href=\"{}\">{}</a>",
                escape(&self.meta.url),
                escape(&self.meta.host)
            ));
        } else {
            out.push_str(&escape(&self.meta.host));
        }
        if !self.meta.crawled_at.is_empty() {
            out.push_str(&format!(" · {}", escape(&self.meta.crawled_at)));
        }
        out.push_str(&format!(
            "</div><p class=\"summary\">{}</p></section>\n",
            escape(&self.summary)
        ));
        out.push_str("<div class=\"layout\">\n<nav class=\"toc\"><ul>");
        for section in &sections {
            out.push_str(&format!(
                "<li><a href=\"#{}\">{}</a></li>",
                section.anchor,
                escape(&section.title)
            ));
        }
        out.push_str("</ul></nav>\n<main>\n");
        for section in &sections {
            out.push_str(&format!(
                "<section class=\"sec\" id=\"{}\"><h2>{}</h2>\n",
                section.anchor,
                escape(&section.title)
            ));
            html_nodes(&section.nodes, &mut out);
            out.push_str("</section>\n");
        }
        out.push_str(&format!(
            "</main>\n</div>\n<footer class=\"foot\">{} <a href=\"https://crawler.siteone.io/\" target=\"_blank\" rel=\"noopener\">SiteOne Crawler</a> \
             (<a href=\"https://github.com/janreges/siteone-crawler\" target=\"_blank\" rel=\"noopener\">GitHub</a>).</footer>\n",
            escape(text(&locale, "label.generated_by"))
        ));
        out.push_str("<script>\n");
        out.push_str(JS);
        out.push_str("\n</script>\n</body>\n</html>\n");
        out
    }

    pub fn to_json(&self) -> Value {
        let locale = self.locale();
        let meta = &self.meta;
        json!({
            "schema": self.schema,
            "meta": {
                "host": meta.host,
                "url": meta.url,
                "siteName": meta.site_name,
                "crawledAt": meta.crawled_at,
                "reportLanguage": meta.report_language,
                "provider": meta.provider,
                "model": meta.model,
                "contextWindow": meta.context_window,
                "crawledPages": meta.crawled_pages,
                "keyPages": meta.key_pages,
                "analyzedPages": meta.analyzed_pages,
                "reducedInput": meta.reduced_input,
                "calls": meta.calls,
                "inputTokens": meta.input_tokens,
                "outputTokens": meta.output_tokens,
                "analysisUnavailable": meta.analysis_unavailable,
                "researchDate": RESEARCH_DATE,
            },
            "evidenceNote": self.evidence_note,
            "summary": self.summary,
            "categories": self.categories.iter().map(|(id, state)| json!({
                "id": id,
                "status": state.status,
                "attempted": state.attempted,
                "succeeded": state.succeeded,
                "failed": state.failed,
                "skipped": state.skipped,
                "reason": state.reason,
            })).collect::<Vec<_>>(),
            "priorities": self.priorities.iter().map(|issue| issue_json(&locale, issue)).collect::<Vec<_>>(),
            "issues": self.issues.iter().map(|issue| issue_json(&locale, issue)).collect::<Vec<_>>(),
            "policy": self.policy.iter().map(origin_json).collect::<Vec<_>>(),
            "access": {
                "issues": self.access.iter().map(|issue| json!({
                    "url": issue.url,
                    "kind": issue.kind.key(),
                    "detail": issue.detail,
                })).collect::<Vec<_>>(),
                "keyPages": self.access_stats.key_pages,
                "otherUrls": self.access_stats.other_urls,
                "otherIssues": self.access_stats.other_issues.iter()
                    .map(|(kind, count)| json!({"kind": kind, "count": count})).collect::<Vec<_>>(),
                "p90Seconds": self.access_stats.p90_seconds,
            },
            "controls": {
                "pages": self.controls.iter().map(|page| json!({
                    "url": page.url,
                    "google": policy_json(&page.google),
                    "bing": policy_json(&page.bing),
                    "nosnippetShare": page.nosnippet_share,
                    "canonicalElsewhere": page.canonical_elsewhere,
                })).collect::<Vec<_>>(),
                "pdfs": self.pdfs.iter().map(|pdf| json!({
                    "url": pdf.url,
                    "google": policy_json(&pdf.google),
                    "bing": policy_json(&pdf.bing),
                })).collect::<Vec<_>>(),
            },
            "render": {
                "checked": self.render.checked,
                "risks": self.render.risks.iter().map(|risk| json!({
                    "url": risk.url,
                    "mainTextChars": risk.main_text_chars,
                    "markers": risk.markers,
                })).collect::<Vec<_>>(),
            },
            "pages": self.pages.iter().map(page_json).collect::<Vec<_>>(),
            "structured": self.structured.iter().map(|page| json!({
                "url": page.url,
                "isHomepage": page.is_homepage,
                "jsonldTypes": page.markup.jsonld_types,
                "nestedTypes": page.markup.nested_types,
                "otherTypes": page.markup.other_types,
                "parseErrors": page.markup.parse_errors,
                "invisibleValues": page.markup.invisible_values.iter()
                    .map(|(path, value)| json!({"path": path, "value": value})).collect::<Vec<_>>(),
                "valuesNotChecked": page.markup.values_not_checked,
                "hasMicrodata": page.markup.has_microdata,
                "hasRdfa": page.markup.has_rdfa,
            })).collect::<Vec<_>>(),
            "discovery": discovery_json(&self.discovery),
            "manualChecks": self.manual_checks.iter().map(|check| json!({
                "key": check.key,
                "text": text(&locale, check.key),
                "url": check.url,
            })).collect::<Vec<_>>(),
            "kit": self.kit,
            "kitFiles": self.kit_files,
            "possibleProfiles": self.possible_profiles,
            "consistency": self.consistency.as_ref().map(|link| json!({
                "file": link.file,
                "possibleInconsistencies": link.possible_inconsistencies,
            })),
            "failedPages": self.failed_pages.iter()
                .map(|(url, error)| json!({"url": url, "error": error})).collect::<Vec<_>>(),
            "sources": self.sources.iter().map(|(url, date)| json!({"url": url, "date": date})).collect::<Vec<_>>(),
        })
    }
}

fn issue_json(locale: &ReportLocale, issue: &Issue) -> Value {
    json!({
        "id": issue.id,
        "category": issue.category,
        "status": issue.status,
        "evidence": issue.evidence,
        "scope": issue.scope,
        "source": {"url": issue.source.0, "date": issue.source.1},
        "titleKey": issue.title_key,
        "title": issue_title(locale, issue),
        "fix": issue_fix(locale, issue),
        "args": issue.args,
        "pages": issue.pages,
    })
}

fn origin_json(policy: &OriginPolicy) -> Value {
    let state = match &policy.state {
        RobotsFetchState::Ok { status, valid_utf8, .. } => {
            json!({"state": "ok", "status": status, "validUtf8": valid_utf8})
        }
        RobotsFetchState::NotFound { status } => json!({"state": "notFound", "status": status}),
        RobotsFetchState::Unavailable { status_or_error } => {
            json!({"state": "unavailable", "statusOrError": status_or_error})
        }
        RobotsFetchState::Skipped => json!({"state": "skipped"}),
        RobotsFetchState::NotAttempted => json!({"state": "notAttempted"}),
    };
    json!({
        "origin": policy.origin,
        "robotsTxt": state,
        "paths": policy.paths,
        "agents": policy.agents.iter().map(|verdict| {
            let agent = verdict.agent;
            let (access, blocked) = match &verdict.access {
                AgentAccess::Allowed => ("allowed", Vec::new()),
                AgentAccess::Blocked => ("blocked", policy.paths.clone()),
                AgentAccess::Partly(paths) => ("partly", paths.clone()),
            };
            json!({
                "token": agent.token,
                "vendor": agent.vendor,
                "purposes": agent.purposes.iter().map(|purpose| match purpose {
                    Purpose::Search => "search",
                    Purpose::UserFetch => "userFetch",
                    Purpose::Training => "training",
                    Purpose::Grounding => "grounding",
                }).collect::<Vec<_>>(),
                "compliance": match agent.compliance {
                    Compliance::Honors => "honors",
                    Compliance::MayNotHonor => "mayNotHonor",
                    Compliance::ControlToken => "controlToken",
                },
                "access": access,
                "blockedPaths": blocked,
                "rule": verdict.rule,
                "docUrl": agent.doc_url,
                "verifiedOn": agent.verified_on,
            })
        }).collect::<Vec<_>>(),
        "contentSignals": policy.content_signals,
        "sitemaps": policy.sitemaps,
    })
}

fn policy_json(policy: &EnginePolicy) -> Value {
    json!({
        "noindex": policy.noindex,
        "nosnippet": policy.nosnippet,
        "maxSnippet": policy.max_snippet,
        "noarchive": policy.noarchive,
        "nocache": policy.nocache,
        "expired": policy.expired,
        "sources": policy.sources,
    })
}

fn block_json(block: &Block) -> Value {
    json!({
        "block": block_ref(block.id),
        "text": block.text,
        "region": block.region,
        "collapsed": block.collapsed,
    })
}

fn page_json(page: &PageAnalysis) -> Value {
    json!({
        "url": page.url,
        "title": page.title,
        "lang": page.lang,
        "pageType": page.page_type,
        "mainTopic": page.main_topic,
        "statesOfferEarly": page.states_offer_early,
        "questions": page.questions,
        "unanswered": page.questions.iter().filter(|question| question.answered == Answered::No).count(),
        "vagueReferences": page.vague_references,
        "improvements": page.improvements,
        "lead": page.lead,
        "faqPairs": page.faq_pairs.iter().map(|pair| json!({
            "question": block_json(&pair.question),
            "answers": pair.answers.iter().map(block_json).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "byline": {
            "author": page.byline.author.as_ref().map(block_json),
            "dateBlock": page.byline.date_block.as_ref().map(block_json),
            "date": page.byline.date.map(|date| date.format("%Y-%m-%d").to_string()),
        },
        "entityDrafts": page.entity_drafts,
        "h1": page.h1.as_ref().map(block_json),
        "coverage": page.coverage,
        "rejected": page.rejected,
    })
}

fn discovery_json(discovery: &Discovery) -> Value {
    json!({
        "sitemaps": discovery.sitemaps.iter().map(|sitemap| {
            let state = match &sitemap.state {
                SitemapState::Parsed { kind, entries, lastmod } => json!({
                    "state": "parsed",
                    "kind": match kind { SitemapKind::UrlSet => "urlSet", SitemapKind::Index => "index" },
                    "entries": entries,
                    "lastmod": {
                        "entries": lastmod.entries,
                        "withLastmod": lastmod.with_lastmod,
                        "invalid": lastmod.invalid,
                        "future": lastmod.future,
                        "allIdentical": lastmod.all_identical,
                        "suspicious": lastmod.suspicious(),
                    },
                }),
                SitemapState::Malformed(why) => json!({"state": "malformed", "reason": why}),
                SitemapState::Failed(code) => json!({"state": "failed", "status": code}),
                SitemapState::NotCrawled => json!({"state": "notCrawled"}),
                SitemapState::NotAbsolute => json!({"state": "notAbsolute"}),
                SitemapState::Redirected(target) => json!({"state": "redirected", "target": target}),
                SitemapState::Unsupported(format) => json!({"state": "unsupported", "format": format}),
            };
            json!({"url": sitemap.url, "declared": sitemap.declared, "state": state})
        }).collect::<Vec<_>>(),
        "listedUrls": discovery.listed_urls,
        "listedNotCrawled": discovery.listed_not_crawled,
        "listedIssues": discovery.listed_issues.iter().map(|listed| {
            let reason = match &listed.reason {
                ListedReason::Status(code) => json!({"kind": "status", "status": code}),
                ListedReason::NoIndex => json!({"kind": "noindex"}),
                ListedReason::CanonicalElsewhere(target) => json!({"kind": "canonicalElsewhere", "target": target}),
            };
            json!({"url": listed.url, "reason": reason})
        }).collect::<Vec<_>>(),
        "keyPagesCompared": discovery.key_pages_compared,
        "missingFromSitemaps": discovery.missing_from_sitemaps,
        "hreflangIssues": discovery.hreflang_issues.iter().map(|issue| json!({
            "page": issue.page,
            "lang": issue.lang,
            "target": issue.target,
            "problem": match issue.problem {
                HreflangProblem::Status(code) => json!({"kind": "status", "status": code}),
                HreflangProblem::NotCrawled => json!({"kind": "notCrawled"}),
                HreflangProblem::OtherSite => json!({"kind": "otherSite"}),
            },
        })).collect::<Vec<_>>(),
        "lastModified": {
            "pages": discovery.last_modified.pages,
            "withHeader": discovery.last_modified.with_header,
            "plausible": discovery.last_modified.plausible,
        },
    })
}

const CSS: &str = r#":root{--bg:#f3f4f6;--surface:#fff;--ink:#111827;--muted:#6b7280;--border:#e5e7eb;--accent:#4e79a7;--chip:#eef2f7;--problem:#b91c1c;--attention:#b45309;--info:#1d4ed8;--ok:#15803d;--na:#6b7280}
[data-theme="dark"]{--bg:#0f172a;--surface:#1f2937;--ink:#e5e7eb;--muted:#9ca3af;--border:#374151;--accent:#7aa8d6;--chip:#243244;--problem:#f87171;--attention:#fbbf24;--info:#93c5fd;--ok:#4ade80;--na:#9ca3af}
*{box-sizing:border-box}body{margin:0;font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink)}
a{color:var(--accent);text-decoration:none;word-break:break-word}a:hover{text-decoration:underline}
.bar{position:sticky;top:0;z-index:10;display:flex;justify-content:space-between;align-items:center;padding:10px 22px;background:var(--surface);border-bottom:1px solid var(--border)}
.brand{font-weight:800;color:var(--ink)}.brand span{color:var(--accent);font-weight:700;margin-left:6px}
.btn{background:var(--chip);color:var(--ink);border:1px solid var(--border);border-radius:8px;padding:5px 11px;cursor:pointer}
.hero{max-width:1200px;margin:24px auto 0;padding:20px 24px;background:var(--surface);border:1px solid var(--border);border-radius:14px}
.hero h1{margin:0 0 6px;font-size:28px;line-height:1.2}.meta{color:var(--muted)}.summary{font-size:16px;margin:12px 0 0}
.layout{display:flex;gap:22px;max-width:1200px;margin:22px auto 0;padding:0 0 50px}
.toc{position:sticky;top:60px;align-self:flex-start;flex:0 0 230px}.toc ul{list-style:none;margin:0;padding:0}.toc li{margin:4px 0;font-size:13.5px}
main{flex:1;min-width:0}
.sec{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 20px;margin:0 0 16px}
.sec h2{margin:0 0 10px;font-size:20px;border-bottom:1px solid var(--border);padding-bottom:8px}.sec h3{font-size:16px;margin:16px 0 6px}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:0 0 12px;font-size:13.5px}
th,td{border:1px solid var(--border);padding:5px 8px;text-align:left;vertical-align:top}th{background:var(--chip)}
pre{background:var(--chip);border:1px solid var(--border);border-radius:8px;padding:10px;overflow-x:auto;font-size:12.5px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
.note{border-left:3px solid var(--accent);padding:6px 12px;background:var(--chip);border-radius:0 8px 8px 0}
.st{display:inline-block;font-weight:700;font-size:12px;padding:1px 8px;border-radius:999px;border:1px solid currentColor}
.st-problem{color:var(--problem)}.st-attention{color:var(--attention)}.st-info{color:var(--info)}.st-ok{color:var(--ok)}.st-not_assessed{color:var(--na)}
.foot{max-width:1200px;margin:0 auto;padding:16px 0 40px;color:var(--muted);font-size:13px}
@media(max-width:800px){.layout{flex-direction:column;padding:0 12px 40px}.toc{position:static}}"#;

const JS: &str = r#"(function(){var html=document.documentElement;
try{if(matchMedia('(prefers-color-scheme: dark)').matches)html.setAttribute('data-theme','dark');}catch(e){}
var b=document.getElementById('themeBtn');if(b)b.addEventListener('click',function(){html.setAttribute('data-theme',html.getAttribute('data-theme')==='dark'?'light':'dark');});})();"#;

/// English texts.
const EN: &[(&str, &str)] = &[
    ("label.title", "AI search readiness"),
    ("label.summary", "Summary"),
    ("label.generated_by", "Generated by"),
    ("label.toggle_theme", "Toggle light/dark theme"),
    ("sec.evidence", "What this report can and cannot tell"),
    ("sec.overview", "Overview"),
    ("sec.priorities", "Priorities"),
    ("sec.policy", "Crawler policy"),
    ("sec.access", "Observed access"),
    ("sec.controls", "Indexing & snippet controls"),
    ("sec.render", "Rendering"),
    ("sec.pages", "Answer extractability & entity clarity"),
    ("sec.structured", "Structured data"),
    ("sec.discovery", "Discovery & freshness"),
    ("sec.manual", "Manual checks"),
    ("sec.kit", "Kit contents"),
    ("sec.method", "Coverage & method"),
    ("cat.crawler_policy", "Crawler policy"),
    ("cat.observed_access", "Observed access"),
    ("cat.indexing_controls", "Indexing & snippet controls"),
    ("cat.rendering", "Rendering"),
    ("cat.answer_extractability", "Answer extractability"),
    ("cat.entity_clarity", "Entity clarity"),
    ("cat.structured_data", "Structured data"),
    ("cat.discovery", "Discovery & freshness"),
    ("cat.manual_checks", "Manual checks"),
    ("status.problem", "Problem"),
    ("status.attention", "Attention"),
    ("status.info", "Info"),
    ("status.ok", "OK"),
    ("status.not_assessed", "Not assessed"),
    ("evidence.strong", "Strong"),
    ("evidence.moderate", "Moderate"),
    ("evidence.weak", "Weak"),
    ("evidence.strong.meaning", "Documented behavior of the engines"),
    ("evidence.moderate.meaning", "Engine guidance, not measured"),
    (
        "evidence.weak.meaning",
        "Vendor claims, observational studies, or no evidence",
    ),
    (
        "evidence.note",
        "This report checks how well AI search and answer engines — Google AI Overviews and AI Mode, Bing Copilot, ChatGPT search, Perplexity and Claude — can reach, read, understand and quote the site. It never promises citations or rankings: the engines decide. Every recommendation shows the strength of its evidence, the engines it concerns and a dated source. Checks that did not run are marked \"not assessed\", never \"OK\".",
    ),
    (
        "evidence.cannot_see",
        "What the crawler cannot see: bot verification in your CDN/WAF, the Search generative AI setting in Search Console, mentions of the site elsewhere on the web, and real citations in AI answers.",
    ),
    (
        "evidence.fact.google",
        "Google: AI Overviews and AI Mode need no special files, AI text files or special schema.org markup; a page must be indexed and eligible for a snippet, and nosnippet, data-nosnippet, max-snippet and noindex apply to AI features too.",
    ),
    (
        "evidence.fact.bing",
        "Bing: grounding and citations in Copilot are eligibility outcomes, not promises; self-contained passages, clear names and accurate lastmod help; NOARCHIVE and NOCACHE restrict Copilot use.",
    ),
    (
        "evidence.fact.js",
        "JavaScript: log studies found no JavaScript execution by GPTBot, OAI-SearchBot, ClaudeBot or PerplexityBot; Google and Bing render pages. A page with almost no text in its HTML is a rendering risk, not proof.",
    ),
    (
        "evidence.fact.jsonld",
        "JSON-LD and llms.txt: an observational comparison showed no demonstrated AI-citation uplift from JSON-LD, and llms.txt is rarely requested; FAQ rich results stopped appearing in Google on 7 May 2026.",
    ),
    ("label.evidence_strength", "Evidence strength"),
    ("label.evidence", "Evidence"),
    ("label.meaning", "Meaning"),
    ("label.sources", "Sources (date)"),
    ("reason.no_key_pages", "no key pages were crawled"),
    ("reason.robots_not_read", "the rules of robots.txt are not known"),
    ("reason.ai_unavailable", "AI not available"),
    ("reason.analysis_failed", "the analysis of every page failed"),
    ("reason.no_pages_analyzed", "no page was analyzed"),
    ("reason.manual_only", "not visible from the website — please check"),
    ("reason.no_pages_checked", "no key page could be checked"),
    ("unit.origins", "origins"),
    ("unit.key_pages", "key pages"),
    ("unit.analyzed_pages", "analyzed pages"),
    ("label.checked", "checked {0} of {1} {2}"),
    ("label.failed_count", "{0} failed"),
    ("label.skipped_count", "{0} skipped"),
    ("label.category", "Category"),
    ("label.status", "Status"),
    ("label.coverage", "Coverage"),
    ("label.no_priorities", "The checks that ran found nothing to fix."),
    ("label.scope", "Engines"),
    ("label.finding", "Finding"),
    ("label.fix", "Fix"),
    ("label.pages", "Pages"),
    ("label.source", "Source"),
    ("label.examples", "e.g."),
    ("label.all_engines", "all engines"),
    ("label.no_issues", "Nothing found."),
    ("label.robots_state", "robots.txt"),
    ("state.ok", "fetched (HTTP {0})"),
    ("state.not_found", "none (HTTP {0}): everything is allowed"),
    ("state.unavailable", "could not be read ({0})"),
    ("state.skipped", "not fetched (--ignore-robots-txt)"),
    ("state.not_attempted", "not fetched"),
    ("purpose.search", "search"),
    ("purpose.user_fetch", "fetch on a user's request"),
    ("purpose.training", "training"),
    ("purpose.grounding", "grounding"),
    ("compliance.honors", "yes"),
    ("compliance.may_not", "may not (per the vendor)"),
    ("compliance.control", "control token"),
    ("verdict.allowed", "allowed"),
    ("verdict.blocked", "blocked"),
    ("verdict.partly", "partly blocked ({0} of {1})"),
    ("label.agent", "Agent"),
    ("label.vendor", "Vendor"),
    ("label.purpose", "Purpose"),
    ("label.compliance", "Honors robots.txt"),
    ("label.verdict", "Verdict"),
    ("label.rule", "Deciding rule"),
    ("label.note", "Note"),
    (
        "agent.googlebot",
        "Google Search including AI Overviews and AI Mode; a disallowed URL may still be indexed without a snippet.",
    ),
    (
        "agent.google_extended",
        "Control token for Gemini training and grounding; no effect on Google Search or AI Overviews.",
    ),
    (
        "agent.bingbot",
        "Bing search and Copilot; Copilot use is controlled by NOARCHIVE and NOCACHE.",
    ),
    (
        "agent.oai_searchbot",
        "ChatGPT search; OpenAI also recommends allowing its published IP ranges.",
    ),
    (
        "agent.chatgpt_user",
        "Fetches pages a ChatGPT user asks for; robots.txt may not apply.",
    ),
    ("agent.gptbot", "Collects training data for OpenAI's models."),
    ("agent.claude_searchbot", "Claude's search index."),
    (
        "agent.claude_user",
        "Fetches pages a Claude user asks for; honors robots.txt.",
    ),
    ("agent.claudebot", "Collects training data for Anthropic's models."),
    (
        "agent.perplexitybot",
        "Perplexity's search index; not used to train foundation models.",
    ),
    (
        "agent.perplexity_user",
        "Fetches pages a Perplexity user asks for; generally ignores robots.txt.",
    ),
    (
        "agent.applebot",
        "Apple search (Siri, Spotlight) and context for Apple's AI; follows Googlebot's rules when robots.txt has none for Applebot.",
    ),
    (
        "agent.applebot_extended",
        "Control token: whether Apple may use content to train its AI models.",
    ),
    ("agent.meta_externalagent", "Collects content for Meta's AI models."),
    ("agent.ccbot", "Common Crawl, a widely used source of AI training data."),
    (
        "label.content_signals",
        "Content-Signal / Content-Usage lines (not honored by major crawlers, mid-2026):",
    ),
    ("label.declared_sitemaps", "Sitemaps declared in robots.txt:"),
    (
        "access.label",
        "As seen by SiteOne Crawler's user agent — your CDN/WAF may treat the crawlers of AI vendors differently (see Manual checks).",
    ),
    ("label.url", "URL"),
    ("label.detail", "Detail"),
    ("kind.denied", "Denied"),
    ("kind.rate_limited", "Rate limited"),
    ("kind.server_error", "Server error"),
    ("kind.transport", "Not fetched"),
    ("kind.redirect_chain", "Redirect chain"),
    ("kind.redirect_loop", "Redirect loop"),
    ("kind.suspected_challenge", "Suspected challenge page"),
    ("kind.suspected_soft_404", "Suspected soft 404"),
    ("kind.slow", "Slow"),
    (
        "label.other_urls",
        "Other internal URLs: {0}, checked by status code and redirects only",
    ),
    ("label.p90", "90th percentile of response times: {0} s"),
    ("label.none", "none"),
    ("label.canonical", "Canonical elsewhere"),
    (
        "label.pdf_note",
        "PDFs with robots restrictions (the PDF text was not inspected):",
    ),
    ("label.expired", "noindex (unavailable_after passed)"),
    (
        "label.render_checked",
        "{0} key page(s) were checked in the HTML as fetched, without running JavaScript.",
    ),
    ("label.main_chars", "Visible text (characters)"),
    ("label.markers", "App-shell signs"),
    (
        "label.ai_unavailable",
        "AI not available: {0}. The per-page analysis did not run.",
    ),
    ("label.no_pages", "No page was analyzed."),
    ("offer.yes", "yes"),
    ("offer.no", "no"),
    ("offer.not_applicable", "not applicable"),
    ("label.coverage_complete", "all blocks"),
    ("label.coverage_reduced", "{0} of {1} blocks (the input was reduced)"),
    ("label.page_type", "Page type"),
    ("label.main_topic", "Main topic"),
    ("label.offer_early", "States its offer early"),
    ("label.inspected", "Inspected"),
    ("label.question", "Likely visitor question"),
    ("label.answer", "Answered"),
    ("label.excerpts", "Excerpts from the page"),
    (
        "label.vague",
        "Passages that rely on \"it\", \"we\" or \"this product\":",
    ),
    ("label.priority", "Priority"),
    ("label.improvement", "Improvement"),
    ("label.blocks", "Blocks"),
    ("priority.high", "high"),
    ("priority.medium", "medium"),
    ("priority.low", "low"),
    ("label.byline", "Author and date"),
    ("label.byline_missing", "not found near the title"),
    ("label.lead", "Lead draft (review before publishing; see leads.md)"),
    ("label.failed_pages", "Pages whose analysis failed"),
    ("label.existing", "Existing markup on the key pages"),
    ("label.types", "JSON-LD types"),
    ("label.parse_errors", "Parse errors"),
    ("label.invisible", "Values not visible on the page"),
    ("label.no_microdata", "No Microdata or RDFa was detected."),
    (
        "label.kit_markup",
        "Markup in the kit (built by the crawler from the pages)",
    ),
    (
        "label.eligibility",
        "Eligibility for a search feature is decided by the engine.",
    ),
    (
        "label.faq_gone",
        "FAQ rich results no longer appear in Google (May 2026).",
    ),
    ("label.no_kit_markup", "The kit has no JSON-LD."),
    ("label.install_html", "Paste"),
    ("label.install_page", "into the page"),
    ("label.install_json", "or use"),
    ("label.evidence_blocks", "Evidence (text of the page):"),
    ("label.sitemaps", "Sitemaps"),
    ("label.declared", "Declared"),
    ("label.state", "State"),
    (
        "label.missing_from_sitemaps",
        "Indexable key pages missing from the sitemaps:",
    ),
    ("label.listed_issues", "Listed URLs that are not indexable:"),
    ("label.hreflang", "hreflang alternates that lead nowhere useful:"),
    ("label.not_checked", "not checked"),
    (
        "label.last_modified",
        "Last-Modified: sent by {0} of {1} key pages, {2} with a usable date.",
    ),
    ("sitemap.urlset", "{0} URLs"),
    ("sitemap.index", "index of {0} sitemaps"),
    ("sitemap.suspicious_lastmod", "suspicious lastmod"),
    ("sitemap.malformed", "malformed: {0}"),
    ("sitemap.not_crawled", "not crawled"),
    ("sitemap.not_absolute", "not an absolute URL"),
    ("sitemap.redirected", "redirects to {0}"),
    ("sitemap.unsupported", "{0} (not read)"),
    ("manual.label", "Not visible from the website — please check:"),
    (
        "manual.search_generative_ai",
        "Google Search Console → Settings → Search generative AI: Include / Exclude / Inherit (check subdomains and property inheritance too).",
    ),
    (
        "manual.gsc_report",
        "The generative-AI performance report in Search Console.",
    ),
    (
        "manual.bing_report",
        "The AI Performance report in Bing Webmaster Tools.",
    ),
    (
        "manual.cdn_waf",
        "CDN/WAF \"block AI bots\" settings and bot verification (Cloudflare and others); when you use a WAF, allow OpenAI's published IP ranges.",
    ),
    ("label.kit_empty", "No kit files were written."),
    ("label.kit_dir", "The deployable kit — start with"),
    (
        "label.possible_profiles",
        "Possible social profiles — add them to the Organization by hand if they are yours:",
    ),
    ("label.pages_crawled", "Pages crawled"),
    ("label.key_pages", "Key pages"),
    ("label.analyzed", "Pages chosen for analysis"),
    ("label.analyzed_ok", "analyzed"),
    ("label.analyzed_failed", "failed"),
    ("label.analyzed_reduced", "with reduced input"),
    ("label.model", "Model"),
    ("label.context", "context"),
    ("label.language", "language"),
    ("label.calls", "LLM calls: {0} · {1} input / {2} output tokens"),
    ("label.research_date", "Research date of the evidence"),
    ("label.attempted", "Attempted"),
    ("label.succeeded", "Succeeded"),
    ("label.failed", "Failed"),
    ("label.skipped", "Skipped"),
    (
        "summary.template",
        "{0} problem(s) and {1} point(s) to review on {2} key pages.",
    ),
    ("summary.lever", "The biggest lever: {0}."),
    (
        "summary.nothing_found",
        "The checks that ran found no problems and no points to review.",
    ),
    (
        "summary.policy_not_assessed",
        "Crawler policy was not assessed, so this report cannot tell whether AI crawlers may fetch the site.",
    ),
    (
        "summary.access_not_assessed",
        "Observed access was not assessed, so this report cannot tell whether the key pages can be fetched.",
    ),
    (
        "issue.search_crawler_blocked",
        "{0} is blocked by robots.txt on {1} ({n} key page(s)): {2}",
    ),
    (
        "fix.search_crawler_blocked",
        "If you want to appear in {0}'s search answers, remove or narrow the rule that blocks it.",
    ),
    (
        "issue.search_crawler_partly_blocked",
        "{0} is blocked on {n} key page(s) of {1}: {2}",
    ),
    (
        "fix.search_crawler_partly_blocked",
        "Check that these pages are meant to stay out of {0}'s answers; otherwise narrow the rule.",
    ),
    (
        "issue.user_fetch_blocked",
        "{0} (fetches pages on a user's request) is blocked on {1}: {2}",
    ),
    (
        "fix.user_fetch_blocked",
        "Allow {0} if assistants should be able to open your pages when a user asks.",
    ),
    (
        "issue.user_fetch_blocked_limited",
        "{0} is blocked on {1}, but its vendor says robots.txt may not apply to it (limited effect): {2}",
    ),
    (
        "fix.user_fetch_blocked_limited",
        "No change is needed for visibility; to really stop these fetches, use your CDN/WAF.",
    ),
    (
        "issue.training_crawler_blocked",
        "The AI-training crawler {0} is blocked on {1}: {2}",
    ),
    (
        "fix.training_crawler_blocked",
        "Nothing to do if intended: blocking training crawlers does not affect visibility in AI search.",
    ),
    (
        "issue.grounding_control_blocked",
        "{0} is blocked on {1}: Gemini may use your pages neither for training nor for grounding ({2})",
    ),
    (
        "fix.grounding_control_blocked",
        "Keep it if intended; it does not affect Google Search or AI Overviews, but Gemini's grounded answers will not use your pages.",
    ),
    (
        "issue.content_signal_present",
        "robots.txt of {0} declares Content-Signal / Content-Usage, which major crawlers do not honor (mid-2026): {1}",
    ),
    (
        "fix.content_signal_present",
        "Do not rely on these lines; use user-agent groups and the engines' own controls.",
    ),
    (
        "issue.robots_unavailable",
        "robots.txt of {0} could not be read ({1}); engines may treat a failing robots.txt as \"disallow all\"",
    ),
    (
        "fix.robots_unavailable",
        "Make /robots.txt answer 200 reliably (or 404 when you have none), for AI crawlers too.",
    ),
    (
        "issue.robots_skipped",
        "robots.txt of {0} was not fetched (--ignore-robots-txt)",
    ),
    (
        "fix.robots_skipped",
        "Crawl without --ignore-robots-txt to assess the crawler policy.",
    ),
    ("issue.robots_not_attempted", "robots.txt of {0} was not fetched"),
    (
        "fix.robots_not_attempted",
        "Crawl the robots.txt of this origin to assess its crawler policy.",
    ),
    (
        "issue.robots_not_utf8",
        "robots.txt of {0} is not valid UTF-8; rules with invalid bytes cannot be evaluated exactly",
    ),
    ("fix.robots_not_utf8", "Save robots.txt as UTF-8 (RFC 9309)."),
    (
        "issue.access_denied",
        "{n} key page(s) answered 401 or 403 to the crawler",
    ),
    (
        "fix.access_denied",
        "Public pages must answer 200 to search and AI crawlers; check your server and CDN/WAF rules.",
    ),
    (
        "issue.access_rate_limited",
        "{n} key page(s) answered 429 (too many requests)",
    ),
    (
        "fix.access_rate_limited",
        "Raise the rate limits for verified search and AI crawlers.",
    ),
    (
        "issue.access_server_error",
        "{n} key page(s) answered with a server error (5xx)",
    ),
    (
        "fix.access_server_error",
        "Fix the server errors; engines drop pages that keep failing.",
    ),
    (
        "issue.access_failed",
        "{n} key page(s) could not be fetched (connection error or timeout)",
    ),
    (
        "fix.access_failed",
        "Check the availability and response times of these URLs.",
    ),
    (
        "issue.access_redirect_chain",
        "{n} key page(s) redirect through more than 2 hops",
    ),
    (
        "fix.access_redirect_chain",
        "Link and redirect straight to the final URL.",
    ),
    ("issue.access_redirect_loop", "{n} key page(s) redirect in a loop"),
    (
        "fix.access_redirect_loop",
        "Break the loop: every redirect must end at a page that answers 200.",
    ),
    (
        "issue.access_suspected_challenge",
        "{n} key page(s) look like a bot challenge or a block page (suspected)",
    ),
    (
        "fix.access_suspected_challenge",
        "Check your bot protection: search and AI crawlers must get the real page.",
    ),
    (
        "issue.access_suspected_soft_404",
        "{n} key page(s) answer 200 but say \"not found\" (suspected soft 404)",
    ),
    (
        "fix.access_suspected_soft_404",
        "Answer 404 for missing pages, or restore their content.",
    ),
    (
        "issue.access_slow",
        "{n} key page(s) were slower than the crawl's 90th percentile and 3 s (observed retrieval risk)",
    ),
    (
        "fix.access_slow",
        "Speed up these pages; slow responses are fetched less reliably.",
    ),
    (
        "issue.other_urls_issues",
        "{0} issue(s) on the other internal URLs (of {1}), counted but not rated",
    ),
    (
        "fix.other_urls_issues",
        "See the counts in the observed-access section.",
    ),
    (
        "issue.google_noindex",
        "{n} key page(s) are noindex for Google (or their unavailable_after has passed)",
    ),
    (
        "fix.google_noindex",
        "Remove noindex from pages that should appear in Google Search and AI Overviews.",
    ),
    (
        "issue.google_nosnippet",
        "{n} key page(s) forbid snippets for Google (nosnippet)",
    ),
    (
        "fix.google_nosnippet",
        "Remove nosnippet: Google cannot quote such pages in AI Overviews or AI Mode.",
    ),
    (
        "issue.google_max_snippet_zero",
        "{n} key page(s) set max-snippet:0 for Google",
    ),
    (
        "fix.google_max_snippet_zero",
        "Raise or remove max-snippet so that Google can quote the page.",
    ),
    (
        "issue.bing_noindex",
        "{n} key page(s) are noindex for Bing (or their unavailable_after has passed)",
    ),
    (
        "fix.bing_noindex",
        "Remove noindex from pages that should appear in Bing and Copilot.",
    ),
    (
        "issue.bing_nosnippet",
        "{n} key page(s) forbid snippets for Bing (nosnippet)",
    ),
    (
        "fix.bing_nosnippet",
        "Remove nosnippet: Bing and Copilot cannot quote such pages.",
    ),
    (
        "issue.bing_max_snippet_zero",
        "{n} key page(s) set max-snippet:0 for Bing",
    ),
    (
        "fix.bing_max_snippet_zero",
        "Raise or remove max-snippet so that Bing can quote the page.",
    ),
    (
        "issue.bing_copilot_restricted",
        "{n} key page(s) restrict Copilot use with NOARCHIVE or NOCACHE for Bing",
    ),
    (
        "fix.bing_copilot_restricted",
        "Remove noarchive / nocache for Bing if Copilot answers should use the whole page.",
    ),
    (
        "issue.short_max_snippet",
        "{n} key page(s) limit snippets to 50 characters or fewer (a product heuristic)",
    ),
    (
        "fix.short_max_snippet",
        "Consider a longer max-snippet so that answers can quote a whole sentence.",
    ),
    (
        "issue.data_nosnippet_large",
        "data-nosnippet hides 30 % or more of the text of {n} key page(s)",
    ),
    (
        "fix.data_nosnippet_large",
        "Keep data-nosnippet for small parts only: hidden text cannot be quoted.",
    ),
    (
        "issue.canonical_elsewhere",
        "{n} key page(s) name another URL as their canonical: {*}",
    ),
    (
        "fix.canonical_elsewhere",
        "Check that the canonical URL is intended; engines index and quote the canonical page.",
    ),
    (
        "issue.pdf_restricted",
        "{n} PDF(s) restrict indexing or snippets with X-Robots-Tag (the PDF text was not inspected)",
    ),
    ("fix.pdf_restricted", "Check that these restrictions are intended."),
    (
        "issue.likely_client_rendered",
        "{n} key page(s) look client-rendered: almost no text in their HTML (rendering risk)",
    ),
    (
        "fix.likely_client_rendered",
        "Render the main content on the server: AI crawlers that do not run JavaScript see an empty page.",
    ),
    (
        "issue.offer_not_early",
        "{n} analyzed page(s) do not state early what they offer or answer",
    ),
    (
        "fix.offer_not_early",
        "Open the page with a sentence that says what it offers or answers.",
    ),
    (
        "issue.unanswered_questions",
        "{0} likely visitor question(s) were not found on {n} analyzed page(s)",
    ),
    (
        "fix.unanswered_questions",
        "Answer the most important questions explicitly in visible text (see leads.md in the kit).",
    ),
    (
        "issue.hidden_content",
        "Half or more of the text of {n} analyzed page(s) is inside collapsed tabs or accordions",
    ),
    (
        "fix.hidden_content",
        "Show the key answers in visible text, not only inside collapsed elements.",
    ),
    (
        "issue.no_contextual_links",
        "{n} analyzed page(s) have no links to other pages of the site in their content",
    ),
    (
        "fix.no_contextual_links",
        "Link related pages from the text so that crawlers and readers find them.",
    ),
    (
        "issue.images_without_alt",
        "{0} image(s) without an alt attribute on {n} analyzed page(s)",
    ),
    (
        "fix.images_without_alt",
        "Describe informative images in their alt attribute.",
    ),
    (
        "issue.vague_references",
        "{n} analyzed page(s) have passages that rely on \"it\", \"we\" or \"this product\"",
    ),
    (
        "fix.vague_references",
        "Name the subject in every passage so that it makes sense when quoted alone.",
    ),
    (
        "issue.article_without_byline",
        "{n} article(s) show no verified author and date near their title",
    ),
    (
        "fix.article_without_byline",
        "Show the author and the publication or update date next to the headline.",
    ),
    (
        "issue.jsonld_parse_error",
        "{0} JSON-LD block(s) on {n} key page(s) are not valid JSON",
    ),
    (
        "fix.jsonld_parse_error",
        "Fix the syntax: engines ignore invalid structured data.",
    ),
    (
        "issue.jsonld_values_not_visible",
        "Structured data on {n} key page(s) has values that do not appear on the page: {*}",
    ),
    (
        "fix.jsonld_values_not_visible",
        "Keep structured data identical to the visible text, or show the values on the page.",
    ),
    (
        "issue.jsonld_values_not_checked",
        "{0} structured-data value(s) on {n} key page(s) were not checked (too many on one page)",
    ),
    ("fix.jsonld_values_not_checked", "Nothing to do; this check is partial."),
    (
        "issue.faq_markup_present",
        "FAQPage markup on {n} key page(s): FAQ rich results no longer appear in Google (since May 2026)",
    ),
    (
        "fix.faq_markup_present",
        "Keep it only for real, visible questions and answers.",
    ),
    (
        "issue.no_site_markup",
        "The homepage has no {*} markup (the kit provides it)",
    ),
    (
        "fix.no_site_markup",
        "Optionally install the site-wide JSON-LD from the kit (evidence: weak).",
    ),
    (
        "issue.no_sitemap",
        "No sitemap was found (neither declared in robots.txt nor crawled)",
    ),
    (
        "fix.no_sitemap",
        "Publish an XML sitemap with accurate lastmod and declare it in robots.txt.",
    ),
    (
        "issue.sitemap_unreadable",
        "{n} sitemap(s) could not be read (malformed, failing or declared without an absolute URL)",
    ),
    (
        "fix.sitemap_unreadable",
        "Fix these sitemaps and declare them with absolute URLs.",
    ),
    (
        "issue.sitemap_not_checked",
        "{n} sitemap(s) were not crawled or are feeds or text files, so they were not checked",
    ),
    (
        "fix.sitemap_not_checked",
        "Nothing to do, or make the crawl reach them to include them in the check.",
    ),
    (
        "issue.sitemap_comparison_not_assessed",
        "Key pages were not compared with the sitemaps: not every sitemap could be read",
    ),
    (
        "fix.sitemap_comparison_not_assessed",
        "Make every sitemap readable to assess the coverage.",
    ),
    (
        "issue.missing_from_sitemap",
        "{n} indexable key page(s) are missing from the sitemaps",
    ),
    ("fix.missing_from_sitemap", "List every indexable page in the sitemap."),
    (
        "issue.sitemap_lists_non_indexable",
        "The sitemaps list {n} URL(s) that are not indexable (redirects, errors, noindex or another canonical)",
    ),
    (
        "fix.sitemap_lists_non_indexable",
        "List only final, indexable, canonical URLs.",
    ),
    (
        "issue.suspicious_lastmod",
        "{n} sitemap(s) have implausible lastmod values (all identical or in the future)",
    ),
    (
        "fix.suspicious_lastmod",
        "Set lastmod to the real date of the last content change; Bing uses it for freshness.",
    ),
    (
        "issue.hreflang_target_broken",
        "hreflang alternates of {n} key page(s) lead to URLs that do not answer 200: {*}",
    ),
    (
        "fix.hreflang_target_broken",
        "Point every hreflang alternate to a live, canonical page.",
    ),
    (
        "issue.hreflang_target_not_crawled",
        "hreflang alternates of {n} key page(s) lead to URLs that were not crawled: {*}",
    ),
    (
        "fix.hreflang_target_not_crawled",
        "Nothing to do, or make the crawl reach them to check them.",
    ),
    (
        "issue.last_modified_missing",
        "Only {0} of {1} key pages send a usable Last-Modified header",
    ),
    (
        "fix.last_modified_missing",
        "Send an accurate Last-Modified header, and an accurate lastmod in the sitemap, for freshness.",
    ),
];

/// Czech texts.
const CS: &[(&str, &str)] = &[
    ("label.title", "Připravenost na AI vyhledávání"),
    ("label.summary", "Shrnutí"),
    ("label.generated_by", "Vygenerováno nástrojem"),
    ("label.toggle_theme", "Přepnout světlý/tmavý motiv"),
    ("sec.evidence", "Co tento report umí a neumí říct"),
    ("sec.overview", "Přehled"),
    ("sec.priorities", "Priority"),
    ("sec.policy", "Pravidla pro crawlery"),
    ("sec.access", "Pozorovaná dostupnost"),
    ("sec.controls", "Indexace a úryvky"),
    ("sec.render", "Vykreslování"),
    ("sec.pages", "Vytěžitelnost odpovědí a srozumitelnost entit"),
    ("sec.structured", "Strukturovaná data"),
    ("sec.discovery", "Objevitelnost a aktuálnost"),
    ("sec.manual", "Ruční kontroly"),
    ("sec.kit", "Obsah sady"),
    ("sec.method", "Pokrytí a metoda"),
    ("cat.crawler_policy", "Pravidla pro crawlery"),
    ("cat.observed_access", "Pozorovaná dostupnost"),
    ("cat.indexing_controls", "Indexace a úryvky"),
    ("cat.rendering", "Vykreslování"),
    ("cat.answer_extractability", "Vytěžitelnost odpovědí"),
    ("cat.entity_clarity", "Srozumitelnost entit"),
    ("cat.structured_data", "Strukturovaná data"),
    ("cat.discovery", "Objevitelnost a aktuálnost"),
    ("cat.manual_checks", "Ruční kontroly"),
    ("status.problem", "Problém"),
    ("status.attention", "Pozor"),
    ("status.info", "Info"),
    ("status.ok", "OK"),
    ("status.not_assessed", "Nehodnoceno"),
    ("evidence.strong", "Silné"),
    ("evidence.moderate", "Střední"),
    ("evidence.weak", "Slabé"),
    ("evidence.strong.meaning", "Zdokumentované chování vyhledávačů"),
    ("evidence.moderate.meaning", "Doporučení vyhledávačů, neměřeno"),
    (
        "evidence.weak.meaning",
        "Tvrzení dodavatelů, pozorovací studie nebo žádné doklady",
    ),
    (
        "evidence.note",
        "Report ověřuje, jak dobře mohou AI vyhledávače a odpovědní systémy — Google AI Overviews a AI Mode, Bing Copilot, vyhledávání v ChatGPT, Perplexity a Claude — web najít, přečíst, pochopit a citovat. Nikdy neslibuje citace ani pozice: rozhodují vyhledávače. Každé doporučení uvádí sílu dokladů, kterých vyhledávačů se týká, a datovaný zdroj. Kontroly, které neproběhly, jsou označené „nehodnoceno“, nikdy „OK“.",
    ),
    (
        "evidence.cannot_see",
        "Co crawler nevidí: ověřování botů ve vašem CDN/WAF, nastavení Search generative AI v Search Console, zmínky o webu jinde na internetu a skutečné citace v odpovědích AI.",
    ),
    (
        "evidence.fact.google",
        "Google: AI Overviews a AI Mode nepotřebují žádné zvláštní soubory, textové soubory pro AI ani zvláštní značky schema.org; stránka musí být zaindexovaná a způsobilá pro úryvek a nosnippet, data-nosnippet, max-snippet a noindex platí i pro funkce AI.",
    ),
    (
        "evidence.fact.bing",
        "Bing: ukotvení a citace v Copilotu jsou výsledkem způsobilosti, ne slibem; pomáhají samostatně srozumitelné pasáže, jasné názvy a přesné lastmod; NOARCHIVE a NOCACHE omezují použití v Copilotu.",
    ),
    (
        "evidence.fact.js",
        "JavaScript: studie serverových logů nezjistily, že by GPTBot, OAI-SearchBot, ClaudeBot nebo PerplexityBot spouštěly JavaScript; Google a Bing stránky vykreslují. Stránka téměř bez textu v HTML je riziko vykreslování, ne důkaz.",
    ),
    (
        "evidence.fact.jsonld",
        "JSON-LD a llms.txt: pozorovací srovnání neprokázalo, že by JSON-LD zvyšoval citace v AI, a o llms.txt se žádá zřídka; rozšířené výsledky FAQ Google přestal zobrazovat 7. května 2026.",
    ),
    ("label.evidence_strength", "Síla dokladů"),
    ("label.evidence", "Doklady"),
    ("label.meaning", "Význam"),
    ("label.sources", "Zdroje (datum)"),
    ("reason.no_key_pages", "nebyla procházena žádná klíčová stránka"),
    ("reason.robots_not_read", "pravidla robots.txt nejsou známa"),
    ("reason.ai_unavailable", "AI není k dispozici"),
    ("reason.analysis_failed", "analýza všech stránek selhala"),
    ("reason.no_pages_analyzed", "žádná stránka nebyla analyzována"),
    ("reason.manual_only", "z webu není vidět — zkontrolujte prosím"),
    ("reason.no_pages_checked", "žádnou klíčovou stránku nešlo zkontrolovat"),
    ("unit.origins", "originy"),
    ("unit.key_pages", "klíčové stránky"),
    ("unit.analyzed_pages", "analyzované stránky"),
    ("label.checked", "{2}: zkontrolováno {0} z {1}"),
    ("label.failed_count", "selhalo {0}"),
    ("label.skipped_count", "přeskočeno {0}"),
    ("label.category", "Kategorie"),
    ("label.status", "Stav"),
    ("label.coverage", "Pokrytí"),
    (
        "label.no_priorities",
        "Kontroly, které proběhly, nenašly nic k nápravě.",
    ),
    ("label.scope", "Vyhledávače"),
    ("label.finding", "Zjištění"),
    ("label.fix", "Náprava"),
    ("label.pages", "Stránky"),
    ("label.source", "Zdroj"),
    ("label.examples", "např."),
    ("label.all_engines", "všechny vyhledávače"),
    ("label.no_issues", "Nic nenalezeno."),
    ("label.robots_state", "robots.txt"),
    ("state.ok", "načten (HTTP {0})"),
    ("state.not_found", "neexistuje (HTTP {0}): vše je povoleno"),
    ("state.unavailable", "nepodařilo se načíst ({0})"),
    ("state.skipped", "nenačten (--ignore-robots-txt)"),
    ("state.not_attempted", "nenačten"),
    ("purpose.search", "vyhledávání"),
    ("purpose.user_fetch", "načtení na žádost uživatele"),
    ("purpose.training", "trénování"),
    ("purpose.grounding", "ukotvení odpovědí"),
    ("compliance.honors", "ano"),
    ("compliance.may_not", "nemusí (podle provozovatele)"),
    ("compliance.control", "řídicí token"),
    ("verdict.allowed", "povolen"),
    ("verdict.blocked", "zakázán"),
    ("verdict.partly", "částečně zakázán ({0} z {1})"),
    ("label.agent", "Agent"),
    ("label.vendor", "Provozovatel"),
    ("label.purpose", "Účel"),
    ("label.compliance", "Respektuje robots.txt"),
    ("label.verdict", "Verdikt"),
    ("label.rule", "Rozhodující pravidlo"),
    ("label.note", "Poznámka"),
    (
        "agent.googlebot",
        "Google Search včetně AI Overviews a AI Mode; zakázaná URL může být i tak zaindexována bez úryvku.",
    ),
    (
        "agent.google_extended",
        "Řídicí token pro trénování a ukotvení odpovědí Gemini; na Google Search ani AI Overviews nemá vliv.",
    ),
    (
        "agent.bingbot",
        "Vyhledávání Bing a Copilot; použití v Copilotu řídí NOARCHIVE a NOCACHE.",
    ),
    (
        "agent.oai_searchbot",
        "Vyhledávání v ChatGPT; OpenAI doporučuje povolit i jeho zveřejněné rozsahy IP adres.",
    ),
    (
        "agent.chatgpt_user",
        "Načítá stránky, o které požádá uživatel ChatGPT; robots.txt se nemusí uplatnit.",
    ),
    ("agent.gptbot", "Sbírá trénovací data pro modely OpenAI."),
    ("agent.claude_searchbot", "Vyhledávací index Claude."),
    (
        "agent.claude_user",
        "Načítá stránky, o které požádá uživatel Claude; respektuje robots.txt.",
    ),
    ("agent.claudebot", "Sbírá trénovací data pro modely Anthropic."),
    (
        "agent.perplexitybot",
        "Vyhledávací index Perplexity; nepoužívá se k trénování základních modelů.",
    ),
    (
        "agent.perplexity_user",
        "Načítá stránky, o které požádá uživatel Perplexity; robots.txt obecně ignoruje.",
    ),
    (
        "agent.applebot",
        "Vyhledávání Apple (Siri, Spotlight) a kontext pro AI od Apple; bez vlastní skupiny v robots.txt se řídí pravidly pro Googlebot.",
    ),
    (
        "agent.applebot_extended",
        "Řídicí token: zda Apple smí obsah použít k trénování svých modelů AI.",
    ),
    (
        "agent.meta_externalagent",
        "Sbírá obsah pro modely AI společnosti Meta.",
    ),
    (
        "agent.ccbot",
        "Common Crawl, hojně používaný zdroj trénovacích dat pro AI.",
    ),
    (
        "label.content_signals",
        "Řádky Content-Signal / Content-Usage (velké crawlery je v polovině roku 2026 nerespektují):",
    ),
    ("label.declared_sitemaps", "Sitemapy uvedené v robots.txt:"),
    (
        "access.label",
        "Tak, jak to viděl user agent SiteOne Crawleru — crawlery firem provozujících AI může vaše CDN/WAF obsluhovat jinak (viz Ruční kontroly).",
    ),
    ("label.url", "URL"),
    ("label.detail", "Detail"),
    ("kind.denied", "Odepřeno"),
    ("kind.rate_limited", "Omezeno počtem požadavků"),
    ("kind.server_error", "Chyba serveru"),
    ("kind.transport", "Nenačteno"),
    ("kind.redirect_chain", "Řetěz přesměrování"),
    ("kind.redirect_loop", "Smyčka přesměrování"),
    ("kind.suspected_challenge", "Podezření na ochranu proti botům"),
    ("kind.suspected_soft_404", "Podezření na soft 404"),
    ("kind.slow", "Pomalé"),
    (
        "label.other_urls",
        "Ostatní interní URL: {0}, kontrolovány jen stavovým kódem a přesměrováním",
    ),
    ("label.p90", "90. percentil doby odezvy: {0} s"),
    ("label.none", "žádné"),
    ("label.canonical", "Kanonická URL jinde"),
    ("label.pdf_note", "PDF s omezeními pro roboty (text PDF nebyl zkoumán):"),
    ("label.expired", "noindex (uplynulo unavailable_after)"),
    (
        "label.render_checked",
        "Klíčové stránky zkontrolované v HTML tak, jak přišlo, bez spuštění JavaScriptu: {0}.",
    ),
    ("label.main_chars", "Viditelný text (znaků)"),
    ("label.markers", "Znaky prázdné aplikace"),
    (
        "label.ai_unavailable",
        "AI není k dispozici: {0}. Analýza jednotlivých stránek neproběhla.",
    ),
    ("label.no_pages", "Žádná stránka nebyla analyzována."),
    ("offer.yes", "ano"),
    ("offer.no", "ne"),
    ("offer.not_applicable", "netýká se"),
    ("label.coverage_complete", "všechny bloky"),
    ("label.coverage_reduced", "{0} z {1} bloků (vstup byl zkrácen)"),
    ("label.page_type", "Typ stránky"),
    ("label.main_topic", "Hlavní téma"),
    ("label.offer_early", "Hned uvádí, co nabízí"),
    ("label.inspected", "Prověřeno"),
    ("label.question", "Pravděpodobná otázka návštěvníka"),
    ("label.answer", "Zodpovězeno"),
    ("label.excerpts", "Výňatky ze stránky"),
    ("label.vague", "Pasáže, které stojí na „to“, „my“ nebo „tento produkt“:"),
    ("label.priority", "Priorita"),
    ("label.improvement", "Zlepšení"),
    ("label.blocks", "Bloky"),
    ("priority.high", "vysoká"),
    ("priority.medium", "střední"),
    ("priority.low", "nízká"),
    ("label.byline", "Autor a datum"),
    ("label.byline_missing", "u titulku nenalezeno"),
    (
        "label.lead",
        "Návrh úvodu (před zveřejněním zkontrolujte; viz leads.md)",
    ),
    ("label.failed_pages", "Stránky, jejichž analýza selhala"),
    ("label.existing", "Stávající strukturovaná data na klíčových stránkách"),
    ("label.types", "Typy JSON-LD"),
    ("label.parse_errors", "Chyby zápisu"),
    ("label.invisible", "Hodnoty, které na stránce nejsou vidět"),
    ("label.no_microdata", "Microdata ani RDFa nebyla nalezena."),
    (
        "label.kit_markup",
        "Strukturovaná data v sadě (sestavil je crawler ze stránek)",
    ),
    (
        "label.eligibility",
        "O způsobilosti pro zobrazení ve vyhledávání rozhoduje vyhledávač.",
    ),
    (
        "label.faq_gone",
        "Rozšířené výsledky FAQ Google nezobrazuje (od května 2026).",
    ),
    ("label.no_kit_markup", "Sada neobsahuje žádný JSON-LD."),
    ("label.install_html", "Vložte"),
    ("label.install_page", "do stránky"),
    ("label.install_json", "nebo použijte"),
    ("label.evidence_blocks", "Doklady (text stránky):"),
    ("label.sitemaps", "Sitemapy"),
    ("label.declared", "Uvedena v robots.txt"),
    ("label.state", "Stav"),
    (
        "label.missing_from_sitemaps",
        "Indexovatelné klíčové stránky, které v sitemapách chybí:",
    ),
    ("label.listed_issues", "Uvedené URL, které nejsou indexovatelné:"),
    ("label.hreflang", "Alternativy hreflang, které nikam užitečně nevedou:"),
    ("label.not_checked", "nekontrolováno"),
    (
        "label.last_modified",
        "Last-Modified: posílá ho {0} z {1} klíčových stránek; s použitelným datem: {2}.",
    ),
    ("sitemap.urlset", "{0} URL"),
    ("sitemap.index", "index {0} sitemap"),
    ("sitemap.suspicious_lastmod", "podezřelé lastmod"),
    ("sitemap.malformed", "chybná: {0}"),
    ("sitemap.not_crawled", "neprocházena"),
    ("sitemap.not_absolute", "není absolutní URL"),
    ("sitemap.redirected", "přesměrovává na {0}"),
    ("sitemap.unsupported", "{0} (nečteno)"),
    ("manual.label", "Z webu není vidět — zkontrolujte prosím:"),
    (
        "manual.search_generative_ai",
        "Google Search Console → Nastavení → Search generative AI: Include / Exclude / Inherit (zkontrolujte i subdomény a dědění mezi službami).",
    ),
    ("manual.gsc_report", "Přehled výkonu generativní AI v Search Console."),
    ("manual.bing_report", "Přehled AI Performance v Bing Webmaster Tools."),
    (
        "manual.cdn_waf",
        "Nastavení CDN/WAF typu „blokovat AI boty“ a ověřování botů (Cloudflare aj.); při použití WAF povolte zveřejněné rozsahy IP adres OpenAI.",
    ),
    ("label.kit_empty", "Žádné soubory sady nebyly zapsány."),
    ("label.kit_dir", "Sada k nasazení — začněte souborem"),
    (
        "label.possible_profiles",
        "Možné profily na sociálních sítích — pokud jsou vaše, přidejte je k organizaci ručně:",
    ),
    ("label.pages_crawled", "Procházené stránky"),
    ("label.key_pages", "Klíčové stránky"),
    ("label.analyzed", "Stránky vybrané k analýze"),
    ("label.analyzed_ok", "analyzováno"),
    ("label.analyzed_failed", "selhalo"),
    ("label.analyzed_reduced", "se zkráceným vstupem"),
    ("label.model", "Model"),
    ("label.context", "kontext"),
    ("label.language", "jazyk"),
    ("label.calls", "Volání LLM: {0} · vstup {1} / výstup {2} tokenů"),
    ("label.research_date", "Datum rešerše dokladů"),
    ("label.attempted", "Pokusy"),
    ("label.succeeded", "Úspěšné"),
    ("label.failed", "Selhané"),
    ("label.skipped", "Přeskočené"),
    (
        "summary.template",
        "{0} problém(y) a {1} bod(y) k prověření na {2} klíčových stránkách.",
    ),
    ("summary.lever", "Největší páka: {0}."),
    (
        "summary.nothing_found",
        "Kontroly, které proběhly, nenašly žádné problémy ani body k prověření.",
    ),
    (
        "summary.policy_not_assessed",
        "Pravidla pro crawlery nebyla hodnocena, takže report neumí říct, zda crawlery AI smějí web načítat.",
    ),
    (
        "summary.access_not_assessed",
        "Pozorovaná dostupnost nebyla hodnocena, takže report neumí říct, zda jdou klíčové stránky načíst.",
    ),
    (
        "issue.search_crawler_blocked",
        "{0} má v robots.txt zakázaný přístup na {1} (klíčové stránky: {n}): {2}",
    ),
    (
        "fix.search_crawler_blocked",
        "Chcete-li se objevovat v odpovědích vyhledávání {0}, odstraňte nebo zužte pravidlo, které ho blokuje.",
    ),
    (
        "issue.search_crawler_partly_blocked",
        "{0} má zakázaný přístup na část klíčových stránek {1} ({n}): {2}",
    ),
    (
        "fix.search_crawler_partly_blocked",
        "Ověřte, že tyto stránky mají v odpovědích {0} opravdu chybět; jinak pravidlo zužte.",
    ),
    (
        "issue.user_fetch_blocked",
        "{0} (načítá stránky na žádost uživatele) má zakázaný přístup na {1}: {2}",
    ),
    (
        "fix.user_fetch_blocked",
        "Povolte {0}, pokud mají asistenti umět otevřít vaše stránky, když o to uživatel požádá.",
    ),
    (
        "issue.user_fetch_blocked_limited",
        "{0} má zakázaný přístup na {1}, ale podle provozovatele se na něj robots.txt nemusí vztahovat (omezený účinek): {2}",
    ),
    (
        "fix.user_fetch_blocked_limited",
        "Kvůli viditelnosti není nutná žádná změna; chcete-li tato načtení opravdu zastavit, použijte CDN/WAF.",
    ),
    (
        "issue.training_crawler_blocked",
        "Crawler pro trénování AI {0} má zakázaný přístup na {1}: {2}",
    ),
    (
        "fix.training_crawler_blocked",
        "Je-li to záměr, nic nedělejte: blokování trénovacích crawlerů neovlivňuje viditelnost v AI vyhledávání.",
    ),
    (
        "issue.grounding_control_blocked",
        "{0} je zakázán na {1}: Gemini nesmí vaše stránky použít k trénování ani k ukotvení odpovědí ({2})",
    ),
    (
        "fix.grounding_control_blocked",
        "Je-li to záměr, ponechte; na Google Search ani AI Overviews to vliv nemá, ale odpovědi Gemini ukotvené ve vyhledávání vaše stránky nepoužijí.",
    ),
    (
        "issue.content_signal_present",
        "robots.txt originu {0} uvádí Content-Signal / Content-Usage, které velké crawlery (v polovině roku 2026) nerespektují: {1}",
    ),
    (
        "fix.content_signal_present",
        "Nespoléhejte na tyto řádky; použijte skupiny user-agent a nastavení samotných vyhledávačů.",
    ),
    (
        "issue.robots_unavailable",
        "robots.txt originu {0} se nepodařilo načíst ({1}); vyhledávače mohou nefunkční robots.txt brát jako „zakázat vše“",
    ),
    (
        "fix.robots_unavailable",
        "Zajistěte, aby /robots.txt spolehlivě odpovídal 200 (nebo 404, pokud ho nemáte), i crawlerům AI.",
    ),
    (
        "issue.robots_skipped",
        "robots.txt originu {0} nebyl načten (--ignore-robots-txt)",
    ),
    (
        "fix.robots_skipped",
        "Pro vyhodnocení pravidel spusťte procházení bez --ignore-robots-txt.",
    ),
    ("issue.robots_not_attempted", "robots.txt originu {0} nebyl načten"),
    (
        "fix.robots_not_attempted",
        "Pro vyhodnocení pravidel projděte i robots.txt tohoto originu.",
    ),
    (
        "issue.robots_not_utf8",
        "robots.txt originu {0} není platné UTF-8; pravidla s neplatnými bajty nelze přesně vyhodnotit",
    ),
    ("fix.robots_not_utf8", "Uložte robots.txt v kódování UTF-8 (RFC 9309)."),
    (
        "issue.access_denied",
        "Klíčové stránky, které crawleru odpověděly 401 nebo 403: {n}",
    ),
    (
        "fix.access_denied",
        "Veřejné stránky musí vyhledávačům a crawlerům AI odpovídat 200; zkontrolujte pravidla serveru a CDN/WAF.",
    ),
    (
        "issue.access_rate_limited",
        "Klíčové stránky s odpovědí 429 (příliš mnoho požadavků): {n}",
    ),
    (
        "fix.access_rate_limited",
        "Zvyšte limity pro ověřené crawlery vyhledávačů a AI.",
    ),
    (
        "issue.access_server_error",
        "Klíčové stránky s chybou serveru (5xx): {n}",
    ),
    (
        "fix.access_server_error",
        "Opravte chyby serveru; stránky, které opakovaně selhávají, vyhledávače vyřazují.",
    ),
    (
        "issue.access_failed",
        "Klíčové stránky, které nešlo načíst (chyba spojení nebo vypršení času): {n}",
    ),
    ("fix.access_failed", "Zkontrolujte dostupnost a dobu odezvy těchto URL."),
    (
        "issue.access_redirect_chain",
        "Klíčové stránky přesměrované přes více než 2 kroky: {n}",
    ),
    (
        "fix.access_redirect_chain",
        "Odkazujte a přesměrovávejte rovnou na cílovou URL.",
    ),
    (
        "issue.access_redirect_loop",
        "Klíčové stránky přesměrované ve smyčce: {n}",
    ),
    (
        "fix.access_redirect_loop",
        "Přerušte smyčku: každé přesměrování musí skončit na stránce, která odpovídá 200.",
    ),
    (
        "issue.access_suspected_challenge",
        "Klíčové stránky, které vypadají jako ochrana proti botům nebo blokační stránka (podezření): {n}",
    ),
    (
        "fix.access_suspected_challenge",
        "Zkontrolujte ochranu proti botům: vyhledávače a crawlery AI musí dostat skutečnou stránku.",
    ),
    (
        "issue.access_suspected_soft_404",
        "Klíčové stránky, které odpovídají 200, ale píšou „nenalezeno“ (podezření na soft 404): {n}",
    ),
    (
        "fix.access_suspected_soft_404",
        "U chybějících stránek odpovídejte 404, nebo obsah obnovte.",
    ),
    (
        "issue.access_slow",
        "Klíčové stránky pomalejší než 90. percentil procházení i 3 s (pozorované riziko načtení): {n}",
    ),
    (
        "fix.access_slow",
        "Zrychlete tyto stránky; pomalé odpovědi se načítají méně spolehlivě.",
    ),
    (
        "issue.other_urls_issues",
        "Potíže na ostatních interních URL (celkem URL: {1}), spočítané, ale nehodnocené: {0}",
    ),
    (
        "fix.other_urls_issues",
        "Počty najdete v části o pozorované dostupnosti.",
    ),
    (
        "issue.google_noindex",
        "Klíčové stránky s noindex pro Google (nebo s uplynulým unavailable_after): {n}",
    ),
    (
        "fix.google_noindex",
        "Odstraňte noindex ze stránek, které se mají objevovat v Google Search a AI Overviews.",
    ),
    (
        "issue.google_nosnippet",
        "Klíčové stránky, které Googlu zakazují úryvky (nosnippet): {n}",
    ),
    (
        "fix.google_nosnippet",
        "Odstraňte nosnippet: takové stránky Google nemůže citovat v AI Overviews ani AI Mode.",
    ),
    (
        "issue.google_max_snippet_zero",
        "Klíčové stránky s max-snippet:0 pro Google: {n}",
    ),
    (
        "fix.google_max_snippet_zero",
        "Zvyšte nebo odstraňte max-snippet, aby Google mohl stránku citovat.",
    ),
    (
        "issue.bing_noindex",
        "Klíčové stránky s noindex pro Bing (nebo s uplynulým unavailable_after): {n}",
    ),
    (
        "fix.bing_noindex",
        "Odstraňte noindex ze stránek, které se mají objevovat v Bingu a Copilotu.",
    ),
    (
        "issue.bing_nosnippet",
        "Klíčové stránky, které Bingu zakazují úryvky (nosnippet): {n}",
    ),
    (
        "fix.bing_nosnippet",
        "Odstraňte nosnippet: takové stránky Bing ani Copilot nemohou citovat.",
    ),
    (
        "issue.bing_max_snippet_zero",
        "Klíčové stránky s max-snippet:0 pro Bing: {n}",
    ),
    (
        "fix.bing_max_snippet_zero",
        "Zvyšte nebo odstraňte max-snippet, aby Bing mohl stránku citovat.",
    ),
    (
        "issue.bing_copilot_restricted",
        "Klíčové stránky, které pomocí NOARCHIVE nebo NOCACHE pro Bing omezují použití v Copilotu: {n}",
    ),
    (
        "fix.bing_copilot_restricted",
        "Odstraňte noarchive / nocache pro Bing, pokud mají odpovědi Copilotu využít celou stránku.",
    ),
    (
        "issue.short_max_snippet",
        "Klíčové stránky, které omezují úryvky na 50 znaků či méně (produktová heuristika): {n}",
    ),
    (
        "fix.short_max_snippet",
        "Zvažte delší max-snippet, aby odpovědi mohly citovat celou větu.",
    ),
    (
        "issue.data_nosnippet_large",
        "Klíčové stránky, kde data-nosnippet skrývá 30 % a více textu: {n}",
    ),
    (
        "fix.data_nosnippet_large",
        "Používejte data-nosnippet jen pro malé části: skrytý text nelze citovat.",
    ),
    (
        "issue.canonical_elsewhere",
        "Klíčové stránky, které jako kanonickou uvádějí jinou URL: {n} ({*})",
    ),
    (
        "fix.canonical_elsewhere",
        "Ověřte, že je kanonická URL záměrná; vyhledávače indexují a citují kanonickou stránku.",
    ),
    (
        "issue.pdf_restricted",
        "Soubory PDF, které hlavičkou X-Robots-Tag omezují indexaci nebo úryvky (text PDF nebyl zkoumán): {n}",
    ),
    ("fix.pdf_restricted", "Ověřte, že jsou tato omezení záměrná."),
    (
        "issue.likely_client_rendered",
        "Klíčové stránky vykreslované zřejmě až v prohlížeči, v HTML téměř bez textu (riziko vykreslování): {n}",
    ),
    (
        "fix.likely_client_rendered",
        "Vykreslujte hlavní obsah na serveru: crawlery AI, které nespouštějí JavaScript, vidí prázdnou stránku.",
    ),
    (
        "issue.offer_not_early",
        "Analyzované stránky, které hned neuvádějí, co nabízejí nebo na co odpovídají: {n}",
    ),
    (
        "fix.offer_not_early",
        "Začněte stránku větou, která řekne, co nabízí nebo na co odpovídá.",
    ),
    (
        "issue.unanswered_questions",
        "Pravděpodobné otázky návštěvníků, které se na analyzovaných stránkách nenašly: {0} (stránky: {n})",
    ),
    (
        "fix.unanswered_questions",
        "Na nejdůležitější otázky odpovězte výslovně ve viditelném textu (viz leads.md v sadě).",
    ),
    (
        "issue.hidden_content",
        "Analyzované stránky s polovinou a více textu ve sbalených záložkách nebo akordeonech: {n}",
    ),
    (
        "fix.hidden_content",
        "Klíčové odpovědi ukažte ve viditelném textu, ne jen ve sbalených prvcích.",
    ),
    (
        "issue.no_contextual_links",
        "Analyzované stránky bez odkazů na jiné stránky webu v obsahu: {n}",
    ),
    (
        "fix.no_contextual_links",
        "Odkazujte související stránky z textu, aby je crawlery i čtenáři našli.",
    ),
    (
        "issue.images_without_alt",
        "Obrázky bez atributu alt na analyzovaných stránkách: {0} (stránky: {n})",
    ),
    ("fix.images_without_alt", "Popište informativní obrázky v atributu alt."),
    (
        "issue.vague_references",
        "Analyzované stránky s pasážemi, které stojí na „to“, „my“ nebo „tento produkt“: {n}",
    ),
    (
        "fix.vague_references",
        "V každé pasáži pojmenujte, o čem mluví, aby dávala smysl i citovaná samostatně.",
    ),
    (
        "issue.article_without_byline",
        "Články bez ověřeného autora a data u titulku: {n}",
    ),
    (
        "fix.article_without_byline",
        "Uveďte vedle titulku autora a datum vydání nebo aktualizace.",
    ),
    (
        "issue.jsonld_parse_error",
        "Bloky JSON-LD, které nejsou platný JSON: {0} (klíčové stránky: {n})",
    ),
    (
        "fix.jsonld_parse_error",
        "Opravte zápis: neplatná strukturovaná data vyhledávače ignorují.",
    ),
    (
        "issue.jsonld_values_not_visible",
        "Strukturovaná data s hodnotami, které na stránce nejsou (klíčové stránky: {n}): {*}",
    ),
    (
        "fix.jsonld_values_not_visible",
        "Udržujte strukturovaná data shodná s viditelným textem, nebo hodnoty na stránce ukažte.",
    ),
    (
        "issue.jsonld_values_not_checked",
        "Nezkontrolované hodnoty strukturovaných dat (příliš mnoho na jedné stránce): {0} (klíčové stránky: {n})",
    ),
    (
        "fix.jsonld_values_not_checked",
        "Není co dělat; kontrola je jen částečná.",
    ),
    (
        "issue.faq_markup_present",
        "Značky FAQPage na klíčových stránkách ({n}): rozšířené výsledky FAQ Google od května 2026 nezobrazuje",
    ),
    (
        "fix.faq_markup_present",
        "Ponechte je jen pro skutečné, viditelné otázky a odpovědi.",
    ),
    (
        "issue.no_site_markup",
        "Úvodní stránka nemá značky {*} (sada je obsahuje)",
    ),
    (
        "fix.no_site_markup",
        "Volitelně nasaďte celowebový JSON-LD ze sady (evidence: weak).",
    ),
    (
        "issue.no_sitemap",
        "Nebyla nalezena žádná sitemapa (ani v robots.txt, ani při procházení)",
    ),
    (
        "fix.no_sitemap",
        "Zveřejněte XML sitemapu s přesným lastmod a uveďte ji v robots.txt.",
    ),
    (
        "issue.sitemap_unreadable",
        "Sitemapy, které nešlo přečíst (chybné, nefunkční nebo uvedené bez absolutní URL): {n}",
    ),
    (
        "fix.sitemap_unreadable",
        "Opravte tyto sitemapy a uvádějte je absolutními URL.",
    ),
    (
        "issue.sitemap_not_checked",
        "Sitemapy, které nebyly procházeny nebo jsou to feedy či textové soubory, a proto nebyly zkontrolovány: {n}",
    ),
    (
        "fix.sitemap_not_checked",
        "Není co dělat, případně je zahrňte do procházení, aby se zkontrolovaly.",
    ),
    (
        "issue.sitemap_comparison_not_assessed",
        "Klíčové stránky nebyly porovnány se sitemapami: ne každou sitemapu šlo přečíst",
    ),
    (
        "fix.sitemap_comparison_not_assessed",
        "Zajistěte čitelnost všech sitemap, aby šlo pokrytí posoudit.",
    ),
    (
        "issue.missing_from_sitemap",
        "Indexovatelné klíčové stránky, které v sitemapách chybí: {n}",
    ),
    (
        "fix.missing_from_sitemap",
        "Uveďte v sitemapě každou indexovatelnou stránku.",
    ),
    (
        "issue.sitemap_lists_non_indexable",
        "URL v sitemapách, které nejsou indexovatelné (přesměrování, chyby, noindex nebo jiná kanonická URL): {n}",
    ),
    (
        "fix.sitemap_lists_non_indexable",
        "Uvádějte jen cílové, indexovatelné a kanonické URL.",
    ),
    (
        "issue.suspicious_lastmod",
        "Sitemapy s nevěrohodnými lastmod (všechna stejná nebo v budoucnosti): {n}",
    ),
    (
        "fix.suspicious_lastmod",
        "Nastavte lastmod na skutečné datum poslední změny obsahu; Bing ho používá k posouzení aktuálnosti.",
    ),
    (
        "issue.hreflang_target_broken",
        "Alternativy hreflang vedou na URL, které neodpovídají 200 (klíčové stránky: {n}): {*}",
    ),
    (
        "fix.hreflang_target_broken",
        "Nasměrujte každou alternativu hreflang na živou kanonickou stránku.",
    ),
    (
        "issue.hreflang_target_not_crawled",
        "Alternativy hreflang vedou na URL, které nebyly procházeny (klíčové stránky: {n}): {*}",
    ),
    (
        "fix.hreflang_target_not_crawled",
        "Není co dělat, případně je zahrňte do procházení, aby se zkontrolovaly.",
    ),
    (
        "issue.last_modified_missing",
        "Klíčové stránky s použitelnou hlavičkou Last-Modified: jen {0} z {1}",
    ),
    (
        "fix.last_modified_missing",
        "Kvůli aktuálnosti posílejte přesnou hlavičku Last-Modified a přesné lastmod v sitemapě.",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::blocks::blocks_from_html;
    use crate::ai::geo::access::{AccessIssue, AccessKind};
    use crate::ai::geo::agents::AI_AGENTS;
    use crate::ai::geo::analyze::{analyzed_page, build_page_request, parse_analysis, verify_analysis};
    use crate::ai::geo::findings::{
        AnalysisRun, CategoryId, CheckStatus, Checks, PageControls, PageMarkup, TITLE_KEYS, origin_policy,
    };
    use crate::ai::geo::jsonld::ExistingMarkup;
    use crate::ai::geo::keys::KeyPage;
    use crate::ai::geo::kit::KitEntry;
    use crate::ai::geo::render::RenderCheck;
    use crate::result::status::RobotsFetchState;
    use serde_json::json;

    const HOME: &str = "https://example.com/";
    const HOSTILE: &str = "<img src=x onerror=alert(1)> | **bold**";

    fn key(url: &str) -> KeyPage {
        KeyPage {
            uq_id: url.to_string(),
            url: url.to_string(),
            status_code: 200,
            score: Some(1.0),
            is_homepage: url == HOME,
        }
    }

    fn analysis(url: &str, html: &str, extra: &str) -> PageAnalysis {
        let page = analyzed_page(url, html);
        let blocks = blocks_from_html(html);
        let (_, coverage) = build_page_request(&page, &blocks, "", false, "cs", 1_000_000, 4_000, 0.0);
        let json = format!(
            r#"{{"page_type":"service","main_topic":"Hypotéky {HOSTILE}","states_offer_early":false,
                 "questions":[{{"question":"Kolik to stojí?","answered":"no","blocks":[]}}]{extra}}}"#
        );
        verify_analysis(parse_analysis(&json).unwrap(), &page, &blocks, &coverage)
    }

    fn checks(ai: bool) -> Checks {
        let pages = vec![key(HOME), key("https://example.com/sluzby")];
        let html = format!(
            "<html lang=\"cs\"><head><title>Služby {}</title></head><body><main><h1>Služby</h1>\
             <p>Nabízíme hypotéky.</p></main></body></html>",
            HOSTILE.replace('<', "&lt;").replace('>', "&gt;")
        );
        Checks {
            policy: vec![origin_policy(
                "https://example.com",
                &RobotsFetchState::Ok {
                    status: 200,
                    content: "User-agent: OAI-SearchBot\nDisallow: /\n".to_string(),
                    valid_utf8: true,
                },
                &["/".to_string(), "/sluzby".to_string()],
            )],
            access: vec![AccessIssue {
                url: "https://example.com/sluzby".to_string(),
                kind: AccessKind::Denied(403),
                detail: format!("HTTP 403 {HOSTILE}"),
            }],
            controls: pages
                .iter()
                .map(|page| PageControls {
                    url: page.url.clone(),
                    ..PageControls::default()
                })
                .collect(),
            render: RenderCheck {
                checked: 2,
                risks: Vec::new(),
            },
            markup: pages
                .iter()
                .map(|page| PageMarkup {
                    url: page.url.clone(),
                    is_homepage: page.is_homepage,
                    markup: ExistingMarkup::default(),
                })
                .collect(),
            analysis: if ai {
                AnalysisRun {
                    selected: 2,
                    pages: vec![analysis("https://example.com/sluzby", &html, "")],
                    failed: vec![(HOME.to_string(), format!("timeout {HOSTILE}"))],
                    ..AnalysisRun::default()
                }
            } else {
                AnalysisRun {
                    selected: 2,
                    unavailable: Some("no model configured".to_string()),
                    ..AnalysisRun::default()
                }
            },
            key_pages: pages,
            ..Checks::default()
        }
    }

    fn kit() -> Vec<KitEntry> {
        let json = json!({
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "@id": "https://example.com/faq#faq",
            "mainEntity": [{"@type": "Question", "name": "Proč?",
                "acceptedAnswer": {"@type": "Answer", "text": "Protože </script><script>alert(1)</script>"}}],
        });
        vec![KitEntry {
            page_url: "https://example.com/faq".to_string(),
            kind: "FAQPage".to_string(),
            file_stem: "page-faq-faq".to_string(),
            json,
            evidence: vec![format!("B2: Proč? {HOSTILE}")],
            notes: vec!["FAQ rich results no longer appear in Google (since 7 May 2026).".to_string()],
        }]
    }

    fn meta(language: &str) -> GeoMeta {
        GeoMeta {
            host: "example.com".to_string(),
            url: HOME.to_string(),
            site_name: format!("Example {HOSTILE}"),
            crawled_at: "2026-09-26 10:00".to_string(),
            report_language: language.to_string(),
            provider: "openai-compatible".to_string(),
            model: "test-model".to_string(),
            context_window: 128_000,
            crawled_pages: 12,
            ..GeoMeta::default()
        }
    }

    fn sample(language: &str, ai: bool) -> GeoDoc {
        let mut doc = GeoDoc::new(
            meta(language),
            checks(ai),
            kit(),
            vec!["https://www.linkedin.com/in/founder".to_string()],
        );
        doc.kit_files = vec!["README.md".to_string(), "jsonld/page-faq-faq.html".to_string()];
        doc
    }

    const KIT_DIR: &str = "ai-geo-kit.example.com.20260926-1000";

    #[test]
    fn the_evidence_box_comes_first_and_the_manual_checks_are_listed() {
        for language in ["en", "cs"] {
            let doc = sample(language, true);
            let locale = ReportLocale::new(language);
            let evidence_title = text(&locale, "sec.evidence");
            let md = doc.to_markdown(KIT_DIR);
            let first_section = md.lines().find(|line| line.starts_with("## ")).unwrap();
            assert_eq!(first_section, format!("## {evidence_title}"));
            let html = doc.to_html(KIT_DIR);
            let first_id = html.find("<section class=\"sec\" id=\"").unwrap();
            assert!(html[first_id..].starts_with("<section class=\"sec\" id=\"s-evidence\""));
            for out in [&md, &html] {
                for check in [
                    "manual.search_generative_ai",
                    "manual.gsc_report",
                    "manual.bing_report",
                    "manual.cdn_waf",
                ] {
                    let wording = text(&locale, check);
                    assert!(!wording.is_empty(), "{check}");
                    assert!(
                        out.contains(&escape(wording)) || out.contains(wording),
                        "{language} {check}"
                    );
                }
                assert!(out.contains("Search generative AI"));
                assert!(out.contains("https://developers.google.com/search/docs/appearance/ai-features"));
                assert!(out.contains("2026-05"));
            }
            let cannot = text(&locale, "evidence.cannot_see");
            assert!(md.contains(cannot), "{language}");
        }
    }

    #[test]
    fn categories_whose_checks_did_not_run_read_not_assessed() {
        let doc = sample("en", false);
        let state = |id: CategoryId| doc.categories.iter().find(|(known, _)| *known == id).unwrap().1.clone();
        assert_eq!(state(CategoryId::AnswerExtractability).status, CheckStatus::NotAssessed);
        let md = doc.to_markdown(KIT_DIR);
        let html = doc.to_html(KIT_DIR);
        for out in [&md, &html] {
            assert!(out.contains("Not assessed"));
            assert!(out.contains(text(&ReportLocale::new("en"), "reason.ai_unavailable")));
            assert!(out.contains("no model configured"));
        }
        let cs = sample("cs", false).to_markdown(KIT_DIR);
        assert!(cs.contains("Nehodnoceno"));
    }

    #[test]
    fn json_ld_install_snippets_are_shown_escaped() {
        let doc = sample("en", true);
        let html = doc.to_html(KIT_DIR);
        assert!(html.contains("&lt;script type=&quot;application/ld+json&quot;&gt;"));
        assert!(!html.contains("<script type=\"application/ld+json\">"));
        assert!(!html.contains("<script>alert(1)"));
        assert!(html.contains("\\u003c/script\\u003e"));
        assert!(html.contains(&format!("href=\"{KIT_DIR}/jsonld/page-faq-faq.html\"")));
        let md = doc.to_markdown(KIT_DIR);
        assert!(md.contains("```html\n<script type=\"application/ld+json\">"));
        assert!(md.contains(&format!("]({KIT_DIR}/jsonld/page-faq-faq.html)")));
    }

    #[test]
    fn hostile_values_are_escaped() {
        let mut doc = sample("en", true);
        doc.pages[0].url = "javascript:alert(2)".to_string();
        let html = doc.to_html(KIT_DIR);
        assert!(!html.contains("<img src=x"), "raw markup from the site");
        assert!(html.contains("&lt;img src=x onerror=alert(1)&gt;"));
        assert!(!html.to_ascii_lowercase().contains("href=\"javascript:"));
        assert!(html.starts_with("<!DOCTYPE html>"));
        assert_eq!(
            html.matches("<script>").count(),
            1,
            "only the report's own theme script"
        );
        let md = doc.to_markdown(KIT_DIR);
        assert!(!md.contains("](javascript:"));
        assert!(
            md.contains("\\| \\*\\*bold\\*\\*") || md.contains("\\| **bold**"),
            "a pipe in a table cell is escaped"
        );
    }

    #[test]
    fn the_json_has_the_schema_and_camel_case_keys() {
        let doc = sample("en", true);
        let value = doc.to_json();
        assert_eq!(value["schema"], "siteone-crawler/ai-geo/2");
        for key in [
            "meta",
            "evidenceNote",
            "summary",
            "categories",
            "priorities",
            "issues",
            "policy",
            "access",
            "controls",
            "render",
            "pages",
            "structured",
            "discovery",
            "manualChecks",
            "kit",
            "kitFiles",
            "possibleProfiles",
            "consistency",
            "failedPages",
            "sources",
        ] {
            assert!(value.get(key).is_some(), "{key}");
        }
        assert_eq!(value["categories"][0]["id"], "crawlerPolicy");
        assert_eq!(value["categories"][0]["status"], "problem");
        assert_eq!(value["priorities"][0]["titleKey"], "search_crawler_blocked");
        assert_eq!(
            value["priorities"][0]["source"]["url"],
            "https://developers.openai.com/api/docs/bots"
        );
        assert_eq!(value["failedPages"][0]["url"], HOME);
        assert_eq!(value["policy"][0]["agents"][3]["token"], "OAI-SearchBot");
        assert_eq!(value["policy"][0]["agents"][3]["access"], "blocked");
        fn keys(value: &serde_json::Value, path: &str) {
            match value {
                serde_json::Value::Object(map) => {
                    for (key, child) in map {
                        assert!(key.starts_with('@') || !key.contains('_'), "{path}.{key}");
                        keys(child, &format!("{path}.{key}"));
                    }
                }
                serde_json::Value::Array(items) => items.iter().for_each(|item| keys(item, path)),
                _ => {}
            }
        }
        keys(&value, "");
    }

    #[test]
    fn every_fixed_text_exists_in_english_and_czech_and_none_promises_a_guarantee() {
        let mut all_keys: Vec<String> = Vec::new();
        for title_key in TITLE_KEYS {
            all_keys.push(format!("issue.{title_key}"));
            all_keys.push(format!("fix.{title_key}"));
        }
        all_keys.extend(AI_AGENTS.iter().map(|agent| agent.note_key.to_string()));
        all_keys.extend(CategoryId::ALL.iter().map(|id| format!("cat.{}", id.key())));
        for reason in [
            "no_key_pages",
            "robots_not_read",
            "ai_unavailable",
            "analysis_failed",
            "no_pages_analyzed",
            "manual_only",
            "no_pages_checked",
        ] {
            all_keys.push(format!("reason.{reason}"));
        }
        for language in ["en", "cs"] {
            let locale = ReportLocale::new(language);
            for key in &all_keys {
                assert!(!text(&locale, key).trim().is_empty(), "{language}: {key}");
            }
            for doc in [sample(language, true), sample(language, false), GeoDoc::default()] {
                let mut doc = doc;
                doc.meta.report_language = language.to_string();
                for out in [
                    doc.to_markdown(KIT_DIR),
                    doc.to_html(KIT_DIR),
                    doc.to_json().to_string(),
                ] {
                    let lower = out.to_lowercase();
                    assert!(!lower.contains("guarantee") && !lower.contains("garant"), "{language}");
                }
            }
        }
        let mut en_keys: Vec<&str> = EN.iter().map(|(key, _)| *key).collect();
        let mut cs_keys: Vec<&str> = CS.iter().map(|(key, _)| *key).collect();
        en_keys.sort_unstable();
        cs_keys.sort_unstable();
        assert_eq!(en_keys.len(), EN.len(), "no key twice in English");
        assert_eq!(en_keys, cs_keys, "the same keys in English and Czech");
        assert_ne!(
            text(&ReportLocale::new("cs"), "issue.access_denied"),
            text(&ReportLocale::new("en"), "issue.access_denied")
        );
    }

    #[test]
    fn the_empty_state_renders() {
        let doc = GeoDoc::default();
        assert_eq!(doc.schema, SCHEMA);
        let md = doc.to_markdown(KIT_DIR);
        let html = doc.to_html(KIT_DIR);
        for out in [&md, &html] {
            assert!(out.contains("Not assessed"));
            assert!(out.contains(text(&ReportLocale::new("en"), "sec.manual")));
        }
        assert!(html.trim_end().ends_with("</html>"));
        assert_eq!(doc.to_json()["pages"], json!([]));
    }

    #[test]
    fn the_summary_counts_problems_and_names_the_biggest_lever() {
        let doc = sample("en", true);
        assert!(
            doc.summary
                .starts_with("2 problem(s) and 1 point(s) to review on 2 key pages. The biggest lever: "),
            "{}",
            doc.summary
        );
        assert!(doc.summary.contains("OAI-SearchBot"), "{}", doc.summary);
        let cs = sample("cs", true);
        assert!(
            cs.summary
                .starts_with("2 problém(y) a 1 bod(y) k prověření na 2 klíčových stránkách."),
            "{}",
            cs.summary
        );

        let mut unknown = checks(true);
        unknown.policy = vec![origin_policy(
            "https://example.com",
            &RobotsFetchState::Skipped,
            &["/".to_string()],
        )];
        unknown.key_pages.clear();
        let doc = GeoDoc::new(meta("en"), unknown, Vec::new(), Vec::new());
        assert!(
            doc.summary
                .contains(text(&ReportLocale::new("en"), "summary.policy_not_assessed")),
            "{}",
            doc.summary
        );
        assert!(
            doc.summary
                .contains(text(&ReportLocale::new("en"), "summary.access_not_assessed")),
            "{}",
            doc.summary
        );
        let nothing = summary(&ReportLocale::new("en"), &[], &[]);
        assert!(nothing.contains(text(&ReportLocale::new("en"), "summary.nothing_found")));
    }

    #[test]
    fn the_kit_is_built_from_the_document() {
        let doc = sample("en", true);
        let files = doc.build_kit("2026-09-26");
        let names: Vec<&str> = files.iter().map(|file| file.relative_path.as_str()).collect();
        assert_eq!(names[0], "README.md");
        assert!(names.contains(&"robots/block-ai-training.snippet.txt"));
        assert!(names.contains(&"robots/robots.proposed.txt"), "{names:?}");
        assert!(names.contains(&"jsonld/page-faq-faq.html"));
        assert!(names.contains(&"llms.txt"));
        let snippet = String::from_utf8(
            files
                .iter()
                .find(|file| file.relative_path == "robots/block-ai-training.snippet.txt")
                .unwrap()
                .bytes
                .clone(),
        )
        .unwrap();
        assert!(snippet.contains("# OAI-SearchBot (OpenAI): blocked"));
    }
}
