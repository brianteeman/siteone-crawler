// SiteOne Crawler - verdicts of the AI search readiness report
// (c) Jan Reges <jan.reges@siteone.cz>
//
// Turns what the checks found into issues and category states, deterministically. Every issue
// carries a status, the strength of the evidence behind it, the engines it concerns and a dated
// source. The deterministic categories are judged on the key pages only, so their verdicts do not
// change with `--ai-max-pages`. A category whose checks were not run, or all failed, is "not
// assessed" — never "OK".

use std::collections::HashMap;

use serde::Serialize;

use crate::ai::geo::access::{AccessIssue, AccessKind, AccessStats};
use crate::ai::geo::agents::{AI_AGENTS, AiAgent, Compliance, Purpose};
use crate::ai::geo::analyze::{Answered, OfferEarly, PageAnalysis, PageType};
use crate::ai::geo::controls::EnginePolicy;
use crate::ai::geo::discovery::{Discovery, HreflangProblem, SitemapKind, SitemapState};
use crate::ai::geo::jsonld::ExistingMarkup;
use crate::ai::geo::keys::KeyPage;
use crate::ai::geo::render::{RenderCheck, RenderRisk};
use crate::ai::geo::robots_ai::{AgentAccess, evaluate, robots_of, state_is_known};
use crate::ai::geo::signals::{PageSignals, PdfRestriction};
use crate::result::status::RobotsFetchState;

/// At most this many priorities.
pub const MAX_PRIORITIES: usize = 10;
/// `data-nosnippet` covering at least this share of a page's own text is reported.
pub const NOSNIPPET_SHARE_ATTENTION: f64 = 0.3;
/// A `max-snippet` of 1 to this many characters is reported (a product heuristic, not a documented
/// threshold).
pub const SHORT_SNIPPET_CHARS: i64 = 50;
/// At least this share of an analyzed page's own text in collapsed blocks is reported…
pub const HIDDEN_SHARE_ATTENTION: f64 = 0.5;
/// …on a page with at least this many characters of its own text.
const HIDDEN_MIN_CHARS: usize = 200;
/// Page types that should state their offer or answer early.
const OFFER_EARLY_TYPES: &[PageType] = &[
    PageType::Product,
    PageType::Service,
    PageType::Pricing,
    PageType::Faq,
    PageType::Article,
];

/// Organization and its common schema.org subtypes (the direct ones, LocalBusiness and its direct
/// subtypes, and frequent deeper ones): a page that declares one of them already has
/// Organization markup.
pub const ORGANIZATION_TYPES: &[&str] = &[
    "Organization",
    "Airline",
    "Consortium",
    "Cooperative",
    "Corporation",
    "EducationalOrganization",
    "CollegeOrUniversity",
    "School",
    "FundingScheme",
    "GovernmentOrganization",
    "LibrarySystem",
    "MedicalOrganization",
    "Hospital",
    "Pharmacy",
    "Physician",
    "NGO",
    "NewsMediaOrganization",
    "OnlineBusiness",
    "OnlineStore",
    "PerformingGroup",
    "MusicGroup",
    "PoliticalParty",
    "Project",
    "ResearchOrganization",
    "SearchRescueOrganization",
    "SportsOrganization",
    "SportsTeam",
    "WorkersUnion",
    "LocalBusiness",
    "AnimalShelter",
    "ArchiveOrganization",
    "AutomotiveBusiness",
    "ChildCare",
    "Dentist",
    "DryCleaningOrLaundry",
    "EmergencyService",
    "EmploymentAgency",
    "EntertainmentBusiness",
    "FinancialService",
    "FoodEstablishment",
    "GovernmentOffice",
    "HealthAndBeautyBusiness",
    "HomeAndConstructionBusiness",
    "InternetCafe",
    "LegalService",
    "Library",
    "LodgingBusiness",
    "MedicalBusiness",
    "ProfessionalService",
    "RadioStation",
    "RealEstateAgent",
    "RecyclingCenter",
    "SelfStorage",
    "ShoppingCenter",
    "SportsActivityLocation",
    "Store",
    "TelevisionStation",
    "TouristInformationCenter",
    "TravelAgency",
    "Restaurant",
    "CafeOrCoffeeShop",
    "BarOrPub",
    "Bakery",
    "Hotel",
    "Attorney",
    "AccountingService",
    "BankOrCreditUnion",
    "InsuranceAgency",
    "AutoDealer",
    "AutoRepair",
    "ClothingStore",
    "ElectronicsStore",
    "Plumber",
    "Electrician",
    "GeneralContractor",
    "HairSalon",
    "BeautySalon",
    "ExerciseGym",
];

/// A source of a recommendation: its URL and the date it was published or last checked.
pub type SourceRef = (&'static str, &'static str);

pub const SRC_GOOGLE_AI_FEATURES: SourceRef = (
    "https://developers.google.com/search/docs/appearance/ai-features",
    "2026-05",
);
pub const SRC_GOOGLE_AI_OPT: SourceRef = (
    "https://developers.google.com/search/docs/fundamentals/ai-optimization-guide",
    "2026-05",
);
pub const SRC_GOOGLE_UPDATES: SourceRef = ("https://developers.google.com/search/updates", "2026-05-07");
pub const SRC_GOOGLE_AI_CONTROL: SourceRef = ("https://support.google.com/webmasters/answer/16908024", "2026-08-31");
pub const SRC_GOOGLE_AI_REPORT: SourceRef = ("https://support.google.com/webmasters/answer/16984139", "2026-08-31");
pub const SRC_BING_GUIDELINES: SourceRef = (
    "https://www.bing.com/webmasters/help/webmaster-guidelines-30fba23a",
    "2026-02",
);
pub const SRC_BING_COPILOT_CONTROLS: SourceRef = (
    "https://blogs.bing.com/webmaster/2023/9/Announcing-new-options-for-webmasters-to-control-usage-of-their-content-in-Bing-Chat/",
    "2023-09",
);
pub const SRC_BING_AI_PERFORMANCE: SourceRef = (
    "https://blogs.bing.com/webmaster/February-2026/Introducing-AI-Performance-in-Bing-Webmaster-Tools-Public-Preview",
    "2026-02",
);
pub const SRC_OPENAI_BOTS: SourceRef = ("https://developers.openai.com/api/docs/bots", "2026-09-26");
pub const SRC_RFC9309: SourceRef = ("https://www.rfc-editor.org/rfc/rfc9309.html", "2022-09");
pub const SRC_CONTENT_SIGNALS: SourceRef = (
    "https://www.seroundtable.com/google-cloudflare-content-signals-41631.html",
    "2026",
);
pub const SRC_JAVASCRIPT: SourceRef = ("https://vercel.com/blog/the-rise-of-the-ai-crawler", "2024-12");
pub const SRC_SCHEMA_CITATIONS: SourceRef = ("https://ahrefs.com/blog/schema-ai-citations/", "2026");
pub const SRC_LLMS_TXT: SourceRef = (
    "https://www.digitalapplied.com/blog/llms-txt-in-practice-adoption-evidence-2026",
    "2026",
);

/// The sources of the report's evidence box, in the order of "Evidence".
pub const SOURCES: &[SourceRef] = &[
    SRC_GOOGLE_AI_FEATURES,
    SRC_GOOGLE_AI_OPT,
    SRC_GOOGLE_AI_CONTROL,
    SRC_GOOGLE_AI_REPORT,
    SRC_BING_GUIDELINES,
    SRC_BING_COPILOT_CONTROLS,
    SRC_BING_AI_PERFORMANCE,
    SRC_OPENAI_BOTS,
    SRC_RFC9309,
    SRC_CONTENT_SIGNALS,
    SRC_JAVASCRIPT,
    SRC_SCHEMA_CITATIONS,
    SRC_GOOGLE_UPDATES,
    SRC_LLMS_TXT,
];

/// Every `Issue::title_key` (the report has a title and a fix text for each).
pub const TITLE_KEYS: &[&str] = &[
    "search_crawler_blocked",
    "search_crawler_partly_blocked",
    "user_fetch_blocked",
    "user_fetch_blocked_limited",
    "training_crawler_blocked",
    "grounding_control_blocked",
    "content_signal_present",
    "robots_unavailable",
    "robots_skipped",
    "robots_not_attempted",
    "robots_not_utf8",
    "access_denied",
    "access_rate_limited",
    "access_server_error",
    "access_failed",
    "access_client_error",
    "access_redirect_chain",
    "access_redirect_loop",
    "access_suspected_challenge",
    "access_suspected_soft_404",
    "access_slow",
    "other_urls_issues",
    "google_noindex",
    "google_nosnippet",
    "google_max_snippet_zero",
    "bing_noindex",
    "bing_nosnippet",
    "bing_max_snippet_zero",
    "bing_copilot_restricted",
    "short_max_snippet",
    "data_nosnippet_large",
    "canonical_elsewhere",
    "pdf_restricted",
    "likely_client_rendered",
    "text_after_rendering",
    "offer_not_early",
    "unanswered_questions",
    "hidden_content",
    "no_contextual_links",
    "images_without_alt",
    "vague_references",
    "article_without_byline",
    "jsonld_parse_error",
    "jsonld_values_not_visible",
    "jsonld_values_not_checked",
    "faq_markup_present",
    "no_site_markup",
    "no_sitemap",
    "sitemap_unreadable",
    "sitemap_not_checked",
    "sitemap_comparison_not_assessed",
    "missing_from_sitemap",
    "sitemap_lists_non_indexable",
    "suspicious_lastmod",
    "hreflang_target_broken",
    "hreflang_target_not_crawled",
    "last_modified_missing",
];

/// The report's categories, in report order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum CategoryId {
    CrawlerPolicy,
    ObservedAccess,
    IndexingControls,
    Rendering,
    AnswerExtractability,
    EntityClarity,
    StructuredData,
    Discovery,
    ManualChecks,
}

impl CategoryId {
    pub const ALL: [CategoryId; 9] = [
        CategoryId::CrawlerPolicy,
        CategoryId::ObservedAccess,
        CategoryId::IndexingControls,
        CategoryId::Rendering,
        CategoryId::AnswerExtractability,
        CategoryId::EntityClarity,
        CategoryId::StructuredData,
        CategoryId::Discovery,
        CategoryId::ManualChecks,
    ];

    pub fn key(&self) -> &'static str {
        match self {
            CategoryId::CrawlerPolicy => "crawler_policy",
            CategoryId::ObservedAccess => "observed_access",
            CategoryId::IndexingControls => "indexing_controls",
            CategoryId::Rendering => "rendering",
            CategoryId::AnswerExtractability => "answer_extractability",
            CategoryId::EntityClarity => "entity_clarity",
            CategoryId::StructuredData => "structured_data",
            CategoryId::Discovery => "discovery",
            CategoryId::ManualChecks => "manual_checks",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum CheckStatus {
    Problem,
    Attention,
    Info,
    Ok,
    NotAssessed,
}

impl CheckStatus {
    /// Worst first.
    pub fn rank(&self) -> u8 {
        match self {
            CheckStatus::Problem => 0,
            CheckStatus::Attention => 1,
            CheckStatus::Info => 2,
            CheckStatus::Ok => 3,
            CheckStatus::NotAssessed => 4,
        }
    }

    pub fn key(&self) -> &'static str {
        match self {
            CheckStatus::Problem => "problem",
            CheckStatus::Attention => "attention",
            CheckStatus::Info => "info",
            CheckStatus::Ok => "ok",
            CheckStatus::NotAssessed => "not_assessed",
        }
    }
}

/// How strong the evidence behind a recommendation is: documented engine behavior (`Strong`),
/// engine guidance that was not measured (`Moderate`), or vendor claims, observational studies or
/// no evidence (`Weak`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Evidence {
    Strong,
    Moderate,
    Weak,
}

impl Evidence {
    pub fn rank(&self) -> u8 {
        match self {
            Evidence::Strong => 0,
            Evidence::Moderate => 1,
            Evidence::Weak => 2,
        }
    }

    pub fn key(&self) -> &'static str {
        match self {
            Evidence::Strong => "strong",
            Evidence::Moderate => "moderate",
            Evidence::Weak => "weak",
        }
    }
}

/// One finding: what, how bad, how well supported, for which engines, according to which source,
/// and on which pages. The report words it from `title_key` and `args`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Issue {
    /// Unique within a report: `category.title_key[.detail]`.
    pub id: String,
    pub category: CategoryId,
    pub status: CheckStatus,
    pub evidence: Evidence,
    /// The engines or vendors concerned: "Google", "Bing", "OpenAI", "All", …
    pub scope: &'static str,
    pub source: SourceRef,
    pub title_key: &'static str,
    pub args: Vec<String>,
    /// The affected pages (key pages for the deterministic categories).
    pub pages: Vec<String>,
}

/// The verdict of a category and how many of its checks ran.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CategoryState {
    pub status: CheckStatus,
    pub attempted: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub skipped: usize,
    /// Why a category is not assessed, as a text key (`ai_unavailable`, `no_key_pages`, …); empty
    /// otherwise.
    pub reason: String,
}

/// The robots.txt verdicts of one origin.
#[derive(Debug, Clone, PartialEq)]
pub struct OriginPolicy {
    /// `scheme://host[:port]`, without a trailing slash.
    pub origin: String,
    pub state: RobotsFetchState,
    /// The key-page paths (path + query) of the origin.
    pub paths: Vec<String>,
    /// One verdict per table agent; empty when the rules are unknown.
    pub agents: Vec<AgentVerdict>,
    pub content_signals: Vec<String>,
    pub sitemaps: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AgentVerdict {
    pub agent: &'static AiAgent,
    pub access: AgentAccess,
    /// The rule that blocks the first blocked path, e.g. "User-agent: GPTBot → Disallow: /".
    pub rule: Option<String>,
}

/// The effective snippet controls of one key page.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PageControls {
    pub url: String,
    pub google: EnginePolicy,
    pub bing: EnginePolicy,
    /// The share of the page's own text inside `data-nosnippet`.
    pub nosnippet_share: Option<f64>,
    pub canonical_elsewhere: Option<String>,
}

/// The existing structured data of one key page.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PageMarkup {
    pub url: String,
    pub is_homepage: bool,
    pub markup: ExistingMarkup,
}

/// The per-page LLM analysis of a run.
#[derive(Debug, Clone, Default)]
pub struct AnalysisRun {
    /// The pages chosen for analysis (`--ai-max-pages`).
    pub selected: usize,
    pub pages: Vec<PageAnalysis>,
    /// (URL, redacted error) of the pages whose analysis failed.
    pub failed: Vec<(String, String)>,
    /// The supporting signals of the chosen pages, by URL.
    pub signals: Vec<(String, PageSignals)>,
    /// Why no page was analyzed at all (no model configured, a dry run, …).
    pub unavailable: Option<String>,
    /// Chosen pages without any text of their own in the HTML (an app shell): not sent, counted
    /// as skipped (the Rendering category covers them).
    pub without_text: Vec<String>,
}

/// What the checks found: the input of the verdicts.
#[derive(Debug, Clone, Default)]
pub struct Checks {
    pub key_pages: Vec<KeyPage>,
    pub policy: Vec<OriginPolicy>,
    /// The access issues of the key pages, in key-page order.
    pub access: Vec<AccessIssue>,
    pub access_stats: AccessStats,
    pub controls: Vec<PageControls>,
    pub render: RenderCheck,
    pub markup: Vec<PageMarkup>,
    pub discovery: Discovery,
    pub pdfs: Vec<PdfRestriction>,
    pub analysis: AnalysisRun,
}

/// The robots.txt verdicts of an origin for every table agent on the key-page `paths`, with the
/// deciding rule of a block. Rules that are unknown (the file could not be read, or was not
/// fetched) give no verdicts.
pub fn origin_policy(origin: &str, state: &RobotsFetchState, paths: &[String]) -> OriginPolicy {
    let robots = robots_of(state);
    let agents = if state_is_known(state) {
        AI_AGENTS
            .iter()
            .map(|agent| {
                let access = evaluate(robots.as_ref(), agent, paths);
                let first_blocked = match &access {
                    AgentAccess::Allowed => None,
                    AgentAccess::Blocked => paths.first(),
                    AgentAccess::Partly(blocked) => blocked.first(),
                };
                let rule = first_blocked.and_then(|path| {
                    let robots = robots.as_ref()?;
                    robots.explain(robots.token_for(agent), path)
                });
                AgentVerdict { agent, access, rule }
            })
            .collect()
    } else {
        Vec::new()
    };
    OriginPolicy {
        origin: origin.trim_end_matches('/').to_string(),
        state: state.clone(),
        paths: paths.to_vec(),
        agents,
        content_signals: robots
            .as_ref()
            .map(|robots| robots.content_signals().to_vec())
            .unwrap_or_default(),
        sitemaps: robots
            .as_ref()
            .map(|robots| robots.sitemaps().to_vec())
            .unwrap_or_default(),
    }
}

/// How many checks of a category ran, and why it is not assessed if none succeeded.
#[derive(Default)]
struct Counts {
    attempted: usize,
    succeeded: usize,
    failed: usize,
    skipped: usize,
    reason: &'static str,
}

impl Counts {
    fn all(checked: usize, reason: &'static str) -> Self {
        Counts {
            attempted: checked,
            succeeded: checked,
            reason,
            ..Counts::default()
        }
    }

    /// A check of the key pages that could run on `checked` of them; the others (no HTML page with
    /// a body: a redirect, an error, another content type) are skipped.
    fn of_key_pages(key_pages: usize, checked: usize, reason: &'static str) -> Self {
        let attempted = key_pages.max(checked);
        Counts {
            attempted,
            succeeded: checked,
            skipped: attempted - checked,
            reason,
            ..Counts::default()
        }
    }
}

/// Collects the issues of one category.
struct Found<'a> {
    category: CategoryId,
    issues: &'a mut Vec<Issue>,
}

impl Found<'_> {
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        title_key: &'static str,
        detail: &str,
        status: CheckStatus,
        evidence: Evidence,
        scope: &'static str,
        source: SourceRef,
        args: Vec<String>,
        pages: Vec<String>,
    ) {
        let id = if detail.is_empty() {
            format!("{}.{}", self.category.key(), title_key)
        } else {
            format!("{}.{}.{}", self.category.key(), title_key, detail)
        };
        self.issues.push(Issue {
            id,
            category: self.category,
            status,
            evidence,
            scope,
            source,
            title_key,
            args,
            pages,
        });
    }
}

/// The state of every category, in report order, and all issues.
pub fn assess(checks: &Checks) -> (Vec<(CategoryId, CategoryState)>, Vec<Issue>) {
    let mut issues: Vec<Issue> = Vec::new();
    let mut states: Vec<(CategoryId, CategoryState)> = Vec::new();
    for category in CategoryId::ALL {
        let mut found = Found {
            category,
            issues: &mut issues,
        };
        let counts = match category {
            CategoryId::CrawlerPolicy => crawler_policy(checks, &mut found),
            CategoryId::ObservedAccess => observed_access(checks, &mut found),
            CategoryId::IndexingControls => indexing_controls(checks, &mut found),
            CategoryId::Rendering => rendering(checks, &mut found),
            CategoryId::AnswerExtractability => answer_extractability(checks, &mut found),
            CategoryId::EntityClarity => entity_clarity(checks, &mut found),
            CategoryId::StructuredData => structured_data(checks, &mut found),
            CategoryId::Discovery => discovery(checks, &mut found),
            CategoryId::ManualChecks => Counts {
                reason: "manual_only",
                ..Counts::default()
            },
        };
        let not_assessed = counts.attempted == 0 || counts.succeeded == 0;
        let status = if not_assessed {
            CheckStatus::NotAssessed
        } else {
            issues
                .iter()
                .filter(|issue| issue.category == category)
                .map(|issue| issue.status)
                .min_by_key(CheckStatus::rank)
                .unwrap_or(CheckStatus::Ok)
        };
        states.push((
            category,
            CategoryState {
                status,
                attempted: counts.attempted,
                succeeded: counts.succeeded,
                failed: counts.failed,
                skipped: counts.skipped,
                reason: if not_assessed {
                    counts.reason.to_string()
                } else {
                    String::new()
                },
            },
        ));
    }
    (states, issues)
}

/// The issues to fix first: Problem, then Attention, then Info; within a status by evidence
/// (Strong first), then by category in report order, then by the number of affected pages; at
/// most `MAX_PRIORITIES`.
pub fn priorities(issues: &[Issue]) -> Vec<&Issue> {
    let mut ranked: Vec<&Issue> = issues
        .iter()
        .filter(|issue| issue.status.rank() <= CheckStatus::Info.rank())
        .collect();
    ranked.sort_by(|a, b| {
        a.status
            .rank()
            .cmp(&b.status.rank())
            .then_with(|| a.evidence.rank().cmp(&b.evidence.rank()))
            .then_with(|| a.category.cmp(&b.category))
            .then_with(|| b.pages.len().cmp(&a.pages.len()))
            .then_with(|| a.id.cmp(&b.id))
    });
    ranked.truncate(MAX_PRIORITIES);
    ranked
}

fn crawler_policy(checks: &Checks, found: &mut Found) -> Counts {
    let mut counts = Counts::default();
    for origin in &checks.policy {
        counts.attempted += 1;
        let name = origin.origin.as_str();
        match &origin.state {
            RobotsFetchState::Ok { valid_utf8, .. } => {
                counts.succeeded += 1;
                if !valid_utf8 {
                    found.add(
                        "robots_not_utf8",
                        name,
                        CheckStatus::Info,
                        Evidence::Strong,
                        "All",
                        SRC_RFC9309,
                        vec![name.to_string()],
                        Vec::new(),
                    );
                }
            }
            RobotsFetchState::NotFound { .. } => counts.succeeded += 1,
            RobotsFetchState::Unavailable { status_or_error } => {
                counts.failed += 1;
                found.add(
                    "robots_unavailable",
                    name,
                    CheckStatus::Attention,
                    Evidence::Strong,
                    "All",
                    SRC_RFC9309,
                    vec![name.to_string(), status_or_error.clone()],
                    Vec::new(),
                );
            }
            RobotsFetchState::Skipped | RobotsFetchState::NotAttempted => {
                counts.skipped += 1;
                let key = if origin.state == RobotsFetchState::Skipped {
                    "robots_skipped"
                } else {
                    "robots_not_attempted"
                };
                found.add(
                    key,
                    name,
                    CheckStatus::Info,
                    Evidence::Strong,
                    "All",
                    SRC_RFC9309,
                    vec![name.to_string()],
                    Vec::new(),
                );
            }
        }
        for verdict in &origin.agents {
            let blocked: &[String] = match &verdict.access {
                AgentAccess::Allowed => continue,
                AgentAccess::Blocked => &origin.paths,
                AgentAccess::Partly(blocked) => blocked,
            };
            let agent = verdict.agent;
            let partly = matches!(verdict.access, AgentAccess::Partly(_));
            let (key, status) = if agent.purposes.contains(&Purpose::Search) {
                if partly {
                    ("search_crawler_partly_blocked", CheckStatus::Problem)
                } else {
                    ("search_crawler_blocked", CheckStatus::Problem)
                }
            } else if agent.purposes.contains(&Purpose::UserFetch) {
                if agent.compliance == Compliance::Honors {
                    ("user_fetch_blocked", CheckStatus::Attention)
                } else {
                    ("user_fetch_blocked_limited", CheckStatus::Info)
                }
            } else if agent.purposes.contains(&Purpose::Grounding) {
                ("grounding_control_blocked", CheckStatus::Info)
            } else {
                ("training_crawler_blocked", CheckStatus::Info)
            };
            found.add(
                key,
                &format!("{}.{}", agent.token.to_ascii_lowercase(), name),
                status,
                Evidence::Strong,
                agent.vendor,
                (agent.doc_url, agent.verified_on),
                vec![
                    agent.token.to_string(),
                    name.to_string(),
                    verdict.rule.clone().unwrap_or_default(),
                ],
                blocked.iter().map(|path| format!("{name}{path}")).collect(),
            );
        }
        if !origin.content_signals.is_empty() {
            found.add(
                "content_signal_present",
                name,
                CheckStatus::Info,
                Evidence::Strong,
                "All",
                SRC_CONTENT_SIGNALS,
                vec![name.to_string(), origin.content_signals.join(" | ")],
                Vec::new(),
            );
        }
    }
    counts.reason = if counts.attempted == 0 {
        "no_key_pages"
    } else {
        "robots_not_read"
    };
    counts
}

fn observed_access(checks: &Checks, found: &mut Found) -> Counts {
    let mut by_kind: Vec<(&'static str, CheckStatus, Vec<String>)> = Vec::new();
    for issue in &checks.access {
        let (key, status) = match issue.kind {
            AccessKind::Denied(_) => ("access_denied", CheckStatus::Problem),
            AccessKind::RateLimited => ("access_rate_limited", CheckStatus::Attention),
            AccessKind::ServerError(_) => ("access_server_error", CheckStatus::Problem),
            AccessKind::Transport(_) => ("access_failed", CheckStatus::Problem),
            AccessKind::ClientError(_) => ("access_client_error", CheckStatus::Problem),
            AccessKind::RedirectChain(_) => ("access_redirect_chain", CheckStatus::Attention),
            AccessKind::RedirectLoop => ("access_redirect_loop", CheckStatus::Problem),
            AccessKind::SuspectedChallenge => ("access_suspected_challenge", CheckStatus::Attention),
            AccessKind::SuspectedSoft404 => ("access_suspected_soft_404", CheckStatus::Attention),
            AccessKind::Slow(_) => ("access_slow", CheckStatus::Info),
        };
        match by_kind.iter_mut().find(|(known, _, _)| *known == key) {
            Some((_, _, pages)) => pages.push(issue.url.clone()),
            None => by_kind.push((key, status, vec![issue.url.clone()])),
        }
    }
    for (key, status, pages) in by_kind {
        found.add(
            key,
            "",
            status,
            Evidence::Strong,
            "All",
            SRC_GOOGLE_AI_FEATURES,
            vec![pages.len().to_string()],
            pages,
        );
    }
    let other: usize = checks.access_stats.other_issues.values().sum();
    if other > 0 {
        found.add(
            "other_urls_issues",
            "",
            CheckStatus::Info,
            Evidence::Strong,
            "All",
            SRC_GOOGLE_AI_FEATURES,
            vec![other.to_string(), checks.access_stats.other_urls.to_string()],
            Vec::new(),
        );
    }
    Counts::all(checks.key_pages.len(), "no_key_pages")
}

/// What identifies a grouped issue: title key, detail, status, scope and source.
type GroupKey = (&'static str, &'static str, CheckStatus, &'static str, SourceRef);

/// An issue that collects pages and arguments.
struct Group {
    key: GroupKey,
    pages: Vec<String>,
    args: Vec<String>,
}

/// Issues that collect pages, in the order they were first seen.
#[derive(Default)]
struct Grouped {
    groups: Vec<Group>,
}

impl Grouped {
    fn add(&mut self, key: GroupKey, page: &str, arg: Option<String>) {
        let index = match self
            .groups
            .iter()
            .position(|group| group.key.0 == key.0 && group.key.1 == key.1)
        {
            Some(index) => index,
            None => {
                self.groups.push(Group {
                    key,
                    pages: Vec::new(),
                    args: Vec::new(),
                });
                self.groups.len() - 1
            }
        };
        let group = &mut self.groups[index];
        if !group.pages.iter().any(|known| known == page) {
            group.pages.push(page.to_string());
        }
        if let Some(arg) = arg
            && !group.args.contains(&arg)
        {
            group.args.push(arg);
        }
    }

    /// Replaces the arguments of the group with this title key.
    fn set_args(&mut self, title_key: &str, args: Vec<String>) {
        for group in &mut self.groups {
            if group.key.0 == title_key {
                group.args = args.clone();
            }
        }
    }

    /// Adds the groups as issues; a group without arguments gets its page count.
    fn emit(self, found: &mut Found, evidence: Evidence) {
        for Group { key, pages, args } in self.groups {
            let (title_key, detail, status, scope, source) = key;
            let args = if args.is_empty() {
                vec![pages.len().to_string()]
            } else {
                args
            };
            found.add(title_key, detail, status, evidence, scope, source, args, pages);
        }
    }
}

fn indexing_controls(checks: &Checks, found: &mut Found) -> Counts {
    let mut grouped = Grouped::default();
    for page in &checks.controls {
        for (engine, policy, scope, source) in [
            ("google", &page.google, "Google", SRC_GOOGLE_AI_FEATURES),
            ("bing", &page.bing, "Bing", SRC_BING_GUIDELINES),
        ] {
            let google = engine == "google";
            if policy.noindex {
                let key = if google { "google_noindex" } else { "bing_noindex" };
                grouped.add((key, "", CheckStatus::Problem, scope, source), &page.url, None);
            }
            if policy.nosnippet {
                let key = if google { "google_nosnippet" } else { "bing_nosnippet" };
                grouped.add((key, "", CheckStatus::Problem, scope, source), &page.url, None);
            }
            match policy.max_snippet {
                Some(0) => {
                    let key = if google {
                        "google_max_snippet_zero"
                    } else {
                        "bing_max_snippet_zero"
                    };
                    grouped.add((key, "", CheckStatus::Problem, scope, source), &page.url, None);
                }
                Some(limit) if (1..=SHORT_SNIPPET_CHARS).contains(&limit) => grouped.add(
                    ("short_max_snippet", engine, CheckStatus::Attention, scope, source),
                    &page.url,
                    None,
                ),
                _ => {}
            }
        }
        if page.bing.noarchive || page.bing.nocache {
            grouped.add(
                (
                    "bing_copilot_restricted",
                    "",
                    CheckStatus::Attention,
                    "Bing",
                    SRC_BING_COPILOT_CONTROLS,
                ),
                &page.url,
                None,
            );
        }
        if page
            .nosnippet_share
            .is_some_and(|share| share >= NOSNIPPET_SHARE_ATTENTION)
        {
            grouped.add(
                (
                    "data_nosnippet_large",
                    "",
                    CheckStatus::Attention,
                    "Google, Bing",
                    SRC_GOOGLE_AI_FEATURES,
                ),
                &page.url,
                None,
            );
        }
        if let Some(target) = &page.canonical_elsewhere {
            grouped.add(
                (
                    "canonical_elsewhere",
                    "",
                    CheckStatus::Attention,
                    "Google, Bing",
                    SRC_GOOGLE_AI_FEATURES,
                ),
                &page.url,
                Some(target.clone()),
            );
        }
    }
    for pdf in &checks.pdfs {
        grouped.add(
            (
                "pdf_restricted",
                "",
                CheckStatus::Info,
                "Google, Bing",
                SRC_GOOGLE_AI_FEATURES,
            ),
            &pdf.url,
            None,
        );
    }
    grouped.emit(found, Evidence::Strong);
    Counts::of_key_pages(checks.key_pages.len(), checks.controls.len(), "no_pages_checked")
}

fn rendering(checks: &Checks, found: &mut Found) -> Counts {
    // An app shell in the HTML (plain crawl), or text that appears only after rendering (browser).
    let (after_rendering, app_shells): (Vec<&RenderRisk>, Vec<&RenderRisk>) = checks
        .render
        .risks
        .iter()
        .partition(|risk| risk.rendered_text_chars.is_some());
    for (title_key, risks) in [
        ("likely_client_rendered", app_shells),
        ("text_after_rendering", after_rendering),
    ] {
        if risks.is_empty() {
            continue;
        }
        let pages: Vec<String> = risks.iter().map(|risk| risk.url.clone()).collect();
        found.add(
            title_key,
            "",
            CheckStatus::Attention,
            Evidence::Moderate,
            "OpenAI, Anthropic, Perplexity",
            SRC_JAVASCRIPT,
            vec![pages.len().to_string()],
            pages,
        );
    }
    let reason = if checks.render.checked == 0 && !checks.render.not_comparable.is_empty() {
        "render_not_comparable"
    } else {
        "no_pages_checked"
    };
    Counts::of_key_pages(checks.key_pages.len(), checks.render.checked, reason)
}

/// The counts of a per-page category: a page succeeded when its answer was parsed and `assessed`
/// holds for its verified analysis; any other attempted page failed.
fn llm_counts(run: &AnalysisRun, assessed: impl Fn(&PageAnalysis) -> bool) -> Counts {
    if run.unavailable.is_some() {
        return Counts {
            skipped: run.selected + run.without_text.len(),
            reason: "ai_unavailable",
            ..Counts::default()
        };
    }
    let attempted = run.pages.len() + run.failed.len();
    let succeeded = run.pages.iter().filter(|page| assessed(page)).count();
    Counts {
        attempted,
        succeeded,
        failed: attempted - succeeded,
        skipped: run.without_text.len(),
        reason: if attempted == 0 {
            "no_pages_analyzed"
        } else if run.pages.is_empty() {
            "analysis_failed"
        } else {
            "no_verified_answers"
        },
    }
}

fn answer_extractability(checks: &Checks, found: &mut Found) -> Counts {
    let run = &checks.analysis;
    let signals: HashMap<&str, &PageSignals> = run
        .signals
        .iter()
        .map(|(url, signals)| (url.as_str(), signals))
        .collect();
    let mut grouped = Grouped::default();
    let mut unanswered = 0;
    let mut images = 0;
    for page in &run.pages {
        if page.states_offer_early == OfferEarly::No && OFFER_EARLY_TYPES.contains(&page.page_type) {
            grouped.add(
                (
                    "offer_not_early",
                    "",
                    CheckStatus::Attention,
                    "Google, Bing",
                    SRC_BING_GUIDELINES,
                ),
                &page.url,
                None,
            );
        }
        let not_answered = page
            .questions
            .iter()
            .filter(|question| question.answered == Answered::No)
            .count();
        if not_answered > 0 {
            unanswered += not_answered;
            grouped.add(
                (
                    "unanswered_questions",
                    "",
                    CheckStatus::Info,
                    "Google, Bing",
                    SRC_GOOGLE_AI_OPT,
                ),
                &page.url,
                None,
            );
        }
        let Some(signals) = signals.get(page.url.as_str()) else {
            continue;
        };
        if signals.main_chars >= HIDDEN_MIN_CHARS
            && signals
                .collapsed_share()
                .is_some_and(|share| share >= HIDDEN_SHARE_ATTENTION)
        {
            grouped.add(
                (
                    "hidden_content",
                    "",
                    CheckStatus::Attention,
                    "Bing",
                    SRC_BING_GUIDELINES,
                ),
                &page.url,
                None,
            );
        }
        if signals.in_content_links == 0 {
            grouped.add(
                (
                    "no_contextual_links",
                    "",
                    CheckStatus::Info,
                    "Google, Bing",
                    SRC_BING_GUIDELINES,
                ),
                &page.url,
                None,
            );
        }
        if signals.images_without_alt > 0 {
            images += signals.images_without_alt;
            grouped.add(
                (
                    "images_without_alt",
                    "",
                    CheckStatus::Info,
                    "Google, Bing",
                    SRC_BING_GUIDELINES,
                ),
                &page.url,
                None,
            );
        }
    }
    grouped.set_args("unanswered_questions", vec![unanswered.to_string()]);
    grouped.set_args("images_without_alt", vec![images.to_string()]);
    grouped.emit(found, Evidence::Moderate);
    // A page whose questions the crawler could not verify tells nothing about extractability.
    llm_counts(run, PageAnalysis::has_verified_questions)
}

fn entity_clarity(checks: &Checks, found: &mut Found) -> Counts {
    let run = &checks.analysis;
    let mut grouped = Grouped::default();
    for page in &run.pages {
        if !page.vague_references.is_empty() {
            grouped.add(
                (
                    "vague_references",
                    "",
                    CheckStatus::Info,
                    "Google, Bing",
                    SRC_BING_GUIDELINES,
                ),
                &page.url,
                None,
            );
        }
        if page.page_type == PageType::Article && (page.byline.author.is_none() || page.byline.date.is_none()) {
            grouped.add(
                (
                    "article_without_byline",
                    "",
                    CheckStatus::Info,
                    "Google, Bing",
                    SRC_GOOGLE_AI_OPT,
                ),
                &page.url,
                None,
            );
        }
    }
    grouped.emit(found, Evidence::Moderate);
    llm_counts(run, |_| true)
}

fn structured_data(checks: &Checks, found: &mut Found) -> Counts {
    let mut grouped = Grouped::default();
    let mut parse_errors = 0;
    let mut not_checked = 0;
    for page in &checks.markup {
        let markup = &page.markup;
        if !markup.parse_errors.is_empty() {
            parse_errors += markup.parse_errors.len();
            grouped.add(
                (
                    "jsonld_parse_error",
                    "",
                    CheckStatus::Problem,
                    "Google, Bing",
                    SRC_GOOGLE_AI_FEATURES,
                ),
                &page.url,
                None,
            );
        }
        for (path, value) in markup.invisible_values.iter().take(3) {
            grouped.add(
                (
                    "jsonld_values_not_visible",
                    "",
                    CheckStatus::Attention,
                    "Google, Bing",
                    SRC_GOOGLE_AI_FEATURES,
                ),
                &page.url,
                Some(format!("{path} = {value}")),
            );
        }
        if markup.values_not_checked > 0 {
            not_checked += markup.values_not_checked;
            grouped.add(
                (
                    "jsonld_values_not_checked",
                    "",
                    CheckStatus::Info,
                    "Google, Bing",
                    SRC_GOOGLE_AI_FEATURES,
                ),
                &page.url,
                None,
            );
        }
        if markup.declares("FAQPage") {
            grouped.add(
                (
                    "faq_markup_present",
                    "",
                    CheckStatus::Info,
                    "Google",
                    SRC_GOOGLE_UPDATES,
                ),
                &page.url,
                None,
            );
        }
    }
    grouped.set_args("jsonld_parse_error", vec![parse_errors.to_string()]);
    grouped.set_args("jsonld_values_not_checked", vec![not_checked.to_string()]);
    grouped.emit(found, Evidence::Strong);
    if let Some(home) = checks.markup.iter().find(|page| page.is_homepage) {
        let has_organization = ORGANIZATION_TYPES.iter().any(|kind| home.markup.declares(kind));
        let mut missing: Vec<String> = Vec::new();
        if !has_organization {
            missing.push("Organization".to_string());
        }
        if !home.markup.declares("WebSite") {
            missing.push("WebSite".to_string());
        }
        if !missing.is_empty() {
            found.add(
                "no_site_markup",
                "",
                CheckStatus::Info,
                Evidence::Weak,
                "Google, Bing",
                SRC_SCHEMA_CITATIONS,
                missing,
                vec![home.url.clone()],
            );
        }
    }
    Counts::of_key_pages(checks.key_pages.len(), checks.markup.len(), "no_pages_checked")
}

fn discovery(checks: &Checks, found: &mut Found) -> Counts {
    let discovery = &checks.discovery;
    let mut grouped = Grouped::default();
    if discovery.sitemaps.is_empty() {
        found.add(
            "no_sitemap",
            "",
            CheckStatus::Info,
            Evidence::Moderate,
            "Google, Bing",
            SRC_BING_GUIDELINES,
            Vec::new(),
            Vec::new(),
        );
    } else {
        for sitemap in &discovery.sitemaps {
            match &sitemap.state {
                SitemapState::Malformed(_) | SitemapState::Failed(_) | SitemapState::NotAbsolute => grouped.add(
                    (
                        "sitemap_unreadable",
                        "",
                        CheckStatus::Attention,
                        "Google, Bing",
                        SRC_BING_GUIDELINES,
                    ),
                    &sitemap.url,
                    None,
                ),
                SitemapState::NotCrawled | SitemapState::Unsupported(_) => grouped.add(
                    (
                        "sitemap_not_checked",
                        "",
                        CheckStatus::Info,
                        "Google, Bing",
                        SRC_BING_GUIDELINES,
                    ),
                    &sitemap.url,
                    None,
                ),
                SitemapState::Parsed { lastmod, .. } if lastmod.suspicious() => grouped.add(
                    (
                        "suspicious_lastmod",
                        "",
                        CheckStatus::Attention,
                        "Bing",
                        SRC_BING_GUIDELINES,
                    ),
                    &sitemap.url,
                    None,
                ),
                SitemapState::Parsed { .. } | SitemapState::Redirected(_) => {}
            }
        }
        // Not compared because a sitemap could not be read or no page list was read; with every
        // sitemap read, 0 compared only means that no key page is indexable.
        let incomplete = discovery.sitemaps.iter().any(|sitemap| {
            matches!(
                sitemap.state,
                SitemapState::Malformed(_)
                    | SitemapState::Failed(_)
                    | SitemapState::NotCrawled
                    | SitemapState::Unsupported(_)
            )
        }) || !discovery.sitemaps.iter().any(|sitemap| {
            matches!(
                sitemap.state,
                SitemapState::Parsed {
                    kind: SitemapKind::UrlSet,
                    ..
                }
            )
        });
        if discovery.key_pages_compared == 0 && incomplete {
            found.add(
                "sitemap_comparison_not_assessed",
                "",
                CheckStatus::Info,
                Evidence::Moderate,
                "Google, Bing",
                SRC_BING_GUIDELINES,
                Vec::new(),
                Vec::new(),
            );
        }
    }
    for page in &discovery.missing_from_sitemaps {
        grouped.add(
            (
                "missing_from_sitemap",
                "",
                CheckStatus::Attention,
                "Google, Bing",
                SRC_BING_GUIDELINES,
            ),
            page,
            None,
        );
    }
    for listed in &discovery.listed_issues {
        grouped.add(
            (
                "sitemap_lists_non_indexable",
                "",
                CheckStatus::Attention,
                "Google, Bing",
                SRC_BING_GUIDELINES,
            ),
            &listed.url,
            None,
        );
    }
    for hreflang in &discovery.hreflang_issues {
        let key = match hreflang.problem {
            HreflangProblem::Status(_) => ("hreflang_target_broken", CheckStatus::Attention),
            HreflangProblem::NotCrawled => ("hreflang_target_not_crawled", CheckStatus::Info),
            HreflangProblem::OtherSite => continue,
        };
        grouped.add(
            (key.0, "", key.1, "Google, Bing", SRC_GOOGLE_AI_FEATURES),
            &hreflang.page,
            Some(hreflang.target.clone()),
        );
    }
    grouped.emit(found, Evidence::Moderate);
    let coverage = discovery.last_modified;
    if coverage.pages > 0 && coverage.plausible * 2 < coverage.pages {
        found.add(
            "last_modified_missing",
            "",
            CheckStatus::Info,
            Evidence::Moderate,
            "Bing",
            SRC_BING_GUIDELINES,
            vec![coverage.plausible.to_string(), coverage.pages.to_string()],
            Vec::new(),
        );
    }
    Counts::of_key_pages(checks.key_pages.len(), coverage.pages, "no_pages_checked")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::blocks::{Block, BlockKind, Region};
    use crate::ai::geo::access::{AccessIssue, AccessKind};
    use crate::ai::geo::analyze::{
        Answered, Byline, Coverage, Excerpt, OfferEarly, PageAnalysis, PageType, Question, Rejected,
    };
    use crate::ai::geo::controls::EnginePolicy;
    use crate::ai::geo::discovery::{
        HreflangIssue, HreflangProblem, LastmodStats, ListedIssue, ListedReason, SitemapFile, SitemapKind, SitemapState,
    };
    use crate::ai::geo::jsonld::ExistingMarkup;
    use crate::ai::geo::keys::KeyPage;
    use crate::ai::geo::render::RenderRisk;
    use crate::ai::geo::signals::{LastModifiedCoverage, PageSignals, PdfRestriction};
    use crate::result::status::RobotsFetchState;

    const HOME: &str = "https://example.com/";

    fn key(url: &str) -> KeyPage {
        KeyPage {
            uq_id: url.to_string(),
            url: url.to_string(),
            status_code: 200,
            score: Some(1.0),
            is_homepage: url == HOME,
        }
    }

    fn paths(list: &[&str]) -> Vec<String> {
        list.iter().map(|path| path.to_string()).collect()
    }

    fn robots(content: &str) -> RobotsFetchState {
        RobotsFetchState::Ok {
            status: 200,
            content: content.to_string(),
            valid_utf8: true,
        }
    }

    /// A site with two key pages whose checks all succeeded and found nothing.
    fn clean() -> Checks {
        let key_pages = vec![key(HOME), key("https://example.com/sluzby")];
        let urls: Vec<String> = key_pages.iter().map(|page| page.url.clone()).collect();
        Checks {
            policy: vec![origin_policy(
                "https://example.com",
                &robots("User-agent: *\nDisallow: /admin\n"),
                &paths(&["/", "/sluzby"]),
            )],
            controls: urls
                .iter()
                .map(|url| PageControls {
                    url: url.clone(),
                    ..PageControls::default()
                })
                .collect(),
            render: RenderCheck {
                checked: 2,
                ..RenderCheck::default()
            },
            markup: urls
                .iter()
                .map(|url| PageMarkup {
                    url: url.clone(),
                    is_homepage: url == HOME,
                    markup: ExistingMarkup {
                        jsonld_types: if url == HOME {
                            vec!["Organization".to_string(), "WebSite".to_string()]
                        } else {
                            Vec::new()
                        },
                        ..ExistingMarkup::default()
                    },
                })
                .collect(),
            discovery: Discovery {
                sitemaps: vec![SitemapFile {
                    url: "https://example.com/sitemap.xml".to_string(),
                    declared: true,
                    state: SitemapState::Parsed {
                        kind: SitemapKind::UrlSet,
                        entries: 2,
                        lastmod: LastmodStats::default(),
                    },
                }],
                listed_urls: 2,
                key_pages_compared: 2,
                last_modified: LastModifiedCoverage {
                    pages: 2,
                    with_header: 2,
                    plausible: 2,
                },
                ..Discovery::default()
            },
            analysis: AnalysisRun {
                selected: 2,
                pages: urls.iter().map(|url| analysis(url, PageType::Service)).collect(),
                signals: urls
                    .iter()
                    .map(|url| {
                        (
                            url.clone(),
                            PageSignals {
                                main_chars: 1_000,
                                collapsed_chars: 0,
                                in_content_links: 3,
                                images_without_alt: 0,
                            },
                        )
                    })
                    .collect(),
                ..AnalysisRun::default()
            },
            key_pages,
            ..Checks::default()
        }
    }

    fn analysis(url: &str, page_type: PageType) -> PageAnalysis {
        PageAnalysis {
            url: url.to_string(),
            title: String::new(),
            lang: "cs".to_string(),
            description: String::new(),
            page_type,
            main_topic: "Služby".to_string(),
            states_offer_early: OfferEarly::Yes,
            questions: vec![Question {
                question: "Co nabízíte?".to_string(),
                answered: Answered::Yes,
                excerpts: vec![Excerpt {
                    block: "B2".to_string(),
                    text: "Nabízíme služby.".to_string(),
                }],
            }],
            vague_references: Vec::new(),
            improvements: Vec::new(),
            lead: None,
            faq_pairs: Vec::new(),
            byline: Byline::default(),
            entity_drafts: Vec::new(),
            h1: None,
            coverage: Coverage::default(),
            rejected: Rejected::default(),
            indexable: true,
        }
    }

    fn state(checks: &Checks, category: CategoryId) -> CategoryState {
        assess(checks)
            .0
            .into_iter()
            .find(|(id, _)| *id == category)
            .map(|(_, state)| state)
            .expect("every category has a state")
    }

    fn issues(checks: &Checks, category: CategoryId) -> Vec<Issue> {
        assess(checks)
            .1
            .into_iter()
            .filter(|issue| issue.category == category)
            .collect()
    }

    fn only(checks: &Checks, category: CategoryId, title_key: &str) -> Issue {
        let found: Vec<Issue> = issues(checks, category)
            .into_iter()
            .filter(|issue| issue.title_key == title_key)
            .collect();
        assert_eq!(found.len(), 1, "{title_key}: {found:?}");
        found.into_iter().next().unwrap()
    }

    #[test]
    fn a_clean_site_is_ok_in_every_assessed_category_and_manual_checks_stay_open() {
        let (states, found) = assess(&clean());
        assert!(found.is_empty(), "{found:?}");
        let order: Vec<CategoryId> = states.iter().map(|(id, _)| *id).collect();
        assert_eq!(order, CategoryId::ALL);
        for (id, state) in &states {
            let expected = if *id == CategoryId::ManualChecks {
                CheckStatus::NotAssessed
            } else {
                CheckStatus::Ok
            };
            assert_eq!(state.status, expected, "{id:?}");
        }
        let policy = &states[0].1;
        assert_eq!(
            (policy.attempted, policy.succeeded, policy.failed, policy.skipped),
            (1, 1, 0, 0)
        );
        assert_eq!(state(&clean(), CategoryId::ObservedAccess).attempted, 2);
    }

    #[test]
    fn key_pages_that_could_not_be_checked_count_as_skipped() {
        let mut checks = clean();
        let mut denied = key("https://example.com/zakazano");
        denied.status_code = 403;
        denied.score = None;
        checks.key_pages.push(denied);
        for category in [
            CategoryId::IndexingControls,
            CategoryId::Rendering,
            CategoryId::StructuredData,
            CategoryId::Discovery,
        ] {
            let counted = state(&checks, category);
            assert_eq!(
                (counted.attempted, counted.succeeded, counted.failed, counted.skipped),
                (3, 2, 0, 1),
                "{category:?}: a 403 key page has no HTML to check"
            );
            assert_eq!(counted.status, CheckStatus::Ok);
        }
    }

    #[test]
    fn a_blocked_search_crawler_is_a_problem_with_its_deciding_rule() {
        let mut checks = clean();
        checks.policy = vec![origin_policy(
            "https://example.com",
            &robots("User-agent: OAI-SearchBot\nDisallow: /\n\nUser-agent: PerplexityBot\nDisallow: /sluzby\n"),
            &paths(&["/", "/sluzby"]),
        )];
        let blocked = only(&checks, CategoryId::CrawlerPolicy, "search_crawler_blocked");
        assert_eq!(blocked.status, CheckStatus::Problem);
        assert_eq!(blocked.evidence, Evidence::Strong);
        assert_eq!(blocked.scope, "OpenAI");
        assert_eq!(
            blocked.source,
            ("https://developers.openai.com/api/docs/bots", "2026-09-26")
        );
        assert!(
            blocked
                .args
                .contains(&"User-agent: OAI-SearchBot → Disallow: /".to_string())
        );
        assert_eq!(blocked.pages, ["https://example.com/", "https://example.com/sluzby"]);
        let partly = only(&checks, CategoryId::CrawlerPolicy, "search_crawler_partly_blocked");
        assert_eq!(partly.status, CheckStatus::Problem);
        assert_eq!(partly.scope, "Perplexity");
        assert_eq!(partly.pages, ["https://example.com/sluzby"]);
        assert_eq!(state(&checks, CategoryId::CrawlerPolicy).status, CheckStatus::Problem);

        // Applebot follows the Googlebot group when it has none of its own.
        let policy = origin_policy(
            "https://example.com",
            &robots("User-agent: Googlebot\nDisallow: /sluzby\n"),
            &paths(&["/", "/sluzby"]),
        );
        let applebot = policy
            .agents
            .iter()
            .find(|verdict| verdict.agent.token == "Applebot")
            .unwrap();
        assert_eq!(applebot.access, AgentAccess::Partly(paths(&["/sluzby"])));
        assert_eq!(
            applebot.rule.as_deref(),
            Some("User-agent: Googlebot → Disallow: /sluzby")
        );
    }

    #[test]
    fn user_fetch_training_and_control_tokens_are_weighed_by_purpose_and_compliance() {
        let mut checks = clean();
        checks.policy = vec![origin_policy(
            "https://example.com",
            &robots(
                "User-agent: Claude-User\nDisallow: /\n\nUser-agent: ChatGPT-User\nDisallow: /\n\n\
                 User-agent: GPTBot\nDisallow: /\n\nUser-agent: Google-Extended\nDisallow: /\n\n\
                 Content-Signal: search=yes, ai-train=no\n",
            ),
            &paths(&["/"]),
        )];
        let status_of = |key: &str| only(&checks, CategoryId::CrawlerPolicy, key).status;
        assert_eq!(
            status_of("user_fetch_blocked"),
            CheckStatus::Attention,
            "Claude-User honors robots.txt"
        );
        assert_eq!(
            status_of("user_fetch_blocked_limited"),
            CheckStatus::Info,
            "ChatGPT-User may not"
        );
        assert_eq!(status_of("training_crawler_blocked"), CheckStatus::Info);
        assert_eq!(status_of("grounding_control_blocked"), CheckStatus::Info);
        assert_eq!(status_of("content_signal_present"), CheckStatus::Info);
        assert_eq!(state(&checks, CategoryId::CrawlerPolicy).status, CheckStatus::Attention);
    }

    #[test]
    fn an_unreadable_robots_txt_is_attention_and_a_skipped_one_is_not_assessed() {
        let mut checks = clean();
        checks.policy = vec![origin_policy(
            "https://example.com",
            &RobotsFetchState::Unavailable {
                status_or_error: "HTTP 503".to_string(),
            },
            &paths(&["/"]),
        )];
        let unavailable = only(&checks, CategoryId::CrawlerPolicy, "robots_unavailable");
        assert_eq!(unavailable.status, CheckStatus::Attention);
        assert!(unavailable.args.contains(&"HTTP 503".to_string()));
        assert!(checks.policy[0].agents.is_empty(), "rules unknown: no verdicts");
        let unavailable_state = state(&checks, CategoryId::CrawlerPolicy);
        assert_eq!(
            (unavailable_state.status, unavailable_state.failed),
            (CheckStatus::NotAssessed, 1),
            "nothing succeeded"
        );

        for (fetch, key) in [
            (RobotsFetchState::Skipped, "robots_skipped"),
            (RobotsFetchState::NotAttempted, "robots_not_attempted"),
        ] {
            checks.policy = vec![origin_policy("https://example.com", &fetch, &paths(&["/"]))];
            assert_eq!(only(&checks, CategoryId::CrawlerPolicy, key).status, CheckStatus::Info);
            let skipped = state(&checks, CategoryId::CrawlerPolicy);
            assert_eq!((skipped.status, skipped.skipped), (CheckStatus::NotAssessed, 1));
        }

        // A 404 means no rules: everything is allowed, and the check succeeded.
        checks.policy = vec![origin_policy(
            "https://example.com",
            &RobotsFetchState::NotFound { status: 404 },
            &paths(&["/"]),
        )];
        assert_eq!(state(&checks, CategoryId::CrawlerPolicy).status, CheckStatus::Ok);
        checks.policy = vec![origin_policy(
            "https://example.com",
            &RobotsFetchState::Ok {
                status: 200,
                content: "User-agent: *\nDisallow:\n".to_string(),
                valid_utf8: false,
            },
            &paths(&["/"]),
        )];
        assert_eq!(
            only(&checks, CategoryId::CrawlerPolicy, "robots_not_utf8").status,
            CheckStatus::Info
        );
    }

    #[test]
    fn observed_access_maps_each_kind_to_its_status() {
        let cases = [
            (AccessKind::Denied(403), "access_denied", CheckStatus::Problem),
            (AccessKind::RateLimited, "access_rate_limited", CheckStatus::Attention),
            (
                AccessKind::ServerError(503),
                "access_server_error",
                CheckStatus::Problem,
            ),
            (AccessKind::Transport(-2), "access_failed", CheckStatus::Problem),
            (
                AccessKind::ClientError(404),
                "access_client_error",
                CheckStatus::Problem,
            ),
            (
                AccessKind::RedirectChain(3),
                "access_redirect_chain",
                CheckStatus::Attention,
            ),
            (AccessKind::RedirectLoop, "access_redirect_loop", CheckStatus::Problem),
            (
                AccessKind::SuspectedChallenge,
                "access_suspected_challenge",
                CheckStatus::Attention,
            ),
            (
                AccessKind::SuspectedSoft404,
                "access_suspected_soft_404",
                CheckStatus::Attention,
            ),
            (AccessKind::Slow(4.2), "access_slow", CheckStatus::Info),
        ];
        for (kind, title_key, status) in cases {
            let mut checks = clean();
            checks.access = vec![
                AccessIssue {
                    url: "https://example.com/sluzby".to_string(),
                    kind: kind.clone(),
                    detail: String::new(),
                },
                AccessIssue {
                    url: HOME.to_string(),
                    kind,
                    detail: String::new(),
                },
            ];
            let issue = only(&checks, CategoryId::ObservedAccess, title_key);
            assert_eq!(issue.status, status, "{title_key}");
            assert_eq!(issue.evidence, Evidence::Strong);
            assert_eq!(
                issue.pages,
                ["https://example.com/sluzby", HOME],
                "one issue per kind, key-page order"
            );
            assert_eq!(state(&checks, CategoryId::ObservedAccess).status, status);
        }

        let mut checks = clean();
        checks.access_stats.other_urls = 40;
        checks.access_stats.other_issues.insert("server_error", 2);
        checks.access_stats.other_issues.insert("denied", 1);
        let other = only(&checks, CategoryId::ObservedAccess, "other_urls_issues");
        assert_eq!(
            other.status,
            CheckStatus::Info,
            "pages outside the key pages only count"
        );
        assert_eq!(other.args, ["3", "40"]);
        assert!(other.pages.is_empty());

        checks.key_pages.clear();
        assert_eq!(
            state(&checks, CategoryId::ObservedAccess).status,
            CheckStatus::NotAssessed
        );
    }

    #[test]
    fn snippet_controls_map_per_engine() {
        let page = "https://example.com/sluzby";
        let with = |google: EnginePolicy, bing: EnginePolicy, share: Option<f64>, canonical: Option<&str>| {
            let mut checks = clean();
            checks.controls[1] = PageControls {
                url: page.to_string(),
                google,
                bing,
                nosnippet_share: share,
                canonical_elsewhere: canonical.map(str::to_string),
            };
            checks
        };
        let noindex = EnginePolicy {
            noindex: true,
            ..EnginePolicy::default()
        };
        let nosnippet = EnginePolicy {
            nosnippet: true,
            ..EnginePolicy::default()
        };
        let zero = EnginePolicy {
            max_snippet: Some(0),
            ..EnginePolicy::default()
        };
        let short = EnginePolicy {
            max_snippet: Some(50),
            ..EnginePolicy::default()
        };
        let long = EnginePolicy {
            max_snippet: Some(51),
            ..EnginePolicy::default()
        };
        let none = EnginePolicy::default;
        let problem = |checks: &Checks, key: &str, scope: &str| {
            let issue = only(checks, CategoryId::IndexingControls, key);
            assert_eq!((issue.status, issue.scope), (CheckStatus::Problem, scope), "{key}");
            assert_eq!(issue.pages, [page]);
        };
        problem(&with(noindex.clone(), none(), None, None), "google_noindex", "Google");
        problem(
            &with(nosnippet.clone(), none(), None, None),
            "google_nosnippet",
            "Google",
        );
        problem(
            &with(zero.clone(), none(), None, None),
            "google_max_snippet_zero",
            "Google",
        );
        problem(&with(none(), noindex, None, None), "bing_noindex", "Bing");
        problem(&with(none(), nosnippet, None, None), "bing_nosnippet", "Bing");
        problem(&with(none(), zero, None, None), "bing_max_snippet_zero", "Bing");

        let attention = |checks: &Checks, key: &str| {
            let issue = only(checks, CategoryId::IndexingControls, key);
            assert_eq!(issue.status, CheckStatus::Attention, "{key}");
            issue
        };
        for bing in [
            EnginePolicy {
                noarchive: true,
                ..EnginePolicy::default()
            },
            EnginePolicy {
                nocache: true,
                ..EnginePolicy::default()
            },
        ] {
            assert_eq!(
                attention(&with(none(), bing, None, None), "bing_copilot_restricted").scope,
                "Bing"
            );
        }
        assert_eq!(
            attention(&with(short.clone(), none(), None, None), "short_max_snippet").scope,
            "Google"
        );
        assert_eq!(
            attention(&with(none(), short, None, None), "short_max_snippet").scope,
            "Bing"
        );
        assert!(issues(&with(long, none(), None, None), CategoryId::IndexingControls).is_empty());
        attention(&with(none(), none(), Some(0.3), None), "data_nosnippet_large");
        assert!(issues(&with(none(), none(), Some(0.29), None), CategoryId::IndexingControls).is_empty());
        let canonical = attention(
            &with(none(), none(), None, Some("https://example.com/jinde")),
            "canonical_elsewhere",
        );
        assert!(canonical.args.contains(&"https://example.com/jinde".to_string()));

        let mut checks = clean();
        checks.pdfs = vec![PdfRestriction {
            url: "https://example.com/cenik.pdf".to_string(),
            google: EnginePolicy {
                noindex: true,
                ..EnginePolicy::default()
            },
            bing: EnginePolicy::default(),
        }];
        let pdf = only(&checks, CategoryId::IndexingControls, "pdf_restricted");
        assert_eq!(pdf.status, CheckStatus::Info);
        assert_eq!(pdf.pages, ["https://example.com/cenik.pdf"]);
    }

    #[test]
    fn a_likely_client_rendered_key_page_is_a_moderate_rendering_risk() {
        let mut checks = clean();
        checks.render.risks.push(RenderRisk {
            url: HOME.to_string(),
            main_text_chars: 12,
            rendered_text_chars: None,
            markers: vec!["div#root"],
        });
        let risk = only(&checks, CategoryId::Rendering, "likely_client_rendered");
        assert_eq!(
            (risk.status, risk.evidence),
            (CheckStatus::Attention, Evidence::Moderate)
        );
        assert_eq!(risk.scope, "OpenAI, Anthropic, Perplexity");
        checks.render.checked = 0;
        checks.render.risks.clear();
        assert_eq!(state(&checks, CategoryId::Rendering).status, CheckStatus::NotAssessed);
    }

    #[test]
    fn text_that_appears_only_after_rendering_is_a_moderate_rendering_risk() {
        let mut checks = clean();
        checks.render.browser = true;
        checks.render.risks.push(RenderRisk {
            url: HOME.to_string(),
            main_text_chars: 40,
            rendered_text_chars: Some(3_000),
            markers: Vec::new(),
        });
        let risk = only(&checks, CategoryId::Rendering, "text_after_rendering");
        assert_eq!(
            (risk.status, risk.evidence),
            (CheckStatus::Attention, Evidence::Moderate)
        );
        assert_eq!(risk.scope, "OpenAI, Anthropic, Perplexity");
        assert_eq!(risk.pages, [HOME]);

        // Rendered pages that cannot be compared leave the category not assessed, and say why.
        checks.render.risks.clear();
        checks.render.checked = 0;
        checks.render.not_comparable = vec![HOME.to_string()];
        let rendering = state(&checks, CategoryId::Rendering);
        assert_eq!(rendering.status, CheckStatus::NotAssessed);
        assert_eq!(rendering.reason, "render_not_comparable");
    }

    #[test]
    fn structured_data_rules() {
        let mut checks = clean();
        checks.markup[1].markup.parse_errors = vec!["JSON-LD block 1: EOF".to_string()];
        checks.markup[1].markup.invisible_values = vec![("Product.name".to_string(), "Tajný".to_string())];
        checks.markup[1].markup.jsonld_types = vec!["FAQPage".to_string()];
        checks.markup[1].markup.values_not_checked = 5;
        let error = only(&checks, CategoryId::StructuredData, "jsonld_parse_error");
        assert_eq!((error.status, error.evidence), (CheckStatus::Problem, Evidence::Strong));
        let invisible = only(&checks, CategoryId::StructuredData, "jsonld_values_not_visible");
        assert_eq!(
            (invisible.status, invisible.evidence),
            (CheckStatus::Attention, Evidence::Strong)
        );
        assert_eq!(
            only(&checks, CategoryId::StructuredData, "faq_markup_present").status,
            CheckStatus::Info
        );
        assert_eq!(
            only(&checks, CategoryId::StructuredData, "jsonld_values_not_checked").status,
            CheckStatus::Info
        );

        let mut bare = clean();
        bare.markup[0].markup.jsonld_types = vec!["WebSite".to_string()];
        let site = only(&bare, CategoryId::StructuredData, "no_site_markup");
        assert_eq!((site.status, site.evidence), (CheckStatus::Info, Evidence::Weak));
        assert_eq!(site.args, ["Organization"]);
        // Microdata counts as existing markup.
        bare.markup[0].markup.other_types = vec!["Organization".to_string()];
        assert!(issues(&bare, CategoryId::StructuredData).is_empty());
        // So does a subtype of Organization.
        for kind in ["LocalBusiness", "Corporation", "Restaurant", "NGO"] {
            bare.markup[0].markup.other_types = vec![kind.to_string()];
            assert!(issues(&bare, CategoryId::StructuredData).is_empty(), "{kind}");
        }
    }

    #[test]
    fn discovery_rules() {
        let mut checks = clean();
        checks.discovery.missing_from_sitemaps = vec!["https://example.com/sluzby".to_string()];
        checks.discovery.listed_issues = vec![ListedIssue {
            url: "https://example.com/stara".to_string(),
            reason: ListedReason::Status(301),
        }];
        checks.discovery.hreflang_issues = vec![
            HreflangIssue {
                page: HOME.to_string(),
                lang: "en".to_string(),
                target: "https://example.com/en/".to_string(),
                problem: HreflangProblem::Status(404),
            },
            HreflangIssue {
                page: HOME.to_string(),
                lang: "de".to_string(),
                target: "https://example.de/".to_string(),
                problem: HreflangProblem::OtherSite,
            },
        ];
        checks.discovery.sitemaps[0].state = SitemapState::Parsed {
            kind: SitemapKind::UrlSet,
            entries: 2,
            lastmod: LastmodStats {
                entries: 3,
                with_lastmod: 3,
                all_identical: true,
                ..LastmodStats::default()
            },
        };
        checks.discovery.last_modified.plausible = 0;
        let status_of = |key: &str| {
            let issue = only(&checks, CategoryId::Discovery, key);
            assert_eq!(issue.evidence, Evidence::Moderate, "{key}");
            issue.status
        };
        assert_eq!(status_of("missing_from_sitemap"), CheckStatus::Attention);
        assert_eq!(status_of("sitemap_lists_non_indexable"), CheckStatus::Attention);
        assert_eq!(status_of("hreflang_target_broken"), CheckStatus::Attention);
        assert_eq!(status_of("suspicious_lastmod"), CheckStatus::Attention);
        assert_eq!(status_of("last_modified_missing"), CheckStatus::Info);
        assert_eq!(
            issues(&checks, CategoryId::Discovery).len(),
            5,
            "another site's hreflang is not checked"
        );

        let mut none = clean();
        none.discovery.sitemaps.clear();
        none.discovery.key_pages_compared = 0;
        assert_eq!(
            only(&none, CategoryId::Discovery, "no_sitemap").status,
            CheckStatus::Info
        );

        // Every sitemap was read, but no key page is indexable: nothing to compare, nothing missing.
        let mut noindex = clean();
        noindex.discovery.key_pages_compared = 0;
        assert!(issues(&noindex, CategoryId::Discovery).is_empty());

        let mut partial = clean();
        partial.discovery.sitemaps.push(SitemapFile {
            url: "https://example.com/sitemap-2.xml".to_string(),
            declared: true,
            state: SitemapState::Failed(500),
        });
        partial.discovery.key_pages_compared = 0;
        assert_eq!(
            only(&partial, CategoryId::Discovery, "sitemap_unreadable").status,
            CheckStatus::Attention
        );
        assert_eq!(
            only(&partial, CategoryId::Discovery, "sitemap_comparison_not_assessed").status,
            CheckStatus::Info
        );
    }

    #[test]
    fn chosen_pages_without_text_count_as_skipped() {
        let mut checks = clean();
        checks.analysis.without_text = vec!["https://example.com/app".to_string()];
        for category in [CategoryId::AnswerExtractability, CategoryId::EntityClarity] {
            let counted = state(&checks, category);
            assert_eq!(
                (counted.attempted, counted.succeeded, counted.failed, counted.skipped),
                (2, 2, 0, 1),
                "{category:?}"
            );
        }
        checks.analysis.pages.clear();
        checks.analysis.unavailable = Some("no model configured".to_string());
        assert_eq!(
            state(&checks, CategoryId::AnswerExtractability).skipped,
            3,
            "none analyzed"
        );
    }

    #[test]
    fn the_llm_categories_are_not_assessed_without_ai_and_ok_only_after_successes() {
        let mut checks = clean();
        checks.analysis.pages.clear();
        checks.analysis.signals.clear();
        checks.analysis.unavailable = Some("no model configured".to_string());
        for category in [CategoryId::AnswerExtractability, CategoryId::EntityClarity] {
            let not_assessed = state(&checks, category);
            assert_eq!(not_assessed.status, CheckStatus::NotAssessed);
            assert_eq!((not_assessed.attempted, not_assessed.skipped), (0, 2));
            assert_eq!(not_assessed.reason, "ai_unavailable");
        }
        assert_eq!(
            state(&checks, CategoryId::CrawlerPolicy).status,
            CheckStatus::Ok,
            "the rest still runs"
        );

        // Every page failed.
        checks.analysis.unavailable = None;
        checks.analysis.failed = vec![
            (HOME.to_string(), "timeout".to_string()),
            ("https://example.com/sluzby".to_string(), "timeout".to_string()),
        ];
        let failed = state(&checks, CategoryId::AnswerExtractability);
        assert_eq!(
            (failed.status, failed.attempted, failed.succeeded, failed.failed),
            (CheckStatus::NotAssessed, 2, 0, 2)
        );

        // Nothing was selected for analysis.
        let mut empty = clean();
        empty.analysis = AnalysisRun::default();
        assert_eq!(
            state(&empty, CategoryId::EntityClarity).status,
            CheckStatus::NotAssessed
        );

        let mut half = clean();
        half.analysis.pages.truncate(1);
        half.analysis.failed = vec![("https://example.com/sluzby".to_string(), "timeout".to_string())];
        let partly = state(&half, CategoryId::AnswerExtractability);
        assert_eq!(
            (partly.status, partly.succeeded, partly.failed),
            (CheckStatus::Ok, 1, 1)
        );
    }

    #[test]
    fn answers_the_crawler_cannot_verify_leave_extractability_not_assessed() {
        let unverified = Question {
            question: "Kolik to stojí?".to_string(),
            answered: Answered::Partly,
            excerpts: Vec::new(),
        };
        let mut checks = clean();
        for page in &mut checks.analysis.pages {
            page.questions = vec![unverified.clone()];
        }
        let state_of = |checks: &Checks| state(checks, CategoryId::AnswerExtractability);
        let unassessed = state_of(&checks);
        assert_eq!(
            (
                unassessed.status,
                unassessed.attempted,
                unassessed.succeeded,
                unassessed.failed
            ),
            (CheckStatus::NotAssessed, 2, 0, 2)
        );
        assert_eq!(unassessed.reason, "no_verified_answers");
        assert_eq!(
            state(&checks, CategoryId::EntityClarity).status,
            CheckStatus::Ok,
            "the rest of the answer was checked"
        );
        for page in &mut checks.analysis.pages {
            page.questions.clear();
        }
        assert_eq!(
            state_of(&checks).status,
            CheckStatus::NotAssessed,
            "no questions at all"
        );

        // One page with a verified answer (or a "no") is enough for a partial assessment.
        checks.analysis.pages[0].questions = vec![Question {
            question: "Jak dlouho to trvá?".to_string(),
            answered: Answered::No,
            excerpts: Vec::new(),
        }];
        let partial = state_of(&checks);
        assert_eq!((partial.succeeded, partial.failed), (1, 1));
        assert_ne!(partial.status, CheckStatus::NotAssessed);
    }

    #[test]
    fn llm_findings_map_to_extractability_and_entity_clarity() {
        let mut checks = clean();
        let excerpt = Excerpt {
            block: "B3".to_string(),
            text: "Tento produkt je skvělý.".to_string(),
        };
        let page = &mut checks.analysis.pages[1];
        page.states_offer_early = OfferEarly::No;
        page.questions = vec![Question {
            question: "Kolik to stojí?".to_string(),
            answered: Answered::No,
            excerpts: Vec::new(),
        }];
        page.vague_references = vec![excerpt];
        let mut article = analysis(HOME, PageType::Article);
        article.byline = Byline::default();
        checks.analysis.pages[0] = article;
        checks.analysis.signals[1].1.collapsed_chars = 500;
        checks.analysis.signals[1].1.in_content_links = 0;
        checks.analysis.signals[1].1.images_without_alt = 2;

        let moderate = |category: CategoryId, key: &str| {
            let issue = only(&checks, category, key);
            assert_eq!(issue.evidence, Evidence::Moderate, "{key}");
            issue.status
        };
        assert_eq!(
            moderate(CategoryId::AnswerExtractability, "offer_not_early"),
            CheckStatus::Attention
        );
        assert_eq!(
            moderate(CategoryId::AnswerExtractability, "hidden_content"),
            CheckStatus::Attention
        );
        assert_eq!(
            moderate(CategoryId::AnswerExtractability, "unanswered_questions"),
            CheckStatus::Info
        );
        assert_eq!(
            moderate(CategoryId::AnswerExtractability, "no_contextual_links"),
            CheckStatus::Info
        );
        assert_eq!(
            moderate(CategoryId::AnswerExtractability, "images_without_alt"),
            CheckStatus::Info
        );
        assert_eq!(
            moderate(CategoryId::EntityClarity, "vague_references"),
            CheckStatus::Info
        );
        assert_eq!(
            moderate(CategoryId::EntityClarity, "article_without_byline"),
            CheckStatus::Info
        );

        // Where "states its offer early" does not apply, a "no" is not reported.
        let mut listing = clean();
        listing.analysis.pages[1].page_type = PageType::Category;
        listing.analysis.pages[1].states_offer_early = OfferEarly::No;
        assert!(issues(&listing, CategoryId::AnswerExtractability).is_empty());

        // A byline with an author and a date is enough.
        let mut signed = clean();
        let mut article = analysis(HOME, PageType::Article);
        let block = Block {
            id: 2,
            region: Region::Main,
            kind: BlockKind::Paragraph,
            heading_path: Vec::new(),
            text: "Jan Novák".to_string(),
            collapsed: false,
            hidden: false,
            landmark: 0,
        };
        article.byline = Byline {
            author: Some(block.clone()),
            date_block: Some(block),
            date: chrono::NaiveDate::from_ymd_opt(2026, 9, 25),
            date_role: Some(crate::ai::geo::analyze::DateRole::Published),
            ..Byline::default()
        };
        signed.analysis.pages[0] = article;
        assert!(issues(&signed, CategoryId::EntityClarity).is_empty());
    }

    #[test]
    fn deterministic_verdicts_do_not_change_with_the_number_of_analyzed_pages() {
        let mut few = clean();
        few.access = vec![AccessIssue {
            url: "https://example.com/sluzby".to_string(),
            kind: AccessKind::Denied(403),
            detail: String::new(),
        }];
        few.markup[1].markup.parse_errors = vec!["JSON-LD block 1: EOF".to_string()];
        let mut many = few.clone();
        few.analysis.selected = 10;
        many.analysis.selected = 100;
        many.analysis.pages = (0..100)
            .map(|i| {
                let mut page = analysis(&format!("https://example.com/p{i}"), PageType::Service);
                page.states_offer_early = OfferEarly::No;
                page
            })
            .collect();
        let deterministic = |checks: &Checks| {
            let (states, found) = assess(checks);
            let states: Vec<(CategoryId, CheckStatus)> = states
                .into_iter()
                .filter(|(id, _)| !matches!(id, CategoryId::AnswerExtractability | CategoryId::EntityClarity))
                .map(|(id, state)| (id, state.status))
                .collect();
            let found: Vec<Issue> = found
                .into_iter()
                .filter(|issue| {
                    !matches!(
                        issue.category,
                        CategoryId::AnswerExtractability | CategoryId::EntityClarity
                    )
                })
                .collect();
            (states, found)
        };
        assert_eq!(deterministic(&few), deterministic(&many));
    }

    #[test]
    fn priorities_are_ordered_by_status_evidence_category_and_pages_and_capped_at_ten() {
        let issue = |id: &str, category: CategoryId, status: CheckStatus, evidence: Evidence, pages: usize| Issue {
            id: id.to_string(),
            category,
            status,
            evidence,
            scope: "All",
            source: SRC_GOOGLE_AI_FEATURES,
            title_key: "access_denied",
            args: Vec::new(),
            pages: (0..pages).map(|i| format!("https://example.com/{i}")).collect(),
        };
        let mut all = vec![
            issue(
                "info",
                CategoryId::CrawlerPolicy,
                CheckStatus::Info,
                Evidence::Strong,
                9,
            ),
            issue(
                "attention-weak",
                CategoryId::CrawlerPolicy,
                CheckStatus::Attention,
                Evidence::Weak,
                9,
            ),
            issue(
                "attention-strong-late",
                CategoryId::Discovery,
                CheckStatus::Attention,
                Evidence::Strong,
                1,
            ),
            issue(
                "attention-strong-early",
                CategoryId::CrawlerPolicy,
                CheckStatus::Attention,
                Evidence::Strong,
                1,
            ),
            issue(
                "problem-few",
                CategoryId::ObservedAccess,
                CheckStatus::Problem,
                Evidence::Strong,
                1,
            ),
            issue(
                "problem-many",
                CategoryId::ObservedAccess,
                CheckStatus::Problem,
                Evidence::Strong,
                5,
            ),
            issue(
                "problem-moderate",
                CategoryId::CrawlerPolicy,
                CheckStatus::Problem,
                Evidence::Moderate,
                9,
            ),
        ];
        let order: Vec<&str> = priorities(&all).iter().map(|issue| issue.id.as_str()).collect();
        assert_eq!(
            order,
            [
                "problem-many",
                "problem-few",
                "problem-moderate",
                "attention-strong-early",
                "attention-strong-late",
                "attention-weak",
                "info"
            ]
        );
        for i in 0..20 {
            all.push(issue(
                &format!("more-{i:02}"),
                CategoryId::Rendering,
                CheckStatus::Attention,
                Evidence::Moderate,
                1,
            ));
        }
        assert_eq!(priorities(&all).len(), MAX_PRIORITIES);
        assert_eq!(MAX_PRIORITIES, 10);
    }

    #[test]
    fn every_issue_carries_a_scope_a_dated_source_and_a_known_title() {
        let mut checks = clean();
        checks.policy = vec![origin_policy(
            "https://example.com",
            &robots("User-agent: *\nDisallow: /\n\nContent-Signal: ai-train=no\n"),
            &paths(&["/", "/sluzby"]),
        )];
        checks.access = vec![AccessIssue {
            url: HOME.to_string(),
            kind: AccessKind::Transport(-1),
            detail: String::new(),
        }];
        checks.controls[0].google.noindex = true;
        checks.controls[0].bing.nocache = true;
        checks.render.risks.push(RenderRisk {
            url: HOME.to_string(),
            main_text_chars: 0,
            rendered_text_chars: None,
            markers: vec!["div#app"],
        });
        checks.markup[0].markup = ExistingMarkup::default();
        checks.discovery.sitemaps.clear();
        checks.analysis.pages[1].states_offer_early = OfferEarly::No;
        let (_, found) = assess(&checks);
        assert!(found.len() >= 12, "{found:?}");
        let mut ids: Vec<&str> = found.iter().map(|issue| issue.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), found.len(), "ids are unique");
        for issue in &found {
            assert!(!issue.scope.trim().is_empty(), "{}", issue.id);
            assert!(issue.source.0.starts_with("https://"), "{}", issue.id);
            assert!(issue.source.1.starts_with("20"), "{}: {}", issue.id, issue.source.1);
            assert!(TITLE_KEYS.contains(&issue.title_key), "{}", issue.title_key);
            assert!(
                matches!(
                    issue.status,
                    CheckStatus::Problem | CheckStatus::Attention | CheckStatus::Info
                ),
                "{}",
                issue.id
            );
        }
    }
}
