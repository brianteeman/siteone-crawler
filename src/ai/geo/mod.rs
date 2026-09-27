// SiteOne Crawler - AI search readiness (--ai-geo)
// (c) Jan Reges <jan.reges@siteone.cz>
//
// How well AI search and answer engines can reach, read, understand and quote a site.
// See docs/superpowers/plans/2026-09-26-ai-search-readiness-geo.md.
//
// `run` drives the checks after the crawl:
//   deterministic — the key pages, the robots.txt verdicts per origin, the observed access, the
//                   snippet controls, the rendering risk, the existing structured data, the
//                   discovery signals (they need no model and run even without a working AI
//                   configuration);
//   per page      — one LLM call per chosen page (`--ai-max-pages`), answered with block ids and
//                   verified against the crawler's own blocks;
//   kit           — JSON-LD built by the crawler from verified blocks;
//   doc           — the findings, the priorities and the report.

pub mod access;
pub mod agents;
pub mod analyze;
pub mod controls;
pub mod discovery;
pub mod doc;
pub mod findings;
pub mod jsonld;
pub mod keys;
pub mod kit;
pub mod prompts;
pub mod render;
pub mod robots_ai;
pub mod signals;
#[cfg(test)]
pub(crate) mod test_support;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use scraper::{Html, Selector};
use tokio::sync::Semaphore;

use crate::ai::blocks::{Block, Region, blocks_from_html};
use crate::ai::client::AiClient;
use crate::ai::config::build_config;
use crate::ai::profile::budget::ContextBudget;
use crate::ai::progress;
use crate::ai::provider::ChatRequest;
use crate::ai::report::locale::ReportLocale;
use crate::ai::selection::build_candidates;
use crate::options::core_options::CoreOptions;
use crate::output::output::Output;
use crate::result::status::{RobotsFetchState, Status};
use crate::types::ContentTypeId;
use crate::utils;

use self::access::observed_access;
use self::analyze::{
    AnalyzedPage, CAT_PAGE, Coverage, RawAnalysis, analyzed_page, build_page_request, has_offered_blocks,
    parse_analysis, signals_text, verify_analysis,
};
use self::controls::{bing_policy, canonical_elsewhere, data_nosnippet_chars, google_policy, sources};
use self::discovery::{CrawlEnd, CrawlScope, discovery, sitemap_proposal};
use self::doc::{GeoDoc, GeoMeta, text};
use self::findings::{AnalysisRun, CheckStatus, Checks, OriginPolicy, PageControls, PageMarkup, origin_policy};
use self::jsonld::{ExistingMarkup, page_name};
use self::keys::{KeyPage, key_pages, normalized_url};
use self::kit::{MarkupPage, SiteMarkup, markup_entries};
use self::render::{browser_render_risks, plain_render_risks};
use self::robots_ai::paths_by_origin;
use self::signals::{PageSignals, main_text_chars, page_signals, pdf_restrictions};

const TASK_PAGES: &str = "geo:pages";
/// Label of the `issue` event (kind `ai`) when a part of the check fails.
const GEO_FAILED: &str = "AI search readiness failed";
const PARSE_ATTEMPTS: u32 = 2;
/// The error text of an answer cut at the output-token limit (see `AiClient::complete_parsed_n`).
const TRUNCATED: &str = "generation stopped at token limit";
/// Dry-run estimates: bytes per token of the input, and the output tokens of one page analysis.
const EST_BYTES_PER_TOKEN: f64 = 2.5;
const EST_PAGE_OUT: (usize, usize) = (400, 1_500);
/// A homepage `og:site_name` longer than this is not taken for the site's name.
const MAX_SITE_NAME_CHARS: usize = 60;
/// At most this many origins get their robots.txt fetched after the crawl.
const MAX_ROBOTS_FETCHES: usize = 10;

static OG_SITE_NAME_SELECTOR: Lazy<Selector> =
    Lazy::new(|| Selector::parse(r#"meta[property="og:site_name"][content]"#).unwrap());
static TITLE_SELECTOR: Lazy<Selector> = Lazy::new(|| Selector::parse("title").unwrap());

/// The input and output budget of a page analysis (Design §8), from `--ai-context-window` and
/// `--ai-max-tokens`; the input budget is bytes of the final escaped user message.
struct Budgets {
    input_bytes: usize,
    max_tokens: u32,
}

impl Budgets {
    fn new(b: &ContextBudget) -> Self {
        Self {
            input_bytes: b.scaled(12, 3),
            max_tokens: b.out_tokens().min(4_000),
        }
    }
}

/// A page chosen for the per-page analysis, with what its prompt and its kit markup need.
struct Chosen {
    page: AnalyzedPage,
    blocks: Vec<Block>,
    signals: PageSignals,
    /// The `<signals>` line of the prompt.
    signals_line: String,
    is_homepage: bool,
    existing: ExistingMarkup,
    /// No `noindex` for Google or Bing and no canonical URL elsewhere: the kit builds markup for it.
    indexable: bool,
}

/// The homepage as the site-wide kit markup needs it.
struct Homepage {
    url: String,
    origin: String,
    existing: ExistingMarkup,
    /// The kit's Organization, and the social profiles of the site chrome it does not claim.
    organization: (serde_json::Value, Vec<String>),
}

/// What the crawl gives the checks: the deterministic results and the pages to analyze.
struct Prepared {
    checks: Checks,
    chosen: Vec<Chosen>,
    /// Chosen pages without any text of their own, which are not analyzed.
    without_text: Vec<String>,
    homepage: Option<Homepage>,
    homepage_url: String,
    site_name: String,
    /// URL as crawled → page name, for the breadcrumbs of the chosen pages.
    crawled_names: HashMap<String, String>,
    crawled_pages: usize,
}

/// The first `max` of the `ranked` pages, with the homepage always among them: when it ranks
/// lower, it replaces the last one and comes first.
fn with_homepage<T>(ranked: &[T], max: usize, is_homepage: impl Fn(&T) -> bool) -> Vec<&T> {
    let mut chosen: Vec<&T> = ranked.iter().take(max).collect();
    if !chosen.iter().any(|page| is_homepage(page))
        && let Some(homepage) = ranked.iter().find(|page| is_homepage(page))
    {
        chosen.truncate(max.saturating_sub(1));
        chosen.insert(0, homepage);
    }
    chosen
}

/// The origins `(scheme, host, port)` of the key pages whose robots.txt the crawl did not fetch,
/// homepage first, at most `MAX_ROBOTS_FETCHES`. The crawler reads the robots.txt of the initial
/// URL only, while the homepage may redirect to its `www` or `https` variant, where the key pages
/// then are; the manager fetches these before `run`, so each origin is judged by its own file.
pub fn robots_origins_to_fetch(status: &Status, include: &[String], exclude: &[String]) -> Vec<(String, String, u16)> {
    let mut origins: Vec<(String, String, u16)> = Vec::new();
    for page in key_pages(status, include, exclude) {
        let Ok(url) = url::Url::parse(&page.url) else {
            continue;
        };
        let (Some(host), Some(port)) = (url.host_str(), url.port_or_known_default()) else {
            continue;
        };
        let origin = (url.scheme().to_string(), host.to_string(), port);
        if origins.contains(&origin)
            || status.get_robots_txt_state(&origin.0, &origin.1, port) != RobotsFetchState::NotAttempted
        {
            continue;
        }
        origins.push(origin);
        if origins.len() == MAX_ROBOTS_FETCHES {
            break;
        }
    }
    origins
}

/// `scheme://host[:port]` as `url::Url::origin` writes it (no default port).
fn origin_of(scheme: &str, host: &str, port: u16) -> String {
    url::Url::parse(&format!("{scheme}://{host}:{port}/"))
        .map(|url| url.origin().ascii_serialization())
        .unwrap_or_else(|_| format!("{scheme}://{host}:{port}"))
}

/// The snippet controls and the existing structured data of a page.
fn page_checks(
    url: &str,
    html: &str,
    headers: Option<&HashMap<String, String>>,
    now: DateTime<Utc>,
) -> (Vec<Block>, PageControls, ExistingMarkup) {
    let document = Html::parse_document(html);
    let blocks = blocks_from_html(html);
    let found = sources(&document, headers);
    let (main_chars, _) = main_text_chars(&blocks);
    let nosnippet = data_nosnippet_chars(&document);
    let controls = PageControls {
        url: url.to_string(),
        google: google_policy(&found, now),
        bing: bing_policy(&found, now),
        nosnippet_share: (main_chars > 0).then(|| nosnippet as f64 / main_chars as f64),
        canonical_elsewhere: canonical_elsewhere(&document, url),
    };
    // Markup must match what the page shows: text hidden for good does not count.
    let visible: Vec<&str> = blocks
        .iter()
        .filter(|block| !block.hidden)
        .map(|block| block.text.as_str())
        .collect();
    let markup = jsonld::existing(&document, &visible.join("\n"));
    (blocks, controls, markup)
}

/// The site's name: the homepage's `og:site_name` when it is short enough to be a name,
/// otherwise the brand segment of its title (the profile's heuristic), otherwise the host.
fn site_name(document: &Html, host: &str) -> String {
    let og = document
        .select(&OG_SITE_NAME_SELECTOR)
        .next()
        .and_then(|meta| meta.value().attr("content"))
        .map(|name| name.split_whitespace().collect::<Vec<_>>().join(" "))
        .unwrap_or_default();
    if !og.is_empty() && og.chars().count() <= MAX_SITE_NAME_CHARS {
        return og;
    }
    let title = document
        .select(&TITLE_SELECTOR)
        .next()
        .map(|title| title.text().collect::<Vec<_>>().join(" "))
        .unwrap_or_default();
    crate::ai::profile::subject_name_from(&title.split_whitespace().collect::<Vec<_>>().join(" "), host)
}

/// The normalized URLs a breadcrumb of `page_url` may use: the page itself and every ancestor
/// path, with and without a trailing slash (see `jsonld::breadcrumb`).
fn breadcrumb_levels(page_url: &str, into: &mut HashSet<String>) {
    into.insert(normalized_url(page_url));
    let Ok(page) = url::Url::parse(page_url) else {
        return;
    };
    let origin = &page[..url::Position::BeforePath];
    into.insert(normalized_url(&format!("{origin}/")));
    let parts: Vec<&str> = page
        .path_segments()
        .map(|segments| segments.filter(|segment| !segment.is_empty()).collect())
        .unwrap_or_default();
    for depth in 1..parts.len() {
        let path = format!("{origin}/{}", parts[..depth].join("/"));
        into.insert(normalized_url(&format!("{path}/")));
        into.insert(normalized_url(&path));
    }
}

/// The deterministic checks over the crawl, and the pages chosen for the analysis with their
/// blocks, signals and markup. Reads `status` only; `crawl_end` and `robots_skipped` (the URLs
/// robots.txt kept the crawler from) say whether the crawl covered the whole site (for the
/// sitemap proposal).
fn prepare(
    options: &CoreOptions,
    status: &Status,
    now: DateTime<Utc>,
    crawl_end: CrawlEnd,
    robots_skipped: &[String],
) -> Prepared {
    let key: Vec<KeyPage> = key_pages(status, &options.ai_include, &options.ai_exclude);
    let key_urls: Vec<String> = key.iter().map(|page| page.url.clone()).collect();
    let policy: Vec<OriginPolicy> = paths_by_origin(&key_urls)
        .into_iter()
        .map(|((scheme, host, port), paths)| {
            let state = status.get_robots_txt_state(&scheme, &host, port);
            origin_policy(&origin_of(&scheme, &host, port), &state, &paths)
        })
        .collect();
    let (access, access_stats) = observed_access(status, &key);
    // With --browser the stored bodies are the rendered pages: compare them with their HTML.
    let render = if options.browser_enabled {
        browser_render_risks(status, &key)
    } else {
        plain_render_risks(status, &key)
    };
    let declared: Vec<String> = policy
        .iter()
        .flat_map(|origin| origin.sitemaps.iter().cloned())
        .collect();
    let mut discovery = discovery(status, &key, &declared, now);
    let pdfs = pdf_restrictions(status, now);

    // Controls and markup of every HTML 200 key page (a key page answering 200 is HTML).
    let mut checked: HashMap<String, (Vec<Block>, PageControls, ExistingMarkup)> = HashMap::new();
    let mut controls = Vec::new();
    let mut markup = Vec::new();
    for page in key.iter().filter(|page| page.status_code == 200) {
        let Some(html) = status.get_url_body_text(&page.uq_id) else {
            continue;
        };
        let headers = status.get_url_headers(&page.uq_id);
        let (blocks, page_controls, page_markup) = page_checks(&page.url, &html, headers.as_ref(), now);
        controls.push(page_controls.clone());
        markup.push(PageMarkup {
            url: page.url.clone(),
            is_homepage: page.is_homepage,
            markup: page_markup.clone(),
        });
        checked.insert(page.url.clone(), (blocks, page_controls, page_markup));
    }

    // The homepage (the initial URL or where its redirects lead), its name and its Organization.
    let homepage_urls: HashSet<&str> = key
        .iter()
        .filter(|page| page.is_homepage)
        .map(|page| page.url.as_str())
        .collect();
    let homepage_page = key
        .iter()
        .find(|page| page.is_homepage && page.status_code == 200 && checked.contains_key(&page.url));
    let homepage_url = homepage_page.map_or_else(|| options.url.clone(), |page| page.url.clone());
    let host = url::Url::parse(&homepage_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_string))
        .unwrap_or_else(|| options.get_initial_host(false));
    // A site without a sitemap gets one proposed when the crawl covered all of it.
    let scope = CrawlScope {
        start_url: utils::redact_url_userinfo(&options.url),
        single_page: options.single_page,
        end: crawl_end,
        max_visited_urls: options.max_visited_urls,
        max_depth: options.max_depth,
        url_filters: !options.include_regex.is_empty() || !options.ignore_regex.is_empty(),
        robots_skipped: robots_skipped.to_vec(),
    };
    let homepage_robots = url::Url::parse(&homepage_url)
        .ok()
        .and_then(|url| {
            let port = url.port_or_known_default()?;
            Some(status.get_robots_txt_state(url.scheme(), url.host_str()?, port))
        })
        .unwrap_or(RobotsFetchState::NotAttempted);
    match sitemap_proposal(status, &discovery, &homepage_url, &homepage_robots, &scope, now) {
        Ok(proposal) => discovery.sitemap_proposal = Some(proposal),
        Err("sitemap_exists") => {}
        Err(why) => discovery.proposal_withheld = Some(why),
    }

    let mut site = host.clone();
    let homepage = homepage_page.and_then(|page| {
        let html = status.get_url_body_text(&page.uq_id)?;
        let document = Html::parse_document(&html);
        site = site_name(&document, &host);
        let base = url::Url::parse(&page.url).ok()?;
        let origin = base.origin().ascii_serialization();
        let (blocks, _, existing) = checked.get(&page.url)?;
        let chrome: Vec<Block> = blocks
            .iter()
            .filter(|block| block.region == Region::Chrome)
            .cloned()
            .collect();
        let organization = jsonld::organization(&site, &origin, &document, &chrome, &base);
        Some(Homepage {
            url: page.url.clone(),
            origin,
            existing: existing.clone(),
            organization,
        })
    });

    // The pages to analyze: the first --ai-max-pages candidates, the homepage always.
    let candidates = build_candidates(status, &options.ai_include, &options.ai_exclude);
    let max_pages = options.ai_max_pages.max(1) as usize;
    let ranked = with_homepage(&candidates.candidates, max_pages, |candidate| {
        homepage_urls.contains(candidate.url.as_str())
    });
    let mut chosen = Vec::new();
    let mut without_text = Vec::new();
    let mut levels: HashSet<String> = HashSet::new();
    for candidate in ranked {
        let Some(html) = status.get_url_body_text(&candidate.uq_id) else {
            continue;
        };
        let (blocks, page_controls, existing) = match checked.get(&candidate.url) {
            Some((blocks, page_controls, existing)) => (blocks.clone(), page_controls.clone(), existing.clone()),
            None => {
                let headers = status.get_url_headers(&candidate.uq_id);
                page_checks(&candidate.url, &html, headers.as_ref(), now)
            }
        };
        let is_homepage = homepage_urls.contains(candidate.url.as_str());
        if !has_offered_blocks(&blocks, is_homepage) {
            without_text.push(candidate.url.clone());
            continue;
        }
        let signals = page_signals(&candidate.url, &html);
        let signals_line = signals_text(
            &existing,
            &page_controls.google,
            &page_controls.bing,
            signals.collapsed_share(),
        );
        breadcrumb_levels(&candidate.url, &mut levels);
        let indexable =
            !page_controls.google.noindex && !page_controls.bing.noindex && page_controls.canonical_elsewhere.is_none();
        chosen.push(Chosen {
            page: analyzed_page(&candidate.url, &html),
            blocks,
            signals,
            signals_line,
            is_homepage,
            existing,
            indexable,
        });
    }

    // The names of the crawled pages a breadcrumb of a chosen page may pass through.
    let mut crawled_names = HashMap::new();
    for visit in status.get_visited_urls() {
        if visit.status_code != 200
            || visit.content_type != ContentTypeId::Html
            || visit.is_external
            || !levels.contains(&normalized_url(&visit.url))
        {
            continue;
        }
        let name = if homepage_urls.contains(visit.url.as_str()) {
            Some(site.clone())
        } else {
            status
                .get_url_body_text(&visit.uq_id)
                .and_then(|html| page_name(&Html::parse_document(&html), &site))
        };
        if let Some(name) = name {
            crawled_names.insert(visit.url.clone(), name);
        }
    }

    Prepared {
        checks: Checks {
            key_pages: key,
            policy,
            access,
            access_stats,
            controls,
            render,
            markup,
            discovery,
            pdfs,
            analysis: AnalysisRun::default(),
        },
        chosen,
        without_text,
        homepage,
        homepage_url,
        site_name: site,
        crawled_names,
        crawled_pages: candidates.total_html_pages,
    }
}

/// Entry point for `--ai-geo`. Fail-soft: never panics, never aborts the crawl. The deterministic
/// checks run even when the AI configuration cannot be built; the per-page categories are then
/// "not assessed". `crawl_end` and `robots_skipped` (the URLs robots.txt kept the crawler from,
/// which reach `status` only after the AI phase) tell how much of the site the crawl covered.
pub async fn run(
    options: &CoreOptions,
    status: &Arc<Mutex<Status>>,
    output: &Arc<Mutex<Box<dyn Output>>>,
    crawl_end: CrawlEnd,
    robots_skipped: &[String],
) {
    let _ = output;
    // Own the usage ledger only when running standalone: the actions and the other pipelines
    // (dispatched before us) already reset it and recorded into it.
    if options.ai_actions.is_empty() && !options.ai_elaborate && !options.ai_profile && !options.ai_consistency {
        crate::ai::usage::reset();
    }
    crate::ai::usage::note_model(&options.ai_model.clone().unwrap_or_default());
    crate::ai::client::reset_rate_limiter().await;

    let locale = ReportLocale::new(&options.ai_report_language);
    let budget = ContextBudget::new(options.ai_context_window, options.ai_max_tokens);
    let budgets = Budgets::new(&budget);
    let temperature = options.ai_temperature as f32;
    let now = Utc::now();

    let prepared = {
        let st = match status.lock() {
            Ok(s) => s,
            Err(_) => return,
        };
        prepare(options, &st, now, crawl_end, robots_skipped)
    };
    let Prepared {
        mut checks,
        chosen,
        without_text,
        homepage,
        homepage_url,
        site_name,
        crawled_names,
        crawled_pages,
    } = prepared;

    if options.ai_dry_run {
        dry_run(
            options,
            status,
            &budget,
            &budgets,
            &chosen,
            checks.key_pages.len(),
            locale.code(),
        );
        print_without_text(&without_text);
        return;
    }
    print_without_text(&without_text);

    // --- The per-page analysis. ---
    let mut analysis = AnalysisRun {
        selected: chosen.len(),
        signals: chosen
            .iter()
            .map(|page| (page.page.url.clone(), page.signals.clone()))
            .collect(),
        without_text,
        ..AnalysisRun::default()
    };
    let mut provider = options.ai_provider.clone();
    match analysis_config(options) {
        Err(error) => {
            let msg = format!(
                "AI search readiness: the per-page analysis was skipped ({error}); the deterministic checks still ran."
            );
            eprintln!("{}", utils::get_color_text(&msg, "yellow", false));
            crate::events::emit_ai_issue(GEO_FAILED, &msg);
            if let Ok(st) = status.lock() {
                st.add_notice_to_summary("ai-geo-analysis", &msg);
            }
            analysis.unavailable = Some(error);
        }
        Ok(config) if !chosen.is_empty() => {
            provider = config.provider.as_str().to_string();
            let model = config.model.clone();
            let client = Arc::new(AiClient::new(config));
            let sem = Arc::new(Semaphore::new(options.ai_max_concurrency.clamp(1, 64) as usize));
            eprintln!(
                "\n{}",
                utils::get_color_text(
                    &format!(
                        "AI search readiness: {} key page(s), {} page(s) to analyze using {provider} / {model} (ctx {} tok)",
                        checks.key_pages.len(),
                        chosen.len(),
                        budget.context_tokens()
                    ),
                    "cyan",
                    true
                )
            );
            let answers = analyze_all(&client, &sem, &chosen, &budgets, locale.code(), temperature).await;
            for (page, answer) in chosen.iter().zip(answers) {
                match answer {
                    Ok((raw, coverage)) => {
                        analysis
                            .pages
                            .push(verify_analysis(raw, &page.page, &page.blocks, &coverage))
                    }
                    Err(error) => analysis.failed.push((page.page.url.clone(), error)),
                }
            }
            if !analysis.failed.is_empty() {
                let msg = format!(
                    "AI search readiness: the analysis of {} of {} page(s) failed; the report is partial.",
                    analysis.failed.len(),
                    chosen.len()
                );
                crate::events::emit_ai_issue(GEO_FAILED, &msg);
                if let Ok(st) = status.lock() {
                    st.add_notice_to_summary("ai-geo-partial", &msg);
                }
            }
        }
        Ok(config) => provider = config.provider.as_str().to_string(),
    }
    checks.analysis = analysis;

    // --- The kit markup, built by the crawler from the homepage and the verified blocks. ---
    let (kit, possible_profiles) = match &homepage {
        Some(homepage) => {
            let site = SiteMarkup {
                site_name: &site_name,
                origin: &homepage.origin,
                homepage_url: &homepage.url,
                organization: homepage.organization.0.clone(),
            };
            let analyses: HashMap<&str, &analyze::PageAnalysis> = checks
                .analysis
                .pages
                .iter()
                .map(|page| (page.url.as_str(), page))
                .collect();
            let pages: Vec<MarkupPage> = chosen
                .iter()
                .map(|page| MarkupPage {
                    url: &page.page.url,
                    analysis: analyses.get(page.page.url.as_str()).copied(),
                    existing: &page.existing,
                    indexable: page.indexable,
                })
                .collect();
            (
                markup_entries(&site, &homepage.existing, &pages, &crawled_names),
                homepage.organization.1.clone(),
            )
        }
        None => (Vec::new(), Vec::new()),
    };

    // --- The document. ---
    let (mut calls, mut input_tokens, mut output_tokens) = (0, 0, 0);
    for (name, usage) in crate::ai::usage::categories() {
        if name == CAT_PAGE {
            calls += usage.calls as usize;
            input_tokens += usage.prompt_tokens;
            output_tokens += usage.completion_tokens;
        }
    }
    let meta = GeoMeta {
        host: options.get_initial_host(false),
        url: utils::redact_url_userinfo(&homepage_url),
        site_name,
        crawled_at: chrono::Local::now().format("%Y-%m-%d %H:%M").to_string(),
        report_language: locale.code().to_string(),
        provider,
        model: options.ai_model.clone().unwrap_or_default(),
        context_window: budget.context_tokens(),
        crawled_pages,
        calls,
        input_tokens,
        output_tokens,
        ..GeoMeta::default()
    };
    let mut doc = GeoDoc::new(meta, checks, kit, possible_profiles);
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    doc.kit_files = doc
        .build_kit(&today)
        .into_iter()
        .map(|file| file.relative_path)
        .collect();

    eprintln!(
        "{}",
        utils::get_color_text(&format!("AI search readiness done: {}", doc.summary), "green", true)
    );
    for (category, state) in &doc.categories {
        let mut line = format!(
            "{}: {}",
            text(&locale, &format!("cat.{}", category.key())),
            text(&locale, &format!("status.{}", state.status.key()))
        );
        if state.status == CheckStatus::NotAssessed && !state.reason.is_empty() {
            line.push_str(&format!(" ({})", text(&locale, &format!("reason.{}", state.reason))));
        }
        eprintln!("  {}", utils::get_color_text(&line, "gray", false));
    }
    if let Ok(st) = status.lock() {
        st.add_info_to_summary("ai-geo", &format!("AI search readiness: {}", doc.summary));
        st.set_ai_geo_doc(doc);
    }
}

/// The AI configuration of the per-page analysis. `--ai-geo` alone may run without a model or an
/// endpoint (the options allow it, for the deterministic checks), which `build_config` does not
/// require; `Err` says what is missing.
fn analysis_config(options: &CoreOptions) -> Result<crate::ai::config::AiConfig, String> {
    if options.ai_model.as_deref().is_none_or(|model| model.trim().is_empty()) {
        return Err("no --ai-model was given".to_string());
    }
    if crate::ai::provider::Provider::parse(&options.ai_provider)
        == Some(crate::ai::provider::Provider::OpenAiCompatible)
        && options.ai_endpoint.is_none()
    {
        return Err("--ai-provider=openai-compatible needs --ai-endpoint=URL".to_string());
    }
    build_config(options)
}

/// Print the plan of a dry run: the calls, an input-token estimate (the final requests at 2.5
/// bytes per token), an output range and, with both rates set, a cost range. No API call is made.
fn dry_run(
    options: &CoreOptions,
    status: &Arc<Mutex<Status>>,
    budget: &ContextBudget,
    budgets: &Budgets,
    chosen: &[Chosen],
    key_pages: usize,
    language: &str,
) {
    let n = chosen.len();
    let input_bytes: usize = chosen
        .iter()
        .map(|page| {
            let (request, _) = page_request(
                page,
                budgets.input_bytes,
                budgets,
                language,
                options.ai_temperature as f32,
            );
            request_bytes(&request)
        })
        .sum();
    let input = (input_bytes as f64 / EST_BYTES_PER_TOKEN).ceil() as usize;
    let low = n * EST_PAGE_OUT.0.min(budgets.max_tokens as usize);
    let high = n * EST_PAGE_OUT.1.min(budgets.max_tokens as usize);
    let cost = match (options.ai_input_cost_per_million, options.ai_output_cost_per_million) {
        (Some(input_rate), Some(output_rate)) => {
            let price = |out: usize| (input as f64 * input_rate + out as f64 * output_rate) / 1_000_000.0;
            format!(", cost ${:.4}–${:.4}", price(low), price(high))
        }
        _ => String::new(),
    };
    eprintln!(
        "{}",
        utils::get_color_text(
            &format!(
                "AI search readiness dry-run: {n} page(s) → {n} analysis call(s), plus the deterministic checks of {key_pages} key page(s); est. input ~{input} tokens, output {low}–{high} tokens{cost}; ctx {} tok. No API calls made.",
                budget.context_tokens()
            ),
            "yellow",
            true
        )
    );
    let extra_body: String = options
        .ai_extra_body
        .as_deref()
        .unwrap_or_default()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    if extra_body.contains("\"enable_thinking\":true") || extra_body.contains("reasoning") {
        eprintln!(
            "  {}",
            utils::get_color_text(
                &format!(
                    "Thinking looks enabled (--ai-extra-body): reasoning tokens count against the {} output tokens of each page; a page cut at the limit is retried once with fewer blocks.",
                    budgets.max_tokens
                ),
                "yellow",
                false
            )
        );
    }
    eprintln!(
        "  {}",
        utils::get_color_text(
            "Answers are cached in --ai-cache-dir, so a rerun over the same pages is cheap.",
            "gray",
            false
        )
    );
    for (i, page) in chosen.iter().take(30).enumerate() {
        eprintln!(
            "  {}",
            utils::get_color_text(&format!("{:>3}. {}", i + 1, page.page.url), "gray", false)
        );
    }
    if let Ok(st) = status.lock() {
        st.add_info_to_summary(
            "ai-geo-dry-run",
            &format!(
                "AI search readiness dry-run: {n} page(s) would be analyzed; the checks of {key_pages} key page(s) need no API call."
            ),
        );
    }
}

/// Name the chosen pages that are not analyzed because their HTML has no text of their own.
fn print_without_text(urls: &[String]) {
    const SHOWN: usize = 10;
    if urls.is_empty() {
        return;
    }
    let mut list = urls.iter().take(SHOWN).cloned().collect::<Vec<_>>().join(", ");
    if urls.len() > SHOWN {
        list.push_str(&format!(" and {} more", urls.len() - SHOWN));
    }
    eprintln!(
        "  {}",
        utils::get_color_text(
            &format!("Not analyzed (no text of their own in the HTML): {list}"),
            "gray",
            false
        )
    );
}

/// The request of a page within `input_bytes`, and what it shows of the page.
fn page_request(
    page: &Chosen,
    input_bytes: usize,
    budgets: &Budgets,
    language: &str,
    temperature: f32,
) -> (ChatRequest, Coverage) {
    build_page_request(
        &page.page,
        &page.blocks,
        &page.signals_line,
        page.is_homepage,
        language,
        input_bytes,
        budgets.max_tokens,
        temperature,
    )
}

/// The bytes of a request's system prompt and messages.
fn request_bytes(request: &ChatRequest) -> usize {
    request.system.as_deref().map_or(0, str::len)
        + request
            .messages
            .iter()
            .map(|message| message.content.len())
            .sum::<usize>()
}

/// Why a page gets no analysis call: none of its blocks fits the input budget next to its URL, title
/// and signals, so the model would see nothing of the page.
const NO_ROOM: &str =
    "no block of the page fits the input budget next to its URL, title and signals (raise --ai-context-window)";

/// The full request of a page and the 30 % smaller one for a retry after a truncated answer; `Err`
/// when the full one would show no block (`NO_ROOM`).
#[allow(clippy::type_complexity)]
fn page_requests(
    page: &Chosen,
    budgets: &Budgets,
    language: &str,
    temperature: f32,
) -> Result<((ChatRequest, Coverage), (ChatRequest, Coverage)), String> {
    let full = page_request(page, budgets.input_bytes, budgets, language, temperature);
    if full.1.included.is_empty() {
        return Err(NO_ROOM.to_string());
    }
    let shown: usize = full.0.messages.iter().map(|message| message.content.len()).sum();
    let reduced = page_request(page, shown * 7 / 10, budgets, language, temperature);
    Ok((full, reduced))
}

/// One analysis call per chosen page, in parallel under the semaphore; each is one unit of
/// `geo:pages` (its subject the page path). An answer cut at the output limit is asked once more
/// with a 30 % smaller block selection. Returns each page's raw answer with what it was shown, or
/// the (redacted) error of its call.
async fn analyze_all(
    client: &Arc<AiClient>,
    sem: &Arc<Semaphore>,
    chosen: &[Chosen],
    budgets: &Budgets,
    language: &str,
    temperature: f32,
) -> Vec<Result<(RawAnalysis, Coverage), String>> {
    progress::start(TASK_PAGES, "GEO: pages", chosen.len() as u64);
    let mut handles = Vec::with_capacity(chosen.len());
    for page in chosen {
        let requests = page_requests(page, budgets, language, temperature);
        let (client, sem) = (client.clone(), sem.clone());
        handles.push(tokio::spawn(progress::unit(
            TASK_PAGES,
            crate::ai::runner::url_path_and_query(&page.page.url),
            async move {
                let (full, reduced) = requests?;
                let _permit = sem.acquire_owned().await.ok();
                match client
                    .complete_parsed_n(&full.0, CAT_PAGE, PARSE_ATTEMPTS, parse_analysis)
                    .await
                {
                    Ok((raw, _)) => Ok((raw, full.1)),
                    Err(error) if error.to_string().contains(TRUNCATED) && !reduced.1.included.is_empty() => client
                        .complete_parsed_n(&reduced.0, CAT_PAGE, PARSE_ATTEMPTS, parse_analysis)
                        .await
                        .map(|(raw, _)| (raw, reduced.1))
                        .map_err(|e| e.to_string()),
                    Err(error) => Err(error.to_string()),
                }
            },
        )));
    }
    let mut answers = Vec::with_capacity(handles.len());
    for handle in handles {
        answers.push(
            handle
                .await
                .unwrap_or_else(|e| Err(format!("analysis task failed: {e}"))),
        );
    }
    progress::finish(TASK_PAGES);
    answers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::geo::test_support::{add, new_status, page};
    use crate::result::visited_url::{SOURCE_A_HREF, SOURCE_INIT_URL, SOURCE_REDIRECT};

    #[test]
    fn the_robots_txt_of_a_key_page_origin_without_a_state_is_to_be_fetched() {
        let mut status = new_status();
        add(
            &mut status,
            page(
                "init",
                "",
                SOURCE_INIT_URL,
                "http://example.com/",
                301,
                Some("https://www.example.com/"),
            ),
            None,
        );
        add(
            &mut status,
            page("home", "init", SOURCE_REDIRECT, "https://www.example.com/", 200, None),
            None,
        );
        add(
            &mut status,
            page(
                "about",
                "home",
                SOURCE_A_HREF,
                "https://www.example.com/about",
                200,
                None,
            ),
            None,
        );
        // The crawl fetched the robots.txt of the initial origin only.
        status.set_robots_txt_state("http", "example.com", 80, RobotsFetchState::NotFound { status: 404 });
        assert_eq!(
            robots_origins_to_fetch(&status, &[], &[]),
            vec![("https".to_string(), "www.example.com".to_string(), 443)]
        );
        status.set_robots_txt_state("https", "www.example.com", 443, RobotsFetchState::Skipped);
        assert!(
            robots_origins_to_fetch(&status, &[], &[]).is_empty(),
            "every origin has a state"
        );
    }

    #[test]
    fn markup_values_hidden_for_good_are_not_visible() {
        let html = r#"<html><head><script type="application/ld+json">
            {"@context":"https://schema.org","@type":"Product","name":"Retired widget","offers":{"@type":"Offer","price":0}}
            </script></head><body><main><h1>Garden tools</h1><div hidden><p>Retired widget costs 0 USD</p></div>
            <details><summary>Current widget</summary><p>The current widget costs 12 USD.</p></details></main></body></html>"#;
        let (_, _, markup) = page_checks("https://example.com/", html, None, Utc::now());
        let invisible: Vec<&str> = markup
            .invisible_values
            .iter()
            .map(|(_, value)| value.as_str())
            .collect();
        assert_eq!(invisible, ["Retired widget", "0"], "{:?}", markup.invisible_values);
        // Text behind a disclosure is visible once opened.
        let html = html
            .replace("Retired widget costs", "Current widget costs")
            .replace(r#""name":"Retired widget""#, r#""name":"Current widget""#);
        let (_, _, markup) = page_checks("https://example.com/", &html, None, Utc::now());
        assert!(
            markup
                .invisible_values
                .iter()
                .all(|(_, value)| value != "Current widget"),
            "{:?}",
            markup.invisible_values
        );
        // Nor is a table row, a cell or text in a cell the page hides.
        for hidden in [
            "<table><tr><th>Product</th><th>Price</th></tr><tr hidden><td>Retired widget</td><td>0 USD</td></tr></table>",
            "<table><tr style=\"visibility:hidden\"><td>Retired widget</td><td>0 USD</td></tr></table>",
            "<table><tr><td>Rake</td><td hidden>Retired widget 0 USD</td></tr></table>",
            "<table><tr><td>Rake <span style=\"display:none\">Retired widget 0 USD</span></td></tr></table>",
        ] {
            let html = format!(
                r#"<html><head><script type="application/ld+json">
                {{"@context":"https://schema.org","@type":"Product","name":"Retired widget","offers":{{"@type":"Offer","price":0}}}}
                </script></head><body><main><h1>Garden tools</h1>{hidden}</main></body></html>"#
            );
            let (_, _, markup) = page_checks("https://example.com/", &html, None, Utc::now());
            let invisible: Vec<&str> = markup
                .invisible_values
                .iter()
                .map(|(_, value)| value.as_str())
                .collect();
            assert_eq!(invisible, ["Retired widget", "0"], "{hidden}");
        }
    }

    #[test]
    fn a_page_without_room_for_any_block_is_not_sent() {
        // A long URL and a title of angle brackets (escaped for the prompt) take the whole budget.
        let url = format!("https://example.com/{}", "a".repeat(1_980));
        let html = format!(
            "<html lang=\"en\"><head><title>{}</title></head><body><main><h1>Tools</h1>\
             <p>Acme sells garden tools.</p></main></body></html>",
            "&lt;".repeat(300)
        );
        let html = html.as_str();
        let blocks = blocks_from_html(html);
        let chosen = Chosen {
            page: analyzed_page(&url, html),
            signals: page_signals(&url, html),
            signals_line: String::new(),
            is_homepage: false,
            existing: ExistingMarkup::default(),
            indexable: true,
            blocks,
        };
        let budgets = |input_bytes: usize| Budgets {
            input_bytes,
            max_tokens: 1_000,
        };
        let refused = page_requests(&chosen, &budgets(3_072), "en", 0.0).map(|_| ());
        assert!(refused.as_ref().is_err_and(|why| why.contains("budget")), "{refused:?}");
        let (full, reduced) = page_requests(&chosen, &budgets(20_000), "en", 0.0).expect("room for the blocks");
        assert!(!full.1.included.is_empty());
        // A retry with 30 % fewer bytes has no room left: `analyze_all` does not send it.
        assert!(reduced.1.included.is_empty());
    }

    #[test]
    fn the_homepage_is_always_among_the_chosen_pages() {
        let ranked = ["/a", "/b", "/", "/c"];
        let home = |url: &&str| *url == "/";
        assert_eq!(with_homepage(&ranked, 2, home), vec![&"/", &"/a"], "added first");
        assert_eq!(
            with_homepage(&ranked, 3, home),
            vec![&"/a", &"/b", &"/"],
            "kept in rank order"
        );
        assert_eq!(with_homepage(&ranked[..2], 1, home), vec![&"/a"], "no homepage to add");
        assert_eq!(with_homepage(&ranked, 0, home), vec![&"/"], "at least the homepage");
    }
}
