// SiteOne Crawler - AI fact consistency across pages (--ai-consistency)
// (c) Jan Reges <jan.reges@siteone.cz>
//
// Finds possible contradictions in hard facts (contacts, prices, rates, fees, conditions, dates,
// figures, identifiers) across the crawled pages and the header/footer lines they share:
//   sources  — numbered evidence blocks per page, plus the unique fact-bearing header/footer
//              lines with their exact page sets;
//   extract  — a few block-grounded facts per source, verified against the crawler's own text;
//   keys     — facts grouped per attribute key into keys (same property of the same subject), in
//              context-sized chunks over several levels;
//   judge    — deterministic candidates (differing values stated in several places), cohorts, a
//              fair review allocation, and the validated, policy-gated review;
//   doc      — the report (Markdown, JSON, HTML, CSV) with its audit trail and completeness state.
// `run` drives the steps after the crawl.

pub mod doc;
pub mod extract;
pub mod judge;
pub mod keys;
pub mod model;
pub mod prompts;
pub mod sources;

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::ops::Range;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Poll;
use std::time::Instant;

use scraper::{Html, Selector};
use tokio::sync::{Semaphore, watch};

use crate::ai::blocks::{Block, blocks_from_html};
use crate::ai::client::AiClient;
use crate::ai::config::build_config;
use crate::ai::grounding::{ValueKey, site_country};
use crate::ai::profile::budget::ContextBudget;
use crate::ai::progress;
use crate::ai::report::locale::ReportLocale;
use crate::ai::selection::build_candidates;
use crate::options::core_options::CoreOptions;
use crate::output::output::Output;
use crate::result::status::Status;
use crate::utils;

use self::doc::{
    ConsistencyDoc, Counts, ExplainedGroup, FailedSource, Finding, FindingValue, KeyOut, Meta, NotJudgedGroup,
    OccurrenceOut, OccurrenceRef, SCHEMA, SourceOut,
};
use self::extract::{CAT_EXTRACT, build_extract_request, parse_facts, verify_facts};
use self::judge::{
    CAT_REVIEW, Candidate, CandidateValue, ReviewResult, ValidatedResult, allocate, build_review_request, cohorts,
    comparisons_done, groups_message, key_outcome, pack_batches, parse_reviews, render_group_within,
    review_call_max_tokens, review_groups_per_call, split_keys, uncovered_values, validate_with_date,
    with_values_first,
};
use self::keys::{
    CAT_GROUP, GroupOutcome, LabelItem, build_group_request, group_key, label_items, max_items_by_output, parse_groups,
};
use self::model::{AnalysisSource, AttributeKey, Disposition, Occurrence, Page, Priority, RawFact, SourceKind};
use self::sources::{chrome_lines, chrome_sources, page_source, source_message};

const TASK_EXTRACT: &str = "consistency:extract";
const TASK_GROUP: &str = "consistency:group";
const TASK_REVIEW: &str = "consistency:review";
/// Label of the `issue` event (kind `ai`) when the check fails or stays partial after a failure.
const CONSISTENCY_FAILED: &str = "AI consistency failed";
const PARSE_ATTEMPTS: u32 = 2;
/// A review batch cut at the output limit is split in half at most this many times.
const MAX_SPLIT_DEPTH: usize = 2;
/// The error text of an answer cut at the output-token limit (see `AiClient::complete_parsed_n`).
const TRUNCATED: &str = "generation stopped at token limit";
/// Dry-run estimates: facts kept per source (typically 2–4), bytes per grouping label line,
/// facts per candidate, bytes per rendered review group, and the output tokens of an extraction
/// (typically 300–600), a grouping label (12, up to twice over the levels) and a review group
/// (200–450).
const EST_FACTS_PER_SOURCE: usize = 3;
const EST_LABEL_BYTES: usize = 80;
const EST_FACTS_PER_CANDIDATE: usize = 10;
const EST_GROUP_BYTES: usize = 1_500;
const EST_BYTES_PER_TOKEN: f64 = 2.5;
const EST_EXTRACT_OUT: (usize, usize) = (300, 600);
const EST_LABEL_OUT: (usize, usize) = (12, 24);
const EST_REVIEW_OUT: (usize, usize) = (200, 450);
/// Buckets that usually need a grouping call, at most.
const EST_MAX_BUCKETS: usize = 12;

/// The input and output budgets of the three LLM steps, from `--ai-context-window` and
/// `--ai-max-tokens`; input budgets are bytes of the final escaped user message.
struct Budgets {
    extract_input_bytes: usize,
    extract_max_tokens: u32,
    group_chunk_bytes: usize,
    group_max_tokens: u32,
    review_batch_bytes: usize,
    review_max_tokens: u32,
}

impl Budgets {
    fn new(b: &ContextBudget) -> Self {
        Self {
            extract_input_bytes: b.scaled(10, 3),
            extract_max_tokens: b.out_tokens().min(2_000),
            group_chunk_bytes: b.scaled(40, 6),
            group_max_tokens: b.out_tokens().min(6_000),
            review_batch_bytes: b.scaled(40, 6),
            review_max_tokens: b.out_tokens(),
        }
    }
}

fn report_error(status: &Arc<Mutex<Status>>, msg: &str) {
    eprintln!("{}", utils::get_color_text(&format!("ERROR: {msg}"), "red", true));
    crate::events::emit_ai_issue(CONSISTENCY_FAILED, msg);
    if let Ok(st) = status.lock() {
        st.add_critical_to_summary("ai-consistency-error", msg);
    }
}

/// The sources of the pages to extract, and the pages that have content blocks of which none fits
/// the extraction budget (e.g. next to a very long URL): those are not sent, but reported (as
/// reduced pages with all their blocks omitted), never silently dropped. A page without content
/// blocks is neither.
fn page_sources(
    pages: &[Page],
    pages_blocks: &[(usize, Vec<Block>)],
    budget: usize,
) -> (Vec<AnalysisSource>, Vec<AnalysisSource>) {
    pages
        .iter()
        .zip(pages_blocks)
        .map(|(page, (_, blocks))| page_source(page, blocks, budget))
        .filter(|source| !source.blocks.is_empty() || source.omitted_blocks > 0)
        .partition(|source| !source.blocks.is_empty())
}

/// The audit record of a page none of whose blocks fit the extraction budget.
fn unfit_source_out(source: &AnalysisSource) -> SourceOut {
    SourceOut {
        id: source.id,
        kind: source.kind,
        url: source.url.clone(),
        path: source.path.clone(),
        blocks: source.blocks.len(),
        omitted_blocks: source.omitted_blocks,
        truncated_blocks: source.truncated_blocks,
        facts: 0,
        ungrounded: 0,
        failed: false,
    }
}

/// The pages analyzed (content sent and extracted) and the pages reduced (blocks left out or cut,
/// all of them for a page over the budget) among the page sources of `outs`.
fn page_counts(outs: &[SourceOut]) -> (usize, usize) {
    let pages = || outs.iter().filter(|s| s.kind == SourceKind::Page && !s.failed);
    (
        pages().filter(|s| s.blocks > 0).count(),
        pages()
            .filter(|s| s.omitted_blocks > 0 || s.truncated_blocks > 0)
            .count(),
    )
}

/// The country whose dialling rules read the national phone numbers of `source` (see
/// `grounding::site_country`): for a page, by its own host and language (`langs[page]`; none: the
/// site's) — a language without a region takes the site's country only when it is the site's
/// language (`site_lang`); for header/footer lines, the country all their pages share, else none.
/// `host` is the crawl's host, for a page URL that does not parse.
fn source_country(
    source: &AnalysisSource,
    pages: &[Page],
    langs: &[String],
    site_lang: &str,
    host: &str,
) -> Option<&'static str> {
    let page_country = |index: usize| {
        let page_host = pages
            .get(index)
            .and_then(|p| url::Url::parse(&p.url).ok())
            .and_then(|url| url.host_str().map(str::to_string))
            .unwrap_or_else(|| host.to_string());
        let lang = langs
            .get(index)
            .map(String::as_str)
            .filter(|lang| !lang.trim().is_empty())
            .unwrap_or(site_lang);
        let primary = |lang: &str| {
            lang.trim()
                .split(['-', '_'])
                .next()
                .unwrap_or_default()
                .to_ascii_lowercase()
        };
        site_country(&page_host, lang).or_else(|| {
            if primary(lang) == primary(site_lang) {
                site_country(&page_host, site_lang)
            } else {
                None
            }
        })
    };
    match source.kind {
        SourceKind::Page => page_country(source.id),
        SourceKind::Chrome => {
            let mut countries = source
                .blocks
                .iter()
                .flat_map(|b| b.pages.iter())
                .map(|&page| page_country(page));
            let first = countries.next()??;
            countries.all(|c| c == Some(first)).then_some(first)
        }
    }
}

/// The `<title>` (on one line) and the `<html lang>` of a page.
fn page_meta(html: &str) -> (String, String) {
    let document = Html::parse_document(html);
    let first = |selector: &str| Selector::parse(selector).ok().and_then(|s| document.select(&s).next());
    let title = first("title")
        .map(|e| {
            e.text()
                .collect::<String>()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();
    let lang = first("html")
        .and_then(|e| e.value().attr("lang"))
        .map(|lang| lang.trim().to_string())
        .unwrap_or_default();
    (title, lang)
}

/// Entry point for `--ai-consistency`. Fail-soft: never panics, never aborts the crawl.
pub async fn run(options: &CoreOptions, status: &Arc<Mutex<Status>>, output: &Arc<Mutex<Box<dyn Output>>>) {
    let _ = output;
    let run_start = Instant::now();
    // Own the usage ledger only when running standalone: the actions, `--ai-elaborate` and
    // `--ai-profile` (dispatched before us) already reset it and recorded into it.
    if options.ai_actions.is_empty() && !options.ai_elaborate && !options.ai_profile {
        crate::ai::usage::reset();
    }
    crate::ai::usage::note_model(&options.ai_model.clone().unwrap_or_default());
    crate::ai::client::reset_rate_limiter().await;

    let host = options.get_initial_host(false);
    let locale = ReportLocale::new(&options.ai_report_language);
    let budget = ContextBudget::new(options.ai_context_window, options.ai_max_tokens);
    let budgets = Budgets::new(&budget);
    let concurrency = options.ai_max_concurrency.clamp(1, 64) as usize;
    let max_pages = options.ai_max_pages.max(1) as usize;

    // --- S0: the selected pages (their HTML under one status lock), their blocks and sources. ---
    let (eligible, chosen) = {
        let st = match status.lock() {
            Ok(s) => s,
            Err(_) => return,
        };
        let cs = build_candidates(&st, &options.ai_include, &options.ai_exclude);
        let chosen: Vec<(String, Option<String>)> = cs
            .candidates
            .iter()
            .take(max_pages)
            .map(|c| (c.url.clone(), st.get_url_body_text(&c.uq_id)))
            .collect();
        (cs.candidates.len(), chosen)
    };
    let selected = chosen.len();
    let mut pages: Vec<Page> = Vec::new();
    let mut langs: Vec<String> = Vec::new();
    let mut pages_blocks: Vec<(usize, Vec<Block>)> = Vec::new();
    let mut missing_body = 0;
    for (url, body) in chosen {
        let Some(html) = body else {
            missing_body += 1;
            continue;
        };
        let (title, lang) = page_meta(&html);
        let index = pages.len();
        pages_blocks.push((index, blocks_from_html(&html)));
        pages.push(Page {
            index,
            path: crate::ai::runner::url_path_and_query(&url),
            url,
            title,
        });
        langs.push(lang);
    }
    // The language of the site (the first page with one, the homepage first) reads the numbers of
    // the header/footer and of pages without a language; each page names the country of its
    // national phone numbers by its own host and language (`source_country`).
    let site_lang = langs.iter().find(|lang| !lang.is_empty()).cloned().unwrap_or_default();

    let (mut sources, unfit) = page_sources(&pages, &pages_blocks, budgets.extract_input_bytes);
    let lines = chrome_lines(&pages_blocks);
    drop(pages_blocks);
    let page_sources = sources.len();
    sources.extend(chrome_sources(&lines, budgets.extract_input_bytes, pages.len()));
    let chrome_chunks = sources.len() - page_sources;

    if sources.is_empty() {
        let msg = "AI consistency: no page content to analyze was crawled.";
        crate::events::emit_ai_issue(CONSISTENCY_FAILED, msg);
        if let Ok(st) = status.lock() {
            st.add_notice_to_summary("ai-consistency", msg);
        }
        return;
    }

    if options.ai_dry_run {
        dry_run(options, status, &budget, &budgets, &sources, &pages, page_sources);
        return;
    }

    let config = match build_config(options) {
        Ok(c) => c,
        Err(e) => {
            report_error(status, &format!("AI consistency skipped: {e}"));
            return;
        }
    };
    let provider = config.provider;
    let model = config.model.clone();
    let client = Arc::new(AiClient::new(config));
    let sem = Arc::new(Semaphore::new(concurrency));
    eprintln!(
        "\n{}",
        utils::get_color_text(
            &format!(
                "AI consistency: {page_sources} page(s) + {chrome_chunks} header/footer chunk(s) using {} / {model} (ctx {} tok)",
                provider.as_str(),
                budget.context_tokens()
            ),
            "cyan",
            true
        )
    );

    // --- S1: extraction, one call per source. ---
    let answers = extract_all(
        &client,
        &sem,
        &sources,
        &pages,
        budgets.extract_max_tokens,
        options.ai_temperature as f32,
    )
    .await;
    let mut occurrences: Vec<Occurrence> = Vec::new();
    let mut next_id = 0;
    let mut source_outs: Vec<SourceOut> = Vec::new();
    let mut failed_sources: Vec<FailedSource> = Vec::new();
    let mut facts_extracted = 0;
    for (source, answer) in sources.iter().zip(answers) {
        let lang = match source.kind {
            SourceKind::Page => langs
                .get(source.id)
                .filter(|lang| !lang.is_empty())
                .unwrap_or(&site_lang),
            SourceKind::Chrome => &site_lang,
        };
        let (facts, ungrounded) = match answer {
            Ok(raw) => {
                facts_extracted += raw.len();
                let country = source_country(source, &pages, &langs, &site_lang, &host);
                let (kept, ungrounded) = verify_facts(raw, source, lang, country, &mut next_id);
                let facts = kept.len();
                occurrences.extend(kept);
                (facts, ungrounded)
            }
            Err(error) => {
                failed_sources.push(FailedSource {
                    id: source.id,
                    kind: source.kind,
                    url: source.url.clone(),
                    path: source.path.clone(),
                    error,
                });
                (0, 0)
            }
        };
        source_outs.push(SourceOut {
            id: source.id,
            kind: source.kind,
            url: source.url.clone(),
            path: source.path.clone(),
            blocks: source.blocks.len(),
            omitted_blocks: source.omitted_blocks,
            truncated_blocks: source.truncated_blocks,
            facts,
            ungrounded,
            failed: failed_sources.last().is_some_and(|f| f.id == source.id),
        });
    }
    // Pages none of whose content fit the budget: nothing was sent, all their blocks are omitted.
    source_outs.extend(unfit.iter().map(unfit_source_out));
    eprintln!(
        "{}",
        utils::get_color_text(
            &format!(
                "AI consistency: {} fact(s) kept from {} source(s) ({} failed)",
                occurrences.len(),
                sources.len(),
                failed_sources.len()
            ),
            "cyan",
            false
        )
    );
    if failed_sources.len() == sources.len() {
        let first = failed_sources.first().map_or("", |f| f.error.as_str());
        report_error(
            status,
            &format!("AI consistency: every extraction call failed; no report produced ({first})."),
        );
        return;
    }

    // --- S2: grouping, per attribute bucket. ---
    let buckets = bucket_items(&occurrences);
    let failed_group_calls = Arc::new(AtomicUsize::new(0));
    let ask = {
        let (client, sem, failed) = (client.clone(), sem.clone(), failed_group_calls.clone());
        let max_tokens = budgets.group_max_tokens;
        move |key: AttributeKey, _level: usize, items: Vec<LabelItem>| {
            let (client, sem, failed) = (client.clone(), sem.clone(), failed.clone());
            let req = build_group_request(key, &items, max_tokens);
            let n = items.len();
            async move {
                let _permit = sem.acquire_owned().await.ok();
                let answer = client
                    .complete_parsed_n(&req, CAT_GROUP, PARSE_ATTEMPTS, |raw| parse_groups(raw, n))
                    .await
                    .map(|(groups, _)| groups)
                    .map_err(|e| e.to_string());
                if answer.is_err() {
                    failed.fetch_add(1, Ordering::Relaxed);
                }
                answer
            }
        }
    };
    let outcomes = group_all(
        buckets,
        budgets.group_chunk_bytes,
        max_items_by_output(budgets.group_max_tokens),
        TASK_GROUP,
        ask,
    )
    .await;
    let mut keys = Vec::new();
    let (mut grouping_incomplete, mut not_cross_compared) = (false, 0);
    for outcome in outcomes {
        grouping_incomplete |= outcome.incomplete;
        not_cross_compared += outcome.not_cross_compared;
        for mut key in outcome.keys {
            key.id = keys.len();
            key.attribute_key = key_attribute(key.attribute_key, &key.occurrence_ids, &occurrences);
            keys.push(key);
        }
    }

    // --- S3: candidates (deterministic). ---
    let (candidates, consistent) = split_keys(&keys, &occurrences);
    let candidate_count = candidates.len();
    let groups: Vec<Candidate> = candidates.into_iter().flat_map(cohorts).collect();
    let (to_review, over_cap) = allocate(groups);

    // --- S4: review. ---
    let per_call = review_groups_per_call(budgets.review_max_tokens);
    let review_results = if per_call == 0 || to_review.is_empty() {
        vec![Vec::new(); to_review.len()]
    } else {
        review_all(
            &client,
            &sem,
            &to_review,
            &occurrences,
            &sources,
            &pages,
            &budgets,
            locale.code(),
            &chrono::Local::now().format("%Y-%m-%d").to_string(),
            per_call,
        )
        .await
    };

    // --- S5: the document. ---
    let (mut llm_calls, mut failed_reviews) = (0, 0);
    for (name, usage) in crate::ai::usage::categories() {
        if name.starts_with("AI consistency") {
            llm_calls += usage.calls as usize;
        }
    }
    let reviewed = review_results.iter().filter(|results| !results.is_empty()).count();
    if per_call > 0 {
        failed_reviews = to_review.len() - reviewed;
    }
    let (pages_analyzed, pages_reduced) = page_counts(&source_outs);
    let chrome_lines_analyzed = sources
        .iter()
        .zip(&source_outs)
        .filter(|(s, out)| s.kind == SourceKind::Chrome && !out.failed)
        .map(|(s, _)| s.blocks.len())
        .sum();
    let meta = Meta {
        host: host.clone(),
        url: utils::redact_url_userinfo(&options.url),
        generated_at: chrono::Local::now().format("%Y-%m-%d %H:%M").to_string(),
        language: locale.code().to_string(),
        provider: provider.as_str().to_string(),
        model,
        context_window: budget.context_tokens(),
        pages_eligible: eligible,
        pages_selected: selected,
        pages_analyzed,
        pages_missing_body: missing_body,
        pages_reduced,
        chrome_lines_analyzed,
        chrome_lines_excluded: lines.excluded,
        sources: sources.len(),
        sources_with_facts: source_outs.iter().filter(|s| s.facts > 0).count(),
        facts_extracted,
        facts_kept: occurrences.len(),
        facts_ungrounded: source_outs.iter().map(|s| s.ungrounded).sum(),
        keys: keys.len(),
        candidates: candidate_count,
        reviewed,
        comparisons_done: to_review
            .iter()
            .zip(&review_results)
            .map(|(c, results)| comparisons_done(c, results))
            .sum(),
        reviews_capped: over_cap.len(),
        reviews_partial: to_review
            .iter()
            .zip(&review_results)
            .filter(|(c, results)| !uncovered_values(c, results).is_empty())
            .count(),
        reviews_failed: failed_reviews,
        reviews_skipped_budget: if per_call == 0 { to_review.len() } else { 0 },
        grouping_incomplete,
        items_not_cross_compared: not_cross_compared,
        llm_calls,
        duration_ms: run_start.elapsed().as_millis() as u64,
        http_cache: http_cache_ttl(options),
    };
    let not_reviewed_reason = if per_call == 0 {
        "not_reviewed_output_budget"
    } else {
        "not_reviewed_call_failed"
    };
    let doc = assemble(
        &locale,
        meta,
        Assembly {
            reviewed: to_review.iter().zip(review_results).collect(),
            not_reviewed_reason,
            over_cap: &over_cap,
            consistent,
            keys: &keys,
            occurrences: &occurrences,
            pages: &pages,
            sources: source_outs,
            failed_sources,
        },
    );

    let findings = doc.findings.len();
    let failed_groupings = failed_group_calls.load(Ordering::Relaxed);
    let failures = [
        (doc.failed_sources.len(), "source(s) could not be analyzed"),
        (failed_groupings, "grouping call(s) failed"),
        (failed_reviews, "difference(s) could not be reviewed"),
    ]
    .iter()
    .filter(|(n, _)| *n > 0)
    .map(|(n, what)| format!("{n} {what}"))
    .collect::<Vec<_>>();
    eprintln!(
        "{}",
        utils::get_color_text(
            &format!(
                "AI consistency done: {findings} possible inconsistenc{} to verify, {} with a plausible explanation, {} not judged ({:?}).",
                if findings == 1 { "y" } else { "ies" },
                doc.counts.explained + doc.counts.not_comparable,
                doc.counts.insufficient + doc.counts.not_reviewed,
                doc.completeness.state
            ),
            "green",
            true
        )
    );
    if let Ok(st) = status.lock() {
        if !failures.is_empty() {
            let msg = format!("AI consistency: {}; the report is partial.", failures.join(", "));
            crate::events::emit_ai_issue(CONSISTENCY_FAILED, &msg);
            st.add_notice_to_summary("ai-consistency-partial", &msg);
        }
        st.add_info_to_summary(
            "ai-consistency",
            &format!(
                "AI consistency: {findings} possible inconsistenc{} to verify across {} page(s) (MD + JSON + HTML + CSV).",
                if findings == 1 { "y" } else { "ies" },
                doc.meta.pages_analyzed
            ),
        );
        st.set_ai_consistency_doc(doc);
    }
}

/// The grouping buckets (`AttributeKey::bucket`) that have facts, with their labels.
fn bucket_items(occurrences: &[Occurrence]) -> Vec<(AttributeKey, Vec<LabelItem>)> {
    AttributeKey::ALL
        .into_iter()
        .filter(|key| key.bucket() == *key)
        .map(|key| (key, label_items(occurrences, key)))
        .filter(|(_, items)| !items.is_empty())
        .collect()
}

/// The attribute key of a grouped key of `bucket`: the most common among its occurrences (a key
/// of the price bucket stating only fees is a fee); a tie goes to the bucket.
fn key_attribute(bucket: AttributeKey, occurrence_ids: &[usize], occurrences: &[Occurrence]) -> AttributeKey {
    let mut counts: BTreeMap<AttributeKey, usize> = BTreeMap::new();
    for o in occurrences.iter().filter(|o| occurrence_ids.contains(&o.id)) {
        *counts.entry(o.attribute_key).or_default() += 1;
    }
    let most = counts.values().copied().max().unwrap_or(0);
    if counts.get(&bucket).copied().unwrap_or(0) == most {
        return bucket;
    }
    counts
        .into_iter()
        .find(|(_, n)| *n == most)
        .map_or(bucket, |(key, _)| key)
}

/// The TTL of the crawler's HTTP cache when it is on (`∞` without expiry); None when off.
fn http_cache_ttl(options: &CoreOptions) -> Option<String> {
    let dir = options.http_cache_dir.as_deref()?;
    if dir.is_empty() || dir == "off" {
        return None;
    }
    Some(match options.http_cache_ttl {
        None => "∞".to_string(),
        Some(secs) if secs > 0 && secs.is_multiple_of(86_400) => format!("{}d", secs / 86_400),
        Some(secs) if secs > 0 && secs.is_multiple_of(3_600) => format!("{}h", secs / 3_600),
        Some(secs) if secs > 0 && secs.is_multiple_of(60) => format!("{}m", secs / 60),
        Some(secs) => format!("{secs}s"),
    })
}

/// Print the plan of a dry run: the calls, an input-token estimate (the final extraction
/// messages plus estimated grouping and review input, at 2.5 bytes per token), an output range
/// and, with both rates set, a cost range. No API call is made.
fn dry_run(
    options: &CoreOptions,
    status: &Arc<Mutex<Status>>,
    budget: &ContextBudget,
    budgets: &Budgets,
    sources: &[AnalysisSource],
    pages: &[Page],
    page_sources: usize,
) {
    let n = sources.len();
    let chrome_chunks = n - page_sources;
    let facts = n * EST_FACTS_PER_SOURCE;
    let candidates = (facts / EST_FACTS_PER_CANDIDATE).clamp(1, judge::MAX_REVIEWED_GROUPS);
    let grouping_calls =
        (facts / 4).clamp(1, EST_MAX_BUCKETS) + (facts * EST_LABEL_BYTES).div_ceil(budgets.group_chunk_bytes.max(1));
    let per_call = review_groups_per_call(budgets.review_max_tokens);
    let review_calls = if per_call == 0 {
        0
    } else {
        candidates
            .div_ceil(per_call)
            .max((candidates * EST_GROUP_BYTES).div_ceil(budgets.review_batch_bytes.max(1)))
    };
    let extract_bytes: usize = sources
        .iter()
        .map(|source| {
            let title = pages
                .iter()
                .find(|p| source.kind == SourceKind::Page && p.index == source.id)
                .map_or("", |p| p.title.as_str());
            prompts::EXTRACT.len() + source_message(source, title).len()
        })
        .sum();
    let group_bytes = grouping_calls * prompts::GROUP.len() + facts * EST_LABEL_BYTES;
    let review_bytes = review_calls * prompts::JUDGE.len() + candidates * EST_GROUP_BYTES;
    let input = ((extract_bytes + group_bytes + review_bytes) as f64 / EST_BYTES_PER_TOKEN).ceil() as usize;
    let output = |(extract, label, review): (usize, usize, usize)| n * extract + facts * label + candidates * review;
    let low = output((EST_EXTRACT_OUT.0, EST_LABEL_OUT.0, EST_REVIEW_OUT.0));
    let high = output((EST_EXTRACT_OUT.1, EST_LABEL_OUT.1, EST_REVIEW_OUT.1));
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
                "AI consistency dry-run: {page_sources} page(s) + {chrome_chunks} header/footer chunk(s) → {n} extraction call(s), ~{grouping_calls} grouping and ~{review_calls} review call(s); est. input ~{input} tokens, output {low}–{high} tokens{cost}; ctx {} tok. No API calls made.",
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
                "Thinking looks enabled (--ai-extra-body): reasoning tokens count against --ai-max-tokens and raise the output; a review batch cut at the limit is split and retried.",
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
    for (i, page) in pages.iter().take(30).enumerate() {
        eprintln!(
            "  {}",
            utils::get_color_text(&format!("{:>3}. {}", i + 1, page.url), "gray", false)
        );
    }
    if let Ok(st) = status.lock() {
        st.add_info_to_summary(
            "ai-consistency-dry-run",
            &format!(
                "AI consistency dry-run: {page_sources} page(s) and {chrome_chunks} header/footer chunk(s) would be analyzed."
            ),
        );
    }
}

/// S1: one extraction call per source, in parallel under the semaphore; each call is one unit of
/// `consistency:extract` (its subject the page path, or `header/footer lines i/n`). Returns the
/// facts of each source, in order, or the error of its call.
async fn extract_all(
    client: &Arc<AiClient>,
    sem: &Arc<Semaphore>,
    sources: &[AnalysisSource],
    pages: &[Page],
    max_tokens: u32,
    temperature: f32,
) -> Vec<Result<Vec<RawFact>, String>> {
    progress::start(TASK_EXTRACT, "Consistency: facts", sources.len() as u64);
    let mut handles = Vec::with_capacity(sources.len());
    for source in sources {
        let req = build_extract_request(source, pages, max_tokens, temperature);
        let (client, sem) = (client.clone(), sem.clone());
        handles.push(tokio::spawn(progress::unit(
            TASK_EXTRACT,
            source.path.clone(),
            async move {
                let _permit = sem.acquire_owned().await.ok();
                client
                    .complete_parsed_n(&req, CAT_EXTRACT, PARSE_ATTEMPTS, parse_facts)
                    .await
                    .map(|(facts, _)| facts)
                    .map_err(|e| e.to_string())
            },
        )));
    }
    let mut answers = Vec::with_capacity(handles.len());
    for handle in handles {
        answers.push(
            handle
                .await
                .unwrap_or_else(|e| Err(format!("extraction task failed: {e}"))),
        );
    }
    progress::finish(TASK_EXTRACT);
    answers
}

/// Where each bucket's grouping stands: the highest level it asked at, whether it finished, and
/// the calls asked per level.
struct Rounds {
    reached: Vec<usize>,
    done: Vec<bool>,
    asks: BTreeMap<usize, usize>,
}

/// S2: group every attribute bucket (`keys::group_key`) concurrently, round by round: the calls of
/// a grouping level (round) start only once every bucket has finished or asked at that level, so
/// the round's number of calls is known when it starts. Each call is one unit of the progress
/// `task` ("Consistency: grouping", restarted per later round as "Consistency: grouping
/// (round N)"); a round without calls starts no task. `ask(key, level, labels)` makes the call.
/// Returns the outcome of each bucket, in order.
pub(crate) async fn group_all<A, Fut>(
    buckets: Vec<(AttributeKey, Vec<LabelItem>)>,
    budget_bytes: usize,
    max_items: usize,
    task: &'static str,
    ask: A,
) -> Vec<GroupOutcome>
where
    A: Fn(AttributeKey, usize, Vec<LabelItem>) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = Result<Vec<(Vec<usize>, String)>, String>> + Send + 'static,
{
    let n = buckets.len();
    let rounds = Arc::new(Mutex::new(Rounds {
        reached: vec![0; n],
        done: vec![false; n],
        asks: BTreeMap::new(),
    }));
    let (open_tx, open_rx) = watch::channel(0usize);
    type Grouping = Pin<Box<dyn Future<Output = GroupOutcome> + Send>>;
    let mut running: Vec<Option<Grouping>> = Vec::with_capacity(n);
    for (bucket, (key, items)) in buckets.into_iter().enumerate() {
        let (rounds, open_rx, ask) = (rounds.clone(), open_rx.clone(), ask.clone());
        let gated = move |level: usize, items: Vec<LabelItem>| {
            if let Ok(mut state) = rounds.lock() {
                state.reached[bucket] = state.reached[bucket].max(level);
                *state.asks.entry(level).or_default() += 1;
            }
            let mut open = open_rx.clone();
            let subject = format!("{} ({} labels)", key.as_str(), items.len());
            let call = ask(key, level, items);
            async move {
                let _ = open.wait_for(|round| *round >= level).await;
                progress::unit(task, subject, call).await
            }
        };
        running.push(Some(Box::pin(group_key(key, items, budget_bytes, max_items, gated))));
    }

    let mut outcomes: Vec<Option<GroupOutcome>> = (0..n).map(|_| None).collect();
    let mut round = 0;
    let mut started = false;
    std::future::poll_fn(|cx| {
        for (bucket, slot) in running.iter_mut().enumerate() {
            if let Some(grouping) = slot
                && let Poll::Ready(outcome) = grouping.as_mut().poll(cx)
            {
                outcomes[bucket] = Some(outcome);
                *slot = None;
                if let Ok(mut state) = rounds.lock() {
                    state.done[bucket] = true;
                }
            }
        }
        // A burst of calls of one level is asked within one poll of its bucket, so after this
        // pass every level every bucket reached has its full number of calls.
        if let Ok(state) = rounds.lock() {
            loop {
                let next = round + 1;
                let unfinished = (0..n).any(|b| !state.done[b]);
                let reached = (0..n).all(|b| state.done[b] || state.reached[b] >= next);
                if !unfinished || !reached {
                    break;
                }
                round = next;
                if let Some(&calls) = state.asks.get(&next) {
                    if started {
                        progress::finish(task);
                    }
                    let label = if next == 1 {
                        "Consistency: grouping".to_string()
                    } else {
                        format!("Consistency: grouping (round {next})")
                    };
                    progress::start(task, &label, calls as u64);
                    started = true;
                }
                open_tx.send_replace(next);
            }
        }
        if running.iter().all(Option::is_none) {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    })
    .await;
    if started {
        progress::finish(task);
    }
    outcomes
        .into_iter()
        .map(|outcome| {
            outcome.unwrap_or(GroupOutcome {
                keys: Vec::new(),
                incomplete: true,
                not_cross_compared: 0,
                calls: 0,
            })
        })
        .collect()
}

/// S4: review the groups in batches (`judge::pack_batches`), in parallel under the semaphore;
/// each batch is one unit of `consistency:review`. Dated articles are judged against
/// `crawl_date`. Returns the valid results of each group, in order (none when its call failed or
/// gave no valid result).
#[allow(clippy::too_many_arguments)]
async fn review_all(
    client: &Arc<AiClient>,
    sem: &Arc<Semaphore>,
    groups: &[Candidate],
    occurrences: &[Occurrence],
    sources: &[AnalysisSource],
    pages: &[Page],
    budgets: &Budgets,
    language: &str,
    crawl_date: &str,
    per_call: usize,
) -> Vec<Vec<ValidatedResult>> {
    let room = budgets.review_batch_bytes.saturating_sub(groups_message(&[]).len() + 1);
    let rendered: Vec<String> = groups
        .iter()
        .enumerate()
        .map(|(i, c)| render_group_within(i + 1, c, occurrences, sources, pages, room))
        .collect();
    let batches = pack_batches(&rendered, budgets.review_batch_bytes, per_call);
    let context = Arc::new(ReviewContext {
        groups: groups.to_vec(),
        rendered,
        occurrences: occurrences.to_vec(),
        sources: sources.to_vec(),
        pages: pages.to_vec(),
        language: language.to_string(),
        crawl_date: crawl_date.to_string(),
        max_tokens: budgets.review_max_tokens,
        room,
    });
    progress::start(TASK_REVIEW, "Consistency: review", batches.len() as u64);
    let mut handles = Vec::with_capacity(batches.len());
    for range in batches {
        let subject = format!("groups {}–{}", range.start + 1, range.end);
        handles.push(tokio::spawn(progress::unit(
            TASK_REVIEW,
            subject,
            review_batch(client.clone(), sem.clone(), context.clone(), range),
        )));
    }
    let mut results: Vec<Vec<ValidatedResult>> = (0..groups.len()).map(|_| Vec::new()).collect();
    for handle in handles {
        for (index, result) in handle.await.unwrap_or_default() {
            if let Some(list) = results.get_mut(index) {
                list.push(result);
            }
        }
    }
    progress::finish(TASK_REVIEW);
    results
}

/// What every review batch shares: the groups under review with their renderings, what renders a
/// group again (occurrences, sources, pages, the byte room of one group), the language of the
/// prose, the crawl date and the output budget.
struct ReviewContext {
    groups: Vec<Candidate>,
    rendered: Vec<String>,
    occurrences: Vec<Occurrence>,
    sources: Vec<AnalysisSource>,
    pages: Vec<Page>,
    language: String,
    crawl_date: String,
    max_tokens: u32,
    room: usize,
}

/// One group of a review call: its index in the groups under review, the candidate as the call
/// shows it, its rendering, and for each value id of that candidate (at index id − 1) the value's
/// id in the group.
#[derive(Clone)]
struct Ask {
    group: usize,
    candidate: Candidate,
    text: String,
    ids: Vec<usize>,
}

/// One review batch. An answer cut at the output limit splits the batch in half and asks again
/// for each half (at most `MAX_SPLIT_DEPTH` times). Once, a call asks again for what a usable
/// answer left out: the groups it did not answer (live: a broken quote ended the JSON early), and
/// the values of a group that its results did not compare with the rest of the group (none judged
/// them, or only apart from the rest: `judge::uncovered_values`) — that group shown again with
/// those values first. Each call asks for `review_call_max_tokens` of its groups. Returns the valid
/// results with the index of their group in `groups`, their value ids those of the group.
async fn review_batch(
    client: Arc<AiClient>,
    sem: Arc<Semaphore>,
    ctx: Arc<ReviewContext>,
    range: Range<usize>,
) -> Vec<(usize, ValidatedResult)> {
    let (groups, rendered) = (&ctx.groups, &ctx.rendered);
    let mut out: Vec<(usize, ValidatedResult)> = Vec::new();
    let asks: Vec<Ask> = range
        .filter_map(|i| {
            let candidate = groups.get(i)?.clone();
            let ids = (1..=candidate.values.len()).collect();
            Some(Ask {
                group: i,
                candidate,
                text: rendered.get(i)?.clone(),
                ids,
            })
        })
        .collect();
    // The groups of a call, its split depth, and whether it asks again for what an answer left out.
    let mut queue: Vec<(Vec<Ask>, usize, bool)> = vec![(asks, 0, false)];
    while let Some((asks, depth, again)) = queue.pop() {
        if asks.is_empty() {
            continue;
        }
        let batch: Vec<Candidate> = asks.iter().map(|a| a.candidate.clone()).collect();
        let texts: Vec<String> = asks.iter().map(|a| a.text.clone()).collect();
        let req = build_review_request(
            &groups_message(&texts),
            &ctx.language,
            &ctx.crawl_date,
            review_call_max_tokens(asks.len(), ctx.max_tokens),
        );
        let answer = {
            let _permit = sem.clone().acquire_owned().await.ok();
            client
                .complete_parsed_n(&req, CAT_REVIEW, PARSE_ATTEMPTS, parse_review_answer)
                .await
        };
        match answer {
            Ok((results, _)) => {
                for result in results {
                    if let Some(mut valid) = validate_with_date(result, &batch, &texts, &ctx.crawl_date) {
                        let ask = &asks[valid.0];
                        valid.4.values = valid
                            .4
                            .values
                            .iter()
                            .filter_map(|&id| ask.ids.get(id - 1).copied())
                            .collect();
                        out.push((ask.group, valid));
                    }
                }
                if again {
                    continue;
                }
                // What the answers so far left out: whole groups, and values of answered groups not
                // compared with the rest of their group.
                let (mut left_out, mut partly) = (Vec::new(), false);
                for ask in &asks {
                    let results: Vec<ValidatedResult> = out
                        .iter()
                        .filter(|(group, _)| *group == ask.group)
                        .map(|(_, result)| result.clone())
                        .collect();
                    let group = &groups[ask.group];
                    if results.is_empty() {
                        left_out.push(ask.clone());
                        continue;
                    }
                    let missing = uncovered_values(group, &results);
                    if !missing.is_empty() {
                        let (candidate, ids) = with_values_first(group, &missing);
                        let text = render_group_within(
                            ask.group + 1,
                            &candidate,
                            &ctx.occurrences,
                            &ctx.sources,
                            &ctx.pages,
                            ctx.room,
                        );
                        left_out.push(Ask {
                            group: ask.group,
                            candidate,
                            text,
                            ids,
                        });
                        partly = true;
                    }
                }
                // Asking for all of the same groups again would repeat the same (cached) request.
                if partly || (!left_out.is_empty() && left_out.len() < asks.len()) {
                    queue.push((left_out, depth, true));
                }
            }
            Err(error) if asks.len() > 1 && depth < MAX_SPLIT_DEPTH && error.to_string().contains(TRUNCATED) => {
                let mut first = asks;
                let second = first.split_off(first.len() / 2);
                queue.push((second, depth + 1, again));
                queue.push((first, depth + 1, again));
            }
            Err(_) => {}
        }
    }
    out
}

/// A review answer with at least one result (an empty one is asked again).
fn parse_review_answer(raw: &str) -> Result<Vec<ReviewResult>, String> {
    let results = parse_reviews(raw)?;
    if results.is_empty() {
        return Err("the answer has no results".to_string());
    }
    Ok(results)
}

/// What the document is built from.
struct Assembly<'a> {
    /// The reviewed groups with their valid results (none: not reviewed).
    reviewed: Vec<(&'a Candidate, Vec<ValidatedResult>)>,
    /// Why a reviewed group without a result was not reviewed.
    not_reviewed_reason: &'static str,
    over_cap: &'a [Candidate],
    consistent: Vec<model::ConsistentFact>,
    keys: &'a [model::FactKey],
    occurrences: &'a [Occurrence],
    pages: &'a [Page],
    sources: Vec<SourceOut>,
    failed_sources: Vec<FailedSource>,
}

/// S5: the document. A result of a finding becomes a finding (its prose localized when it was
/// replaced by deterministic text); an explainable or not-comparable one an explained group; an
/// insufficient-context one, and every group not reviewed, a not-judged group. The findings are
/// sorted and numbered `F1…`; the counts, completeness state and summary follow from them.
fn assemble(locale: &ReportLocale, meta: Meta, a: Assembly) -> ConsistencyDoc {
    let by_id: HashMap<usize, &Occurrence> = a.occurrences.iter().map(|o| (o.id, o)).collect();
    let values_of = |c: &Candidate, ids: &[usize]| -> Vec<FindingValue> {
        ids.iter()
            .filter_map(|&id| c.values.get(id.wrapping_sub(1)))
            .map(|v| finding_value(v, &by_id, a.pages))
            .collect()
    };
    let all_ids = |c: &Candidate| (1..=c.values.len()).collect::<Vec<_>>();
    let mut findings = Vec::new();
    let mut explained = Vec::new();
    let mut not_judged = Vec::new();
    let mut counts = Counts {
        consistent: a.consistent.len(),
        ..Counts::default()
    };
    for (candidate, results) in &a.reviewed {
        if results.is_empty() {
            counts.not_reviewed += 1;
            not_judged.push(NotJudgedGroup {
                key_id: candidate.key_id,
                key: candidate.name.clone(),
                attribute_key: candidate.attribute_key,
                status: a.not_reviewed_reason,
                reason: String::new(),
                values: values_of(candidate, &all_ids(candidate)),
            });
            continue;
        }
        // A second result for the same values of a group repeats the first (the first wins).
        let mut judged: Vec<Vec<usize>> = Vec::new();
        for (_, disposition, priority, confidence, result, replaced) in results {
            let mut values = result.values.clone();
            values.sort_unstable();
            if judged.contains(&values) {
                continue;
            }
            judged.push(values);
            let (title, explanation, check, benign) = if *replaced {
                // The reason of a non-finding says what the review decided.
                let explanation = match disposition {
                    Disposition::Explainable => "prose_reason_explainable",
                    Disposition::NotComparable => "prose_reason_not_comparable",
                    Disposition::InsufficientContext => "prose_reason_insufficient",
                    _ => "prose_explanation",
                };
                (
                    doc::text(locale, "prose_title").replace("{name}", &candidate.name),
                    doc::text(locale, explanation).to_string(),
                    doc::text(locale, "prose_check").to_string(),
                    Vec::new(),
                )
            } else {
                (
                    result.title.clone(),
                    result.explanation.clone(),
                    result.check.clone(),
                    result.benign.clone(),
                )
            };
            let values = values_of(candidate, &result.values);
            match (disposition, confidence) {
                (Disposition::Finding, Some(confidence)) => {
                    let priority = priority.unwrap_or(Priority::Medium);
                    match priority {
                        Priority::Critical => counts.critical += 1,
                        Priority::High => counts.high += 1,
                        Priority::Medium => counts.medium += 1,
                        Priority::Low => counts.low += 1,
                    }
                    findings.push(Finding {
                        id: String::new(),
                        key_id: candidate.key_id,
                        priority,
                        confidence: *confidence,
                        attribute_key: candidate.attribute_key,
                        key: candidate.name.clone(),
                        title,
                        explanation,
                        benign_explanations: benign,
                        check,
                        prose_replaced: *replaced,
                        values,
                    });
                }
                (Disposition::Explainable | Disposition::NotComparable, _) => {
                    let not_comparable = *disposition == Disposition::NotComparable;
                    if not_comparable {
                        counts.not_comparable += 1;
                    } else {
                        counts.explained += 1;
                    }
                    explained.push(ExplainedGroup {
                        key_id: candidate.key_id,
                        key: candidate.name.clone(),
                        attribute_key: candidate.attribute_key,
                        disposition: if not_comparable {
                            "not_comparable"
                        } else {
                            "explainable"
                        },
                        title,
                        reason: explanation,
                        benign_explanations: benign,
                        values,
                    });
                }
                _ => {
                    counts.insufficient += 1;
                    not_judged.push(NotJudgedGroup {
                        key_id: candidate.key_id,
                        key: candidate.name.clone(),
                        attribute_key: candidate.attribute_key,
                        status: "insufficient_context",
                        reason: explanation,
                        values,
                    });
                }
            }
        }
        // Values no result compared with the rest of the group, even when asked again.
        let left_out = uncovered_values(candidate, results);
        if !left_out.is_empty() {
            counts.not_reviewed += 1;
            not_judged.push(NotJudgedGroup {
                key_id: candidate.key_id,
                key: candidate.name.clone(),
                attribute_key: candidate.attribute_key,
                status: "not_reviewed_left_out",
                reason: String::new(),
                values: values_of(candidate, &left_out),
            });
        }
    }
    for candidate in a.over_cap {
        counts.not_reviewed += 1;
        not_judged.push(NotJudgedGroup {
            key_id: candidate.key_id,
            key: candidate.name.clone(),
            attribute_key: candidate.attribute_key,
            status: "not_reviewed_cap",
            reason: String::new(),
            values: values_of(candidate, &all_ids(candidate)),
        });
    }
    doc::sort_findings(&mut findings);
    for (i, finding) in findings.iter_mut().enumerate() {
        finding.id = format!("F{}", i + 1);
    }
    let completeness = doc::completeness(&meta, &counts, a.failed_sources.len(), meta.sources);
    let key_of: HashMap<usize, usize> = a
        .keys
        .iter()
        .flat_map(|k| k.occurrence_ids.iter().map(move |&o| (o, k.id)))
        .collect();
    let keys = a
        .keys
        .iter()
        .map(|k| KeyOut {
            id: k.id,
            attribute_key: k.attribute_key,
            name: k.name.clone(),
            aliases: k.aliases.clone(),
            occurrence_ids: k.occurrence_ids.clone(),
            outcome: key_outcome(k, a.occurrences),
        })
        .collect();
    let occurrences = a
        .occurrences
        .iter()
        .map(|o| {
            let (value_key, value_exact) = match &o.value_key {
                ValueKey::Exact(key) => (key.clone(), true),
                ValueKey::Uncertain(key) => (key.clone(), false),
            };
            OccurrenceOut {
                id: o.id,
                key_id: key_of.get(&o.id).copied(),
                source: o.source,
                region: o.region,
                block_ref: o.block_ref.clone(),
                attribute_key: o.attribute_key,
                subject: o.subject.clone(),
                attribute: o.attribute.clone(),
                value: o.value.clone(),
                value_key,
                value_exact,
                qualifiers: o.qualifiers.clone(),
                evidence: o.evidence.clone(),
                heading_path: o.heading_path.clone(),
                urls: urls_of(&o.pages, a.pages),
            }
        })
        .collect();
    ConsistencyDoc {
        schema: SCHEMA,
        summary: doc::summary(locale, &meta, &counts, &completeness),
        disclaimer: doc::text(locale, "disclaimer").to_string(),
        caution: doc::text(locale, "caution").to_string(),
        by_page: doc::by_page(&findings),
        meta,
        completeness,
        counts,
        findings,
        explained,
        not_judged,
        consistent: a.consistent,
        keys,
        sources: a.sources,
        occurrences,
        failed_sources: a.failed_sources,
    }
}

/// A value with its occurrences (the widest first) and every page showing it.
fn finding_value(value: &CandidateValue, by_id: &HashMap<usize, &Occurrence>, pages: &[Page]) -> FindingValue {
    let mut occurrences: Vec<OccurrenceRef> = value
        .occurrence_ids
        .iter()
        .filter_map(|id| by_id.get(id))
        .map(|o| OccurrenceRef {
            occurrence_id: o.id,
            region: o.region,
            path: match o.region {
                SourceKind::Page => o
                    .pages
                    .first()
                    .and_then(|&p| pages.get(p))
                    .map(|p| p.path.clone())
                    .unwrap_or_default(),
                SourceKind::Chrome => String::new(),
            },
            qualifiers: o.qualifiers.clone(),
            heading_path: o.heading_path.clone(),
            evidence: o.evidence.clone(),
            urls: urls_of(&o.pages, pages),
        })
        .collect();
    occurrences.sort_by(|a, b| {
        b.urls
            .len()
            .cmp(&a.urls.len())
            .then(a.occurrence_id.cmp(&b.occurrence_id))
    });
    FindingValue {
        text: value.text.clone(),
        occurrences,
        urls: urls_of(&value.pages, pages),
    }
}

/// The URLs of page indexes, sorted.
fn urls_of(indexes: &[usize], pages: &[Page]) -> Vec<String> {
    let mut urls: Vec<String> = indexes
        .iter()
        .filter_map(|&i| pages.get(i))
        .map(|p| p.url.clone())
        .collect();
    urls.sort();
    urls.dedup();
    urls
}

#[cfg(test)]
mod tests {
    //! Budget stress tests: every final user message fits its byte budget, and every batch its
    //! output budget, from small to large context windows (the budgets of Design §4).

    use super::group_all;
    use super::judge::{
        Candidate, CandidateValue, Origin, cohorts, groups_message, pack_batches, render_group_within,
        review_groups_per_call,
    };
    use super::keys::{LabelItem, group_message, max_items_by_output, pack_chunks};
    use super::model::{AnalysisSource, AttributeKey, Occurrence, Page, SourceKind};
    use super::sources::{chrome_lines, chrome_sources, page_source, source_message};
    use crate::ai::blocks::blocks_from_html;
    use crate::ai::grounding::ValueKey;
    use crate::ai::profile::budget::ContextBudget;

    const CONTEXTS: [i64; 5] = [8_000, 16_000, 32_000, 128_000, 262_144];
    const WORDS: &str = "Příliš žluťoučký kůň úpěl ďábelské ódy 🐴 <b>&</b>";

    struct Budgets {
        extract_input_bytes: usize,
        group_chunk_bytes: usize,
        group_max_tokens: u32,
        review_batch_bytes: usize,
        review_max_tokens: u32,
    }

    fn budgets(ctx: i64) -> Budgets {
        let b = ContextBudget::new(ctx, 32_000);
        Budgets {
            extract_input_bytes: b.scaled(10, 3),
            group_chunk_bytes: b.scaled(40, 6),
            group_max_tokens: b.out_tokens().min(6_000),
            review_batch_bytes: b.scaled(40, 6),
            review_max_tokens: b.out_tokens(),
        }
    }

    #[test]
    fn fees_and_prices_share_a_bucket_and_a_key_keeps_its_own_attribute() {
        use super::{bucket_items, key_attribute};
        let fact = |id: usize, key: AttributeKey, subject: &str| Occurrence {
            id,
            source: id,
            region: SourceKind::Page,
            block_ref: "B1".to_string(),
            attribute_key: key,
            subject: subject.to_string(),
            attribute: "cena".to_string(),
            value: format!("{id}90 Kč"),
            value_key: ValueKey::Exact(format!("num:{id}90:CZK:")),
            qualifiers: String::new(),
            evidence: format!("{id}90 Kč"),
            value_span: (0, 6),
            heading_path: Vec::new(),
            pages: vec![id],
        };
        let occurrences = vec![
            fact(0, AttributeKey::Price, "Instalace"),
            fact(1, AttributeKey::Fee, "Instalace"),
            fact(2, AttributeKey::Fee, "Výjezd"),
            fact(3, AttributeKey::Phone, "Linka"),
        ];
        let buckets = bucket_items(&occurrences);
        let keys: Vec<(AttributeKey, usize)> = buckets.iter().map(|(key, items)| (*key, items.len())).collect();
        assert_eq!(keys, vec![(AttributeKey::Phone, 1), (AttributeKey::Price, 2)]);
        assert_eq!(
            key_attribute(AttributeKey::Price, &[2], &occurrences),
            AttributeKey::Fee
        );
        assert_eq!(
            key_attribute(AttributeKey::Price, &[1, 2], &occurrences),
            AttributeKey::Fee
        );
        assert_eq!(
            key_attribute(AttributeKey::Price, &[0, 1], &occurrences),
            AttributeKey::Price,
            "a tie goes to the bucket"
        );
    }

    /// A review whose prose was replaced (a verdict word, a foreign number, a missing text) keeps
    /// its disposition; the reason shown for an explained or not-judged group then says what the
    /// review decided, not the finding text "the pages state different values".
    #[test]
    fn replaced_prose_of_a_non_finding_gets_a_reason_that_fits_its_disposition() {
        use super::doc::Meta;
        use super::judge::{Candidate, CandidateValue, Origin, ReviewResult};
        use super::model::{Confidence, Disposition, Priority};
        use super::{Assembly, assemble, doc, judge};
        use crate::ai::report::locale::ReportLocale;
        let pages: Vec<Page> = (0..2)
            .map(|i| Page {
                index: i,
                url: format!("https://example.com/{i}"),
                path: format!("/{i}"),
                title: String::new(),
            })
            .collect();
        let occurrences: Vec<Occurrence> = (0..2)
            .map(|i| Occurrence {
                id: i,
                source: i,
                region: SourceKind::Page,
                block_ref: "B1".to_string(),
                attribute_key: AttributeKey::Price,
                subject: "Tarif".to_string(),
                attribute: "cena".to_string(),
                value: format!("{} Kč", 100 + i),
                value_key: ValueKey::Exact(format!("num:{}:CZK:", 100 + i)),
                qualifiers: String::new(),
                evidence: format!("Tarif {} Kč", 100 + i),
                value_span: (6, 12),
                heading_path: Vec::new(),
                pages: vec![i],
            })
            .collect();
        let candidate = Candidate {
            key_id: 0,
            name: "Tarif – cena".to_string(),
            attribute_key: AttributeKey::Price,
            values: (0..2)
                .map(|i| CandidateValue {
                    id: i + 1,
                    key: occurrences[i].value_key.clone(),
                    text: occurrences[i].value.clone(),
                    occurrence_ids: vec![i],
                    source_ids: vec![i],
                    origins: vec![Origin::Page(i)],
                    pages: vec![i],
                    qualified: false,
                })
                .collect(),
            baseline: None,
        };
        let result = |disposition: Disposition, confidence: Option<Confidence>| {
            let review = ReviewResult {
                group: 1,
                values: vec![1, 2],
                confidence: String::new(),
                priority: String::new(),
                title: "Tarif – cena: values differ".to_string(),
                explanation: "The pages state different values for the same property.".to_string(),
                benign: Vec::new(),
                check: String::new(),
            };
            (
                0,
                disposition,
                confidence.map(|_| Priority::Medium),
                confidence,
                review,
                true,
            )
        };
        for (language, expect) in [("en", "plausible legitimate reason"), ("cs", "oprávněný důvod")] {
            let locale = ReportLocale::new(language);
            let doc = assemble(
                &locale,
                Meta::default(),
                Assembly {
                    reviewed: vec![
                        (&candidate, vec![result(Disposition::Explainable, None)]),
                        (&candidate, vec![result(Disposition::NotComparable, None)]),
                        (&candidate, vec![result(Disposition::InsufficientContext, None)]),
                        (
                            &candidate,
                            vec![
                                result(Disposition::Finding, Some(Confidence::Possible)),
                                result(Disposition::Finding, Some(Confidence::Possible)),
                            ],
                        ),
                    ],
                    not_reviewed_reason: "not_reviewed_call_failed",
                    over_cap: &[],
                    consistent: Vec::new(),
                    keys: &[],
                    occurrences: &occurrences,
                    pages: &pages,
                    sources: Vec::new(),
                    failed_sources: Vec::new(),
                },
            );
            // The finding's result came twice for the same values (live: a thinking model); the
            // second copy is dropped.
            assert_eq!(doc.findings.len(), 1, "one finding per group and value set");
            assert_eq!(doc.counts.medium, 1);
            assert_eq!(doc.explained.len(), 2);
            assert_eq!(doc.explained[0].reason, doc::text(&locale, "prose_reason_explainable"));
            assert!(doc.explained[0].reason.contains(expect), "{}", doc.explained[0].reason);
            assert_eq!(
                doc.explained[1].reason,
                doc::text(&locale, "prose_reason_not_comparable")
            );
            assert_eq!(
                doc.not_judged[0].reason,
                doc::text(&locale, "prose_reason_insufficient")
            );
            // A finding keeps the deterministic finding text.
            assert_eq!(doc.findings[0].explanation, doc::text(&locale, "prose_explanation"));
            for reason in [
                "prose_reason_explainable",
                "prose_reason_not_comparable",
                "prose_reason_insufficient",
            ] {
                assert!(!judge::has_forbidden_word(doc::text(&locale, reason)), "{reason}");
            }
        }
    }

    /// Grouping runs round by round across the attribute buckets: a round's calls start only
    /// once every bucket has finished or reached that round, so its progress total is exact, and
    /// the progress task is restarted per round.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn grouping_rounds_start_only_when_every_bucket_reached_them() {
        use std::sync::{Arc, Mutex};
        let item = |subject: &str, id: usize| LabelItem {
            subject: subject.to_string(),
            attribute: "tel".to_string(),
            example: String::new(),
            aliases: Vec::new(),
            occurrence_ids: vec![id],
        };
        let phones = vec![item("Alfa", 0), item("Beta", 1), item("Gama", 2), item("Delta", 3)];
        let emails = vec![item("Info", 4), item("Podpora", 5)];
        let fees = vec![item("Poplatek", 6)];
        let log: Arc<Mutex<Vec<(AttributeKey, usize, &'static str)>>> = Arc::new(Mutex::new(Vec::new()));
        let ask = {
            let log = log.clone();
            move |key: AttributeKey, level: usize, items: Vec<LabelItem>| {
                let log = log.clone();
                async move {
                    log.lock().unwrap().push((key, level, "start"));
                    if key == AttributeKey::Email {
                        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                    }
                    log.lock().unwrap().push((key, level, "end"));
                    Ok(vec![((1..=items.len()).collect(), String::new())])
                }
            }
        };
        let task = "test:consistency-rounds";
        let outcomes = group_all(
            vec![
                (AttributeKey::Phone, phones),
                (AttributeKey::Email, emails),
                (AttributeKey::Fee, fees),
            ],
            100_000,
            2,
            task,
            ask,
        )
        .await;
        // Phone: 4 labels in 2 chunks of 2 (round 1), merged into 2 components that meet in one
        // chunk (round 2). Email: one chunk. Fee: a single label needs no call.
        let calls: Vec<usize> = outcomes.iter().map(|o| o.calls).collect();
        assert_eq!(calls, vec![3, 1, 0]);
        let keys: Vec<usize> = outcomes.iter().map(|o| o.keys.len()).collect();
        assert_eq!(keys, vec![1, 1, 1]);
        assert!(outcomes.iter().all(|o| !o.incomplete));
        let log = log.lock().unwrap().clone();
        let email_done = log
            .iter()
            .position(|e| *e == (AttributeKey::Email, 1, "end"))
            .expect("the email call");
        let round_two = log
            .iter()
            .position(|(_, level, _)| *level == 2)
            .expect("a round-2 call");
        assert!(
            email_done < round_two,
            "round 2 waited for the slow round-1 call: {log:?}"
        );
        assert_eq!(
            crate::ai::progress::label(task).as_deref(),
            Some("Consistency: grouping (round 2)")
        );
        assert_eq!(crate::ai::progress::current(task), Some((1, 1)));
    }

    #[test]
    fn extraction_messages_fit_every_context() {
        let paragraphs: String = (0..300)
            .map(|i| {
                format!(
                    "<p>{WORDS} odstavec {i}{}</p>",
                    if i % 40 == 0 { ", cena 1 290 Kč" } else { "" }
                )
            })
            .collect();
        let html = format!(
            "<html><body><main><h1>{WORDS}</h1>{paragraphs}<table><tr><th>Tarif</th><th>Cena</th></tr>\
             <tr><td>{WORDS}</td><td>290 Kč</td></tr></table><p>{}</p></main></body></html>",
            WORDS.repeat(200)
        );
        let blocks = blocks_from_html(&html);
        let page = Page {
            index: 0,
            url: "https://example.com/cenik".to_string(),
            path: "/cenik".to_string(),
            title: WORDS.repeat(10),
        };
        let footers: Vec<(usize, Vec<_>)> = (0..50)
            .map(|i| {
                let lines: String = (0..30)
                    .map(|j| format!("<p>{WORDS} pobočka {j}: tel. 800 {i:03} {j:03}</p>"))
                    .collect();
                (i, blocks_from_html(&format!("<body><footer>{lines}</footer></body>")))
            })
            .collect();
        let lines = chrome_lines(&footers);
        for ctx in CONTEXTS {
            let budget = budgets(ctx).extract_input_bytes;
            let source = page_source(&page, &blocks, budget);
            assert!(
                source.omitted_blocks > 0,
                "ctx {ctx}: the page is larger than the budget"
            );
            assert!(
                source.blocks.iter().any(|b| b.text.contains("290 Kč")),
                "ctx {ctx}: facts kept"
            );
            let message = source_message(&source, &page.title);
            assert!(message.len() <= budget, "ctx {ctx}: page {} > {budget}", message.len());
            for chunk in chrome_sources(&lines, budget, 1) {
                let message = source_message(&chunk, "");
                assert!(
                    message.len() <= budget,
                    "ctx {ctx}: chrome {} > {budget}",
                    message.len()
                );
            }
        }
    }

    #[test]
    fn grouping_chunks_fit_the_input_and_the_output_budget() {
        let items: Vec<LabelItem> = (0..1_000)
            .map(|i| LabelItem {
                subject: format!("{i} {}", WORDS.repeat(3)).chars().take(120).collect(),
                attribute: format!("{} {i}", WORDS.repeat(3)).chars().take(120).collect(),
                example: format!("{i} 290 Kč"),
                aliases: vec![format!("{WORDS} {i}"), format!("alias {i} {WORDS}")],
                occurrence_ids: vec![i],
            })
            .collect();
        for ctx in CONTEXTS {
            let b = budgets(ctx);
            let max_items = max_items_by_output(b.group_max_tokens);
            let chunks = pack_chunks(&items, AttributeKey::Price, b.group_chunk_bytes, max_items);
            assert!(chunks.len() > 1, "ctx {ctx}");
            assert_eq!(chunks.iter().map(|r| r.len()).sum::<usize>(), items.len());
            for range in chunks {
                assert!(range.len() <= max_items, "ctx {ctx}");
                let message = group_message(AttributeKey::Price, &items[range]);
                assert!(message.len() <= b.group_chunk_bytes, "ctx {ctx}: {}", message.len());
            }
        }
    }

    #[test]
    fn review_batches_fit_the_input_and_the_output_budget() {
        // One key with 40 values, each stated on 6 pages with long multibyte context.
        let pages: Vec<Page> = (0..240)
            .map(|i| Page {
                index: i,
                url: format!("https://example.com/{i}"),
                path: format!("/{}/{i}", "sekce-ž".repeat(8)),
                title: String::new(),
            })
            .collect();
        let sources: Vec<AnalysisSource> = pages
            .iter()
            .map(|p| AnalysisSource {
                id: p.index,
                kind: SourceKind::Page,
                url: p.url.clone(),
                path: p.path.clone(),
                blocks: Vec::new(),
                omitted_blocks: 0,
                truncated_blocks: 0,
            })
            .collect();
        let mut occ = Vec::new();
        let mut values = Vec::new();
        for v in 0..40 {
            let text = format!("{} Kč", 1_000 + v * 10);
            let ids: Vec<usize> = (0..6).map(|n| v * 6 + n).collect();
            for &id in &ids {
                occ.push(Occurrence {
                    id,
                    source: id,
                    region: SourceKind::Page,
                    block_ref: "B1".to_string(),
                    attribute_key: AttributeKey::Price,
                    subject: WORDS.to_string(),
                    attribute: WORDS.to_string(),
                    value: text.clone(),
                    value_key: ValueKey::Exact(format!("num:{text}")),
                    qualifiers: format!("{} {id}", WORDS.repeat(4)),
                    evidence: format!("{} {text} {}", WORDS.repeat(3), WORDS.repeat(3)),
                    value_span: (WORDS.repeat(3).len() + 1, WORDS.repeat(3).len() + 1 + text.len()),
                    heading_path: vec![WORDS.repeat(2), format!("{WORDS} {id}")],
                    pages: vec![id],
                });
            }
            values.push(CandidateValue {
                id: v + 1,
                key: ValueKey::Exact(format!("num:{text}")),
                text,
                occurrence_ids: ids.clone(),
                source_ids: ids.clone(),
                origins: ids.iter().map(|&id| Origin::Page(id)).collect(),
                pages: ids,
                qualified: true,
            });
        }
        let candidate = Candidate {
            key_id: 0,
            name: WORDS.repeat(5),
            attribute_key: AttributeKey::Price,
            values,
            baseline: None,
        };
        let cohorts = cohorts(candidate);
        assert_eq!(cohorts.len(), 8, "a baseline against 39 others, 5 at a time");
        for ctx in CONTEXTS {
            let b = budgets(ctx);
            let per_call = review_groups_per_call(b.review_max_tokens);
            assert!(per_call >= 1, "ctx {ctx}");
            let room = b.review_batch_bytes - groups_message(&[]).len() - 1;
            let rendered: Vec<String> = cohorts
                .iter()
                .enumerate()
                .map(|(i, c)| render_group_within(i + 1, c, &occ, &sources, &pages, room))
                .collect();
            for (i, group) in rendered.iter().enumerate() {
                assert!(group.len() <= room, "ctx {ctx}: group {i} {} > {room}", group.len());
                assert_eq!(group.matches("<value id=").count(), cohorts[i].values.len());
            }
            for range in pack_batches(&rendered, b.review_batch_bytes, per_call) {
                assert!(range.len() <= per_call, "ctx {ctx}");
                let message = groups_message(&rendered[range]);
                assert!(message.len() <= b.review_batch_bytes, "ctx {ctx}: {}", message.len());
            }
        }
    }
}

#[cfg(test)]
mod review_budget_tests {
    use super::judge::{Candidate, CandidateValue, Origin, groups_message, render_group_within};
    use super::model::{AnalysisSource, AttributeKey, Occurrence, Page, SourceKind};
    use crate::ai::grounding::ValueKey;
    use crate::ai::profile::budget::ContextBudget;

    /// Six values, each stated once on its own page with the path `path(i)`, `field` for every
    /// crawler text of the occurrence, and the value text `value(i)`.
    fn case(
        path: impl Fn(usize) -> String,
        field: &str,
        value: impl Fn(usize) -> String,
    ) -> (Candidate, Vec<Occurrence>, Vec<AnalysisSource>, Vec<Page>) {
        let pages: Vec<Page> = (0..6)
            .map(|i| Page {
                index: i,
                url: format!("https://example.com{}", path(i)),
                path: path(i),
                title: String::new(),
            })
            .collect();
        let sources: Vec<AnalysisSource> = pages
            .iter()
            .map(|p| AnalysisSource {
                id: p.index,
                kind: SourceKind::Page,
                url: p.url.clone(),
                path: p.path.clone(),
                blocks: Vec::new(),
                omitted_blocks: 0,
                truncated_blocks: 0,
            })
            .collect();
        let occ: Vec<Occurrence> = (0..6)
            .map(|i| Occurrence {
                id: i,
                source: i,
                region: SourceKind::Page,
                block_ref: "B1".to_string(),
                attribute_key: AttributeKey::Price,
                subject: field.to_string(),
                attribute: field.to_string(),
                value: value(i),
                value_key: ValueKey::Exact(format!("num:{i}")),
                qualifiers: field.to_string(),
                evidence: format!("{field} {} {field}", value(i)),
                value_span: (field.len() + 1, field.len() + 1 + value(i).len()),
                heading_path: vec![field.to_string(), field.to_string()],
                pages: vec![i],
            })
            .collect();
        let values = (0..6)
            .map(|i| CandidateValue {
                id: i + 1,
                key: ValueKey::Exact(format!("num:{i}")),
                text: value(i),
                occurrence_ids: vec![i],
                source_ids: vec![i],
                origins: vec![Origin::Page(i)],
                pages: vec![i],
                qualified: true,
            })
            .collect();
        let candidate = Candidate {
            key_id: 0,
            name: field.to_string(),
            attribute_key: AttributeKey::Price,
            values,
            baseline: None,
        };
        (candidate, occ, sources, pages)
    }

    fn room(ctx: i64) -> usize {
        ContextBudget::new(ctx, 2_000).scaled(40, 6) - groups_message(&[]).len() - 1
    }

    #[test]
    fn long_page_paths_do_not_push_a_group_over_the_review_budget() {
        // Review: six prices on paths of 1,760 characters at an 8K context sent 9,728 bytes for a
        // 6,144-byte budget, and without any evidence.
        let (candidate, occ, sources, pages) = case(
            |i| format!("/section-{}-{i}", "abcdefghijklmnpqrstuvwxyz".repeat(70)),
            "Tariff Mini: regular monthly price",
            |i| format!("{} EUR", 100 * (i + 1)),
        );
        let room = room(8_192);
        let group = render_group_within(1, &candidate, &occ, &sources, &pages, room);
        assert!(group.len() <= room, "{} > {room}", group.len());
        assert_eq!(group.matches("<evidence>").count(), 6, "the evidence is kept");
    }

    #[test]
    fn the_leanest_group_fits_the_smallest_review_budget() {
        // Every text as long as it can be and escaping to 4 bytes per character.
        let hostile = "<".repeat(2_000);
        let (candidate, occ, sources, pages) = case(
            |i| format!("/{}{i}", "<".repeat(3_000)),
            &hostile,
            |i| format!("{hostile}{i}"),
        );
        for ctx in [8_192, 16_000, 32_000] {
            let room = room(ctx);
            let group = render_group_within(1, &candidate, &occ, &sources, &pages, room);
            assert!(group.len() <= room, "ctx {ctx}: {} > {room}", group.len());
            assert_eq!(group.matches("<value id=").count(), 6, "every value is shown");
        }
    }
}

#[cfg(test)]
mod country_tests {
    use super::model::{AnalysisSource, Page, SourceBlock, SourceKind};
    use super::source_country;

    fn page(index: usize, url: &str) -> Page {
        Page {
            index,
            url: url.to_string(),
            path: "/".to_string(),
            title: String::new(),
        }
    }

    fn source(kind: SourceKind, id: usize, pages: &[usize]) -> AnalysisSource {
        AnalysisSource {
            id,
            kind,
            url: String::new(),
            path: String::new(),
            blocks: vec![SourceBlock {
                ref_id: "L1".to_string(),
                text: "020 7946 0123".to_string(),
                heading_path: Vec::new(),
                pages: pages.to_vec(),
            }],
            omitted_blocks: 0,
            truncated_blocks: 0,
        }
    }

    #[test]
    fn each_page_reads_national_numbers_by_its_own_country() {
        let pages = vec![
            page(0, "https://example.com/"),
            page(1, "https://example.com/uk"),
            page(2, "https://example.com/de"),
            page(3, "https://example.com/de/kontakt"),
            page(4, "https://example.com/en"),
            page(5, "https://example.com/x"),
            page(6, "https://example.cz/"),
        ];
        let langs: Vec<String> = ["de-DE", "en-GB", "de-DE", "de", "en", "", "en-GB"]
            .map(str::to_string)
            .to_vec();
        let country = |kind: SourceKind, id: usize, on: &[usize]| {
            source_country(&source(kind, id, on), &pages, &langs, "de-DE", "example.com")
        };
        assert_eq!(country(SourceKind::Page, 0, &[0]), Some("DE"));
        assert_eq!(country(SourceKind::Page, 1, &[1]), Some("GB"), "the page's own region");
        assert_eq!(country(SourceKind::Page, 3, &[3]), Some("DE"), "the site's language");
        assert_eq!(country(SourceKind::Page, 4, &[4]), None, "another language, no region");
        assert_eq!(
            country(SourceKind::Page, 5, &[5]),
            Some("DE"),
            "no language: the site's"
        );
        assert_eq!(country(SourceKind::Page, 6, &[6]), Some("CZ"), "the page's own TLD");
        // Header/footer lines: the country all their pages share, else none.
        assert_eq!(country(SourceKind::Chrome, 9, &[0, 2, 3]), Some("DE"));
        assert_eq!(country(SourceKind::Chrome, 9, &[0, 1, 2]), None);
    }
}

#[cfg(test)]
mod coverage_tests {
    use super::model::{Page, SourceKind};
    use super::{page_counts, page_sources, unfit_source_out};
    use crate::ai::blocks::blocks_from_html;

    #[test]
    fn a_page_whose_content_fits_no_budget_is_reported_not_dropped() {
        let pages = vec![
            Page {
                index: 0,
                url: "https://example.com/".to_string(),
                path: "/".to_string(),
                title: "Service".to_string(),
            },
            Page {
                index: 1,
                url: format!("https://example.com/?q={}", "x".repeat(4_000)),
                path: "/?q=…".to_string(),
                title: "Service".to_string(),
            },
            Page {
                index: 2,
                url: "https://example.com/empty".to_string(),
                path: "/empty".to_string(),
                title: String::new(),
            },
        ];
        let html = "<main><h1>Service</h1><p>Company ID 12345678</p></main>";
        let pages_blocks = vec![
            (0, blocks_from_html(html)),
            (1, blocks_from_html(html)),
            (2, blocks_from_html("<main></main>")),
        ];
        let (sources, unfit) = page_sources(&pages, &pages_blocks, 3_072);
        assert_eq!(sources.iter().map(|s| s.id).collect::<Vec<_>>(), [0]);
        assert_eq!(
            unfit.iter().map(|s| s.id).collect::<Vec<_>>(),
            [1],
            "not the empty page"
        );
        assert_eq!(unfit[0].omitted_blocks, 2);

        let mut outs: Vec<_> = Vec::new();
        outs.extend(unfit.iter().map(unfit_source_out));
        assert_eq!(outs[0].kind, SourceKind::Page);
        assert_eq!((outs[0].blocks, outs[0].omitted_blocks, outs[0].failed), (0, 2, false));
        // Analyzed: pages with content sent; reduced: pages with blocks not inspected.
        assert_eq!(page_counts(&outs), (0, 1));
    }
}
