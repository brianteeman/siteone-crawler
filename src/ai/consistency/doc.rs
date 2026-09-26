// SiteOne Crawler - AI fact consistency: the report document and its renderers
// (c) Jan Reges <jan.reges@siteone.cz>
//
// The stored result of `--ai-consistency`: the findings (possible inconsistencies to verify), the
// differences with a plausible explanation, the ones that could not be judged, the values seen
// alike in several places, and the full audit trail (every source, every kept occurrence, every
// failed source) with an explicit completeness state. Rendered to Markdown, JSON, self-contained
// HTML and CSV. Every fixed text (disclaimer, caution line, labels, summary templates) comes from
// `text` in English or Czech; the prose the model wrote follows `--ai-report-language`.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use serde_json::Value;

use crate::ai::profile::doc::{LOGO_SVG, esc, is_safe_url};
use crate::ai::report::locale::ReportLocale;

use super::model::{AttributeKey, Confidence, ConsistentFact, Priority, SourceKind};

/// The `schema` of the JSON document.
pub const SCHEMA: &str = "siteone-crawler/ai-consistency/2";
/// URLs listed openly per value; the rest is collapsed.
pub const URLS_SHOWN: usize = 5;
/// Occurrences shown openly per value; the rest is collapsed.
pub const OCCURRENCES_SHOWN: usize = 3;
/// Facts with the same value listed in the Markdown and HTML report (the JSON has all of them).
pub const CONSISTENT_SHOWN: usize = 20;
/// Fewer sources with facts than this is too little evidence for a comparison.
const MIN_SOURCES_WITH_FACTS: usize = 3;

/// The whole report.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConsistencyDoc {
    pub schema: &'static str,
    pub meta: Meta,
    pub completeness: Completeness,
    pub disclaimer: String,
    pub caution: String,
    pub summary: String,
    pub counts: Counts,
    /// Ordered by `sort_findings`, numbered `F1…`.
    pub findings: Vec<Finding>,
    pub by_page: Vec<PageFindings>,
    /// Differences the review found explainable or not comparable, with the reason.
    pub explained: Vec<ExplainedGroup>,
    /// Differences the evidence did not allow to judge, and those not reviewed.
    pub not_judged: Vec<NotJudgedGroup>,
    /// Facts stated with one exact value in several places.
    pub consistent: Vec<ConsistentFact>,
    pub sources: Vec<SourceOut>,
    /// Every kept fact occurrence.
    pub occurrences: Vec<OccurrenceOut>,
    pub failed_sources: Vec<FailedSource>,
}

/// What was analyzed and how.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Meta {
    pub host: String,
    pub url: String,
    pub generated_at: String,
    /// The report language (`--ai-report-language`).
    pub language: String,
    pub provider: String,
    pub model: String,
    pub context_window: i64,
    /// Candidate pages after the include/exclude masks.
    pub pages_eligible: usize,
    /// The first `--ai-max-pages` of them.
    pub pages_selected: usize,
    /// Selected pages whose content was extracted.
    pub pages_analyzed: usize,
    /// Selected pages without a stored HTML body.
    pub pages_missing_body: usize,
    /// Selected pages whose input was reduced (blocks left out or cut; all of them for a page
    /// whose content fit no extraction request, which is then not analyzed).
    pub pages_reduced: usize,
    pub chrome_lines_analyzed: usize,
    pub chrome_lines_excluded: usize,
    /// Extraction sources: pages with content, and header/footer chunks.
    pub sources: usize,
    /// Sources that yielded at least one kept fact.
    pub sources_with_facts: usize,
    /// Facts the model returned, facts kept (verified), facts dropped as ungrounded.
    pub facts_extracted: usize,
    pub facts_kept: usize,
    pub facts_ungrounded: usize,
    /// Fact keys after grouping, and keys with differing values.
    pub keys: usize,
    pub candidates: usize,
    /// Review groups (candidates or their cohorts) with a valid review result.
    pub reviewed: usize,
    /// Values compared in the reviewed groups (each against the others: values − 1 per group).
    pub comparisons_done: usize,
    /// Review groups not reviewed: over the review cap, after a failed call, or because
    /// `--ai-max-tokens` is too low for a review.
    pub reviews_capped: usize,
    pub reviews_failed: usize,
    pub reviews_skipped_budget: usize,
    /// Reviewed groups with values that no result judged, even when asked again.
    pub reviews_partial: usize,
    pub grouping_incomplete: bool,
    pub items_not_cross_compared: usize,
    pub llm_calls: usize,
    pub duration_ms: u64,
    /// The TTL of the crawler's HTTP cache when it was on (pages may be older than the run).
    pub http_cache: Option<String>,
}

/// How many results of each kind the review produced.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Counts {
    pub critical: usize,
    pub high: usize,
    pub medium: usize,
    pub low: usize,
    pub explained: usize,
    pub not_comparable: usize,
    pub insufficient: usize,
    pub not_reviewed: usize,
    pub consistent: usize,
}

/// How much of the selected scope the report covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletenessState {
    /// Every selected source was extracted, grouping finished, and every candidate was reviewed.
    Complete,
    Partial,
    /// Fewer than 3 sources yielded facts, or at least half of the sources failed.
    Insufficient,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Completeness {
    pub state: CompletenessState,
    pub reasons: Vec<CompletenessReason>,
}

/// One reason the report is partial: a stable `code` (see `reason_text`) and its count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CompletenessReason {
    pub code: &'static str,
    pub count: usize,
}

/// A possible inconsistency to verify.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Finding {
    pub id: String,
    pub priority: Priority,
    pub confidence: Confidence,
    pub attribute_key: AttributeKey,
    /// The name of the fact key (subject – attribute).
    pub key: String,
    pub title: String,
    pub explanation: String,
    pub benign_explanations: Vec<String>,
    pub check: String,
    /// The model's prose was replaced by deterministic text.
    pub prose_replaced: bool,
    pub values: Vec<FindingValue>,
}

/// One of the differing values, with where it appears.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FindingValue {
    /// The value as the pages write it.
    pub text: String,
    pub occurrences: Vec<OccurrenceRef>,
    /// Every page showing the value, sorted.
    pub urls: Vec<String>,
}

/// One occurrence of a value: its conditions, headings and the crawler's text around it.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OccurrenceRef {
    pub occurrence_id: usize,
    pub region: SourceKind,
    /// The page path of a page occurrence; empty for a header/footer line.
    pub path: String,
    pub qualifiers: String,
    pub heading_path: Vec<String>,
    /// The crawler's copy of the text around the value (never text written by the model).
    pub evidence: String,
    /// The pages showing this occurrence.
    pub urls: Vec<String>,
}

/// A page with the findings that mention it.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PageFindings {
    pub url: String,
    pub findings: Vec<PageFinding>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PageFinding {
    pub finding_id: String,
    pub priority: Priority,
    pub title: String,
    /// The values of the finding this page shows.
    pub values: Vec<String>,
}

/// A difference the review found explainable (`explainable`) or not comparable (`not_comparable`).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExplainedGroup {
    pub key: String,
    pub attribute_key: AttributeKey,
    pub disposition: &'static str,
    pub title: String,
    pub reason: String,
    pub benign_explanations: Vec<String>,
    pub values: Vec<FindingValue>,
}

/// A difference that could not be judged: `insufficient_context`, or not reviewed
/// (`not_reviewed_cap`, `not_reviewed_call_failed`, `not_reviewed_output_budget`, and
/// `not_reviewed_left_out` for the values of a reviewed group that no result judged).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NotJudgedGroup {
    pub key: String,
    pub attribute_key: AttributeKey,
    pub status: &'static str,
    /// The review's explanation of `insufficient_context`; empty otherwise.
    pub reason: String,
    pub values: Vec<FindingValue>,
}

/// One extraction source.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceOut {
    pub id: usize,
    pub kind: SourceKind,
    pub url: String,
    pub path: String,
    pub blocks: usize,
    pub omitted_blocks: usize,
    pub truncated_blocks: usize,
    /// Facts kept from this source, and facts dropped as ungrounded.
    pub facts: usize,
    pub ungrounded: usize,
    pub failed: bool,
}

/// One kept fact occurrence with its block reference.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OccurrenceOut {
    pub id: usize,
    pub source: usize,
    pub region: SourceKind,
    pub block_ref: String,
    pub attribute_key: AttributeKey,
    pub subject: String,
    pub attribute: String,
    pub value: String,
    /// The comparison key of the value, and whether it is exact (an uncertain one never
    /// establishes "consistent").
    pub value_key: String,
    pub value_exact: bool,
    pub qualifiers: String,
    pub evidence: String,
    pub heading_path: Vec<String>,
    pub urls: Vec<String>,
}

/// A source whose extraction failed.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FailedSource {
    pub id: usize,
    pub kind: SourceKind,
    pub url: String,
    pub path: String,
    pub error: String,
}

/// The fixed texts: `(key, English, Czech)`. A text with `|` holds plural forms (English: one,
/// other; Czech: one, 2–4, other) with `{n}` for the number; `{name}` marks other placeholders.
const TEXTS: &[(&str, &str, &str)] = &[
    (
        "title",
        "Fact consistency across pages",
        "Soulad faktů napříč stránkami",
    ),
    ("kicker", "AI fact-consistency check", "AI kontrola souladu faktů"),
    (
        "disclaimer",
        "This report is an automated comparison of facts found on the website. It is not proof of an error: values can differ for legitimate reasons (another product variant, period, region or condition, rounding), and the automated review can misread the context or miss a difference. Please verify every finding manually before changing any content.",
        "Tato zpráva je automatické porovnání faktů nalezených na webu. Není důkazem chyby: hodnoty se mohou lišit oprávněně (jiná varianta produktu, období, region či podmínky, zaokrouhlení) a automatické posouzení může kontext pochopit špatně nebo rozdíl přehlédnout. Každé zjištění prosím před úpravou obsahu ručně ověřte.",
    ),
    (
        "caution",
        "Possible inconsistency — please verify manually. The values may apply to different circumstances.",
        "Možný nesoulad — ověřte prosím ručně. Hodnoty se mohou vztahovat k různým situacím.",
    ),
    ("completeness", "Completeness", "Úplnost"),
    ("state_complete", "Complete within scope", "Úplné v rámci rozsahu"),
    ("state_partial", "Partial", "Částečné"),
    ("state_insufficient", "Insufficient evidence", "Nedostatek podkladů"),
    ("summary", "Summary", "Shrnutí"),
    ("findings", "Findings", "Zjištění"),
    (
        "priority_critical",
        "Critical — check first",
        "Kritické — ověřte přednostně",
    ),
    ("priority_high", "High", "Vysoké"),
    ("priority_medium", "Medium", "Střední"),
    ("priority_low", "Low", "Nízké"),
    ("filter_all", "All", "Vše"),
    ("filter_critical", "Critical", "Kritické"),
    ("confidence_likely", "Likely inconsistent", "Pravděpodobný nesoulad"),
    ("confidence_possible", "Possibly inconsistent", "Možný nesoulad"),
    ("fact", "Fact", "Fakt"),
    ("benign", "Possible legitimate reasons", "Možná oprávněná vysvětlení"),
    ("check", "What to check", "Co ověřit"),
    ("values", "Values", "Hodnoty"),
    ("col_value", "Value", "Hodnota"),
    ("col_where", "Where it appears", "Kde se vyskytuje"),
    ("col_pages", "Pages", "Stránky"),
    ("region_page", "Page", "Stránka"),
    ("region_chrome", "Header/footer line", "Řádek hlavičky/patičky"),
    ("section", "Section", "Sekce"),
    ("conditions", "Conditions", "Podmínky"),
    (
        "on_pages",
        "on {n} page|on {n} pages",
        "na {n} stránce|na {n} stránkách",
    ),
    (
        "more_pages",
        "{n} more page|{n} more pages",
        "další {n} stránka|další {n} stránky|dalších {n} stránek",
    ),
    (
        "more_occurrences",
        "{n} more occurrence|{n} more occurrences",
        "další {n} výskyt|další {n} výskyty|dalších {n} výskytů",
    ),
    ("by_page", "By page", "Podle stránek"),
    ("col_page", "Page", "Stránka"),
    ("col_findings", "Findings", "Zjištění"),
    (
        "explained",
        "Differences with a plausible explanation",
        "Rozdíly s věrohodným vysvětlením",
    ),
    ("disp_explainable", "Explainable", "Vysvětlitelné"),
    ("disp_not_comparable", "Not comparable", "Nesrovnatelné"),
    ("not_judged", "Could not be judged", "Nešlo posoudit"),
    (
        "status_insufficient_context",
        "The evidence does not allow a judgment",
        "Podklady neumožňují posouzení",
    ),
    (
        "status_not_reviewed_cap",
        "Not reviewed: over the review limit",
        "Neposouzeno: nad limitem posouzení",
    ),
    (
        "status_not_reviewed_call_failed",
        "Not reviewed: the review call failed",
        "Neposouzeno: dotaz na posouzení selhal",
    ),
    (
        "status_not_reviewed_output_budget",
        "Not reviewed: --ai-max-tokens is too low for a review",
        "Neposouzeno: --ai-max-tokens je na posouzení příliš nízké",
    ),
    (
        "status_not_reviewed_left_out",
        "Not reviewed: the review left these values out",
        "Neposouzeno: posouzení tyto hodnoty vynechalo",
    ),
    (
        "show_items",
        "Show {n} item|Show {n} items",
        "Zobrazit {n} položku|Zobrazit {n} položky|Zobrazit {n} položek",
    ),
    (
        "consistent",
        "Same value observed in sampled occurrences",
        "Stejná hodnota ve vzorku výskytů",
    ),
    ("col_fact", "Fact", "Fakt"),
    ("col_places", "Places", "Míst"),
    ("col_pages_count", "Pages", "Stránek"),
    (
        "consistent_more",
        "{n} more fact is in the JSON report.|{n} more facts are in the JSON report.",
        "Další {n} fakt najdete v JSON reportu.|Další {n} fakta najdete v JSON reportu.|Dalších {n} faktů najdete v JSON reportu.",
    ),
    ("coverage", "Coverage & method", "Rozsah a metoda"),
    (
        "cov_pages",
        "Pages eligible / selected / analyzed",
        "Stránky způsobilé / vybrané / analyzované",
    ),
    (
        "cov_missing_body",
        "Selected pages without stored HTML",
        "Vybrané stránky bez uloženého HTML",
    ),
    (
        "cov_reduced",
        "Pages with reduced input (blocks not inspected)",
        "Stránky se zkráceným vstupem (část bloků neprověřena)",
    ),
    (
        "cov_chrome",
        "Header/footer lines analyzed / excluded",
        "Řádky hlavičky/patičky analyzované / vynechané",
    ),
    (
        "cov_facts",
        "Facts extracted / kept / dropped as ungrounded",
        "Fakta vytěžená / ponechaná / vyřazená jako nedoložená",
    ),
    (
        "cov_keys",
        "Fact keys / keys with differing values",
        "Klíče faktů / klíče s rozdílnými hodnotami",
    ),
    (
        "cov_reviews",
        "Groups reviewed / value comparisons",
        "Posouzené skupiny / porovnání hodnot",
    ),
    ("cov_grouping", "Grouping", "Seskupení"),
    ("grouping_complete", "complete", "úplné"),
    (
        "grouping_incomplete",
        "incomplete ({n} label not compared with every other one)|incomplete ({n} labels not compared with every other one)",
        "neúplné ({n} označení nebylo porovnáno se všemi ostatními)|neúplné ({n} označení nebyla porovnána se všemi ostatními)|neúplné ({n} označení nebylo porovnáno se všemi ostatními)",
    ),
    (
        "cov_failed",
        "Sources that could not be analyzed",
        "Zdroje, které se nepodařilo analyzovat",
    ),
    ("cov_model", "Model", "Model"),
    ("cov_context", "context window {n} tokens", "kontextové okno {n} tokenů"),
    ("cov_calls", "LLM calls / duration", "Volání LLM / doba běhu"),
    (
        "cov_facts_note",
        "Up to 5 facts were extracted per page and per group of header/footer lines (usually 2–4); facts that were not picked were not compared.",
        "Z každé stránky a z každé skupiny řádků hlavičky/patičky bylo vybráno nejvýše 5 faktů (obvykle 2–4); fakta, která vybrána nebyla, se neporovnávala.",
    ),
    (
        "cov_http_cache",
        "Pages may come from the crawler's HTTP cache (TTL {ttl}); for a fresh comparison run again with `--http-cache-dir=`.",
        "Stránky mohou pocházet z HTTP cache crawleru (TTL {ttl}); pro čerstvé porovnání spusťte kontrolu znovu s `--http-cache-dir=`.",
    ),
    (
        "failed_sources",
        "Sources that could not be analyzed",
        "Zdroje, které se nepodařilo analyzovat",
    ),
    ("toggle_theme", "Toggle light/dark theme", "Přepnout světlý/tmavý motiv"),
    ("generated_by", "Generated by", "Vygenerováno nástrojem"),
    // Summary.
    (
        "sum_none",
        "No potential inconsistencies were identified among the compared facts.",
        "Mezi porovnanými fakty jsme nenašli žádný možný nesoulad.",
    ),
    (
        "sum_insufficient",
        "There was too little evidence for a reliable comparison.",
        "Podkladů bylo na spolehlivé porovnání příliš málo.",
    ),
    (
        "sum_compared",
        "We compared {facts} from {pages}.",
        "Porovnali jsme {facts} na {pages}.",
    ),
    (
        "sum_compared_chrome",
        "We compared {facts} from {pages} and the shared header/footer ({lines}).",
        "Porovnali jsme {facts} na {pages} a ve sdílené hlavičce/patičce ({lines}).",
    ),
    (
        "n_facts",
        "{n} fact|{n} facts",
        "{n} fakt uvedený|{n} fakty uvedené|{n} faktů uvedených",
    ),
    ("n_pages", "{n} page|{n} pages", "{n} stránce|{n} stránkách"),
    (
        "n_lines",
        "{n} distinct fact line|{n} distinct fact lines",
        "{n} odlišný řádek s fakty|{n} odlišné řádky s fakty|{n} odlišných řádků s fakty",
    ),
    (
        "n_worth",
        "{n} difference is|{n} differences are",
        "{n} rozdíl stojí|{n} rozdíly stojí|{n} rozdílů stojí",
    ),
    (
        "sum_worth",
        "{diffs} worth checking: {breakdown}.",
        "{diffs} za ověření ({breakdown}).",
    ),
    ("bd_critical", "{n} critical", "kritické {n}"),
    ("bd_high", "{n} high", "vysoké {n}"),
    ("bd_medium", "{n} medium", "střední {n}"),
    ("bd_low", "{n} low", "nízké {n}"),
    (
        "n_explained",
        "{n} difference has|{n} differences have",
        "{n} rozdíl má|{n} rozdíly mají|{n} rozdílů má",
    ),
    (
        "sum_explained",
        "{explained} a plausible legitimate explanation.",
        "{explained} věrohodné vysvětlení.",
    ),
    (
        "sum_explained_judged",
        "{explained} a plausible legitimate explanation; {judged} could not be judged from the evidence.",
        "{explained} věrohodné vysvětlení; {judged} nešlo z podkladů posoudit.",
    ),
    (
        "sum_judged",
        "{n} difference could not be judged from the evidence.|{n} differences could not be judged from the evidence.",
        "{n} rozdíl nešlo z podkladů posoudit.|{n} rozdíly nešlo z podkladů posoudit.|{n} rozdílů nešlo z podkladů posoudit.",
    ),
    // Completeness reasons (see `reason_text`).
    (
        "reason_no_facts",
        "no source yielded any facts",
        "žádný zdroj neposkytl fakta",
    ),
    (
        "reason_insufficient_sources",
        "facts were found in only {n} source|facts were found in only {n} sources",
        "fakta se našla jen v {n} zdroji|fakta se našla jen v {n} zdrojích",
    ),
    (
        "reason_failed_sources",
        "{n} source could not be analyzed|{n} sources could not be analyzed",
        "{n} zdroj se nepodařilo analyzovat|{n} zdroje se nepodařilo analyzovat|{n} zdrojů se nepodařilo analyzovat",
    ),
    (
        "reason_missing_body",
        "{n} selected page had no stored HTML|{n} selected pages had no stored HTML",
        "{n} vybraná stránka neměla uložené HTML|{n} vybrané stránky neměly uložené HTML|{n} vybraných stránek nemělo uložené HTML",
    ),
    (
        "reason_pages_reduced",
        "{n} page was too long: some or all of its blocks were not inspected|{n} pages were too long: some or all of their blocks were not inspected",
        "{n} stránka byla příliš dlouhá: některé nebo všechny její bloky nebyly prověřeny|{n} stránky byly příliš dlouhé: některé nebo všechny jejich bloky nebyly prověřeny|{n} stránek bylo příliš dlouhých: některé nebo všechny jejich bloky nebyly prověřeny",
    ),
    (
        "reason_chrome_lines_excluded",
        "{n} header/footer line was not analyzed (limit)|{n} header/footer lines were not analyzed (limit)",
        "{n} řádek hlavičky/patičky nebyl analyzován (limit)|{n} řádky hlavičky/patičky nebyly analyzovány (limit)|{n} řádků hlavičky/patičky nebylo analyzováno (limit)",
    ),
    (
        "reason_grouping_incomplete",
        "grouping was incomplete: {n} fact label was not compared with every other one|grouping was incomplete: {n} fact labels were not compared with every other one",
        "seskupení nebylo úplné: {n} označení faktu nebylo porovnáno se všemi ostatními|seskupení nebylo úplné: {n} označení faktů nebyla porovnána se všemi ostatními|seskupení nebylo úplné: {n} označení faktů nebylo porovnáno se všemi ostatními",
    ),
    (
        "reason_grouping_failed",
        "grouping was incomplete: a grouping call failed",
        "seskupení nebylo úplné: dotaz na seskupení selhal",
    ),
    (
        "reason_not_reviewed_cap",
        "{n} difference was not reviewed (review limit)|{n} differences were not reviewed (review limit)",
        "{n} rozdíl nebyl posouzen (limit posouzení)|{n} rozdíly nebyly posouzeny (limit posouzení)|{n} rozdílů nebylo posouzeno (limit posouzení)",
    ),
    (
        "reason_not_reviewed_call_failed",
        "{n} difference could not be reviewed (the review call failed)|{n} differences could not be reviewed (the review call failed)",
        "{n} rozdíl se nepodařilo posoudit (dotaz selhal)|{n} rozdíly se nepodařilo posoudit (dotaz selhal)|{n} rozdílů se nepodařilo posoudit (dotaz selhal)",
    ),
    (
        "reason_not_reviewed_left_out",
        "{n} difference was reviewed only in part: the review left some of its values out|{n} differences were reviewed only in part: the review left some of their values out",
        "{n} rozdíl byl posouzen jen zčásti: posouzení vynechalo některé jeho hodnoty|{n} rozdíly byly posouzeny jen zčásti: posouzení vynechalo některé jejich hodnoty|{n} rozdílů bylo posouzeno jen zčásti: posouzení vynechalo některé jejich hodnoty",
    ),
    (
        "reason_not_reviewed_output_budget",
        "the review was skipped for {n} difference: --ai-max-tokens is too low|the review was skipped for {n} differences: --ai-max-tokens is too low",
        "posouzení {n} rozdílu bylo vynecháno: --ai-max-tokens je příliš nízké|posouzení {n} rozdílů bylo vynecháno: --ai-max-tokens je příliš nízké",
    ),
    // The deterministic prose that replaces a review's prose (see `judge::validate`).
    ("prose_title", "{name}: values differ", "{name}: hodnoty se liší"),
    (
        "prose_explanation",
        "The pages state different values for the same property.",
        "Stránky uvádějí pro stejnou vlastnost různé hodnoty.",
    ),
    (
        "prose_check",
        "Compare the values on the listed pages and decide which one is current.",
        "Porovnejte hodnoty na uvedených stránkách a rozhodněte, která z nich je aktuální.",
    ),
    (
        "prose_reason_explainable",
        "The review saw a plausible legitimate reason for the difference, but its wording could not be used; please compare the values and the evidence.",
        "Posouzení vidí pro rozdíl pravděpodobný oprávněný důvod, jeho znění ale nešlo použít; porovnejte prosím hodnoty a podklady.",
    ),
    (
        "prose_reason_not_comparable",
        "The review judged that the values describe different things, but its wording could not be used; please compare the values and the evidence.",
        "Posouzení vyhodnotilo, že hodnoty popisují různé věci, jeho znění ale nešlo použít; porovnejte prosím hodnoty a podklady.",
    ),
    (
        "prose_reason_insufficient",
        "The evidence did not let the review tell whether the values conflict.",
        "Z dostupných podkladů nešlo posoudit, zda si hodnoty odporují.",
    ),
];

/// The fixed text `key` in Czech for a Czech report, else in English; "" for an unknown key.
pub fn text(locale: &ReportLocale, key: &str) -> &'static str {
    TEXTS
        .iter()
        .find(|(k, _, _)| *k == key)
        .map(|(_, en, cs)| if locale.is_czech() { *cs } else { *en })
        .unwrap_or("")
}

/// The plural form of `key` for `n`, with `{n}` replaced by the formatted number.
fn count_text(locale: &ReportLocale, key: &str, n: usize) -> String {
    let forms: Vec<&str> = text(locale, key).split('|').collect();
    let index = match (locale.is_czech(), n) {
        (_, 1) => 0,
        (true, 2..=4) => 1,
        (true, _) => 2,
        (false, _) => 1,
    };
    forms
        .get(index)
        .or(forms.last())
        .copied()
        .unwrap_or_default()
        .replace("{n}", &fmt_count(locale, n))
}

/// A number with thousands separators: `2,184` in English, `2 184` (no-break space) in Czech.
fn fmt_count(locale: &ReportLocale, n: usize) -> String {
    let separator = if locale.is_czech() { '\u{a0}' } else { ',' };
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + 4);
    for (i, digit) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(separator);
        }
        out.push(digit);
    }
    out
}

/// `template` with each `{name}` replaced by its value.
fn fill(template: &str, values: &[(&str, &str)]) -> String {
    let mut out = template.to_string();
    for (name, value) in values {
        out = out.replace(&format!("{{{name}}}"), value);
    }
    out
}

/// The deterministic summary: coverage first (or, with nothing found, "no potential
/// inconsistencies" immediately followed by the coverage — never a bare "all good"), then the
/// findings by priority, then the explained and the not judged differences.
pub fn summary(locale: &ReportLocale, meta: &Meta, counts: &Counts, completeness: &Completeness) -> String {
    let mut parts: Vec<String> = Vec::new();
    if completeness.state == CompletenessState::Insufficient {
        parts.push(text(locale, "sum_insufficient").to_string());
    }
    let findings = counts.critical + counts.high + counts.medium + counts.low;
    if findings == 0 {
        parts.push(text(locale, "sum_none").to_string());
    }
    let facts = count_text(locale, "n_facts", meta.facts_kept);
    let pages = count_text(locale, "n_pages", meta.pages_analyzed);
    parts.push(if meta.chrome_lines_analyzed > 0 {
        let lines = count_text(locale, "n_lines", meta.chrome_lines_analyzed);
        fill(
            text(locale, "sum_compared_chrome"),
            &[("facts", &facts), ("pages", &pages), ("lines", &lines)],
        )
    } else {
        fill(text(locale, "sum_compared"), &[("facts", &facts), ("pages", &pages)])
    });
    if findings > 0 {
        let breakdown: Vec<String> = [
            (counts.critical, "bd_critical"),
            (counts.high, "bd_high"),
            (counts.medium, "bd_medium"),
            (counts.low, "bd_low"),
        ]
        .into_iter()
        .filter(|(n, _)| *n > 0)
        .map(|(n, key)| count_text(locale, key, n))
        .collect();
        parts.push(fill(
            text(locale, "sum_worth"),
            &[
                ("diffs", &count_text(locale, "n_worth", findings)),
                ("breakdown", &breakdown.join(", ")),
            ],
        ));
    }
    let explained = counts.explained + counts.not_comparable;
    let judged = counts.insufficient + counts.not_reviewed;
    let explained_text = count_text(locale, "n_explained", explained);
    match (explained > 0, judged > 0) {
        (true, true) => parts.push(fill(
            text(locale, "sum_explained_judged"),
            &[("explained", &explained_text), ("judged", &fmt_count(locale, judged))],
        )),
        (true, false) => parts.push(fill(text(locale, "sum_explained"), &[("explained", &explained_text)])),
        (false, true) => parts.push(count_text(locale, "sum_judged", judged)),
        (false, false) => {}
    }
    parts.join(" ")
}

/// The completeness state: `Insufficient` when fewer than 3 sources yielded facts or at least
/// half of the `sources` failed; `Complete` when every selected source was extracted, grouping
/// finished and every candidate was reviewed; `Partial` otherwise. The reasons, in a fixed
/// order: too few sources with facts, failed sources, pages without HTML, reduced pages,
/// excluded header/footer lines, incomplete grouping, and reviews capped, failed, skipped or
/// left partly undone. A not-reviewed count in `counts` beyond the capped, skipped and partly
/// undone ones counts as failed.
pub fn completeness(meta: &Meta, counts: &Counts, failed_sources: usize, sources: usize) -> Completeness {
    let reason = |code: &'static str, count: usize| CompletenessReason { code, count };
    let mut reasons = Vec::new();
    if meta.sources_with_facts == 0 {
        reasons.push(reason("no_facts", 0));
    } else if meta.sources_with_facts < MIN_SOURCES_WITH_FACTS {
        reasons.push(reason("insufficient_sources", meta.sources_with_facts));
    }
    let failed_reviews = meta.reviews_failed.max(
        counts
            .not_reviewed
            .saturating_sub(meta.reviews_capped + meta.reviews_skipped_budget + meta.reviews_partial),
    );
    let counted = [
        ("failed_sources", failed_sources),
        ("missing_body", meta.pages_missing_body),
        ("pages_reduced", meta.pages_reduced),
        ("chrome_lines_excluded", meta.chrome_lines_excluded),
        ("grouping_incomplete", usize::from(meta.grouping_incomplete)),
        ("not_reviewed_cap", meta.reviews_capped),
        ("not_reviewed_call_failed", failed_reviews),
        ("not_reviewed_output_budget", meta.reviews_skipped_budget),
        ("not_reviewed_left_out", meta.reviews_partial),
    ];
    for (code, count) in counted {
        if count > 0 {
            let count = if code == "grouping_incomplete" {
                meta.items_not_cross_compared
            } else {
                count
            };
            reasons.push(reason(code, count));
        }
    }
    let insufficient =
        sources == 0 || meta.sources_with_facts < MIN_SOURCES_WITH_FACTS || failed_sources * 2 >= sources;
    let state = if insufficient {
        CompletenessState::Insufficient
    } else if reasons.is_empty() {
        CompletenessState::Complete
    } else {
        CompletenessState::Partial
    };
    Completeness { state, reasons }
}

/// The localized text of a completeness reason.
pub fn reason_text(locale: &ReportLocale, reason: &CompletenessReason) -> String {
    match reason.code {
        "no_facts" => text(locale, "reason_no_facts").to_string(),
        "grouping_incomplete" if reason.count == 0 => text(locale, "reason_grouping_failed").to_string(),
        code => count_text(locale, &format!("reason_{code}"), reason.count),
    }
}

/// Order the findings: by priority (critical first), then confidence (likely first), then the
/// importance of the attribute, then reach (the number of pages showing any of the values); ties
/// by the key name and the values.
pub fn sort_findings(f: &mut [Finding]) {
    f.sort_by(|a, b| {
        a.priority
            .cmp(&b.priority)
            .then(a.confidence.cmp(&b.confidence))
            .then(b.attribute_key.importance().cmp(&a.attribute_key.importance()))
            .then(reach(b).cmp(&reach(a)))
            .then_with(|| a.key.cmp(&b.key))
            .then_with(|| value_texts(a).cmp(&value_texts(b)))
    });
}

fn reach(f: &Finding) -> usize {
    f.values
        .iter()
        .flat_map(|v| v.urls.iter())
        .collect::<BTreeSet<_>>()
        .len()
}

fn value_texts(f: &Finding) -> Vec<&str> {
    f.values.iter().map(|v| v.text.as_str()).collect()
}

/// Every page URL (sorted) with the findings that mention it, in the findings' order, each with
/// the values of the finding that page shows.
pub fn by_page(findings: &[Finding]) -> Vec<PageFindings> {
    let mut pages: BTreeMap<&str, Vec<PageFinding>> = BTreeMap::new();
    for finding in findings {
        for value in &finding.values {
            for url in &value.urls {
                let list = pages.entry(url.as_str()).or_default();
                match list.iter_mut().find(|p| p.finding_id == finding.id) {
                    Some(entry) => {
                        if !entry.values.contains(&value.text) {
                            entry.values.push(value.text.clone());
                        }
                    }
                    None => list.push(PageFinding {
                        finding_id: finding.id.clone(),
                        priority: finding.priority,
                        title: finding.title.clone(),
                        values: vec![value.text.clone()],
                    }),
                }
            }
        }
    }
    pages
        .into_iter()
        .map(|(url, findings)| PageFindings {
            url: url.to_string(),
            findings,
        })
        .collect()
}

fn priority_code(p: Priority) -> &'static str {
    match p {
        Priority::Critical => "critical",
        Priority::High => "high",
        Priority::Medium => "medium",
        Priority::Low => "low",
    }
}

fn confidence_code(c: Confidence) -> &'static str {
    match c {
        Confidence::Likely => "likely",
        Confidence::Possible => "possible",
    }
}

fn region_code(r: SourceKind) -> &'static str {
    match r {
        SourceKind::Page => "page",
        SourceKind::Chrome => "chrome",
    }
}

fn state_key(state: CompletenessState) -> &'static str {
    match state {
        CompletenessState::Complete => "state_complete",
        CompletenessState::Partial => "state_partial",
        CompletenessState::Insufficient => "state_insufficient",
    }
}

fn state_code(state: CompletenessState) -> &'static str {
    match state {
        CompletenessState::Complete => "complete",
        CompletenessState::Partial => "partial",
        CompletenessState::Insufficient => "insufficient",
    }
}

/// Human duration: "6 min 3 s" for a minute or more, else "12.4 s" ("12,4 s" in Czech).
fn fmt_duration(locale: &ReportLocale, ms: u64) -> String {
    let secs = ms as f64 / 1000.0;
    if secs >= 60.0 {
        format!("{} min {} s", (secs as u64) / 60, (secs as u64) % 60)
    } else if locale.is_czech() {
        format!("{secs:.1} s").replace('.', ",")
    } else {
        format!("{secs:.1} s")
    }
}

impl ConsistencyDoc {
    fn locale(&self) -> ReportLocale {
        ReportLocale::new(&self.meta.language)
    }

    /// The whole document as JSON (`schema`, camelCase keys).
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// The coverage rows: `(label, value)`.
    fn coverage_rows(&self, locale: &ReportLocale) -> Vec<(&'static str, String)> {
        let m = &self.meta;
        let n = |v: usize| fmt_count(locale, v);
        vec![
            (
                text(locale, "cov_pages"),
                format!(
                    "{} / {} / {}",
                    n(m.pages_eligible),
                    n(m.pages_selected),
                    n(m.pages_analyzed)
                ),
            ),
            (text(locale, "cov_missing_body"), n(m.pages_missing_body)),
            (text(locale, "cov_reduced"), n(m.pages_reduced)),
            (
                text(locale, "cov_chrome"),
                format!("{} / {}", n(m.chrome_lines_analyzed), n(m.chrome_lines_excluded)),
            ),
            (
                text(locale, "cov_facts"),
                format!(
                    "{} / {} / {}",
                    n(m.facts_extracted),
                    n(m.facts_kept),
                    n(m.facts_ungrounded)
                ),
            ),
            (text(locale, "cov_keys"), format!("{} / {}", n(m.keys), n(m.candidates))),
            (
                text(locale, "cov_reviews"),
                format!("{} / {}", n(m.reviewed), n(m.comparisons_done)),
            ),
            (
                text(locale, "cov_grouping"),
                if m.grouping_incomplete {
                    count_text(locale, "grouping_incomplete", m.items_not_cross_compared)
                } else {
                    text(locale, "grouping_complete").to_string()
                },
            ),
            (text(locale, "cov_failed"), n(self.failed_sources.len())),
            (
                text(locale, "cov_model"),
                format!(
                    "{} / {} · {}",
                    m.provider,
                    m.model,
                    text(locale, "cov_context").replace("{n}", &fmt_count(locale, m.context_window.max(0) as usize))
                ),
            ),
            (
                text(locale, "cov_calls"),
                format!("{} / {}", n(m.llm_calls), fmt_duration(locale, m.duration_ms)),
            ),
        ]
    }

    /// The coverage notes: facts per page, and the HTTP-cache note when the cache was on.
    fn coverage_notes(&self, locale: &ReportLocale) -> Vec<String> {
        let mut notes = vec![text(locale, "cov_facts_note").to_string()];
        if let Some(ttl) = &self.meta.http_cache {
            notes.push(fill(text(locale, "cov_http_cache"), &[("ttl", ttl)]));
        }
        notes
    }

    fn consistent_shown(&self) -> Vec<&ConsistentFact> {
        let mut facts: Vec<&ConsistentFact> = self.consistent.iter().collect();
        facts.sort_by(|a, b| {
            b.sources
                .cmp(&a.sources)
                .then(b.pages.cmp(&a.pages))
                .then_with(|| a.name.cmp(&b.name))
        });
        facts.truncate(CONSISTENT_SHOWN);
        facts
    }

    // ---- CSV ----

    /// One row per finding × value × affected URL:
    /// `finding_id,priority,confidence,attribute,subject,value,qualifiers,url,region,heading_path`,
    /// with the qualifiers, region and heading path of the value's occurrence on that page (the
    /// page's own content first, else a header/footer line). Cells
    /// are quoted when they hold a comma, a quote or a line break; a cell a spreadsheet would run
    /// as a formula gets a leading `'`. UTF-8 with a byte-order mark, for spreadsheets.
    pub fn to_csv(&self) -> String {
        let mut out = String::from(
            "\u{feff}finding_id,priority,confidence,attribute,subject,value,qualifiers,url,region,heading_path\n",
        );
        for finding in &self.findings {
            for value in &finding.values {
                for url in &value.urls {
                    let on_page = |o: &&OccurrenceRef| o.urls.contains(url);
                    let occurrence = value
                        .occurrences
                        .iter()
                        .filter(on_page)
                        .find(|o| o.region == SourceKind::Page)
                        .or_else(|| value.occurrences.iter().find(on_page));
                    let row = [
                        finding.id.as_str(),
                        priority_code(finding.priority),
                        confidence_code(finding.confidence),
                        finding.attribute_key.as_str(),
                        finding.key.as_str(),
                        value.text.as_str(),
                        occurrence.map_or("", |o| o.qualifiers.as_str()),
                        url.as_str(),
                        occurrence.map_or("", |o| region_code(o.region)),
                        &occurrence.map(|o| o.heading_path.join(" > ")).unwrap_or_default(),
                    ];
                    let cells: Vec<String> = row.iter().map(|cell| csv_cell(cell)).collect();
                    out.push_str(&cells.join(","));
                    out.push('\n');
                }
            }
        }
        out
    }

    // ---- Markdown ----

    pub fn to_markdown(&self) -> String {
        let locale = self.locale();
        let t = |key: &str| text(&locale, key);
        let mut out = String::with_capacity(16 * 1024);
        out.push_str(&format!("# {}\n\n", t("title")));
        out.push_str(&format!(
            "**{}** · {}\n\n",
            md(&self.meta.host),
            md(&self.meta.generated_at)
        ));
        out.push_str(&format!("> {}\n\n", md(&self.disclaimer)));

        out.push_str(&format!(
            "## {}: {}\n\n",
            t("completeness"),
            t(state_key(self.completeness.state))
        ));
        for reason in &self.completeness.reasons {
            out.push_str(&format!("- {}\n", md(&reason_text(&locale, reason))));
        }
        if !self.completeness.reasons.is_empty() {
            out.push('\n');
        }
        out.push_str(&format!("## {}\n\n{}\n\n", t("summary"), md(&self.summary)));

        out.push_str(&format!("## {} ({})\n\n", t("findings"), self.findings.len()));
        if self.findings.is_empty() {
            out.push_str(&format!("{}\n\n", t("sum_none")));
        }
        for finding in &self.findings {
            self.md_finding(&mut out, &locale, finding);
        }

        if !self.by_page.is_empty() {
            out.push_str(&format!(
                "## {}\n\n| {} | {} |\n|---|---|\n",
                t("by_page"),
                t("col_page"),
                t("col_findings")
            ));
            for page in &self.by_page {
                let refs: Vec<String> = page
                    .findings
                    .iter()
                    .map(|f| {
                        let values: Vec<String> = f.values.iter().map(|v| md(v)).collect();
                        format!(
                            "{} ({}): {}",
                            md(&f.finding_id),
                            t(priority_key(f.priority)),
                            values.join(" · ")
                        )
                    })
                    .collect();
                out.push_str(&format!("| {} | {} |\n", md_url(&page.url), refs.join("<br>")));
            }
            out.push('\n');
        }

        if !self.explained.is_empty() {
            out.push_str(&format!(
                "## {} ({})\n\n<details>\n<summary>{}</summary>\n\n",
                t("explained"),
                self.explained.len(),
                count_text(&locale, "show_items", self.explained.len())
            ));
            for group in &self.explained {
                let label = if group.disposition == "not_comparable" {
                    t("disp_not_comparable")
                } else {
                    t("disp_explainable")
                };
                out.push_str(&format!(
                    "- **{}** — {}: {}. {}\n",
                    md(&group.key),
                    label,
                    md_values(&group.values),
                    md(&group.reason)
                ));
            }
            out.push_str("\n</details>\n\n");
        }

        if !self.not_judged.is_empty() {
            out.push_str(&format!(
                "## {} ({})\n\n<details>\n<summary>{}</summary>\n\n",
                t("not_judged"),
                self.not_judged.len(),
                count_text(&locale, "show_items", self.not_judged.len())
            ));
            for group in &self.not_judged {
                let status = t(&format!("status_{}", group.status));
                let reason = if group.reason.is_empty() {
                    String::new()
                } else {
                    format!(" {}", md(&group.reason))
                };
                out.push_str(&format!(
                    "- **{}** — {}: {}.{}\n",
                    md(&group.key),
                    status,
                    md_values(&group.values),
                    reason
                ));
            }
            out.push_str("\n</details>\n\n");
        }

        if !self.consistent.is_empty() {
            out.push_str(&format!(
                "## {}\n\n| {} | {} | {} | {} |\n|---|---|---:|---:|\n",
                t("consistent"),
                t("col_fact"),
                t("col_value"),
                t("col_places"),
                t("col_pages_count")
            ));
            for fact in self.consistent_shown() {
                out.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    md(&fact.name),
                    md(&fact.value),
                    fmt_count(&locale, fact.sources),
                    fmt_count(&locale, fact.pages)
                ));
            }
            out.push('\n');
            if self.consistent.len() > CONSISTENT_SHOWN {
                out.push_str(&format!(
                    "{}\n\n",
                    count_text(&locale, "consistent_more", self.consistent.len() - CONSISTENT_SHOWN)
                ));
            }
        }

        out.push_str(&format!("## {}\n\n", t("coverage")));
        for (label, value) in self.coverage_rows(&locale) {
            out.push_str(&format!("- {label}: {}\n", md(&value)));
        }
        out.push('\n');
        for note in self.coverage_notes(&locale) {
            out.push_str(&format!("{note}\n\n"));
        }
        if !self.failed_sources.is_empty() {
            out.push_str(&format!("### {}\n\n", t("failed_sources")));
            for failed in &self.failed_sources {
                out.push_str(&format!("- {} — {}\n", md(&failed.path), md(&failed.error)));
            }
            out.push('\n');
        }
        out.push_str(&format!(
            "---\n\n{} [SiteOne Crawler](https://crawler.siteone.io/).\n",
            t("generated_by")
        ));
        out
    }

    fn md_finding(&self, out: &mut String, locale: &ReportLocale, f: &Finding) {
        let t = |key: &str| text(locale, key);
        out.push_str(&format!(
            "### {} · {} · {} — {}\n\n",
            md(&f.id),
            t(priority_key(f.priority)),
            t(confidence_key(f.confidence)),
            md(&f.title)
        ));
        out.push_str(&format!("> **{}**\n\n", self.caution));
        out.push_str(&format!("*{}: {}*\n\n", t("fact"), md(&f.key)));
        if !f.explanation.is_empty() {
            out.push_str(&format!("{}\n\n", md(&f.explanation)));
        }
        if !f.benign_explanations.is_empty() {
            out.push_str(&format!("**{}**\n\n", t("benign")));
            for reason in &f.benign_explanations {
                out.push_str(&format!("- {}\n", md(reason)));
            }
            out.push('\n');
        }
        if !f.check.is_empty() {
            out.push_str(&format!("**{}:** {}\n\n", t("check"), md(&f.check)));
        }
        out.push_str(&format!("**{}**\n\n", t("values")));
        for value in &f.values {
            out.push_str(&format!(
                "- **{}** — {}\n",
                md(&value.text),
                count_text(locale, "on_pages", value.urls.len())
            ));
            for occurrence in value.occurrences.iter().take(OCCURRENCES_SHOWN) {
                out.push_str(&format!("  - {}\n", md(&occurrence_line(locale, occurrence))));
                if !occurrence.evidence.is_empty() {
                    out.push_str(&format!("    > {}\n", md(&occurrence.evidence)));
                }
            }
            if value.occurrences.len() > OCCURRENCES_SHOWN {
                out.push_str(&format!(
                    "  - {}\n",
                    count_text(locale, "more_occurrences", value.occurrences.len() - OCCURRENCES_SHOWN)
                ));
            }
            if !value.urls.is_empty() {
                let shown: Vec<String> = value.urls.iter().take(URLS_SHOWN).map(|u| md_url(u)).collect();
                out.push_str(&format!("  - {}: {}\n", t("col_pages"), shown.join(", ")));
                if value.urls.len() > URLS_SHOWN {
                    let rest: Vec<String> = value.urls[URLS_SHOWN..].iter().map(|u| md_url(u)).collect();
                    out.push_str(&format!(
                        "    <details><summary>{}</summary>\n\n    {}\n\n    </details>\n",
                        count_text(locale, "more_pages", rest.len()),
                        rest.join(", ")
                    ));
                }
            }
        }
        out.push('\n');
    }

    // ---- HTML ----

    pub fn to_html(&self) -> String {
        let locale = self.locale();
        let t = |key: &str| text(&locale, key);
        let mut h = String::with_capacity(64 * 1024);
        h.push_str("<!DOCTYPE html>\n<html lang=\"");
        h.push_str(&esc(locale.code()));
        h.push_str("\" data-theme=\"light\">\n<head>\n<meta charset=\"utf-8\">\n");
        h.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
        h.push_str(&format!(
            "<title>{} · {}</title>\n<style>\n{}\n</style>\n</head>\n<body>\n",
            esc(t("title")),
            esc(&self.meta.host),
            CSS
        ));
        h.push_str(&format!(
            "<header class=\"bar\"><a class=\"logo\" href=\"https://crawler.siteone.io/\" target=\"_blank\" rel=\"noopener\" aria-label=\"SiteOne Crawler\">{}<span class=\"wordmark\">SiteOne&nbsp;Crawler<span class=\"sub\">AI&nbsp;Consistency</span></span></a><button id=\"themeBtn\" class=\"btn\" aria-label=\"{}\">◐</button></header>\n",
            LOGO_SVG,
            esc(t("toggle_theme"))
        ));
        h.push_str("<section class=\"hero\"><div class=\"hero-card\">");
        h.push_str(&format!("<h1 class=\"hero-title\">{}</h1>", esc(t("title"))));
        h.push_str(&format!("<div class=\"hero-kicker\">{}</div>", esc(t("kicker"))));
        h.push_str("<div class=\"hero-meta\">");
        h.push_str(&link(&self.meta.url, &self.meta.host));
        if !self.meta.generated_at.is_empty() {
            h.push_str(&format!("<span class=\"dot\">·</span>{}", esc(&self.meta.generated_at)));
        }
        h.push_str("</div></div></section>\n<main class=\"wrap\">\n");

        // Disclaimer, completeness state and summary — always before the findings.
        h.push_str(&format!(
            "<section class=\"sec disclaimer\"><p>{}</p></section>\n",
            esc(&self.disclaimer)
        ));
        h.push_str(&format!(
            "<section class=\"sec\" id=\"summary\"><div class=\"state state-{}\"><span>{}:</span> <strong>{}</strong></div>",
            state_code(self.completeness.state),
            esc(t("completeness")),
            esc(t(state_key(self.completeness.state)))
        ));
        if !self.completeness.reasons.is_empty() {
            h.push_str("<ul class=\"reasons\">");
            for reason in &self.completeness.reasons {
                h.push_str(&format!("<li>{}</li>", esc(&reason_text(&locale, reason))));
            }
            h.push_str("</ul>");
        }
        h.push_str(&format!(
            "<h2>{}</h2><p class=\"summary\">{}</p></section>\n",
            esc(t("summary")),
            esc(&self.summary)
        ));

        // Findings.
        h.push_str(&format!(
            "<section class=\"sec\" id=\"findings\"><h2>{} ({})</h2>\n",
            esc(t("findings")),
            self.findings.len()
        ));
        if self.findings.is_empty() {
            h.push_str(&format!("<p>{}</p>\n", esc(t("sum_none"))));
        } else {
            h.push_str("<div class=\"filter\" role=\"group\">");
            h.push_str(&format!(
                "<button class=\"btn on\" data-filter=\"all\">{} ({})</button>",
                esc(t("filter_all")),
                self.findings.len()
            ));
            for (priority, label) in [
                (Priority::Critical, t("filter_critical")),
                (Priority::High, t("priority_high")),
                (Priority::Medium, t("priority_medium")),
                (Priority::Low, t("priority_low")),
            ] {
                let n = self.findings.iter().filter(|f| f.priority == priority).count();
                if n > 0 {
                    h.push_str(&format!(
                        "<button class=\"btn\" data-filter=\"{}\">{} ({n})</button>",
                        priority_code(priority),
                        esc(label)
                    ));
                }
            }
            h.push_str("</div>\n");
            for finding in &self.findings {
                self.html_finding(&mut h, &locale, finding);
            }
        }
        h.push_str("</section>\n");

        // By page.
        if !self.by_page.is_empty() {
            h.push_str(&format!(
                "<section class=\"sec\" id=\"by-page\"><h2>{}</h2><table class=\"grid\"><thead><tr><th>{}</th><th>{}</th></tr></thead><tbody>",
                esc(t("by_page")),
                esc(t("col_page")),
                esc(t("col_findings"))
            ));
            for page in &self.by_page {
                h.push_str(&format!(
                    "<tr><td class=\"url\">{}</td><td>",
                    link(&page.url, &page.url)
                ));
                for f in &page.findings {
                    let values: Vec<String> = f.values.iter().map(|v| esc(v)).collect();
                    h.push_str(&format!(
                        "<div><a href=\"#{}\">{}</a> <span class=\"badge b-{}\">{}</span> {} <span class=\"muted\">({})</span></div>",
                        esc(&f.finding_id),
                        esc(&f.finding_id),
                        priority_code(f.priority),
                        esc(t(priority_key(f.priority))),
                        esc(&f.title),
                        values.join(" · ")
                    ));
                }
                h.push_str("</td></tr>");
            }
            h.push_str("</tbody></table></section>\n");
        }

        // Differences with a plausible explanation, and those not judged (collapsed).
        if !self.explained.is_empty() {
            h.push_str(&format!(
                "<section class=\"sec\" id=\"explained\"><h2>{} ({})</h2><details><summary>{}</summary><ul class=\"groups\">",
                esc(t("explained")),
                self.explained.len(),
                esc(&count_text(&locale, "show_items", self.explained.len()))
            ));
            for group in &self.explained {
                let label = if group.disposition == "not_comparable" {
                    t("disp_not_comparable")
                } else {
                    t("disp_explainable")
                };
                h.push_str(&format!(
                    "<li><strong>{}</strong> <span class=\"tag\">{}</span> {}<div class=\"muted\">{}</div></li>",
                    esc(&group.key),
                    esc(label),
                    html_values(&group.values),
                    esc(&group.reason)
                ));
            }
            h.push_str("</ul></details></section>\n");
        }
        if !self.not_judged.is_empty() {
            h.push_str(&format!(
                "<section class=\"sec\" id=\"not-judged\"><h2>{} ({})</h2><details><summary>{}</summary><ul class=\"groups\">",
                esc(t("not_judged")),
                self.not_judged.len(),
                esc(&count_text(&locale, "show_items", self.not_judged.len()))
            ));
            for group in &self.not_judged {
                h.push_str(&format!(
                    "<li><strong>{}</strong> <span class=\"tag\">{}</span> {}",
                    esc(&group.key),
                    esc(t(&format!("status_{}", group.status))),
                    html_values(&group.values)
                ));
                if !group.reason.is_empty() {
                    h.push_str(&format!("<div class=\"muted\">{}</div>", esc(&group.reason)));
                }
                h.push_str("</li>");
            }
            h.push_str("</ul></details></section>\n");
        }

        // The same value seen in several places.
        if !self.consistent.is_empty() {
            h.push_str(&format!(
                "<section class=\"sec\" id=\"consistent\"><h2>{}</h2><table class=\"grid\"><thead><tr><th>{}</th><th>{}</th><th class=\"num\">{}</th><th class=\"num\">{}</th></tr></thead><tbody>",
                esc(t("consistent")),
                esc(t("col_fact")),
                esc(t("col_value")),
                esc(t("col_places")),
                esc(t("col_pages_count"))
            ));
            for fact in self.consistent_shown() {
                h.push_str(&format!(
                    "<tr><td>{}</td><td class=\"val\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td></tr>",
                    esc(&fact.name),
                    esc(&fact.value),
                    fmt_count(&locale, fact.sources),
                    fmt_count(&locale, fact.pages)
                ));
            }
            h.push_str("</tbody></table>");
            if self.consistent.len() > CONSISTENT_SHOWN {
                h.push_str(&format!(
                    "<p class=\"muted\">{}</p>",
                    esc(&count_text(
                        &locale,
                        "consistent_more",
                        self.consistent.len() - CONSISTENT_SHOWN
                    ))
                ));
            }
            h.push_str("</section>\n");
        }

        // Coverage & method.
        h.push_str(&format!(
            "<section class=\"sec\" id=\"coverage\"><h2>{}</h2><table class=\"grid cov\"><tbody>",
            esc(t("coverage"))
        ));
        for (label, value) in self.coverage_rows(&locale) {
            h.push_str(&format!("<tr><th>{}</th><td>{}</td></tr>", esc(label), esc(&value)));
        }
        h.push_str("</tbody></table>");
        for note in self.coverage_notes(&locale) {
            h.push_str(&format!("<p class=\"muted\">{}</p>", code_spans(&note)));
        }
        if !self.failed_sources.is_empty() {
            h.push_str(&format!("<h3>{}</h3><ul>", esc(t("failed_sources"))));
            for failed in &self.failed_sources {
                h.push_str(&format!(
                    "<li>{} — <span class=\"muted\">{}</span></li>",
                    esc(&failed.path),
                    esc(&failed.error)
                ));
            }
            h.push_str("</ul>");
        }
        h.push_str("</section>\n</main>\n");
        h.push_str(&format!(
            "<footer class=\"foot\"><p class=\"gen\">{} <a href=\"https://crawler.siteone.io/\" target=\"_blank\" rel=\"noopener\">SiteOne Crawler</a> (<a href=\"https://github.com/janreges/siteone-crawler\" target=\"_blank\" rel=\"noopener\">GitHub</a>).</p></footer>\n",
            esc(t("generated_by"))
        ));
        h.push_str("<script>\n");
        h.push_str(JS);
        h.push_str("\n</script>\n</body>\n</html>\n");
        h
    }

    fn html_finding(&self, h: &mut String, locale: &ReportLocale, f: &Finding) {
        let t = |key: &str| text(locale, key);
        h.push_str(&format!(
            "<article class=\"finding\" id=\"{}\" data-priority=\"{}\"><div class=\"fhead\"><span class=\"badge b-{}\">{}</span><span class=\"conf\">{}</span><span class=\"fid\">{}</span></div>",
            esc(&f.id),
            priority_code(f.priority),
            priority_code(f.priority),
            esc(t(priority_key(f.priority))),
            esc(t(confidence_key(f.confidence))),
            esc(&f.id)
        ));
        h.push_str(&format!("<p class=\"caution\">{}</p>", esc(&self.caution)));
        h.push_str(&format!("<h3>{}</h3>", esc(&f.title)));
        h.push_str(&format!("<p class=\"fact\">{}: {}</p>", esc(t("fact")), esc(&f.key)));
        if !f.explanation.is_empty() {
            h.push_str(&format!("<p>{}</p>", esc(&f.explanation)));
        }
        if !f.benign_explanations.is_empty() {
            h.push_str(&format!("<h4>{}</h4><ul>", esc(t("benign"))));
            for reason in &f.benign_explanations {
                h.push_str(&format!("<li>{}</li>", esc(reason)));
            }
            h.push_str("</ul>");
        }
        if !f.check.is_empty() {
            h.push_str(&format!("<h4>{}</h4><p>{}</p>", esc(t("check")), esc(&f.check)));
        }
        h.push_str(&format!(
            "<table class=\"grid values\"><thead><tr><th>{}</th><th>{}</th><th>{}</th></tr></thead><tbody>",
            esc(t("col_value")),
            esc(t("col_where")),
            esc(t("col_pages"))
        ));
        for value in &f.values {
            h.push_str(&format!("<tr><td class=\"val\">{}</td><td>", esc(&value.text)));
            for occurrence in value.occurrences.iter().take(OCCURRENCES_SHOWN) {
                html_occurrence(h, locale, occurrence, &value.text);
            }
            if value.occurrences.len() > OCCURRENCES_SHOWN {
                h.push_str(&format!(
                    "<details><summary>{}</summary>",
                    esc(&count_text(
                        locale,
                        "more_occurrences",
                        value.occurrences.len() - OCCURRENCES_SHOWN
                    ))
                ));
                for occurrence in &value.occurrences[OCCURRENCES_SHOWN..] {
                    html_occurrence(h, locale, occurrence, &value.text);
                }
                h.push_str("</details>");
            }
            h.push_str(&format!(
                "</td><td><div class=\"muted\">{}</div><ul class=\"urls\">",
                esc(&count_text(locale, "on_pages", value.urls.len()))
            ));
            for url in value.urls.iter().take(URLS_SHOWN) {
                h.push_str(&format!("<li>{}</li>", link(url, &display_path(url))));
            }
            h.push_str("</ul>");
            if value.urls.len() > URLS_SHOWN {
                h.push_str(&format!(
                    "<details><summary>{}</summary><ul class=\"urls\">",
                    esc(&count_text(locale, "more_pages", value.urls.len() - URLS_SHOWN))
                ));
                for url in &value.urls[URLS_SHOWN..] {
                    h.push_str(&format!("<li>{}</li>", link(url, &display_path(url))));
                }
                h.push_str("</ul></details>");
            }
            h.push_str("</td></tr>");
        }
        h.push_str("</tbody></table></article>\n");
    }
}

fn priority_key(p: Priority) -> &'static str {
    match p {
        Priority::Critical => "priority_critical",
        Priority::High => "priority_high",
        Priority::Medium => "priority_medium",
        Priority::Low => "priority_low",
    }
}

fn confidence_key(c: Confidence) -> &'static str {
    match c {
        Confidence::Likely => "confidence_likely",
        Confidence::Possible => "confidence_possible",
    }
}

/// Where an occurrence appears, with its section and conditions:
/// `Page /kontakt · Section: A > B · Conditions: …` or `Header/footer line · on 57 pages · …`.
fn occurrence_line(locale: &ReportLocale, o: &OccurrenceRef) -> String {
    let mut parts = vec![match o.region {
        SourceKind::Page => format!("{} {}", text(locale, "region_page"), o.path),
        SourceKind::Chrome => format!(
            "{} · {}",
            text(locale, "region_chrome"),
            count_text(locale, "on_pages", o.urls.len())
        ),
    }];
    if !o.heading_path.is_empty() {
        parts.push(format!("{}: {}", text(locale, "section"), o.heading_path.join(" > ")));
    }
    if !o.qualifiers.trim().is_empty() {
        parts.push(format!("{}: {}", text(locale, "conditions"), o.qualifiers));
    }
    parts.join(" · ")
}

fn html_occurrence(h: &mut String, locale: &ReportLocale, o: &OccurrenceRef, value: &str) {
    h.push_str(&format!(
        "<div class=\"occ\"><div class=\"occ-where\">{}</div>",
        esc(&occurrence_line(locale, o))
    ));
    if !o.evidence.is_empty() {
        h.push_str(&format!(
            "<blockquote class=\"ev\">{}</blockquote>",
            mark(&o.evidence, value)
        ));
    }
    h.push_str("</div>");
}

/// The escaped evidence with the first occurrence of the value highlighted.
fn mark(evidence: &str, value: &str) -> String {
    match evidence.find(value).filter(|_| !value.is_empty()) {
        Some(at) => format!(
            "{}<mark>{}</mark>{}",
            esc(&evidence[..at]),
            esc(value),
            esc(&evidence[at + value.len()..])
        ),
        None => esc(evidence),
    }
}

/// An `<a>` to `url` when it is a safe web URL, else the escaped label.
fn link(url: &str, label: &str) -> String {
    let lower = url.trim().to_ascii_lowercase();
    if (lower.starts_with("http://") || lower.starts_with("https://")) && is_safe_url(url) {
        format!("<a href=\"{}\">{}</a>", esc(url), esc(label))
    } else {
        esc(label)
    }
}

/// The path and query of a URL, for a compact link text.
fn display_path(url: &str) -> String {
    match url::Url::parse(url) {
        Ok(parsed) => match parsed.query() {
            Some(query) => format!("{}?{query}", parsed.path()),
            None => parsed.path().to_string(),
        },
        Err(_) => url.to_string(),
    }
}

/// The values of a group as escaped `<code>` chips.
fn html_values(values: &[FindingValue]) -> String {
    values
        .iter()
        .map(|v| format!("<code>{}</code>", esc(&v.text)))
        .collect::<Vec<_>>()
        .join(" · ")
}

/// Escaped text with `` `code` `` spans (from the fixed texts) as `<code>`.
fn code_spans(text: &str) -> String {
    text.split('`')
        .enumerate()
        .map(|(i, part)| {
            if i % 2 == 1 {
                format!("<code>{}</code>", esc(part))
            } else {
                esc(part)
            }
        })
        .collect()
}

/// Website or model text for Markdown: on one line, with `<` (which could open an HTML tag) and
/// the characters Markdown would interpret escaped.
fn md(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for (i, word) in text.split_whitespace().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        for c in word.chars() {
            match c {
                '<' => out.push_str("&lt;"),
                '\\' | '`' | '*' | '_' | '[' | ']' | '|' | '#' => {
                    out.push('\\');
                    out.push(c);
                }
                _ => out.push(c),
            }
        }
    }
    out
}

/// A web URL as a Markdown autolink; anything else as escaped text.
fn md_url(url: &str) -> String {
    let lower = url.trim().to_ascii_lowercase();
    if (lower.starts_with("http://") || lower.starts_with("https://"))
        && !url.contains(|c: char| c.is_whitespace() || c == '<' || c == '>')
    {
        format!("<{url}>")
    } else {
        md(url)
    }
}

fn md_values(values: &[FindingValue]) -> String {
    values
        .iter()
        .map(|v| format!("**{}**", md(&v.text)))
        .collect::<Vec<_>>()
        .join(" · ")
}

/// One CSV cell: a leading `'` for a cell a spreadsheet would run as a formula, and quotes
/// (doubled inside) when it holds a comma, a quote or a line break.
fn csv_cell(value: &str) -> String {
    let mut cell = value.to_string();
    let formula = cell
        .trim_start_matches(|c: char| c <= ' ')
        .starts_with(['=', '+', '-', '@']);
    if formula || cell.starts_with(['\t', '\r', '\n']) {
        cell.insert(0, '\'');
    }
    if cell.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", cell.replace('"', "\"\""))
    } else {
        cell
    }
}

const CSS: &str = r#":root{--bg:#f3f4f6;--surface:#fff;--ink:#111827;--muted:#6b7280;--border:#e5e7eb;--accent:#4e79a7;--accent-ink:#fff;--chipbg:#eef2f7;--logo-a:#111827;--logo-b:#4e79a7;--note:#fff8e6;--note-border:#ecd9a9;--crit-bg:#fbe9e7;--crit:#8f2d1f;--high-bg:#fcefe0;--high:#8a4b12;--med-bg:#fbf5dc;--med:#6f5a10;--low-bg:#eef0f3;--low:#374151;--mark:#fff1a8}
[data-theme="dark"]{--bg:#0f172a;--surface:#1f2937;--ink:#e5e7eb;--muted:#9ca3af;--border:#374151;--accent:#7aa8d6;--accent-ink:#0f172a;--chipbg:#243244;--logo-a:#e5e7eb;--logo-b:#7aa8d6;--note:#2a2415;--note-border:#5c4c25;--crit-bg:#3b1f1b;--crit:#f3b4a8;--high-bg:#3a2a18;--high:#f2c38f;--med-bg:#353018;--med:#e8d48a;--low-bg:#273142;--low:#cbd5e1;--mark:#5a4d12}
*{box-sizing:border-box}body{margin:0;font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink)}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.bar{position:sticky;top:0;z-index:10;display:flex;align-items:center;justify-content:space-between;gap:16px;padding:11px 22px;background:var(--surface);border-bottom:1px solid var(--border)}
.logo{display:inline-flex;align-items:center;gap:11px;text-decoration:none}.logo:hover{text-decoration:none}
.logo-svg{display:block;height:30px;width:auto}
.wordmark{font-weight:800;font-size:16px;letter-spacing:-.01em;color:var(--ink);line-height:1.05}
.wordmark .sub{display:block;font-weight:700;font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin-top:3px}
.btn{background:var(--chipbg);color:var(--ink);border:1px solid var(--border);border-radius:8px;padding:6px 12px;cursor:pointer;font-size:14px}
.btn.on{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
.hero{max-width:1100px;margin:26px auto 0;padding:0 22px}
.hero-card{position:relative;background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:22px 26px 20px;overflow:hidden}
.hero-card::before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--accent)}
.hero-title{font-size:28px;line-height:1.2;margin:0 0 4px;font-weight:800;letter-spacing:-.02em}
.hero-kicker{color:var(--muted);font-size:12.5px;font-weight:600;letter-spacing:.08em;text-transform:uppercase}
.hero-meta{margin-top:10px;color:var(--muted);font-size:14px}.hero-meta a{font-weight:600;word-break:break-all}.hero-meta .dot{margin:0 8px;opacity:.5}
.wrap{max-width:1100px;margin:20px auto 0;padding:0 22px 40px}
.sec{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px 22px;margin:0 0 18px}
.sec h2{margin:0 0 10px;font-size:20px;border-bottom:1px solid var(--border);padding-bottom:8px}
.sec h3{font-size:17px;margin:8px 0 4px}.sec h4{font-size:14px;margin:12px 0 4px;color:var(--muted)}
.sec p{margin:0 0 10px}.sec ul{margin:0 0 10px;padding-left:20px}
.disclaimer{background:var(--note);border-color:var(--note-border)}.disclaimer p{margin:0}
.state{font-size:15px;margin:0 0 6px}.reasons{color:var(--muted);font-size:14px}
.summary{font-size:16px}
.filter{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}
.finding{border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin:0 0 14px}
.fhead{display:flex;flex-wrap:wrap;align-items:center;gap:10px}.fid{margin-left:auto;color:var(--muted);font-size:13px}
.badge{display:inline-block;font-size:12px;font-weight:700;padding:2px 10px;border-radius:999px}
.b-critical{background:var(--crit-bg);color:var(--crit)}.b-high{background:var(--high-bg);color:var(--high)}
.b-medium{background:var(--med-bg);color:var(--med)}.b-low{background:var(--low-bg);color:var(--low)}
.conf{font-size:13px;color:var(--muted);font-weight:600}
.caution{font-size:13.5px;color:var(--muted);border-left:3px solid var(--note-border);padding:2px 10px;margin:10px 0}
.fact{font-size:13px;color:var(--muted)}
.grid{border-collapse:collapse;width:100%;font-size:14px;margin:6px 0 4px}
.grid th,.grid td{border:1px solid var(--border);padding:6px 9px;text-align:left;vertical-align:top}
.grid thead th{background:var(--chipbg)}.cov th{width:45%;font-weight:600;background:var(--chipbg)}
.num{text-align:right!important}.val{font-weight:700;white-space:nowrap}.url{word-break:break-all}
.occ{margin:0 0 8px}.occ-where{font-size:13px;color:var(--muted)}
.ev{margin:3px 0 0;padding:4px 10px;border-left:3px solid var(--border);font-size:13.5px}
mark{background:var(--mark);color:inherit;padding:0 2px;border-radius:3px}
.urls{list-style:none;padding:0!important;margin:4px 0!important;font-size:13px;word-break:break-all}
details summary{cursor:pointer;color:var(--accent);font-size:13.5px}
.groups li{margin:0 0 8px}.tag{font-size:12px;background:var(--chipbg);border-radius:6px;padding:1px 7px}
code{background:var(--chipbg);border-radius:4px;padding:1px 5px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}
.muted{color:var(--muted);font-size:13.5px}
.foot{max-width:1100px;margin:10px auto 0;padding:16px 22px 40px;color:var(--muted);border-top:1px solid var(--border)}
.gen{font-size:12px;margin:5px 0 0}.foot a{font-weight:600}
@media(max-width:760px){.hero-title{font-size:23px}.val{white-space:normal}}"#;

const JS: &str = r#"(function(){var html=document.documentElement;
try{if(matchMedia('(prefers-color-scheme: dark)').matches)html.setAttribute('data-theme','dark');}catch(e){}
var b=document.getElementById('themeBtn');if(b)b.addEventListener('click',function(){html.setAttribute('data-theme',html.getAttribute('data-theme')==='dark'?'light':'dark');});
var buttons=document.querySelectorAll('.filter button');
buttons.forEach(function(button){button.addEventListener('click',function(){var p=button.getAttribute('data-filter');
buttons.forEach(function(x){x.classList.toggle('on',x===button);});
document.querySelectorAll('.finding').forEach(function(f){f.hidden=!(p==='all'||f.getAttribute('data-priority')===p);});});});})();"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::consistency::judge::has_forbidden_word;
    use crate::ai::consistency::model::{AttributeKey, Confidence, ConsistentFact, Priority, SourceKind};
    use crate::ai::report::locale::ReportLocale;

    fn en() -> ReportLocale {
        ReportLocale::new("en")
    }

    fn cs() -> ReportLocale {
        ReportLocale::new("cs")
    }

    fn occurrence(id: usize, region: SourceKind, path: &str, evidence: &str, urls: &[&str]) -> OccurrenceRef {
        OccurrenceRef {
            occurrence_id: id,
            region,
            path: path.to_string(),
            qualifiers: String::new(),
            heading_path: vec!["Kontakt".to_string()],
            evidence: evidence.to_string(),
            urls: urls.iter().map(|u| u.to_string()).collect(),
        }
    }

    fn value(text: &str, occurrences: Vec<OccurrenceRef>) -> FindingValue {
        let mut urls: Vec<String> = occurrences.iter().flat_map(|o| o.urls.clone()).collect();
        urls.sort();
        urls.dedup();
        FindingValue {
            text: text.to_string(),
            occurrences,
            urls,
        }
    }

    fn finding(key: &str, priority: Priority, confidence: Confidence, attribute_key: AttributeKey) -> Finding {
        Finding {
            id: String::new(),
            priority,
            confidence,
            attribute_key,
            key: key.to_string(),
            title: format!("{key}: values differ"),
            explanation: "The pages state different values for the same property.".to_string(),
            benign_explanations: vec!["The values might apply to different branches.".to_string()],
            check: "Compare the values on the listed pages.".to_string(),
            prose_replaced: false,
            values: vec![
                value(
                    "800 123 456",
                    vec![occurrence(
                        1,
                        SourceKind::Chrome,
                        "",
                        "Zákaznická linka 800 123 456",
                        &["https://example.com/", "https://example.com/kontakt"],
                    )],
                ),
                value(
                    "800 123 465",
                    vec![occurrence(
                        2,
                        SourceKind::Page,
                        "/reklamace",
                        "Volejte 800 123 465.",
                        &["https://example.com/reklamace"],
                    )],
                ),
            ],
        }
    }

    fn meta(language: &str) -> Meta {
        Meta {
            host: "example.com".to_string(),
            url: "https://example.com/".to_string(),
            generated_at: "2026-09-26 10:00".to_string(),
            language: language.to_string(),
            provider: "openai-compatible".to_string(),
            model: "test-model".to_string(),
            context_window: 128_000,
            pages_eligible: 12,
            pages_selected: 10,
            pages_analyzed: 10,
            pages_missing_body: 0,
            pages_reduced: 0,
            chrome_lines_analyzed: 3,
            chrome_lines_excluded: 0,
            sources: 11,
            sources_with_facts: 9,
            facts_extracted: 30,
            facts_kept: 28,
            facts_ungrounded: 2,
            keys: 14,
            candidates: 3,
            reviewed: 3,
            comparisons_done: 3,
            reviews_capped: 0,
            reviews_failed: 0,
            reviews_skipped_budget: 0,
            reviews_partial: 0,
            grouping_incomplete: false,
            items_not_cross_compared: 0,
            llm_calls: 14,
            duration_ms: 12_345,
            http_cache: None,
        }
    }

    fn counts_of(findings: &[Finding]) -> Counts {
        let of = |p: Priority| findings.iter().filter(|f| f.priority == p).count();
        Counts {
            critical: of(Priority::Critical),
            high: of(Priority::High),
            medium: of(Priority::Medium),
            low: of(Priority::Low),
            explained: 1,
            not_comparable: 0,
            insufficient: 1,
            not_reviewed: 0,
            consistent: 1,
        }
    }

    /// A document built the way the pipeline builds it: sorted, numbered findings, the by-page
    /// view, the summary and the completeness state.
    fn sample(locale: &ReportLocale, mut findings: Vec<Finding>) -> ConsistencyDoc {
        let meta = meta(locale.code());
        sort_findings(&mut findings);
        for (i, f) in findings.iter_mut().enumerate() {
            f.id = format!("F{}", i + 1);
        }
        let counts = counts_of(&findings);
        let completeness = completeness(&meta, &counts, 0, meta.sources);
        ConsistencyDoc {
            schema: SCHEMA,
            summary: summary(locale, &meta, &counts, &completeness),
            disclaimer: text(locale, "disclaimer").to_string(),
            caution: text(locale, "caution").to_string(),
            by_page: by_page(&findings),
            explained: vec![ExplainedGroup {
                key: "Doprava zdarma – hranice".to_string(),
                attribute_key: AttributeKey::FreeShippingThreshold,
                disposition: "explainable",
                title: "Different thresholds for different countries".to_string(),
                reason: "The thresholds might apply to different countries.".to_string(),
                benign_explanations: Vec::new(),
                values: vec![value("1 500 Kč", Vec::new()), value("60 €", Vec::new())],
            }],
            not_judged: vec![NotJudgedGroup {
                key: "Pobočka – otevírací doba".to_string(),
                attribute_key: AttributeKey::OpeningHours,
                status: "insufficient_context",
                reason: "The evidence does not say which branch.".to_string(),
                values: vec![value("8–16", Vec::new()), value("9–17", Vec::new())],
            }],
            consistent: vec![ConsistentFact {
                key_id: 0,
                name: "Example s.r.o. – IČO".to_string(),
                attribute_key: AttributeKey::CompanyId,
                value: "12345678".to_string(),
                sources: 3,
                pages: 10,
                occurrence_ids: vec![5, 6, 7],
            }],
            sources: Vec::new(),
            occurrences: Vec::new(),
            failed_sources: Vec::new(),
            meta,
            completeness,
            counts,
            findings,
        }
    }

    fn mixed() -> Vec<Finding> {
        vec![
            finding(
                "Zákaznická linka – telefon",
                Priority::Medium,
                Confidence::Possible,
                AttributeKey::Phone,
            ),
            finding("Hypotéka – RPSN", Priority::High, Confidence::Likely, AttributeKey::Apr),
        ]
    }

    #[test]
    fn disclaimer_and_completeness_come_before_the_findings() {
        for locale in [en(), cs()] {
            let doc = sample(&locale, mixed());
            let first_title = &doc.findings[0].title;
            let state = text(&locale, "state_complete");
            for (format, rendered) in [("md", doc.to_markdown()), ("html", doc.to_html())] {
                let disclaimer = rendered
                    .find(&esc(&doc.disclaimer))
                    .or_else(|| rendered.find(&doc.disclaimer))
                    .unwrap_or_else(|| panic!("{format}: no disclaimer"));
                let completeness = rendered.find(state).unwrap_or_else(|| panic!("{format}: no state"));
                let finding = rendered
                    .find(&esc(first_title))
                    .unwrap_or_else(|| panic!("{format}: no finding"));
                assert!(disclaimer < completeness, "{format}: disclaimer before completeness");
                assert!(completeness < finding, "{format}: completeness before the findings");
                assert!(
                    rendered.find(&esc(&doc.summary)).is_some_and(|at| at < finding),
                    "{format}: the summary before the findings"
                );
            }
        }
    }

    #[test]
    fn every_finding_carries_the_caution_line() {
        for locale in [en(), cs()] {
            let doc = sample(&locale, mixed());
            let caution = text(&locale, "caution");
            assert!(caution.starts_with("Possible inconsistency") || caution.starts_with("Možný nesoulad"));
            for (format, rendered) in [("md", doc.to_markdown()), ("html", doc.to_html())] {
                assert_eq!(
                    rendered.matches(&esc(caution)).count(),
                    doc.findings.len(),
                    "{format}: one caution line per finding"
                );
            }
        }
    }

    #[test]
    fn fixed_texts_are_exact_and_calm() {
        assert_eq!(
            text(&en(), "disclaimer"),
            "This report is an automated comparison of facts found on the website. It is not proof of an error: values can differ for legitimate reasons (another product variant, period, region or condition, rounding), and the automated review can misread the context or miss a difference. Please verify every finding manually before changing any content."
        );
        assert_eq!(
            text(&cs(), "disclaimer"),
            "Tato zpráva je automatické porovnání faktů nalezených na webu. Není důkazem chyby: hodnoty se mohou lišit oprávněně (jiná varianta produktu, období, region či podmínky, zaokrouhlení) a automatické posouzení může kontext pochopit špatně nebo rozdíl přehlédnout. Každé zjištění prosím před úpravou obsahu ručně ověřte."
        );
        assert_eq!(
            text(&en(), "caution"),
            "Possible inconsistency — please verify manually. The values may apply to different circumstances."
        );
        assert_eq!(
            text(&cs(), "caution"),
            "Možný nesoulad — ověřte prosím ručně. Hodnoty se mohou vztahovat k různým situacím."
        );
        for (key, en_text, cs_text) in [
            (
                "priority_critical",
                "Critical — check first",
                "Kritické — ověřte přednostně",
            ),
            ("priority_high", "High", "Vysoké"),
            ("priority_medium", "Medium", "Střední"),
            ("priority_low", "Low", "Nízké"),
            ("confidence_likely", "Likely inconsistent", "Pravděpodobný nesoulad"),
            ("confidence_possible", "Possibly inconsistent", "Možný nesoulad"),
        ] {
            assert_eq!(text(&en(), key), en_text);
            assert_eq!(text(&cs(), key), cs_text);
        }
        // Other report languages get the English fixed texts.
        assert_eq!(text(&ReportLocale::new("de"), "priority_high"), "High");
        // The disclaimer quotes the words in a negation ("not proof of an error"), as required
        // verbatim; every other fixed text avoids them.
        for (key, en_text, cs_text) in TEXTS {
            if *key == "disclaimer" {
                continue;
            }
            for (language, text) in [("en", en_text), ("cs", cs_text)] {
                assert!(!text.is_empty(), "{key} ({language}) is empty");
                assert!(!has_forbidden_word(text), "{key} ({language}): {text}");
            }
        }
    }

    #[test]
    fn summaries_are_calm_in_the_zero_mixed_and_insufficient_cases() {
        let zero_counts = Counts {
            critical: 0,
            high: 0,
            medium: 0,
            low: 0,
            explained: 0,
            not_comparable: 0,
            insufficient: 0,
            not_reviewed: 0,
            consistent: 4,
        };
        let mixed_counts = Counts {
            critical: 0,
            high: 1,
            medium: 4,
            low: 2,
            explained: 30,
            not_comparable: 8,
            insufficient: 3,
            not_reviewed: 2,
            consistent: 9,
        };
        let mut big = meta("en");
        big.facts_kept = 2184;
        big.pages_analyzed = 612;
        let complete = completeness(&big, &zero_counts, 0, big.sources);
        assert_eq!(
            summary(&en(), &big, &zero_counts, &complete),
            "No potential inconsistencies were identified among the compared facts. We compared 2,184 facts from 612 pages and the shared header/footer (3 distinct fact lines)."
        );
        assert_eq!(
            summary(&en(), &big, &mixed_counts, &complete),
            "We compared 2,184 facts from 612 pages and the shared header/footer (3 distinct fact lines). 7 differences are worth checking: 1 high, 4 medium, 2 low. 38 differences have a plausible legitimate explanation; 5 could not be judged from the evidence."
        );
        assert_eq!(
            summary(&cs(), &big, &zero_counts, &complete),
            "Mezi porovnanými fakty jsme nenašli žádný možný nesoulad. Porovnali jsme 2\u{a0}184 faktů uvedených na 612 stránkách a ve sdílené hlavičce/patičce (3 odlišné řádky s fakty)."
        );
        assert_eq!(
            summary(&cs(), &big, &mixed_counts, &complete),
            "Porovnali jsme 2\u{a0}184 faktů uvedených na 612 stránkách a ve sdílené hlavičce/patičce (3 odlišné řádky s fakty). 7 rozdílů stojí za ověření (vysoké 1, střední 4, nízké 2). 38 rozdílů má věrohodné vysvětlení; 5 nešlo z podkladů posoudit."
        );
        let mut one = meta("en");
        one.facts_kept = 1;
        one.pages_analyzed = 1;
        one.chrome_lines_analyzed = 0;
        let one_finding = Counts {
            medium: 1,
            ..zero_counts.clone()
        };
        assert_eq!(
            summary(&en(), &one, &one_finding, &complete),
            "We compared 1 fact from 1 page. 1 difference is worth checking: 1 medium."
        );
        assert_eq!(
            summary(&cs(), &one, &one_finding, &complete),
            "Porovnali jsme 1 fakt uvedený na 1 stránce. 1 rozdíl stojí za ověření (střední 1)."
        );

        let mut thin = meta("en");
        thin.sources_with_facts = 2;
        let insufficient = completeness(&thin, &zero_counts, 0, thin.sources);
        assert_eq!(insufficient.state, CompletenessState::Insufficient);
        for locale in [en(), cs()] {
            for (counts, state) in [
                (&zero_counts, &complete),
                (&mixed_counts, &complete),
                (&zero_counts, &insufficient),
            ] {
                let text = summary(&locale, &big, counts, state);
                assert!(!has_forbidden_word(&text), "{text}");
            }
            let text = summary(&locale, &thin, &zero_counts, &insufficient);
            assert!(
                text.starts_with("There was too little evidence") || text.starts_with("Podkladů bylo"),
                "{text}"
            );
            // Never a bare "all good": the zero case always states the coverage too.
            assert!(text.contains("28"), "{text}");
        }
    }

    #[test]
    fn completeness_states_follow_the_coverage() {
        let counts = counts_of(&mixed());
        let full = meta("en");
        let state = completeness(&full, &counts, 0, full.sources);
        assert_eq!(state.state, CompletenessState::Complete);
        assert!(state.reasons.is_empty());

        let mut partial = meta("en");
        partial.pages_reduced = 2;
        partial.chrome_lines_excluded = 4;
        partial.grouping_incomplete = true;
        partial.items_not_cross_compared = 7;
        partial.reviews_capped = 5;
        partial.reviews_failed = 1;
        partial.pages_missing_body = 1;
        let state = completeness(&partial, &counts, 1, partial.sources);
        assert_eq!(state.state, CompletenessState::Partial);
        let codes: Vec<(&str, usize)> = state.reasons.iter().map(|r| (r.code, r.count)).collect();
        assert_eq!(
            codes,
            vec![
                ("failed_sources", 1),
                ("missing_body", 1),
                ("pages_reduced", 2),
                ("chrome_lines_excluded", 4),
                ("grouping_incomplete", 7),
                ("not_reviewed_cap", 5),
                ("not_reviewed_call_failed", 1),
            ]
        );
        for locale in [en(), cs()] {
            for reason in &state.reasons {
                let text = reason_text(&locale, reason);
                assert!(!text.is_empty() && !text.contains('{'), "{text}");
                assert!(!has_forbidden_word(&text), "{text}");
            }
        }
        assert_eq!(
            reason_text(&en(), &state.reasons[3]),
            "4 header/footer lines were not analyzed (limit)"
        );
        assert_eq!(
            reason_text(&cs(), &state.reasons[3]),
            "4 řádky hlavičky/patičky nebyly analyzovány (limit)"
        );

        let mut skipped = meta("en");
        skipped.reviews_skipped_budget = 3;
        let state = completeness(&skipped, &counts, 0, skipped.sources);
        assert_eq!(state.state, CompletenessState::Partial);
        assert_eq!(state.reasons[0].code, "not_reviewed_output_budget");

        // Fewer than 3 sources with facts, or at least half of the sources failed.
        let mut thin = meta("en");
        thin.sources_with_facts = 2;
        let state = completeness(&thin, &counts, 0, thin.sources);
        assert_eq!(state.state, CompletenessState::Insufficient);
        assert_eq!(state.reasons[0].code, "insufficient_sources");
        let state = completeness(&full, &counts, 6, 12);
        assert_eq!(state.state, CompletenessState::Insufficient);
        let state = completeness(&full, &counts, 5, 12);
        assert_eq!(state.state, CompletenessState::Partial);
        let mut none = meta("en");
        none.sources_with_facts = 0;
        let state = completeness(&none, &counts, 0, 0);
        assert_eq!(state.state, CompletenessState::Insufficient);
        assert_eq!(state.reasons[0].code, "no_facts");
    }

    #[test]
    fn findings_sort_by_priority_confidence_importance_and_reach() {
        let mut wide = finding("B wide", Priority::Medium, Confidence::Possible, AttributeKey::Phone);
        wide.values[0].urls.push("https://example.com/extra".to_string());
        let mut findings = vec![
            finding("Z low", Priority::Low, Confidence::Likely, AttributeKey::Price),
            finding(
                "A figure",
                Priority::Medium,
                Confidence::Possible,
                AttributeKey::CustomersCount,
            ),
            finding("C narrow", Priority::Medium, Confidence::Possible, AttributeKey::Phone),
            wide,
            finding("D likely", Priority::Medium, Confidence::Likely, AttributeKey::Rating),
            finding("E critical", Priority::Critical, Confidence::Likely, AttributeKey::Apr),
        ];
        sort_findings(&mut findings);
        let keys: Vec<&str> = findings.iter().map(|f| f.key.as_str()).collect();
        assert_eq!(
            keys,
            vec!["E critical", "D likely", "B wide", "C narrow", "A figure", "Z low"]
        );
    }

    #[test]
    fn by_page_lists_every_page_with_the_findings_that_mention_it() {
        let doc = sample(&en(), mixed());
        let pages: Vec<(&str, Vec<&str>)> = doc
            .by_page
            .iter()
            .map(|p| {
                (
                    p.url.as_str(),
                    p.findings.iter().map(|f| f.finding_id.as_str()).collect(),
                )
            })
            .collect();
        assert_eq!(
            pages,
            vec![
                ("https://example.com/", vec!["F1", "F2"]),
                ("https://example.com/kontakt", vec!["F1", "F2"]),
                ("https://example.com/reklamace", vec!["F1", "F2"]),
            ]
        );
        let home = &doc.by_page[0].findings[0];
        assert_eq!(home.values, vec!["800 123 456".to_string()]);
        assert_eq!(home.priority, Priority::High);
        assert_eq!(home.title, doc.findings[0].title);
        let html = doc.to_html();
        let md = doc.to_markdown();
        for rendered in [&html, &md] {
            assert!(rendered.contains(text(&en(), "by_page")));
            assert!(rendered.contains("https://example.com/reklamace"));
        }
    }

    #[test]
    fn csv_has_a_row_per_finding_value_and_url_and_quotes_what_it_must() {
        let mut hostile = finding(
            "Tarif, \"Basic\"",
            Priority::High,
            Confidence::Likely,
            AttributeKey::Price,
        );
        hostile.values[0].text = "290 Kč,\n\"akce\"".to_string();
        hostile.values[0].occurrences[0].qualifiers = "od 1. 10., \"bez DPH\"".to_string();
        hostile.values[0].occurrences[0].heading_path = vec!["Ceník".to_string(), "Tarify".to_string()];
        hostile.values[1].text = "=HYPERLINK(\"x\")".to_string();
        let doc = sample(&en(), vec![hostile]);
        let csv = doc.to_csv();
        let mut lines = csv.split('\n');
        assert_eq!(
            lines.next(),
            Some("\u{feff}finding_id,priority,confidence,attribute,subject,value,qualifiers,url,region,heading_path")
        );
        assert!(csv.contains(
            "F1,high,likely,price,\"Tarif, \"\"Basic\"\"\",\"290 Kč,\n\"\"akce\"\"\",\"od 1. 10., \"\"bez DPH\"\"\",https://example.com/,chrome,Ceník > Tarify\n"
        ), "{csv}");
        assert!(csv.contains(
            "F1,high,likely,price,\"Tarif, \"\"Basic\"\"\",\"290 Kč,\n\"\"akce\"\"\",\"od 1. 10., \"\"bez DPH\"\"\",https://example.com/kontakt,chrome,Ceník > Tarify\n"
        ), "{csv}");
        // A cell a spreadsheet would run as a formula is neutralized.
        assert!(
            csv.contains(",\"'=HYPERLINK(\"\"x\"\")\",,https://example.com/reklamace,page,Kontakt\n"),
            "{csv}"
        );
        // One row per finding × value × affected URL, plus the header.
        assert_eq!(csv.matches("\nF1,").count(), 3, "{csv}");
    }

    #[test]
    fn html_escapes_hostile_values_evidence_and_prose() {
        let mut hostile = finding(
            "<script>alert(1)</script>",
            Priority::High,
            Confidence::Likely,
            AttributeKey::Price,
        );
        hostile.title = "<script>alert(2)</script>".to_string();
        hostile.explanation = "<img src=x onerror=alert(3)>".to_string();
        hostile.check = "<script>alert(4)</script>".to_string();
        hostile.benign_explanations = vec!["<script>alert(5)</script>".to_string()];
        hostile.values[0].text = "<script>alert(6)</script>".to_string();
        hostile.values[0].occurrences[0].evidence = "Cena <script>alert(7)</script>".to_string();
        hostile.values[0].occurrences[0].qualifiers = "<b>od</b>".to_string();
        hostile.values[0].urls.push("javascript:alert(8)".to_string());
        let doc = sample(&en(), vec![hostile]);
        let html = doc.to_html();
        for n in 1..=7 {
            assert!(!html.contains(&format!("<script>alert({n})")), "alert({n}) is live");
            assert!(
                html.contains(&format!("&lt;script&gt;alert({n})")) || n == 3,
                "alert({n}) is shown"
            );
        }
        assert!(!html.contains("<img src=x"));
        assert!(!html.contains("<b>od</b>"));
        assert!(!html.contains("href=\"javascript:"));
        // Self-contained: no external assets.
        assert!(!html.contains("<script src") && !html.contains("<link rel=\"stylesheet\""));
        assert!(html.contains("data-priority=\"high\""));
        let md = doc.to_markdown();
        assert!(!md.contains("<script>alert("), "{md}");
    }

    #[test]
    fn json_has_the_schema_camel_case_keys_and_full_url_lists() {
        let mut many = finding("Linka", Priority::Medium, Confidence::Possible, AttributeKey::Phone);
        let urls: Vec<String> = (0..12).map(|i| format!("https://example.com/p{i:02}")).collect();
        many.values[0].occurrences[0].urls = urls.clone();
        many.values[0].urls = urls.clone();
        let doc = sample(&en(), vec![many]);
        let json = doc.to_json();
        assert_eq!(json["schema"], "siteone-crawler/ai-consistency/2");
        assert_eq!(json["meta"]["pagesAnalyzed"], 10);
        assert_eq!(json["meta"]["factsUngrounded"], 2);
        assert_eq!(json["completeness"]["state"], "complete");
        assert_eq!(json["counts"]["notComparable"], 0);
        let first = &json["findings"][0];
        assert_eq!(first["attributeKey"], "phone");
        assert_eq!(first["priority"], "medium");
        assert_eq!(first["confidence"], "possible");
        assert_eq!(
            first["benignExplanations"][0],
            "The values might apply to different branches."
        );
        assert_eq!(first["values"][0]["urls"].as_array().map(Vec::len), Some(12));
        assert_eq!(first["values"][0]["occurrences"][0]["region"], "chrome");
        assert_eq!(first["values"][0]["occurrences"][0]["occurrenceId"], 1);
        assert_eq!(json["byPage"].as_array().map(Vec::len), Some(13));
        assert_eq!(json["consistent"][0]["attributeKey"], "company_id");
        assert_eq!(json["notJudged"][0]["status"], "insufficient_context");
        // Only 5 URLs are listed openly; the rest is collapsed but present.
        let html = doc.to_html();
        assert!(html.contains("https://example.com/p11"));
        assert!(html.contains("<details"));
        let md = doc.to_markdown();
        assert!(md.contains("https://example.com/p11"));
        fn keys_are_camel_case(value: &serde_json::Value, path: &str) {
            match value {
                serde_json::Value::Object(map) => {
                    for (key, inner) in map {
                        assert!(
                            !key.contains('_') && !key.starts_with(|c: char| c.is_ascii_uppercase()),
                            "{path}.{key}"
                        );
                        keys_are_camel_case(inner, &format!("{path}.{key}"));
                    }
                }
                serde_json::Value::Array(items) => {
                    for item in items {
                        keys_are_camel_case(item, path);
                    }
                }
                _ => {}
            }
        }
        keys_are_camel_case(&json, "");
    }

    #[test]
    fn the_http_cache_note_appears_only_when_the_cache_is_set() {
        let mut doc = sample(&en(), mixed());
        let note = "Pages may come from the crawler's HTTP cache";
        assert!(!doc.to_markdown().contains(note));
        assert!(!doc.to_html().contains(note));
        doc.meta.http_cache = Some("24h".to_string());
        for rendered in [doc.to_markdown(), doc.to_html()] {
            assert!(rendered.contains(note) || rendered.contains(&esc(note)), "{rendered}");
            assert!(rendered.contains("(TTL 24h)"));
            assert!(rendered.contains("--http-cache-dir="));
        }
    }

    #[test]
    fn a_report_without_findings_says_so_calmly_and_keeps_the_other_sections() {
        let doc = sample(&cs(), Vec::new());
        for rendered in [doc.to_markdown(), doc.to_html()] {
            assert!(rendered.contains("Mezi porovnanými fakty jsme nenašli žádný možný nesoulad."));
            assert!(rendered.contains(text(&cs(), "explained")));
            assert!(rendered.contains(text(&cs(), "not_judged")));
            assert!(rendered.contains(text(&cs(), "consistent")));
            assert!(rendered.contains(text(&cs(), "coverage")));
            assert!(rendered.contains("12345678"));
            assert!(!rendered.contains(text(&cs(), "caution")));
        }
    }
}
