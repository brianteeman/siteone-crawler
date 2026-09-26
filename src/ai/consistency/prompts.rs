// SiteOne Crawler - AI fact consistency: embedded prompts
// (c) Jan Reges <jan.reges@siteone.cz>
//
// The static system prompts of the pipeline's LLM steps (repo-root `prompts/consistency/`). They
// are identical for every call of a step, so providers can prefix-cache them; everything from the
// website or from an earlier step goes into the user message, inside the step's XML envelope.

/// Extraction: a few block-grounded facts from one page or one chunk of header/footer lines.
pub const EXTRACT: &str = include_str!("../../../prompts/consistency/extract.md");

/// Grouping: labels of one attribute key that name the same property of the same subject.
pub const GROUP: &str = include_str!("../../../prompts/consistency/group.md");

/// Review: whether the values of each group may contradict each other; `judge_system` appends the
/// output language.
pub const JUDGE: &str = include_str!("../../../prompts/consistency/judge.md");

/// The review system prompt: `JUDGE` plus the (escaped) language of the prose it writes.
pub fn judge_system(language: &str) -> String {
    format!(
        "{JUDGE}\n\n<output_language>\nWrite \"title\", \"explanation\", \"benign_explanations\" and \"check\" in the language '{}'. Quote values exactly as they appear in <groups>.\n</output_language>",
        crate::ai::prompt::sanitize_for_prompt(language)
    )
}
