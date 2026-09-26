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
