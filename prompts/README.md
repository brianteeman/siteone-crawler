# Prompt packs

Embedded LLM prompt content, compiled into the binary via `include_str!`.

## `profile/` — `--ai-profile` pipeline

- `shared/*.md` — one system prompt per pipeline phase. `{{placeholder}}` tokens are substituted
  at call time (see `src/ai/profile/promptpack.rs::render`).
- `types/<key>.md` — front matter `name_cs` / `name_en` + a classifier description body.
- `chapters/<key>/<NN-id>.md` — front matter `heading` / `target_chars` / `max_pages` + two
  sections `## selection_focus` and `## synthesis_instructions`.

`types/` and `chapters/` are GENERATED from
`docs/superpowers/specs/2026-07-21-ai-profile-chapters-research.md` by
`scripts/gen_profile_prompts.py`. Edit the research doc (or the generator) and re-run it; do not
hand-edit generated files. `src/ai/profile/embedded.rs` (the `include_str!` table) is generated too.

## `consistency/` — `--ai-consistency` pipeline

Static system prompts, one per LLM step, embedded by `src/ai/consistency/prompts.rs`. They take no
placeholders: everything from the website or from an earlier step goes into the user message,
inside the step's XML envelope, escaped with `sanitize_for_prompt`.

- `extract.md` — a few block-grounded facts from one page (`<page_data>`) or one chunk of
  header/footer lines (`<chrome_data>`).
- `group.md` — groups the labels of one attribute key (`<labels>`) that name the same property of
  the same subject.
