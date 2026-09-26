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

## `geo/` — `--ai-geo` pipeline

- `page.md` — the system prompt of the per-page analysis (one call per analyzed page), embedded by
  `src/ai/geo/prompts.rs`, which appends the output language. The page itself is sent in the user
  message as numbered blocks inside `<page_data>`; the model answers with block ids, which the
  crawler verifies (`src/ai/geo/analyze.rs`). Hand-written, not generated.
