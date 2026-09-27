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
- `judge.md` — reviews groups of differing values (`<groups>`) with a lenient rubric;
  `prompts::judge_system` appends the `<crawl_date>` (dated articles are judged against it) and the
  `<output_language>` of the prose.

Each prompt has a `<security>` block that declares its envelope untrusted data, and the unit tests
check that a forged closing tag or a natural-language instruction inside the data stays inert. The
code never trusts the answers: facts are verified against the crawler's blocks, ids against the
supplied lists, and the review's priority, numbers and wording are gated in `judge.rs`.

After changing a prompt, run the live evaluation on the committed corpus
(`tests/fixtures/consistency/`, expectations in `expected.json`):
`SITEONE_LIVE_AI_ENDPOINT=… SITEONE_LIVE_AI_MODEL=… cargo test --test integration_crawl consistency_corpus_live -- --ignored --nocapture`
(optional `SITEONE_LIVE_AI_EXTRA_BODY`, `SITEONE_LIVE_AI_CONTEXT_WINDOW`, `SITEONE_LIVE_AI_REPORT_DIR`). It
prints recall, false positives and missed cases.

## `geo/` — `--ai-geo` pipeline

- `page.md` — the system prompt of the per-page analysis (one call per analyzed page), embedded by
  `src/ai/geo/prompts.rs`, which appends the output language. The page itself is sent in the user
  message as numbered blocks inside `<page_data>`; the model answers with block ids, which the
  crawler verifies (`src/ai/geo/analyze.rs`). Hand-written, not generated.

The output language is named (`Czech ('cs')`, falling back to the bare code for an unknown one):
with a bare code, a local model wrote most questions and improvements in the page's language. The
lead draft stays in the page's language. The prompt's rules are enforced by the crawler, not trusted:
every block id, quote, lead number, FAQ pair and byline is checked, a lead is kept only for a page
that does not state its offer early, and the kit builds FAQPage markup only from at least two real
questions (ending with a question mark) on an indexable page.

After changing the prompt, check it live on a small site, e.g.
`--ai-geo --ai-max-pages=5 --ai-report-language=cs`, and read the per-page section of the report
and `leads.md` / `drafts/entity-drafts.md` in the kit; `ai_geo_end_to_end` in
`tests/integration_crawl.rs` covers the verification with a mock model.
