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
//              fair review allocation, and the validated, policy-gated review.

pub mod doc;
pub mod extract;
pub mod judge;
pub mod keys;
pub mod model;
pub mod prompts;
pub mod sources;

#[cfg(test)]
mod tests {
    //! Budget stress tests: every final user message fits its byte budget, and every batch its
    //! output budget, from small to large context windows (the budgets of Design §4).

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
