// SiteOne Crawler - AI fact consistency across pages (--ai-consistency)
// (c) Jan Reges <jan.reges@siteone.cz>
//
// Finds possible contradictions in hard facts (contacts, prices, rates, fees, conditions, dates,
// figures, identifiers) across the crawled pages and the header/footer lines they share:
//   sources  — numbered evidence blocks per page, plus the unique fact-bearing header/footer
//              lines with their exact page sets;
//   extract  — a few block-grounded facts per source, verified against the crawler's own text.

pub mod extract;
pub mod model;
pub mod prompts;
pub mod sources;
