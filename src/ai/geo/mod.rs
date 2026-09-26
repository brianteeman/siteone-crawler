// SiteOne Crawler - AI search readiness (--ai-geo)
// (c) Jan Reges <jan.reges@siteone.cz>
//
// How well AI search and answer engines can reach, read, understand and quote a site.
// See docs/superpowers/plans/2026-09-26-ai-search-readiness-geo.md.

pub mod access;
pub mod agents;
pub mod analyze;
pub mod controls;
pub mod discovery;
pub mod jsonld;
pub mod keys;
pub mod prompts;
pub mod render;
pub mod robots_ai;
pub mod signals;
#[cfg(test)]
pub(crate) mod test_support;
