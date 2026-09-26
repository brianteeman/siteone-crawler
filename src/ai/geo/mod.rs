// SiteOne Crawler - AI search readiness (--ai-geo)
// (c) Jan Reges <jan.reges@siteone.cz>
//
// How well AI search and answer engines can reach, read, understand and quote a site.
// See docs/superpowers/plans/2026-09-26-ai-search-readiness-geo.md.

pub mod access;
pub mod agents;
pub mod controls;
pub mod keys;
pub mod robots_ai;
#[cfg(test)]
pub(crate) mod test_support;
