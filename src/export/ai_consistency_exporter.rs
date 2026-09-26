// SiteOne Crawler - AI fact-consistency exporter (Markdown + JSON + HTML + CSV)
// (c) Jan Reges <jan.reges@siteone.cz>
//
// Consumes the `ConsistencyDoc` stored in Status by the `--ai-consistency` pipeline and writes the
// four files of one report (`ai-consistency.<host>.<run-id>.md|json|html|csv`) together: created
// without overwriting anything, and rolled back as a set when one of them fails.

use std::path::{Path, PathBuf};

use crate::error::{CrawlerError, CrawlerResult};
use crate::export::exporter::Exporter;
use crate::export::utils::atomic_files::write_files_no_clobber;
use crate::output::output::Output;
use crate::result::status::Status;

/// The artifact kind and label of each file, in the order of `paths`.
const ARTIFACTS: [(&str, &str); 4] = [
    ("ai-consistency-md", "AI consistency (Markdown)"),
    ("ai-consistency-json", "AI consistency (JSON)"),
    ("ai-consistency-html", "AI consistency (HTML)"),
    ("ai-consistency-csv", "AI consistency (CSV)"),
];

/// Writes the four files of the fact-consistency report.
pub struct AiConsistencyExporter {
    paths: [PathBuf; 4],
}

impl AiConsistencyExporter {
    pub fn new(paths: [PathBuf; 4]) -> Self {
        AiConsistencyExporter { paths }
    }

    /// The Markdown, JSON, HTML and CSV paths in `output_dir`, sharing the stem
    /// `ai-consistency.<host>.<run-id>` (filesystem-safe; without a host component for an empty
    /// host).
    pub fn paths(output_dir: &str, host: &str, run_id: &str) -> [PathBuf; 4] {
        let mut components = vec!["ai-consistency".to_string()];
        if !host.is_empty() {
            components.push(sanitize_component(host));
        }
        components.push(sanitize_component(run_id));
        let stem = components.join(".");
        let directory = Path::new(output_dir);
        ["md", "json", "html", "csv"].map(|extension| directory.join(format!("{stem}.{extension}")))
    }
}

/// Keep a filename component filesystem-safe.
fn sanitize_component(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

impl Exporter for AiConsistencyExporter {
    fn get_name(&self) -> &str {
        "AiConsistencyExporter"
    }

    fn should_be_activated(&self) -> bool {
        // The manager constructs the exporter only when the pipeline stored a document.
        true
    }

    fn export(&mut self, status: &Status, _output: &dyn Output) -> CrawlerResult<()> {
        let Some(doc) = status.get_ai_consistency_doc() else {
            return Ok(());
        };
        for path in &self.paths {
            if let Some(parent) = path.parent()
                && !parent.as_os_str().is_empty()
                && !parent.exists()
            {
                std::fs::create_dir_all(parent)
                    .map_err(|e| CrawlerError::Export(format!("Cannot create AI consistency dir: {e}")))?;
            }
        }

        // Render all four before writing any.
        let json = serde_json::to_string_pretty(&doc.to_json())
            .map_err(|e| CrawlerError::Export(format!("AI consistency JSON serialization: {e}")))?;
        let contents = [doc.to_markdown(), json, doc.to_html(), doc.to_csv()];
        let files: Vec<(&Path, &[u8])> = self
            .paths
            .iter()
            .zip(&contents)
            .map(|(path, content)| (path.as_path(), content.as_bytes()))
            .collect();
        write_files_no_clobber(&files)
            .map_err(|e| CrawlerError::Export(format!("Cannot write the AI consistency report: {e}")))?;
        // Announced once all four exist: a failed set is rolled back, so none of them remains.
        for ((kind, label), path) in ARTIFACTS.iter().zip(&self.paths) {
            crate::events::emit_ai_artifact(kind, label, path);
        }

        let [md, json, html, csv] = self.paths.each_ref().map(|path| path.display().to_string());
        eprintln!(
            "{}",
            crate::utils::get_color_text(
                &format!("AI consistency report saved to: {md}, {json}, {html} and {csv}"),
                "green",
                false
            )
        );
        status.add_info_to_summary(
            "ai-consistency-files",
            &format!("AI consistency report saved to {md}, {json}, {html} and {csv}."),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_four_paths_share_the_stem_and_differ_in_the_extension() {
        let paths = AiConsistencyExporter::paths("tmp/reports", "www.acme.cz", "2026-09-26.10-11-12.123-99");
        let stems: Vec<_> = paths.iter().map(|p| p.file_stem().map(|s| s.to_os_string())).collect();
        assert!(stems.windows(2).all(|pair| pair[0] == pair[1]), "{paths:?}");
        assert_eq!(
            paths[0],
            std::path::Path::new("tmp/reports").join("ai-consistency.www-acme-cz.2026-09-26-10-11-12-123-99.md")
        );
        let extensions: Vec<&str> = paths
            .iter()
            .map(|p| p.extension().and_then(|e| e.to_str()).unwrap_or_default())
            .collect();
        assert_eq!(extensions, vec!["md", "json", "html", "csv"]);
        // Without a host the stem has no host component.
        let paths = AiConsistencyExporter::paths("out", "", "run/1");
        assert_eq!(paths[3], std::path::Path::new("out").join("ai-consistency.run-1.csv"));
    }
}
