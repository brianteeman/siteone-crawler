// SiteOne Crawler - AI search readiness exporter (report + kit)
// (c) Jan Reges <jan.reges@siteone.cz>
//
// Consumes the `GeoDoc` stored in Status by the `--ai-geo` pipeline and publishes the report
// (`ai-geo.<host>.<run-id>.md|json|html`) and the deployable kit directory
// (`ai-geo-kit.<host>.<run-id>/`) together, without overwriting anything.

use std::path::{Component, Path, PathBuf};

use crate::ai::geo::doc::GeoDoc;
use crate::ai::geo::kit::KitFile;
use crate::error::{CrawlerError, CrawlerResult};
use crate::export::exporter::Exporter;
use crate::export::utils::atomic_files::write_files_no_clobber;
use crate::output::output::Output;
use crate::result::status::Status;

/// The artifact kind and label of each report file, in the order of `report`.
const REPORT_ARTIFACTS: [(&str, &str); 3] = [
    ("ai-geo-md", "AI search readiness (Markdown)"),
    ("ai-geo-json", "AI search readiness (JSON)"),
    ("ai-geo-html", "AI search readiness (HTML)"),
];
/// The artifact kind and label of the kit directory.
const KIT_ARTIFACT: (&str, &str) = ("ai-geo-kit", "AI search readiness kit");

/// Publishes the AI search readiness report and its kit.
pub struct AiGeoExporter {
    /// The Markdown, JSON and HTML report.
    report: [PathBuf; 3],
    kit_dir: PathBuf,
}

impl AiGeoExporter {
    pub fn new(report: [PathBuf; 3], kit_dir: PathBuf) -> Self {
        AiGeoExporter { report, kit_dir }
    }

    /// The report paths (`ai-geo.<host>.<run-id>.md|json|html`) and the kit directory
    /// (`ai-geo-kit.<host>.<run-id>`) in `output_dir` (filesystem-safe; without a host component
    /// for an empty host).
    pub fn paths(output_dir: &str, host: &str, run_id: &str) -> ([PathBuf; 3], PathBuf) {
        let mut components = Vec::new();
        if !host.is_empty() {
            components.push(sanitize_component(host));
        }
        components.push(sanitize_component(run_id));
        let suffix = components.join(".");
        let directory = Path::new(output_dir);
        (
            ["md", "json", "html"].map(|extension| directory.join(format!("ai-geo.{suffix}.{extension}"))),
            directory.join(format!("ai-geo-kit.{suffix}")),
        )
    }

    /// Write the kit of `doc` (see `write_kit`), then its report with `write_files_no_clobber`,
    /// the report linking into the kit directory. Nothing that existed is overwritten; when a step
    /// fails, whatever this call created is removed again, the kit included.
    pub fn publish(&self, doc: &GeoDoc, today: &str) -> std::io::Result<()> {
        let kit_name = self
            .kit_dir
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        // Render everything before writing anything.
        let json = serde_json::to_string_pretty(&doc.to_json()).map_err(std::io::Error::other)?;
        let contents = [doc.to_markdown(&kit_name), json, doc.to_html(&kit_name)];
        let created = write_kit_tracked(&self.kit_dir, &doc.build_kit(today))?;
        let files: Vec<(&Path, &[u8])> = self
            .report
            .iter()
            .zip(&contents)
            .map(|(path, content)| (path.as_path(), content.as_bytes()))
            .collect();
        if let Err(error) = write_files_no_clobber(&files) {
            rollback(&self.kit_dir, &created);
            return Err(error);
        }
        Ok(())
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

/// Write the kit `files` into `dir`, which must not exist yet: the directory is created with
/// `create_dir` (an existing one is an `AlreadyExists` error and stays untouched), its
/// subdirectories as needed, and every file with `create_new`. A relative path that could leave
/// the directory is an `InvalidInput` error. On any failure, the files and directories this call
/// created are removed and the error names the failing path.
pub fn write_kit(dir: &Path, files: &[KitFile]) -> std::io::Result<()> {
    write_kit_tracked(dir, files).map(|_| ())
}

/// `write_kit`, returning the files and directories it created inside `dir`, in creation order.
fn write_kit_tracked(dir: &Path, files: &[KitFile]) -> std::io::Result<Vec<PathBuf>> {
    for file in files {
        let path = Path::new(&file.relative_path);
        if file.relative_path.is_empty() || !path.components().all(|part| matches!(part, Component::Normal(_))) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{}: not a path inside the kit", file.relative_path),
            ));
        }
    }
    std::fs::create_dir(dir)
        .map_err(|error| std::io::Error::new(error.kind(), format!("{}: {error}", dir.display())))?;
    let mut created: Vec<PathBuf> = Vec::new();
    for file in files {
        let path = dir.join(&file.relative_path);
        if let Err(error) = write_new(&path, &file.bytes, dir, &mut created) {
            rollback(dir, &created);
            return Err(std::io::Error::new(
                error.kind(),
                format!("{}: {error}", path.display()),
            ));
        }
    }
    Ok(created)
}

/// Create the missing parent directories of `path` below `dir` and the file itself, recording
/// every directory and the file in `created` (in creation order).
fn write_new(path: &Path, bytes: &[u8], dir: &Path, created: &mut Vec<PathBuf>) -> std::io::Result<()> {
    use std::io::Write;
    let mut missing: Vec<&Path> = path
        .ancestors()
        .skip(1)
        .take_while(|ancestor| *ancestor != dir)
        .filter(|ancestor| !ancestor.is_dir())
        .collect();
    missing.reverse();
    for ancestor in missing {
        std::fs::create_dir(ancestor)?;
        created.push(ancestor.to_path_buf());
    }
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    created.push(path.to_path_buf());
    file.write_all(bytes)?;
    file.sync_all()
}

/// Remove what `write_kit` created (`created`, in creation order), newest first, then `dir`.
/// Only empty directories are removed, so nothing this call did not create is ever deleted.
fn rollback(dir: &Path, created: &[PathBuf]) {
    for path in created.iter().rev() {
        if path.is_dir() {
            let _ = std::fs::remove_dir(path);
        } else {
            let _ = std::fs::remove_file(path);
        }
    }
    let _ = std::fs::remove_dir(dir);
}

impl Exporter for AiGeoExporter {
    fn get_name(&self) -> &str {
        "AiGeoExporter"
    }

    fn should_be_activated(&self) -> bool {
        // The manager constructs the exporter only when the pipeline stored a document.
        true
    }

    fn export(&mut self, status: &Status, _output: &dyn Output) -> CrawlerResult<()> {
        let Some(doc) = status.get_ai_geo_doc() else {
            return Ok(());
        };
        if let Some(parent) = self.kit_dir.parent()
            && !parent.as_os_str().is_empty()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| CrawlerError::Export(format!("Cannot create AI search readiness dir: {e}")))?;
        }
        let today = chrono::Local::now().format("%Y-%m-%d").to_string();
        self.publish(&doc, &today)
            .map_err(|e| CrawlerError::Export(format!("Cannot write the AI search readiness report and kit: {e}")))?;
        // Announced once everything exists: a failed publication is rolled back.
        for ((kind, label), path) in REPORT_ARTIFACTS.iter().zip(&self.report) {
            crate::events::emit_ai_artifact(kind, label, path);
        }
        crate::events::emit_ai_artifact(KIT_ARTIFACT.0, KIT_ARTIFACT.1, &self.kit_dir);

        let [md, json, html] = self.report.each_ref().map(|path| path.display().to_string());
        let kit = self.kit_dir.display();
        eprintln!(
            "{}",
            crate::utils::get_color_text(
                &format!("AI search readiness report saved to: {md}, {json} and {html}; kit: {kit}"),
                "green",
                false
            )
        );
        status.add_info_to_summary(
            "ai-geo-files",
            &format!("AI search readiness report saved to {md}, {json} and {html}; deployable kit in {kit}."),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TODAY: &str = "2026-09-27";

    fn file(path: &str, text: &str) -> KitFile {
        KitFile {
            relative_path: path.to_string(),
            bytes: text.as_bytes().to_vec(),
        }
    }

    fn exporter(dir: &Path) -> AiGeoExporter {
        let (report, kit_dir) = AiGeoExporter::paths(&dir.to_string_lossy(), "www.acme.cz", "run-1");
        AiGeoExporter::new(report, kit_dir)
    }

    /// Every file and directory under `dir`, relative to it, sorted.
    fn tree(dir: &Path) -> Vec<String> {
        let mut out = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(current) = stack.pop() {
            for entry in std::fs::read_dir(&current).unwrap() {
                let path = entry.unwrap().path();
                out.push(path.strip_prefix(dir).unwrap().to_string_lossy().replace('\\', "/"));
                if path.is_dir() {
                    stack.push(path);
                }
            }
        }
        out.sort();
        out
    }

    #[test]
    fn the_report_files_and_the_kit_share_the_stem() {
        let (report, kit_dir) = AiGeoExporter::paths("tmp/reports", "www.acme.cz", "2026-09-26.10-11-12.123-99");
        assert_eq!(
            report[0],
            Path::new("tmp/reports").join("ai-geo.www-acme-cz.2026-09-26-10-11-12-123-99.md")
        );
        let extensions: Vec<&str> = report
            .iter()
            .map(|p| p.extension().and_then(|e| e.to_str()).unwrap_or_default())
            .collect();
        assert_eq!(extensions, vec!["md", "json", "html"]);
        assert_eq!(
            kit_dir,
            Path::new("tmp/reports").join("ai-geo-kit.www-acme-cz.2026-09-26-10-11-12-123-99")
        );
        // Without a host the names have no host component.
        let (report, kit_dir) = AiGeoExporter::paths("out", "", "run/1");
        assert_eq!(report[2], Path::new("out").join("ai-geo.run-1.html"));
        assert_eq!(kit_dir, Path::new("out").join("ai-geo-kit.run-1"));
    }

    #[test]
    fn the_report_and_the_kit_are_published_together() {
        let dir = tempfile::tempdir().unwrap();
        let exporter = exporter(dir.path());
        let doc = GeoDoc::default();
        exporter.publish(&doc, TODAY).expect("published");
        let kit = doc.build_kit(TODAY);
        assert!(!kit.is_empty(), "the kit has at least its README");
        for kit_file in &kit {
            let path = exporter.kit_dir.join(&kit_file.relative_path);
            assert_eq!(std::fs::read(&path).unwrap(), kit_file.bytes, "{}", path.display());
        }
        let md = std::fs::read_to_string(&exporter.report[0]).unwrap();
        assert_eq!(md, doc.to_markdown("ai-geo-kit.www-acme-cz.run-1"));
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&exporter.report[1]).unwrap()).unwrap();
        assert_eq!(json["schema"], "siteone-crawler/ai-geo/2");
        let html = std::fs::read_to_string(&exporter.report[2]).unwrap();
        assert_eq!(html, doc.to_html("ai-geo-kit.www-acme-cz.run-1"));
    }

    #[test]
    fn a_pre_existing_kit_dir_fails_and_stays_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let exporter = exporter(dir.path());
        std::fs::create_dir(&exporter.kit_dir).unwrap();
        std::fs::write(exporter.kit_dir.join("README.md"), "mine").unwrap();
        let error = exporter
            .publish(&GeoDoc::default(), TODAY)
            .expect_err("the kit dir exists");
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists, "{error}");
        assert_eq!(
            std::fs::read_to_string(exporter.kit_dir.join("README.md")).unwrap(),
            "mine"
        );
        assert_eq!(
            tree(dir.path()),
            vec!["ai-geo-kit.www-acme-cz.run-1", "ai-geo-kit.www-acme-cz.run-1/README.md"],
            "no report file and nothing else in the kit dir"
        );
    }

    #[test]
    fn a_pre_existing_report_file_fails_and_removes_only_the_new_kit() {
        let dir = tempfile::tempdir().unwrap();
        let exporter = exporter(dir.path());
        std::fs::write(&exporter.report[1], "old report").unwrap();
        let error = exporter
            .publish(&GeoDoc::default(), TODAY)
            .expect_err("the JSON exists");
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists, "{error}");
        assert_eq!(std::fs::read_to_string(&exporter.report[1]).unwrap(), "old report");
        assert_eq!(
            tree(dir.path()),
            vec!["ai-geo.www-acme-cz.run-1.json"],
            "the kit and the other report files of this call are removed"
        );
    }

    #[test]
    fn a_failure_mid_way_removes_only_the_files_this_call_created() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("other.txt"), "keep").unwrap();
        let kit_dir = dir.path().join("kit");
        let files = [
            file("README.md", "readme"),
            file("jsonld/site-website.json", "{}"),
            file("jsonld/site-website.json", "{}"),
        ];
        let error = write_kit(&kit_dir, &files).expect_err("the second copy of a file fails");
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists, "{error}");
        assert!(error.to_string().contains("site-website.json"), "{error}");
        assert_eq!(tree(dir.path()), vec!["other.txt"]);
        assert_eq!(std::fs::read_to_string(dir.path().join("other.txt")).unwrap(), "keep");
    }

    #[test]
    fn a_kit_path_outside_the_kit_dir_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let kit_dir = dir.path().join("kit");
        for path in ["../escape.txt", "/tmp/escape.txt", "jsonld/../../escape.txt", ""] {
            let error = write_kit(&kit_dir, &[file("README.md", "readme"), file(path, "x")]).expect_err("refused");
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput, "{path}: {error}");
            assert!(tree(dir.path()).is_empty(), "{path}: {:?}", tree(dir.path()));
        }
    }
}
