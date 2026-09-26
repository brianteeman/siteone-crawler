// SiteOne Crawler - No-clobber multi-file writer
// (c) Jan Reges <jan.reges@siteone.cz>
//
// Writes a set of report files that belong together (e.g. the Markdown, JSON, HTML and CSV of one
// run) so that a run never overwrites an earlier report and never leaves a partial set behind.

use std::io::Write;
use std::path::Path;

/// Write every `(path, content)` pair, in order, without ever overwriting a file:
/// - each final path is created with `OpenOptions::create_new` (an existing file is an
///   `AlreadyExists` error), then written with `write_all` and flushed with `sync_all`;
/// - on any failure, the files this call created are removed and the error is returned, with the
///   failing path in its message (its kind is kept);
/// - files that existed before the call are never modified or removed.
///
/// The same path listed twice fails on its second entry, so nothing of the set remains.
pub fn write_files_no_clobber(files: &[(&Path, &[u8])]) -> std::io::Result<()> {
    let mut created: Vec<&Path> = Vec::with_capacity(files.len());
    for (path, content) in files {
        let written = match std::fs::OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(mut file) => {
                created.push(path);
                file.write_all(content).and_then(|_| file.sync_all())
            }
            Err(error) => Err(error),
        };
        if let Err(error) = written {
            for path in &created {
                let _ = std::fs::remove_file(path);
            }
            return Err(std::io::Error::new(
                error.kind(),
                format!("{}: {error}", path.display()),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const NAMES: [&str; 4] = ["r.md", "r.json", "r.html", "r.csv"];

    fn paths(dir: &Path) -> Vec<PathBuf> {
        NAMES.iter().map(|name| dir.join(name)).collect()
    }

    fn contents() -> Vec<Vec<u8>> {
        NAMES
            .iter()
            .map(|name| format!("content of {name}").into_bytes())
            .collect()
    }

    fn write(paths: &[PathBuf], contents: &[Vec<u8>]) -> std::io::Result<()> {
        let files: Vec<(&Path, &[u8])> = paths
            .iter()
            .zip(contents)
            .map(|(path, content)| (path.as_path(), content.as_slice()))
            .collect();
        write_files_no_clobber(&files)
    }

    #[test]
    fn writes_every_file_with_its_content() {
        let dir = tempfile::tempdir().unwrap();
        let (paths, contents) = (paths(dir.path()), contents());
        write(&paths, &contents).expect("all four are written");
        for (path, content) in paths.iter().zip(&contents) {
            assert_eq!(&std::fs::read(path).unwrap(), content, "{}", path.display());
        }
        assert!(write_files_no_clobber(&[]).is_ok(), "nothing to write is fine");
    }

    #[test]
    fn a_pre_existing_file_at_any_position_fails_the_whole_set_and_stays_untouched() {
        for position in 0..NAMES.len() {
            let dir = tempfile::tempdir().unwrap();
            let (paths, contents) = (paths(dir.path()), contents());
            std::fs::write(&paths[position], b"pre-existing").unwrap();

            let error = write(&paths, &contents).expect_err("an existing destination is an error");
            assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists, "position {position}");
            assert!(
                error.to_string().contains(NAMES[position]),
                "the error names the file: {error}"
            );
            assert_eq!(
                std::fs::read(&paths[position]).unwrap(),
                b"pre-existing",
                "position {position}: the existing file is never touched"
            );
            for (other, path) in paths.iter().enumerate().filter(|(other, _)| *other != position) {
                assert!(
                    !path.exists(),
                    "position {position}: {} was created by this call and must be removed",
                    NAMES[other]
                );
            }
        }
    }

    #[test]
    fn the_same_path_twice_keeps_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("twice.json");
        let error = write_files_no_clobber(&[
            (path.as_path(), b"first".as_slice()),
            (path.as_path(), b"second".as_slice()),
        ])
        .expect_err("the second create finds the first");
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert!(!path.exists(), "the file this call created is removed");
    }

    #[test]
    fn a_missing_directory_rolls_back_the_earlier_files() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first.md");
        let missing = dir.path().join("no-such-dir").join("second.json");
        let error = write_files_no_clobber(&[
            (first.as_path(), b"first".as_slice()),
            (missing.as_path(), b"second".as_slice()),
        ])
        .expect_err("the directory does not exist");
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
        assert!(!first.exists());
        assert!(!missing.exists());
    }
}
