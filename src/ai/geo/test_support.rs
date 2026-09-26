// SiteOne Crawler - test helpers of the AI search readiness pipeline
// (c) Jan Reges <jan.reges@siteone.cz>

use crate::info::Info;
use crate::result::status::Status;
use crate::result::storage::memory_storage::MemoryStorage;
use crate::result::visited_url::VisitedUrl;
use crate::types::ContentTypeId;

/// An empty crawl result that keeps bodies.
pub(crate) fn new_status() -> Status {
    let info = Info::new(
        "SiteOne Crawler".to_string(),
        "test".to_string(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        "https://example.com/".to_string(),
    );
    Status::new(
        Box::new(MemoryStorage::new(false)),
        true,
        info,
        std::time::Instant::now(),
    )
}

/// An internal HTML response fetched in 0.1 s; `location` becomes the stored `Location` of a
/// redirect, as the crawler records it.
pub(crate) fn page(
    uq_id: &str,
    source_uq_id: &str,
    source_attr: i32,
    url: &str,
    status_code: i32,
    location: Option<&str>,
) -> VisitedUrl {
    let extras =
        location.map(|location| std::collections::HashMap::from([("Location".to_string(), location.to_string())]));
    let content_type = if location.is_some() {
        ContentTypeId::Redirect
    } else {
        ContentTypeId::Html
    };
    VisitedUrl::new(
        uq_id.to_string(),
        source_uq_id.to_string(),
        source_attr,
        url.to_string(),
        status_code,
        0.1,
        Some(100),
        content_type,
        Some("text/html".to_string()),
        None,
        extras,
        false,
        true,
        0,
        None,
    )
}

pub(crate) fn add(status: &mut Status, visited: VisitedUrl, body: Option<&str>) {
    status.add_visited_url(visited, body.map(str::as_bytes), None);
}

/// Like `add`, with response headers (lowercase names, as the HTTP client stores them).
pub(crate) fn add_with_headers(status: &mut Status, visited: VisitedUrl, body: Option<&str>, headers: &[(&str, &str)]) {
    let headers: std::collections::HashMap<String, String> = headers
        .iter()
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect();
    status.add_visited_url(visited, body.map(str::as_bytes), Some(&headers));
}
