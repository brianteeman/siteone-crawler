// SiteOne Crawler - robots.txt evaluation per AI crawler (RFC 9309)
// (c) Jan Reges <jan.reges@siteone.cz>
//
// Reads robots.txt the way RFC 9309 and Google's parser (github.com/google/robotstxt) do: a group
// is one or more consecutive user-agent lines followed by rules (only an allow or disallow line
// ends the list of user agents), the groups of one product token merge, a named group makes a
// crawler ignore `*`, the longest matching pattern wins and Allow wins a tie. Like Google, it also
// reads the misspelled keys Google accepts, a two-word line without the colon, and only the first
// 500 KiB of the file and 16,663 bytes of a line. Patterns and paths are compared after the
// percent-encoding normalization of RFC 9309 §2.2.2, with a matcher that is linear in the input
// (no regex per rule, no backtracking). Used for the crawler-policy verdicts and for the safety
// check of a proposed robots.txt.

use std::collections::BTreeMap;

use crate::ai::geo::agents::{AiAgent, find_agent};
use crate::result::status::RobotsFetchState;

/// Google ignores the content of a robots.txt after 500 KiB.
const MAX_PARSED_BYTES: usize = 500 * 1024;
/// Google ignores the bytes of a line after 16,663 (2083 × 8 − 1, see google/robotstxt).
const MAX_LINE_BYTES: usize = 2083 * 8 - 1;
/// `policy_equivalent` gives up (and fails) above this many rule matches: tokens × paths × rules.
const MAX_EQUIVALENCE_MATCHES: usize = 20_000_000;

/// What a robots.txt line is, as Google's parser reads its key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Field {
    UserAgent,
    Allow,
    Disallow,
    Sitemap,
    ContentSignal,
    Other,
}

/// One allow or disallow line.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Rule {
    allow: bool,
    /// The value as written in the file.
    value: String,
    /// The value after `normalize_path`, as matched.
    pattern: String,
}

/// The product tokens of a group's user-agent lines (`*` for the global group) and its rules.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Group {
    agents: Vec<String>,
    rules: Vec<Rule>,
}

/// A parsed robots.txt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiRobots {
    groups: Vec<Group>,
    sitemaps: Vec<String>,
    content_signals: Vec<String>,
    raw: String,
    ends_with_ruleless_group: bool,
}

/// Whether an agent may fetch a set of paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentAccess {
    Allowed,
    Blocked,
    /// Some paths are blocked: these.
    Partly(Vec<String>),
}

impl AiRobots {
    pub fn parse(raw: &str) -> Self {
        let mut groups: Vec<Group> = Vec::new();
        let mut sitemaps = Vec::new();
        let mut content_signals = Vec::new();
        // The last group still takes user-agent lines until its first rule.
        let mut taking_agents = false;
        // For `ends_with_ruleless_group`, over the whole file: the last user-agent line in any
        // spelling Google accepts, and the last allow or disallow line that a strict parser reads.
        let (mut last_agent_line, mut last_strict_rule_line) = (None, None);
        let text = raw.trim_start_matches('\u{feff}');
        let mut offset = 0;
        for (index, line) in text.split(['\n', '\r']).enumerate() {
            let start = offset;
            offset += line.len() + 1;
            let Some((field, key, _, has_colon)) = record(prefix(line, MAX_LINE_BYTES)) else {
                continue;
            };
            if field == Field::UserAgent {
                last_agent_line = Some(index);
            }
            if has_colon && (key.eq_ignore_ascii_case("allow") || key.eq_ignore_ascii_case("disallow")) {
                last_strict_rule_line = Some(index);
            }
            if start >= MAX_PARSED_BYTES {
                continue;
            }
            // The line that crosses the 500 KiB limit counts up to the limit only.
            let Some((field, key, value, _)) = record(prefix(line, MAX_LINE_BYTES.min(MAX_PARSED_BYTES - start)))
            else {
                continue;
            };
            match field {
                Field::UserAgent => {
                    if !taking_agents {
                        groups.push(Group::default());
                        taking_agents = true;
                    }
                    if let Some(group) = groups.last_mut() {
                        group.agents.push(product_token(value));
                    }
                }
                Field::Allow | Field::Disallow => {
                    // Rules before the first user-agent line belong to no group.
                    let Some(group) = groups.last_mut() else {
                        continue;
                    };
                    taking_agents = false;
                    let allow = field == Field::Allow;
                    group.rules.push(Rule {
                        allow,
                        value: value.to_string(),
                        pattern: normalize_path(value),
                    });
                    // Google: allowing `/dir/index.htm(l)` also allows `/dir/` itself.
                    if allow
                        && let Some(slash) = value.rfind('/')
                        && value[slash..].starts_with("/index.htm")
                    {
                        group.rules.push(Rule {
                            allow,
                            value: value.to_string(),
                            pattern: format!("{}$", normalize_path(&value[..=slash])),
                        });
                    }
                }
                Field::Sitemap => sitemaps.push(value.to_string()),
                Field::ContentSignal => content_signals.push(format!("{}: {}", key, value)),
                Field::Other => {}
            }
        }
        let ends_with_ruleless_group = match (last_agent_line, last_strict_rule_line) {
            (Some(agent), Some(rule)) => agent > rule,
            (agent, _) => agent.is_some(),
        };
        Self {
            groups,
            sitemaps,
            content_signals,
            raw: raw.to_string(),
            ends_with_ruleless_group,
        }
    }

    /// Whether the crawler with product token `token` may fetch `path_and_query`.
    pub fn is_allowed(&self, token: &str, path_and_query: &str) -> bool {
        self.deciding_rule(token, path_and_query)
            .is_none_or(|(_, rule)| rule.allow)
    }

    /// The group and rule that decide `is_allowed`, e.g. "User-agent: OAI-SearchBot → Disallow: /";
    /// `None` when no rule matches (the path is allowed).
    pub fn explain(&self, token: &str, path_and_query: &str) -> Option<String> {
        let (group, rule) = self.deciding_rule(token, path_and_query)?;
        let agent = group
            .agents
            .iter()
            .find(|agent| agent.as_str() != "*" && agent.eq_ignore_ascii_case(token))
            .map_or("*", String::as_str);
        let field = if rule.allow { "Allow" } else { "Disallow" };
        Some(format!("User-agent: {} → {}: {}", agent, field, rule.value))
    }

    pub fn has_named_group(&self, token: &str) -> bool {
        self.groups
            .iter()
            .any(|group| group.agents.iter().any(|agent| is_named(agent, token)))
    }

    /// Every product token that has a group of its own, once, as first written.
    pub fn named_tokens(&self) -> Vec<String> {
        let mut tokens: Vec<String> = Vec::new();
        for agent in self.groups.iter().flat_map(|group| &group.agents) {
            if agent != "*" && !agent.is_empty() && !tokens.iter().any(|token| token.eq_ignore_ascii_case(agent)) {
                tokens.push(agent.clone());
            }
        }
        tokens
    }

    /// Whether a group appended to the file could merge into the last one: a user-agent line (in
    /// any spelling Google accepts) comes after the last `Allow:` / `Disallow:` line that a strict
    /// RFC 9309 parser reads. Checked over the whole file, also past the 500 KiB that are parsed.
    pub fn ends_with_ruleless_group(&self) -> bool {
        self.ends_with_ruleless_group
    }

    /// The token whose groups apply to `agent`: its own, or its fallback token when the file
    /// has no group for the agent but has one for the fallback (Applebot → Googlebot).
    pub fn token_for(&self, agent: &AiAgent) -> &'static str {
        match agent.fallback_token {
            Some(fallback) if !self.has_named_group(agent.token) && self.has_named_group(fallback) => fallback,
            _ => agent.token,
        }
    }

    pub fn sitemaps(&self) -> &[String] {
        &self.sitemaps
    }

    /// The `Content-Signal` and `Content-Usage` lines, as written.
    pub fn content_signals(&self) -> &[String] {
        &self.content_signals
    }

    pub fn raw(&self) -> &str {
        &self.raw
    }

    /// The groups of `token`: its named groups merged, otherwise the `*` groups merged.
    fn groups_for(&self, token: &str) -> Vec<&Group> {
        let named: Vec<&Group> = self
            .groups
            .iter()
            .filter(|group| group.agents.iter().any(|agent| is_named(agent, token)))
            .collect();
        if !named.is_empty() {
            return named;
        }
        self.groups
            .iter()
            .filter(|group| group.agents.iter().any(|agent| agent == "*"))
            .collect()
    }

    /// The longest matching rule of the groups of `token` (Allow on a tie); `/robots.txt` and an
    /// empty pattern are never decided by a rule.
    fn deciding_rule(&self, token: &str, path_and_query: &str) -> Option<(&Group, &Rule)> {
        let path = normalize_path(if path_and_query.is_empty() { "/" } else { path_and_query });
        if path == "/robots.txt" {
            return None;
        }
        let mut best: Option<(&Group, &Rule)> = None;
        for group in self.groups_for(token) {
            for rule in &group.rules {
                if rule.pattern.is_empty() || !pattern_matches(&rule.pattern, &path) {
                    continue;
                }
                let wins = best.is_none_or(|(_, current)| {
                    rule.pattern.len() > current.pattern.len()
                        || (rule.pattern.len() == current.pattern.len() && rule.allow && !current.allow)
                });
                if wins {
                    best = Some((group, rule));
                }
            }
        }
        best
    }
}

/// A robots.txt line as Google's parser reads it: the comment is dropped, the key ends at the first
/// colon — or, without a colon, the line must be exactly two words — and the key is recognized by
/// its start, including the misspellings Google accepts. Returns the field, the key, the value
/// and whether a colon separated them.
fn record(line: &str) -> Option<(Field, &str, &str, bool)> {
    let line = line.split('#').next().unwrap_or_default().trim();
    let (key, value, has_colon) = match line.split_once(':') {
        Some((key, value)) => (key.trim(), value.trim(), true),
        None => {
            let mut words = line.split_ascii_whitespace();
            match (words.next(), words.next(), words.next()) {
                (Some(key), Some(value), None) => (key, value, false),
                _ => return None,
            }
        }
    };
    if key.is_empty() {
        return None;
    }
    let lower = key.to_ascii_lowercase();
    let starts = |prefixes: &[&str]| prefixes.iter().any(|prefix| lower.starts_with(prefix));
    let field = if starts(&["user-agent", "useragent", "user agent"]) {
        Field::UserAgent
    } else if starts(&["allow"]) {
        Field::Allow
    } else if starts(&["disallow", "dissallow", "dissalow", "disalow", "diasllow", "disallaw"]) {
        Field::Disallow
    } else if starts(&["sitemap", "site-map"]) {
        Field::Sitemap
    } else if lower == "content-signal" || lower == "content-usage" {
        Field::ContentSignal
    } else {
        Field::Other
    };
    Some((field, key, value, has_colon))
}

/// The longest prefix of `text` of at most `max_bytes` bytes that ends on a character boundary.
fn prefix(text: &str, max_bytes: usize) -> &str {
    let mut end = max_bytes.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// The product token of a user-agent value: `*` for the global group, otherwise its leading
/// letters, underscores and hyphens (`googlebot/1.2` and `googlebot*` both name `googlebot`).
fn product_token(value: &str) -> String {
    let mut chars = value.chars();
    if chars.next() == Some('*') && chars.next().is_none_or(char::is_whitespace) {
        return "*".to_string();
    }
    value
        .chars()
        .take_while(|c| c.is_ascii_alphabetic() || *c == '_' || *c == '-')
        .collect()
}

fn is_named(agent: &str, token: &str) -> bool {
    agent != "*" && !agent.is_empty() && agent.eq_ignore_ascii_case(token)
}

/// Percent-encoding normalization of RFC 9309 §2.2.2, applied to patterns and paths alike:
/// `%XX` of an unreserved character is decoded, any other `%XX` stays encoded with uppercase hex,
/// non-ASCII bytes and characters a URI cannot hold are encoded, a `%` without two hex digits
/// becomes `%25`. Reserved characters (`/`, `?`, `=`, …) and the wildcards `*` and `$` stay as
/// they are, so `/a%2Fb` never equals `/a/b` while `/%7Ejoe` equals `/~joe`.
pub fn normalize_path(p: &str) -> String {
    let bytes = p.as_bytes();
    let mut out = String::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let byte = bytes[i];
        if byte == b'%' {
            let decoded = match (
                bytes.get(i + 1).and_then(hex_value),
                bytes.get(i + 2).and_then(hex_value),
            ) {
                (Some(high), Some(low)) => Some(high * 16 + low),
                _ => None,
            };
            match decoded {
                Some(value) if value.is_ascii_alphanumeric() || matches!(value, b'-' | b'.' | b'_' | b'~') => {
                    out.push(value as char);
                    i += 3;
                }
                Some(value) => {
                    push_encoded(&mut out, value);
                    i += 3;
                }
                None => {
                    out.push_str("%25");
                    i += 1;
                }
            }
            continue;
        }
        if byte >= 0x80
            || byte <= b' '
            || byte == 0x7f
            || matches!(byte, b'"' | b'<' | b'>' | b'\\' | b'^' | b'`' | b'{' | b'|' | b'}')
        {
            push_encoded(&mut out, byte);
        } else {
            out.push(byte as char);
        }
        i += 1;
    }
    out
}

fn hex_value(byte: &u8) -> Option<u8> {
    (*byte as char).to_digit(16).map(|digit| digit as u8)
}

fn push_encoded(out: &mut String, byte: u8) {
    out.push_str(&format!("%{:02X}", byte));
}

/// Whether a normalized pattern matches the start of a normalized path: `*` matches any run of
/// characters, a trailing `$` anchors the end. The literal pieces between the `*`s are matched at
/// their first occurrence, which is exact for patterns whose only wildcard is `*` and keeps the
/// work linear in the input (`str::find`), however hostile the pattern.
fn pattern_matches(pattern: &str, path: &str) -> bool {
    let (pattern, anchored) = match pattern.strip_suffix('$') {
        Some(rest) => (rest, true),
        None => (pattern, false),
    };
    let mut pieces = pattern.split('*');
    let Some(mut rest) = path.strip_prefix(pieces.next().unwrap_or_default()) else {
        return false;
    };
    let pieces: Vec<&str> = pieces.collect();
    let Some((last, middle)) = pieces.split_last() else {
        // No `*`: a prefix of the path, or the whole path when anchored.
        return !anchored || rest.is_empty();
    };
    for piece in middle {
        let Some(at) = rest.find(piece) else {
            return false;
        };
        rest = &rest[at + piece.len()..];
    }
    if anchored {
        rest.ends_with(last)
    } else {
        rest.contains(last)
    }
}

/// The access of `agent` to `paths` (path + query each), with its fallback token applied. No
/// robots.txt (`None`) blocks nothing, as does an empty `paths`.
pub fn evaluate(robots: Option<&AiRobots>, agent: &AiAgent, paths: &[String]) -> AgentAccess {
    let Some(robots) = robots else {
        return AgentAccess::Allowed;
    };
    let token = robots.token_for(agent);
    let blocked: Vec<String> = paths
        .iter()
        .filter(|path| !robots.is_allowed(token, path))
        .cloned()
        .collect();
    if blocked.is_empty() {
        AgentAccess::Allowed
    } else if blocked.len() == paths.len() {
        AgentAccess::Blocked
    } else {
        AgentAccess::Partly(blocked)
    }
}

/// Checks that `after` gives every crawler the same allow/deny answers as `before` (`None` = no
/// robots.txt). Checked are `tokens`, `*` and every agent named in `before` (with the fallback of
/// a table agent applied), on `paths`, `/` and one literal path per rule pattern of either file.
/// `Err` describes the first answer that would change, or says that the files have too many rules
/// to check them all (`MAX_EQUIVALENCE_MATCHES`): the check fails closed rather than run for
/// minutes.
pub fn policy_equivalent(
    before: Option<&AiRobots>,
    after: &AiRobots,
    tokens: &[String],
    paths: &[String],
) -> Result<(), String> {
    let mut all_tokens: Vec<String> = vec!["*".to_string()];
    all_tokens.extend(before.map(AiRobots::named_tokens).unwrap_or_default());
    all_tokens.extend(tokens.iter().cloned());
    let mut all_paths: Vec<String> = vec!["/".to_string()];
    all_paths.extend(paths.iter().cloned());
    for robots in before.into_iter().chain([after]) {
        for rule in robots.groups.iter().flat_map(|group| &group.rules) {
            if let Some(path) = literal_path(&rule.pattern) {
                all_paths.push(path);
            }
        }
    }
    all_tokens.sort();
    all_tokens.dedup();
    all_paths.sort();
    all_paths.dedup();
    let rules: usize = before
        .into_iter()
        .chain([after])
        .flat_map(|robots| &robots.groups)
        .map(|group| group.rules.len())
        .sum();
    if all_tokens.len().saturating_mul(all_paths.len()).saturating_mul(rules) > MAX_EQUIVALENCE_MATCHES {
        return Err(format!(
            "robots.txt has too many rules ({}) to check that the proposal changes nothing else",
            rules
        ));
    }
    for token in &all_tokens {
        for path in &all_paths {
            let was = allowed_with_fallback(before, token, path);
            let is = allowed_with_fallback(Some(after), token, path);
            if was != is {
                let verdict = |allowed: bool| if allowed { "allowed" } else { "blocked" };
                return Err(format!(
                    "User-agent {}: {} would change from {} to {}",
                    token,
                    path,
                    verdict(was),
                    verdict(is)
                ));
            }
        }
    }
    Ok(())
}

fn allowed_with_fallback(robots: Option<&AiRobots>, token: &str, path: &str) -> bool {
    let Some(robots) = robots else {
        return true;
    };
    let token = find_agent(token).map_or(token, |agent| robots.token_for(agent));
    robots.is_allowed(token, path)
}

/// A path that a normalized pattern matches: the pattern without its wildcards and end anchor.
fn literal_path(pattern: &str) -> Option<String> {
    let literal: String = pattern.strip_suffix('$').unwrap_or(pattern).replace('*', "");
    if literal.is_empty() {
        return None;
    }
    Some(if literal.starts_with('/') {
        literal
    } else {
        format!("/{literal}")
    })
}

/// Whether a fetch state says what the rules are: a fetched file (`Ok`) or none (`NotFound`).
pub fn state_is_known(state: &RobotsFetchState) -> bool {
    matches!(state, RobotsFetchState::Ok { .. } | RobotsFetchState::NotFound { .. })
}

/// The parsed robots.txt of an `Ok` fetch state.
pub fn robots_of(state: &RobotsFetchState) -> Option<AiRobots> {
    match state {
        RobotsFetchState::Ok { content, .. } => Some(AiRobots::parse(content)),
        _ => None,
    }
}

/// URLs grouped by origin `(scheme, host, port)`, each as path + query, in input order.
/// URLs that cannot be parsed or have no host are left out.
pub fn paths_by_origin(urls: &[String]) -> BTreeMap<(String, String, u16), Vec<String>> {
    let mut grouped: BTreeMap<(String, String, u16), Vec<String>> = BTreeMap::new();
    for url in urls {
        let Ok(parsed) = url::Url::parse(url) else {
            continue;
        };
        let (Some(host), Some(port)) = (parsed.host_str(), parsed.port_or_known_default()) else {
            continue;
        };
        let path = match parsed.query() {
            Some(query) => format!("{}?{}", parsed.path(), query),
            None => parsed.path().to_string(),
        };
        grouped
            .entry((parsed.scheme().to_string(), host.to_string(), port))
            .or_default()
            .push(path);
    }
    grouped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::geo::agents::find_agent;
    use crate::result::status::RobotsFetchState;

    fn paths(list: &[&str]) -> Vec<String> {
        list.iter().map(|path| path.to_string()).collect()
    }

    fn table_tokens_except(excluded: &[&str]) -> Vec<String> {
        crate::ai::geo::agents::AI_AGENTS
            .iter()
            .map(|agent| agent.token.to_string())
            .filter(|token| !excluded.contains(&token.as_str()))
            .collect()
    }

    /// RFC 9309 §5.1.
    const RFC_SIMPLE_EXAMPLE: &str = "User-Agent: *\n\
Disallow: *.gif$\n\
Disallow: /example/\n\
Allow: /publications/\n\
\n\
User-Agent: foobot\n\
Disallow:/\n\
Allow:/example/page.html\n\
Allow:/example/allowed.gif\n\
\n\
User-Agent: barbot\n\
User-Agent: bazbot\n\
Disallow: /example/page.html\n\
\n\
User-Agent: quxbot\n";

    #[test]
    fn rfc9309_simple_example() {
        let robots = AiRobots::parse(RFC_SIMPLE_EXAMPLE);

        // `*`: every agent without a group of its own.
        assert!(!robots.is_allowed("otherbot", "/example/index.html"));
        assert!(!robots.is_allowed("otherbot", "/images/logo.gif"));
        // `Allow: /publications/` is longer than `*.gif$`.
        assert!(robots.is_allowed("otherbot", "/publications/logo.gif"));
        assert!(robots.is_allowed("otherbot", "/publications/2026/report.pdf"));
        assert!(robots.is_allowed("otherbot", "/about"));
        assert!(robots.is_allowed("otherbot", "/images/logo.gif?v=2"));

        // foobot: only two URL prefixes.
        assert!(!robots.is_allowed("foobot", "/"));
        assert!(!robots.is_allowed("foobot", "/example/"));
        assert!(!robots.is_allowed("foobot", "/publications/"));
        assert!(robots.is_allowed("foobot", "/example/page.html"));
        assert!(robots.is_allowed("foobot", "/example/allowed.gif"));

        // barbot and bazbot share one group.
        for bot in ["barbot", "bazbot"] {
            assert!(!robots.is_allowed(bot, "/example/page.html"), "{bot}");
            assert!(robots.is_allowed(bot, "/example/other.html"), "{bot}");
            assert!(robots.is_allowed(bot, "/images/logo.gif"), "{bot}");
        }

        // quxbot: an empty group at the end of the file gives unrestricted access.
        assert!(robots.is_allowed("quxbot", "/example/index.html"));
        assert!(robots.is_allowed("quxbot", "/images/logo.gif"));
        assert!(robots.ends_with_ruleless_group());
    }

    #[test]
    fn rfc9309_longest_match_example() {
        let robots = AiRobots::parse(
            "User-Agent: foobot\n\
Allow: /example/page/\n\
Disallow: /example/page/disallowed.gif\n",
        );
        assert!(robots.is_allowed("foobot", "/example/page/allowed.gif"));
        assert!(!robots.is_allowed("foobot", "/example/page/disallowed.gif"));
    }

    #[test]
    fn longest_match_wins_and_allow_wins_a_tie() {
        // Google's precedence examples.
        let cases = [
            ("allow: /p\ndisallow: /", "/page", true),
            ("allow: /folder\ndisallow: /folder", "/folder/page", true),
            ("allow: /page\ndisallow: /*.htm", "/page.htm", false),
            ("allow: /page\ndisallow: /*.ph", "/page.php5", true),
            ("allow: /$\ndisallow: /", "/", true),
            ("allow: /$\ndisallow: /", "/page.htm", false),
            ("disallow: /folder\nallow: /folder", "/folder/page", true),
        ];
        for (rules, path, allowed) in cases {
            let robots = AiRobots::parse(&format!("user-agent: *\n{rules}\n"));
            assert_eq!(robots.is_allowed("anybot", path), allowed, "{rules:?} on {path}");
        }
    }

    #[test]
    fn wildcards_and_the_end_anchor_follow_googles_examples() {
        let cases: [(&str, &[&str], &[&str]); 5] = [
            (
                "/fish",
                &[
                    "/fish",
                    "/fish.html",
                    "/fish/salmon.html",
                    "/fishheads",
                    "/fish.php?id=anything",
                ],
                &["/Fish.asp", "/catfish", "/?id=fish", "/desert/fish"],
            ),
            (
                "/fish*",
                &["/fish", "/fish.html", "/fishheads/yummy.html"],
                &["/Fish.asp", "/catfish"],
            ),
            (
                "/fish/",
                &["/fish/", "/fish/?id=anything", "/fish/salmon.htm"],
                &["/fish", "/fish.html", "/animals/fish/", "/Fish/Salmon.asp"],
            ),
            (
                "/*.php$",
                &["/filename.php", "/folder/filename.php"],
                &[
                    "/filename.php?parameters",
                    "/filename.php/",
                    "/filename.php5",
                    "/windows.PHP",
                ],
            ),
            (
                "/fish*.php",
                &["/fish.php", "/fishheads/catfish.php?parameters"],
                &["/Fish.PHP", "/fish.html"],
            ),
        ];
        for (pattern, matching, not_matching) in cases {
            let robots = AiRobots::parse(&format!("User-agent: *\nDisallow: {pattern}\n"));
            for path in matching {
                assert!(!robots.is_allowed("anybot", path), "{pattern} should match {path}");
            }
            for path in not_matching {
                assert!(robots.is_allowed("anybot", path), "{pattern} should not match {path}");
            }
        }
    }

    #[test]
    fn a_dollar_inside_a_pattern_is_literal() {
        let robots = AiRobots::parse("User-agent: *\nDisallow: /price$list\n");
        assert!(!robots.is_allowed("anybot", "/price$list/2026"));
        assert!(robots.is_allowed("anybot", "/price"));
    }

    #[test]
    fn the_matcher_backtracks_without_blowing_up() {
        let robots = AiRobots::parse("User-agent: *\nDisallow: /*a*a*a*a*a*a*a*a*a*a*b$\n");
        let long = format!("/{}", "a".repeat(5_000));
        assert!(robots.is_allowed("anybot", &long));
        assert!(!robots.is_allowed("anybot", &format!("{long}b")));
        assert!(!robots.is_allowed("anybot", "/xaxaxaxaxaxaxaxaxaxab"));
    }

    #[test]
    fn hostile_patterns_are_matched_in_linear_time() {
        // One `*` before a long literal that almost matches: a backtracking matcher retries the
        // literal at every position of the path.
        let long = "a".repeat(16_000);
        let robots = AiRobots::parse(&format!("User-agent: *\nDisallow: /*{long}b\n"));
        let path = format!("/{long}");
        let started = std::time::Instant::now();
        for _ in 0..5 {
            assert!(robots.is_allowed("anybot", &path));
        }
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "{:?}",
            started.elapsed()
        );
    }

    /// The definition of a robots.txt pattern, tried every way: `*` takes any number of bytes, and
    /// the pattern must match a prefix of the path (the whole path when anchored).
    fn matches_by_definition(pattern: &[u8], path: &[u8], anchored: bool) -> bool {
        match pattern.split_first() {
            None => !anchored || path.is_empty(),
            Some((b'*', rest)) => (0..=path.len()).any(|taken| matches_by_definition(rest, &path[taken..], anchored)),
            Some((byte, rest)) => path.first() == Some(byte) && matches_by_definition(rest, &path[1..], anchored),
        }
    }

    #[test]
    fn the_matcher_agrees_with_the_definition() {
        // A small deterministic generator (LCG), so the cases are the same on every run.
        let mut seed: u64 = 0x5eed;
        let mut next = move |bound: u64| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) % bound
        };
        for _ in 0..50_000 {
            let pattern: String = (0..next(8)).map(|_| ['a', 'b', '/', '*'][next(4) as usize]).collect();
            let anchored = next(3) == 0;
            let path: String = (0..next(10)).map(|_| ['a', 'b', '/'][next(3) as usize]).collect();
            let full = if anchored {
                format!("{pattern}$")
            } else {
                pattern.clone()
            };
            assert_eq!(
                pattern_matches(&full, &path),
                matches_by_definition(pattern.as_bytes(), path.as_bytes(), anchored),
                "pattern {full:?} on {path:?}"
            );
        }
    }

    #[test]
    fn policy_equivalence_fails_closed_when_the_check_would_be_too_large() {
        let rules: String = (0..2_000).map(|i| format!("Disallow: /p{i}/\n")).collect();
        let before = AiRobots::parse(&format!("User-agent: *\n{rules}"));
        let after = AiRobots::parse(&format!("{}\nUser-agent: GPTBot\nDisallow: /\n", before.raw()));
        let started = std::time::Instant::now();
        let result = policy_equivalent(Some(&before), &after, &table_tokens_except(&["GPTBot"]), &[]);
        assert!(
            result.as_ref().is_err_and(|why| why.contains("too many rules")),
            "{result:?}"
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "{:?}",
            started.elapsed()
        );
    }

    #[test]
    fn normalize_path_follows_rfc9309_percent_encoding() {
        // RFC 9309 §2.2.2 table.
        assert_eq!(normalize_path("/foo/bar?baz=quz"), "/foo/bar?baz=quz");
        assert_eq!(normalize_path("/foo/bar/ツ"), "/foo/bar/%E3%83%84");
        assert_eq!(normalize_path("/foo/bar/%E3%83%84"), "/foo/bar/%E3%83%84");
        assert_eq!(normalize_path("/foo/bar/%62%61%7A"), "/foo/bar/baz");
        // Unreserved characters are decoded, reserved ones stay encoded, hex is uppercased.
        assert_eq!(normalize_path("/%7Ejoe/%2d%2E%5f"), "/~joe/-._");
        assert_eq!(normalize_path("/a%2fb"), "/a%2Fb");
        assert_eq!(normalize_path("/a%3Fb%3d"), "/a%3Fb%3D");
        assert_eq!(normalize_path("/caf%c3%a9"), "/caf%C3%A9");
        // Characters a URI cannot hold are encoded; a stray `%` too.
        assert_eq!(normalize_path("/a b\"c"), "/a%20b%22c");
        assert_eq!(normalize_path("/100%"), "/100%25");
        assert_eq!(normalize_path("/100%zz"), "/100%25zz");
        // Wildcards survive.
        assert_eq!(normalize_path("/*.pdf$"), "/*.pdf$");
    }

    #[test]
    fn percent_encoding_is_normalized_on_patterns_and_paths() {
        let robots = AiRobots::parse("User-agent: *\nDisallow: /a%2Fb\nDisallow: /%7Ejoe/\nDisallow: /café\n");
        assert!(!robots.is_allowed("anybot", "/a%2Fb"));
        assert!(!robots.is_allowed("anybot", "/a%2fb"));
        assert!(robots.is_allowed("anybot", "/a/b"), "%2F is not a path separator");
        assert!(!robots.is_allowed("anybot", "/~joe/index.html"));
        assert!(!robots.is_allowed("anybot", "/%7ejoe/index.html"));
        assert!(!robots.is_allowed("anybot", "/café/menu"));
        assert!(!robots.is_allowed("anybot", "/caf%C3%A9/menu"));
        assert!(!robots.is_allowed("anybot", "/caf%c3%a9"));
        assert!(robots.is_allowed("anybot", "/cafe"));

        let slash = AiRobots::parse("User-agent: *\nDisallow: /a/b\n");
        assert!(slash.is_allowed("anybot", "/a%2Fb"));
    }

    #[test]
    fn groups_of_one_token_merge_and_a_named_group_ignores_star() {
        let robots = AiRobots::parse(
            "User-agent: GPTBot\nDisallow: /a/\n\nUser-agent: *\nDisallow: /\n\nuser-agent: gptbot\nDisallow: /b/\n",
        );
        assert!(!robots.is_allowed("GPTBot", "/a/x"));
        assert!(!robots.is_allowed("GPTBot", "/b/x"));
        assert!(
            robots.is_allowed("GPTBot", "/c"),
            "the * group does not apply to GPTBot"
        );
        assert!(!robots.is_allowed("ClaudeBot", "/c"));

        // A named group without a matching rule still shadows `*`.
        let shadow = AiRobots::parse("User-agent: *\nDisallow: /\n\nUser-agent: ClaudeBot\nDisallow: /tmp/\n");
        assert!(shadow.is_allowed("ClaudeBot", "/pricing"));
    }

    #[test]
    fn user_agent_values_match_case_insensitively_by_their_product_token() {
        let robots =
            AiRobots::parse("user-agent: oai-searchbot/1.3\nDISALLOW: /\n\nUser-agent: Googlebot-Image\nDisallow: /\n");
        assert!(!robots.is_allowed("OAI-SearchBot", "/x"));
        assert!(robots.has_named_group("OAI-SEARCHBOT"));
        // A longer token is another crawler.
        assert!(robots.is_allowed("Googlebot", "/x"));
        assert!(!robots.has_named_group("Googlebot"));
    }

    #[test]
    fn only_rules_end_a_list_of_user_agents() {
        let robots = AiRobots::parse(
            "User-agent: a\nSitemap: https://example.com/sitemap.xml\nCrawl-delay: 5\nUser-agent: b\nDisallow: /x\nUser-agent: c\nDisallow: /y\n",
        );
        assert!(!robots.is_allowed("a", "/x"));
        assert!(!robots.is_allowed("b", "/x"));
        assert!(robots.is_allowed("a", "/y"));
        assert!(!robots.is_allowed("c", "/y"));
        assert!(robots.is_allowed("c", "/x"));
        assert_eq!(robots.sitemaps(), ["https://example.com/sitemap.xml"]);

        // An empty Disallow is a rule too: it ends the list and allows everything.
        let empty = AiRobots::parse("User-agent: a\nDisallow:\nUser-agent: b\nDisallow: /\n");
        assert!(empty.is_allowed("a", "/x"));
        assert!(!empty.is_allowed("b", "/x"));
    }

    #[test]
    fn rules_outside_a_group_are_ignored() {
        let robots = AiRobots::parse("Disallow: /\nAllow: /x\n\nUser-agent: a\nDisallow: /private\n");
        assert!(robots.is_allowed("otherbot", "/"));
        assert!(!robots.is_allowed("a", "/private/1"));
        assert!(robots.is_allowed("a", "/public"));
    }

    #[test]
    fn comments_blank_lines_bom_and_line_endings_are_handled() {
        let robots = AiRobots::parse(
            "\u{feff}# robots\r\nUser-agent: * # everyone\r\n\r\nDisallow: /private # secret\r\nAllow: /private/open\rSitemap: https://example.com/s.xml\r\n",
        );
        assert!(!robots.is_allowed("anybot", "/private/x"));
        assert!(robots.is_allowed("anybot", "/private/open/x"));
        assert!(robots.is_allowed("anybot", "/public"));
        assert_eq!(robots.sitemaps(), ["https://example.com/s.xml"]);
        assert!(!robots.ends_with_ruleless_group());
    }

    #[test]
    fn empty_disallow_allows_and_robots_txt_is_always_allowed() {
        let open = AiRobots::parse("User-agent: *\nDisallow:\n");
        assert!(open.is_allowed("anybot", "/anything"));
        assert_eq!(open.explain("anybot", "/anything"), None);

        let closed = AiRobots::parse("User-agent: *\nDisallow: /\n");
        assert!(!closed.is_allowed("anybot", "/"));
        assert!(!closed.is_allowed("anybot", ""), "an empty path is the root");
        assert!(closed.is_allowed("anybot", "/robots.txt"));
    }

    #[test]
    fn explain_names_the_deciding_group_and_rule() {
        let robots = AiRobots::parse("User-agent: *\nAllow: /\n\nUser-agent: OAI-SearchBot\nDisallow: /\n");
        assert_eq!(
            robots.explain("OAI-SearchBot", "/pricing").as_deref(),
            Some("User-agent: OAI-SearchBot → Disallow: /")
        );
        assert_eq!(
            robots.explain("oai-searchbot", "/pricing").as_deref(),
            Some("User-agent: OAI-SearchBot → Disallow: /")
        );
        assert_eq!(
            robots.explain("GPTBot", "/pricing").as_deref(),
            Some("User-agent: * → Allow: /")
        );
        let partial = AiRobots::parse("User-agent: GPTBot\nDisallow: /docs/*.pdf$\n");
        assert_eq!(
            partial.explain("GPTBot", "/docs/a.pdf").as_deref(),
            Some("User-agent: GPTBot → Disallow: /docs/*.pdf$")
        );
        assert_eq!(partial.explain("GPTBot", "/docs/a.html"), None);
    }

    #[test]
    fn applebot_follows_googlebot_only_without_a_group_of_its_own() {
        let applebot = find_agent("Applebot").unwrap();

        let googlebot_only = AiRobots::parse("User-agent: Googlebot\nDisallow: /g/\n\nUser-agent: *\nDisallow: /\n");
        assert_eq!(googlebot_only.token_for(applebot), "Googlebot");
        assert_eq!(
            evaluate(Some(&googlebot_only), applebot, &paths(&["/g/x", "/other"])),
            AgentAccess::Partly(paths(&["/g/x"]))
        );
        assert_eq!(
            googlebot_only
                .explain(googlebot_only.token_for(applebot), "/g/x")
                .as_deref(),
            Some("User-agent: Googlebot → Disallow: /g/")
        );

        let own_group = AiRobots::parse("User-agent: Applebot\nDisallow: /a/\n\nUser-agent: Googlebot\nDisallow: /\n");
        assert_eq!(own_group.token_for(applebot), "Applebot");
        assert_eq!(
            evaluate(Some(&own_group), applebot, &paths(&["/a/x", "/b"])),
            AgentAccess::Partly(paths(&["/a/x"]))
        );

        // Without a Googlebot group the `*` group applies, as for any other crawler.
        let star_only = AiRobots::parse("User-agent: *\nDisallow: /\n");
        assert_eq!(star_only.token_for(applebot), "Applebot");
        assert_eq!(
            evaluate(Some(&star_only), applebot, &paths(&["/", "/b"])),
            AgentAccess::Blocked
        );
    }

    #[test]
    fn evaluate_reports_allowed_blocked_and_partly() {
        let gptbot = find_agent("GPTBot").unwrap();
        let robots = AiRobots::parse("User-agent: GPTBot\nDisallow: /private/\n");
        assert_eq!(
            evaluate(Some(&robots), gptbot, &paths(&["/", "/about"])),
            AgentAccess::Allowed
        );
        assert_eq!(
            evaluate(Some(&robots), gptbot, &paths(&["/private/a", "/private/b"])),
            AgentAccess::Blocked
        );
        assert_eq!(
            evaluate(Some(&robots), gptbot, &paths(&["/", "/private/a"])),
            AgentAccess::Partly(paths(&["/private/a"]))
        );
        // No robots.txt: nothing is blocked.
        assert_eq!(evaluate(None, gptbot, &paths(&["/private/a"])), AgentAccess::Allowed);
    }

    #[test]
    fn named_tokens_lists_every_named_agent_once() {
        let robots = AiRobots::parse(
            "User-agent: Alpha\nUser-agent: *\nDisallow: /\n\nUser-agent: alpha\nUser-agent: Beta/2.0\nDisallow: /x\n",
        );
        assert_eq!(robots.named_tokens(), ["Alpha", "Beta"]);
        assert!(!robots.has_named_group("*"));
    }

    #[test]
    fn a_trailing_group_without_rules_is_detected() {
        assert!(AiRobots::parse("User-agent: *\nDisallow: /\n\nUser-agent: Example\n").ends_with_ruleless_group());
        assert!(AiRobots::parse("User-agent: *\nDisallow: /\n\nUser-agent: Example").ends_with_ruleless_group());
        assert!(AiRobots::parse("User-agent: X\nCrawl-delay: 5\n# end\n\n").ends_with_ruleless_group());
        assert!(!AiRobots::parse("User-agent: *\nDisallow: /\n").ends_with_ruleless_group());
        assert!(
            !AiRobots::parse("User-agent: *\nDisallow: /\nSitemap: https://e.com/s.xml\n").ends_with_ruleless_group()
        );
        assert!(!AiRobots::parse("").ends_with_ruleless_group());
        assert!(!AiRobots::parse("Sitemap: https://e.com/s.xml\n").ends_with_ruleless_group());
    }

    #[test]
    fn googles_accepted_typos_and_missing_colons_are_read() {
        let robots = AiRobots::parse(
            "useragent: a\nDisalow: /1\n\nUser agent: b\nDissallow: /2\n\nUser-agents: c\nDisallow /3\n\n\
User-agent d\nDisallow: /4\nAllowed: /4/open\nDisallow /5 extra\nsite-map: https://example.com/s.xml\n",
        );
        assert!(!robots.is_allowed("a", "/1"));
        assert!(!robots.is_allowed("b", "/2"));
        assert!(!robots.is_allowed("c", "/3"));
        assert!(!robots.is_allowed("d", "/4"));
        assert!(robots.is_allowed("d", "/4/open/x"));
        // Without a colon, only exactly two words make a record.
        assert!(robots.is_allowed("d", "/5"));
        assert_eq!(robots.sitemaps(), ["https://example.com/s.xml"]);
        assert_eq!(robots.named_tokens(), ["a", "b", "c", "d"]);
    }

    #[test]
    fn a_trailing_user_agent_line_in_any_spelling_leaves_the_group_open() {
        for tail in [
            "useragent: *",
            "User agent: Example",
            "User-agent Googlebot",
            "User-Agent; Googlebot",
        ] {
            let robots = AiRobots::parse(&format!("User-agent: *\nDisallow: /private/\n\n{tail}\n"));
            assert!(robots.ends_with_ruleless_group(), "{tail}");
        }
        // A rule only Google's parser reads does not close the group for a strict parser.
        assert!(AiRobots::parse("User-agent: X\nDisalow: /x\n").ends_with_ruleless_group());
        assert!(AiRobots::parse("User-agent: X\nDisallow /x\n").ends_with_ruleless_group());
        assert!(!AiRobots::parse("User-agent: X\nDisallow: /x\n# end\n").ends_with_ruleless_group());

        let before = AiRobots::parse("User-agent: *\nDisallow: /private/\n\nuseragent: *\n");
        let after = AiRobots::parse(&format!("{}User-agent: GPTBot\nDisallow: /\n", before.raw()));
        let result = policy_equivalent(Some(&before), &after, &table_tokens_except(&["GPTBot"]), &[]);
        assert!(result.is_err(), "{result:?}");
    }

    #[test]
    fn only_the_first_500_kib_and_16663_bytes_of_a_line_are_read() {
        let padding = format!("# {}\n", "x".repeat(600 * 1024));
        let late = AiRobots::parse(&format!("User-agent: *\nDisallow: /a\n{padding}Disallow: /b\n"));
        assert!(!late.is_allowed("anybot", "/a"));
        assert!(late.is_allowed("anybot", "/b"), "a rule after 500 KiB is ignored");
        assert!(late.raw().ends_with("Disallow: /b\n"), "the raw file is kept whole");

        let long = AiRobots::parse(&format!("User-agent: *\nDisallow: /{}\n", "a".repeat(20_000)));
        // "Disallow: /" takes 11 of the 16,663 bytes of the line.
        assert!(!long.is_allowed("anybot", &format!("/{}", "a".repeat(16_652))));
        assert!(long.is_allowed("anybot", &format!("/{}", "a".repeat(16_651))));
    }

    #[test]
    fn allowing_an_index_page_allows_its_directory() {
        let robots = AiRobots::parse("User-agent: *\nDisallow: /dir/\nAllow: /dir/index.html\n");
        assert!(robots.is_allowed("anybot", "/dir/"));
        assert!(robots.is_allowed("anybot", "/dir/index.html"));
        assert!(!robots.is_allowed("anybot", "/dir/other.html"));
        assert_eq!(
            robots.explain("anybot", "/dir/").as_deref(),
            Some("User-agent: * → Allow: /dir/index.html")
        );
    }

    #[test]
    fn content_signals_are_captured_raw() {
        let robots = AiRobots::parse(
            "User-Agent: *\nContent-Signal: search=yes, ai-train=no\ncontent-usage: train-ai=n\nAllow: /\n",
        );
        assert_eq!(
            robots.content_signals(),
            ["Content-Signal: search=yes, ai-train=no", "content-usage: train-ai=n"]
        );
        assert!(robots.is_allowed("GPTBot", "/"));
        assert_eq!(
            robots.raw(),
            "User-Agent: *\nContent-Signal: search=yes, ai-train=no\ncontent-usage: train-ai=n\nAllow: /\n"
        );
    }

    #[test]
    fn policy_equivalence_holds_when_a_group_is_appended_after_rules() {
        let before = AiRobots::parse("User-agent: *\nDisallow: /private/\n\nUser-agent: Googlebot\nAllow: /\n");
        let after = AiRobots::parse(&format!("{}\nUser-agent: GPTBot\nDisallow: /\n", before.raw()));
        assert_eq!(
            policy_equivalent(
                Some(&before),
                &after,
                &table_tokens_except(&["GPTBot"]),
                &paths(&["/private/a", "/about"])
            ),
            Ok(())
        );
    }

    #[test]
    fn policy_equivalence_fails_on_a_trailing_ruleless_group() {
        let before = AiRobots::parse("User-agent: *\nDisallow: /private/\n\nUser-agent: Example\n");
        let after = AiRobots::parse(&format!("{}User-agent: GPTBot\nDisallow: /\n", before.raw()));
        // `Example` is named in the file, so it is checked without being listed.
        let result = policy_equivalent(Some(&before), &after, &[], &[]);
        assert!(result.as_ref().is_err_and(|why| why.contains("Example")), "{result:?}");
    }

    #[test]
    fn policy_equivalence_checks_star_and_a_literal_path_per_rule() {
        let before = AiRobots::parse("User-agent: *\nDisallow: /secret/*.pdf$\n");
        // Another `*` group merges with the first one and, on a tie, Allow wins.
        let after = AiRobots::parse(&format!("{}\nUser-agent: *\nAllow: /secret/*.pdf$\n", before.raw()));
        let result = policy_equivalent(Some(&before), &after, &[], &[]);
        assert!(
            result.as_ref().is_err_and(|why| why.contains("/secret/.pdf")),
            "{result:?}"
        );
    }

    #[test]
    fn policy_equivalence_from_a_missing_robots_txt() {
        let proposed = AiRobots::parse("User-agent: *\nAllow: /\n\nUser-agent: GPTBot\nDisallow: /\n");
        assert_eq!(
            policy_equivalent(None, &proposed, &table_tokens_except(&["GPTBot"]), &paths(&["/about"])),
            Ok(())
        );
        let closing = AiRobots::parse("User-agent: *\nDisallow: /\n");
        assert!(policy_equivalent(None, &closing, &[], &[]).is_err());
    }

    #[test]
    fn policy_equivalence_applies_the_applebot_fallback() {
        // Before: Applebot follows the Googlebot group. After: a new Applebot-Extended group does
        // not change that, but a new Applebot group would.
        let before = AiRobots::parse("User-agent: Googlebot\nDisallow: /g/\n");
        let fine = AiRobots::parse(&format!(
            "{}\nUser-agent: Applebot-Extended\nDisallow: /\n",
            before.raw()
        ));
        assert_eq!(
            policy_equivalent(
                Some(&before),
                &fine,
                &table_tokens_except(&["Applebot-Extended"]),
                &paths(&["/g/1"])
            ),
            Ok(())
        );
        let changed = AiRobots::parse(&format!("{}\nUser-agent: Applebot\nDisallow: /a/\n", before.raw()));
        let result = policy_equivalent(Some(&before), &changed, &["Applebot".to_string()], &paths(&["/g/1"]));
        assert!(result.as_ref().is_err_and(|why| why.contains("Applebot")), "{result:?}");
    }

    #[test]
    fn only_ok_and_not_found_states_tell_the_rules() {
        let ok = RobotsFetchState::Ok {
            status: 200,
            content: "User-agent: *\nDisallow: /x\n".into(),
            valid_utf8: true,
        };
        assert!(state_is_known(&ok));
        assert!(robots_of(&ok).is_some_and(|robots| !robots.is_allowed("anybot", "/x/1")));

        let not_found = RobotsFetchState::NotFound { status: 404 };
        assert!(state_is_known(&not_found));
        assert!(robots_of(&not_found).is_none());

        for unknown in [
            RobotsFetchState::Unavailable {
                status_or_error: "HTTP 503".into(),
            },
            RobotsFetchState::Unavailable {
                status_or_error: "-2:TIMEOUT".into(),
            },
            RobotsFetchState::Skipped,
            RobotsFetchState::NotAttempted,
        ] {
            assert!(!state_is_known(&unknown), "{unknown:?}");
            assert!(robots_of(&unknown).is_none(), "{unknown:?}");
        }
    }

    #[test]
    fn key_pages_are_evaluated_per_origin_with_its_own_state() {
        let urls = paths(&[
            "https://example.com/",
            "https://example.com/pricing?plan=pro",
            "https://blog.example.com/post",
            "http://example.com:8080/",
            "https://example.com:443/about",
        ]);
        let grouped = paths_by_origin(&urls);
        assert_eq!(
            grouped.keys().cloned().collect::<Vec<_>>(),
            [
                ("http".to_string(), "example.com".to_string(), 8080),
                ("https".to_string(), "blog.example.com".to_string(), 443),
                ("https".to_string(), "example.com".to_string(), 443),
            ]
        );
        assert_eq!(
            grouped[&("https".to_string(), "example.com".to_string(), 443)],
            paths(&["/", "/pricing?plan=pro", "/about"])
        );

        let states = |host: &str| match host {
            "example.com" => RobotsFetchState::Ok {
                status: 200,
                content: "User-agent: OAI-SearchBot\nDisallow: /\n".into(),
                valid_utf8: true,
            },
            "blog.example.com" => RobotsFetchState::NotFound { status: 404 },
            _ => RobotsFetchState::NotAttempted,
        };
        let agent = find_agent("OAI-SearchBot").unwrap();
        let verdicts: Vec<Option<AgentAccess>> = grouped
            .iter()
            .map(|((scheme, host, _), paths)| {
                let state = if scheme == "https" {
                    states(host)
                } else {
                    RobotsFetchState::NotAttempted
                };
                state_is_known(&state).then(|| evaluate(robots_of(&state).as_ref(), agent, paths))
            })
            .collect();
        assert_eq!(verdicts, [None, Some(AgentAccess::Allowed), Some(AgentAccess::Blocked)]);
    }
}
