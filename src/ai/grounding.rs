// SiteOne Crawler - AI grounding helpers
// (c) Jan Reges <jan.reges@siteone.cz>
//
// Meaning-preserving matching of values an LLM quoted from a website against the crawler's
// own copy of the text, shared by the AI pipelines that must never report a value the page does
// not state:
//   - `normalize_for_match` / `contains_normalized` / `find_token_bounded` check that a quote
//     and a value really occur in a block, with a map back to the original text;
//   - `value_key` turns a value into a comparison key that never merges values of different
//     meaning (`ValueKey::Exact`), and marks an equivalence it cannot be sure of as
//     `ValueKey::Uncertain`, which may be shown as a difference but never establishes "equal";
//   - `fact_signals`, `numbers_in`, `date_mentions` and `snippet_of` support block selection,
//     prose validation and evidence display; `normalize_label` compares the names of facts.

use std::collections::HashSet;
use std::ops::{Range, RangeInclusive};

use chrono::NaiveDate;
use once_cell::sync::Lazy;
use regex::Regex;
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;

// ---------------------------------------------------------------------------
// Normalization and matching
// ---------------------------------------------------------------------------

/// Text normalized by `normalize_for_match`, with the way back to the original.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Normalized {
    /// The normalized text.
    pub text: String,
    /// For every byte of `text`, the byte range of the original char (or whitespace run) it came
    /// from.
    pub origins: Vec<Range<usize>>,
}

impl Normalized {
    /// The byte range of the original text that the normalized range `start..end` came from.
    /// `None` for an empty or out-of-bounds range.
    pub fn original_span(&self, start: usize, end: usize) -> Option<(usize, usize)> {
        if start >= end {
            return None;
        }
        let first = self.origins.get(start)?;
        let last = self.origins.get(end - 1)?;
        Some((first.start, last.end))
    }
}

/// Normalize text for matching a quote or a value against the crawler's copy of a block:
/// lowercases; turns every whitespace char (NBSP, U+2007, U+2009, U+202F, `\t`, `\r`, `\n`, …)
/// into a space; drops the emphasis and escape marks `*`, `_`, `` ` `` and `\`; maps `–`/`—` to
/// `-`; collapses whitespace runs and trims. The result carries a byte-offset map back to `s`.
pub fn normalize_for_match(s: &str) -> Normalized {
    let mut text = String::with_capacity(s.len());
    let mut origins = Vec::with_capacity(s.len());
    // A whitespace run not yet written: it becomes one space before the next kept char.
    let mut pending_space: Option<Range<usize>> = None;
    for (at, ch) in s.char_indices() {
        let range = at..at + ch.len_utf8();
        let mapped = match ch {
            '*' | '_' | '`' | '\\' => continue,
            '–' | '—' => '-',
            c if c.is_whitespace() => {
                pending_space = Some(match pending_space {
                    Some(run) => run.start..range.end,
                    None => range,
                });
                continue;
            }
            c => c,
        };
        if let Some(space) = pending_space.take()
            && !text.is_empty()
        {
            text.push(' ');
            origins.push(space);
        }
        for lower in mapped.to_lowercase() {
            let before = text.len();
            text.push(lower);
            origins.extend(std::iter::repeat_n(range.clone(), text.len() - before));
        }
    }
    Normalized { text, origins }
}

/// True when the normalized `needle` occurs in the normalized `hay` (see `normalize_for_match`).
/// An empty needle is never contained, so an empty quote can never pass as evidence.
pub fn contains_normalized(hay: &str, needle: &str) -> bool {
    let needle = normalize_for_match(needle);
    !needle.text.is_empty() && normalize_for_match(hay).text.contains(&needle.text)
}

/// Find `needle` in `hay` after normalization, as a whole token: the chars next to the match may
/// not be a letter or a digit, nor a `.`/`,` joined to a digit, nor — next to a digit of the match —
/// a space or an apostrophe joined to a digit (so `5 %` never matches inside `15 %` or `1,5 %`, and
/// `290` never inside `1.290`, `1 290` or `1'290`). Returns the first such match as a byte range
/// of the ORIGINAL `hay` (e.g. for `snippet_of`); `None` when there is none or `needle` is empty.
pub fn find_token_bounded(hay: &str, needle: &str) -> Option<(usize, usize)> {
    let needle = normalize_for_match(needle).text;
    if needle.is_empty() {
        return None;
    }
    let hay = normalize_for_match(hay);
    let text = hay.text.as_str();
    let mut from = 0;
    while let Some(found) = text.get(from..).and_then(|rest| rest.find(&needle)) {
        let start = from + found;
        let end = start + needle.len();
        if token_bounded(text, start, end) {
            return hay.original_span(start, end);
        }
        from = start
            + text
                .get(start..)
                .and_then(|rest| rest.chars().next())
                .map_or(1, char::len_utf8);
    }
    None
}

/// A label (a subject or an attribute of a fact) reduced for comparing names: decomposed (NFKD)
/// with the diacritics dropped, lowercased, `+` read as the word `plus`, every other char that is
/// not a letter or a digit turned into a space, whitespace collapsed and trimmed.
/// `Zákaznická  linka!` → `zakaznicka linka`, `Tarif S+` → `tarif s plus`.
pub fn normalize_label(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut gap = false;
    for c in s.nfkd().filter(|c| !is_combining_mark(*c)).flat_map(char::to_lowercase) {
        if c.is_alphanumeric() {
            if gap && !out.is_empty() {
                out.push(' ');
            }
            gap = false;
            out.push(c);
        } else if c == '+' {
            // `Tarif S+` is another variant than `Tarif S`.
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str("plus");
            gap = true;
        } else {
            gap = true;
        }
    }
    out
}

/// The span of a number value in `text`, extended over an operator written right before it (`od`,
/// `do`, `from`, `up to`, `nad`, `více než`, `<`, `≥`, … — the operators `parse_number` reads) and
/// over a `+` right after a final digit (`18+`), so that the value keeps the meaning the page gives
/// it: `od 18 let` is not `do 18 let`. A word operator must stand apart from the word before it
/// and from the value; a symbol may touch the value (`≥18`). A span that is not a range of `text`
/// is returned as it is.
pub fn extend_over_operator(text: &str, span: (usize, usize)) -> (usize, usize) {
    let (start, end) = span;
    let (Some(before), Some(value)) = (text.get(..start), text.get(start..end)) else {
        return span;
    };
    let trimmed = before.trim_end();
    let spaced = trimmed.len() < before.len();
    let mut extended = (start, end);
    let mut longest = 0;
    for (word, _) in OPERATORS {
        let chars = word.chars().count();
        let Some((at, _)) = trimmed.char_indices().rev().nth(chars.saturating_sub(1)) else {
            continue;
        };
        let tail = trimmed.get(at..).unwrap_or_default();
        if tail.to_lowercase() != *word || chars <= longest {
            continue;
        }
        let alphabetic = word.starts_with(char::is_alphabetic);
        let joined = trimmed
            .get(..at)
            .unwrap_or_default()
            .chars()
            .next_back()
            .is_some_and(char::is_alphanumeric);
        if alphabetic && (joined || !spaced) {
            continue;
        }
        longest = chars;
        extended.0 = at;
    }
    let after = text.get(end..).unwrap_or_default();
    if value.ends_with(|c: char| c.is_ascii_digit())
        && let Some(rest) = after.strip_prefix('+')
        && !rest.starts_with(char::is_alphanumeric)
    {
        extended.1 = end + 1;
    }
    extended
}

/// The span of a number value in `text`, widened so that the value keeps the meaning the page
/// gives it: over a sign joined to it (`-5 %`, `−5 %`), over the other end of a range it is one
/// end of (`10 %` or `5` of `5–10 %` → `5–10 %`, `20 EUR` of `10 EUR – 20 EUR`), and then over an
/// operator before it (`extend_over_operator`). A dash is a sign only when it touches the number
/// and follows no letter or digit (`COVID-19` has no `-19`, `Sleva – 5 %` no `-5 %`); a dash
/// between two numbers is a range unless the two ends name different units (`290 Kč – 10 GB` is
/// no range). A span that is not a range of `text` is returned as it is.
pub fn extend_number_span(text: &str, span: (usize, usize)) -> (usize, usize) {
    let (Some(before), Some(value), Some(after)) = (text.get(..span.0), text.get(span.0..span.1), text.get(span.1..))
    else {
        return span;
    };
    let (mut start, mut end) = span;
    let value_unit = unit_suffix(value).map(|(_, unit)| unit);

    // Left: the lower end of a range, or a sign.
    let left = before.trim_end_matches(char::is_whitespace);
    if let Some(dash) = left.chars().next_back().filter(|&c| is_dash(c)) {
        let pre = left.get(..left.len() - dash.len_utf8()).unwrap_or_default();
        let low = pre.trim_end_matches(char::is_whitespace);
        let (body, low_unit) = match unit_suffix(low) {
            Some((at, unit)) => (low.get(..at).unwrap_or_default().trim_end(), Some(unit)),
            None => (low, None),
        };
        let low_start = numeral_spans(body).find(|r| r.end == body.len()).map(|r| r.start);
        match low_start {
            Some(at) if units_agree(value_unit, low_unit) => {
                start = at
                    - body
                        .get(..at)
                        .and_then(|head| head.chars().next_back())
                        .filter(|c| matches!(c, '€' | '$' | '£'))
                        .map_or(0, char::len_utf8);
            }
            Some(_) => {}
            None if left.len() == before.len()
                && value.starts_with(|c: char| c.is_ascii_digit())
                && !pre.chars().next_back().is_some_and(char::is_alphanumeric) =>
            {
                start = pre.len();
            }
            None => {}
        }
    }

    // Right: the upper end of a range.
    let right = after.trim_start_matches(char::is_whitespace);
    if let Some(dash) = right.chars().next().filter(|&c| is_dash(c)) {
        let rest = right.get(dash.len_utf8()..).unwrap_or_default().trim_start();
        let rest = rest.strip_prefix(['€', '$', '£']).unwrap_or(rest);
        if rest.starts_with(|c: char| c.is_ascii_digit()) {
            let high_end = text.len() - rest.len() + scan_numeral(rest, 0);
            let tail = text.get(high_end..).unwrap_or_default();
            let spaced = tail.trim_start_matches(char::is_whitespace);
            let high_unit = unit_prefix(spaced);
            let agree = match (value_unit, high_unit) {
                (Some(a), Some((_, b))) => a == b,
                (Some(_), None) => false,
                (None, _) => true,
            };
            if agree {
                end = high_end + high_unit.map_or(0, |(len, _)| tail.len() - spaced.len() + len);
            }
        }
    }
    extend_over_operator(text, (start, end))
}

fn is_dash(c: char) -> bool {
    matches!(c, '-' | '–' | '—' | '−')
}

/// The two ends of a range agree on their units unless both name one and they differ.
fn units_agree(a: Option<&str>, b: Option<&str>) -> bool {
    a.zip(b).is_none_or(|(a, b)| a == b)
}

/// A currency or `%` (`UNITS`) that `s` ends with, as a whole word: where it starts, and its
/// canonical unit. The longest one wins.
fn unit_suffix(s: &str) -> Option<(usize, &'static str)> {
    UNITS
        .iter()
        .filter_map(|(word, unit)| {
            let chars = word.chars().count();
            let (at, _) = s.char_indices().rev().nth(chars.checked_sub(1)?)?;
            let tail = s.get(at..)?;
            let joined = s
                .get(..at)
                .and_then(|head| head.chars().next_back())
                .is_some_and(char::is_alphabetic);
            (tail.to_lowercase() == *word && !(word.starts_with(char::is_alphabetic) && joined))
                .then_some((at, *unit, chars))
        })
        .max_by_key(|(_, _, chars)| *chars)
        .map(|(at, unit, _)| (at, unit))
}

/// A currency or `%` (`UNITS`) that `s` starts with, as a whole word: its length in bytes, and
/// its canonical unit.
fn unit_prefix(s: &str) -> Option<(usize, &'static str)> {
    UNITS.iter().find_map(|(word, unit)| {
        let chars = word.chars().count();
        let len = s.char_indices().nth(chars).map_or(s.len(), |(at, _)| at);
        let head = s.get(..len)?;
        let runs_on = s.get(len..).is_some_and(|rest| rest.starts_with(char::is_alphabetic));
        (head.chars().count() == chars
            && head.to_lowercase() == *word
            && !(word.ends_with(char::is_alphabetic) && runs_on))
            .then_some((len, *unit))
    })
}

/// Locate a value the model quoted from a block: `value` as a whole token of `block` (see
/// `find_token_bounded`, checked against the block's text, not the quote's edges) inside an
/// occurrence of `quote` in `block`, both compared after `normalize_for_match`. Returns the
/// value's byte range in the ORIGINAL `block`; `None` when the quote is not in the block, or the
/// value is not a token of it there, or either is empty.
pub fn locate_quoted_value(block: &str, quote: &str, value: &str) -> Option<(usize, usize)> {
    let quote = normalize_for_match(quote).text;
    let value = normalize_for_match(value).text;
    if quote.is_empty() || value.is_empty() {
        return None;
    }
    let block = normalize_for_match(block);
    let text = block.text.as_str();
    let mut from = 0;
    while let Some(found) = text.get(from..).and_then(|rest| rest.find(&quote)) {
        let quote_start = from + found;
        let quote_end = quote_start + quote.len();
        let mut at = quote_start;
        while let Some(offset) = text.get(at..quote_end).and_then(|inside| inside.find(&value)) {
            let start = at + offset;
            let end = start + value.len();
            if token_bounded(text, start, end) {
                return block.original_span(start, end);
            }
            at = start + next_char_len(text, start);
        }
        from = quote_start + next_char_len(text, quote_start);
    }
    None
}

fn next_char_len(text: &str, at: usize) -> usize {
    text.get(at..)
        .and_then(|rest| rest.chars().next())
        .map_or(1, char::len_utf8)
}

fn token_bounded(text: &str, start: usize, end: usize) -> bool {
    let first = text.get(start..).and_then(|rest| rest.chars().next());
    let last = text.get(..end).and_then(|head| head.chars().next_back());
    let mut before = text.get(..start).unwrap_or_default().chars().rev();
    let mut after = text.get(end..).unwrap_or_default().chars();
    free_edge(first, before.next(), before.next()) && free_edge(last, after.next(), after.next())
}

/// A match edge is free when the char next to it is not a letter or a digit, nor a `.`/`,`
/// followed (away from the match) by a digit; and, when the match's own edge char is a digit, nor
/// a space or an apostrophe followed by a digit (a thousands separator: `290` is no token of
/// `1 290`, `123 456` none of `800 123 456`).
fn free_edge(edge: Option<char>, neighbour: Option<char>, beyond: Option<char>) -> bool {
    match neighbour {
        Some(c) if c.is_alphanumeric() => false,
        Some('.' | ',') => !beyond.is_some_and(|c| c.is_numeric()),
        Some(' ' | '\'' | '’') if edge.is_some_and(|c| c.is_numeric()) => !beyond.is_some_and(|c| c.is_numeric()),
        _ => true,
    }
}

// ---------------------------------------------------------------------------
// Fact signals
// ---------------------------------------------------------------------------

/// Which kinds of hard facts a text appears to contain (heuristic, for choosing blocks and lines).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Signals {
    /// An amount with a currency (`1 290 Kč`, `290,-`, `€ 29.90`, `$5`).
    pub money: bool,
    /// A percentage (`15 %`, `4,59 procent`).
    pub percent: bool,
    /// A phone number: 9 to 15 digits in one run of digits and separators.
    pub phone: bool,
    /// An e-mail address.
    pub email: bool,
    /// A date (`31. 12. 2026`, `2026-12-31`, `25. září`, `September 25, 2026`).
    pub date: bool,
    /// A company, VAT or bank identifier (`IČO 12345678`, `CZ12345678`, an IBAN, `123/0100`).
    pub ids: bool,
    /// Opening hours or a time range (`Po–Pá 8:00–17:00`, `po-pá 8-18`, `8–18 h`).
    pub hours: bool,
    /// The number of ASCII digits in the text.
    pub digits: usize,
}

static MONEY: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        \d(?:[\d\ .,'’]*\d)?\ ?(?:kč|czk|eur|usd|gbp|chf|pln|zł|huf|ft|korun[ay]?)\b
        | \d(?:[\d\ .,'’]*\d)?\ ?(?:€|\$|£|,-)
        | (?:€|\$|£)\ ?\d
        | \b(?:eur|usd|czk|gbp|chf|pln|huf|kč)\ ?\d",
    )
    .expect("money regex")
});

static PERCENT: Lazy<Regex> = Lazy::new(|| Regex::new(r"\d\ ?(?:%|procent|percent)").expect("percent regex"));

/// A run that may be a phone number; `fact_signals` and `phone_key` count its digits.
static PHONE_RUN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\+?\(?\d[\d ()\-./]*\d").expect("phone regex"));

/// An e-mail address in lowercased text that still has its underscores.
static EMAIL_ADDRESS: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[a-z0-9._%+\-]+@[a-z0-9\-]+(?:\.[a-z0-9\-]+)*\.[a-z]{2,}").expect("e-mail regex"));

static IDS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        \b(?:ičo|ič|ico|dič|dic|vat(?:\ id)?|ust-idnr|nip|regon|krs|crn|company\ (?:no|number|id))\b
            \.?:?\ ?(?:[a-z]{2})?\d{6,}
        | \b(?:cz|sk|de|at|pl|hu|gb)\d{8,12}\b
        | \b[a-z]{2}\d{2}(?:\ ?[a-z0-9]{4}){3,7}\b
        | \b(?:\d{1,6}-)?\d{2,10}/\d{4}\b",
    )
    .expect("ids regex")
});

static HOURS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        \b\d{1,2}[:.]\d{2}\ ?-\ ?\d{1,2}[:.]\d{2}\b
        | \b(?:po|út|ut|st|čt|ct|pá|pa|so|ne|mo|tu|we|th|fr|sa|su|di|mi|do)[a-zá-ž]*\.?\ ?-\ ?
            (?:po|út|ut|st|čt|ct|pá|pa|so|ne|mo|tu|we|th|fr|sa|su|di|mi|do)[a-zá-ž]*\.?:?\ ?
            \d{1,2}(?:[:.]\d{2})?\ ?-\ ?\d{1,2}\b
        | \b\d{1,2}\ ?-\ ?\d{1,2}\ ?(?:h|hod)\b",
    )
    .expect("hours regex")
});

const MONTHS_EN: &str = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?";
const MONTHS_CS: &str = r"ledna|února|března|dubna|května|června|července|srpna|září|října|listopadu|prosince";

/// Date patterns over normalized (lowercased) text, each with the capture order of its parts.
static DATE_PATTERNS: Lazy<Vec<(Regex, DateOrder)>> = Lazy::new(|| {
    vec![
        (
            Regex::new(r"\b(\d{1,2})\.\ ?(\d{1,2})\.\ ?(\d{4})\b").expect("date regex"),
            DateOrder::DayMonthYear,
        ),
        (
            Regex::new(r"\b(\d{4})-(\d{2})-(\d{2})(?:t|\b)").expect("date regex"),
            DateOrder::YearMonthDay,
        ),
        (
            Regex::new(&format!(r"\b(\d{{1,2}})\. ?({MONTHS_CS})(?: (\d{{4}}))?\b")).expect("date regex"),
            DateOrder::DayMonthYear,
        ),
        (
            Regex::new(&format!(
                r"\b({MONTHS_EN})\.? (\d{{1,2}})(?:st|nd|rd|th)?,? (\d{{4}})\b"
            ))
            .expect("date regex"),
            DateOrder::MonthDayYear,
        ),
        (
            Regex::new(&format!(
                r"\b(\d{{1,2}})(?:st|nd|rd|th)? (?:of )?({MONTHS_EN})\.?,? (\d{{4}})\b"
            ))
            .expect("date regex"),
            DateOrder::DayMonthYear,
        ),
    ]
});

#[derive(Clone, Copy)]
enum DateOrder {
    DayMonthYear,
    MonthDayYear,
    YearMonthDay,
}

/// Detect the kinds of hard facts in `text` (see `Signals`). The patterns are compiled once.
pub fn fact_signals(text: &str) -> Signals {
    let n = normalize_for_match(text).text;
    // A run of 9 to 15 digits that is not part of a longer token and holds no date
    // ("31.12.2026 10:00" is not a phone number).
    let phone = PHONE_RUN.find_iter(&n).any(|m| {
        let joined = n
            .get(..m.start())
            .unwrap_or_default()
            .chars()
            .next_back()
            .is_some_and(char::is_alphanumeric);
        !joined
            && (9..=15).contains(&digit_string(m.as_str()).len())
            && !DATE_PATTERNS.iter().any(|(re, _)| re.is_match(m.as_str()))
    });
    Signals {
        money: MONEY.is_match(&n),
        percent: PERCENT.is_match(&n),
        phone,
        email: EMAIL_ADDRESS.is_match(&n),
        date: DATE_PATTERNS.iter().any(|(re, _)| re.is_match(&n)),
        ids: IDS.is_match(&n),
        hours: HOURS.is_match(&n),
        digits: text.chars().filter(char::is_ascii_digit).count(),
    }
}

// ---------------------------------------------------------------------------
// Value keys
// ---------------------------------------------------------------------------

/// How a value should be read when it is turned into a `ValueKey`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueHint {
    /// A phone number or an e-mail address.
    Contact,
    /// An amount, rate, count or other figure, with its operator, unit and time base.
    Number,
    /// A calendar date.
    Date,
    /// Anything else, compared as text.
    Text,
}

/// A comparison key of a value. Two `Exact` keys are equal exactly when the values mean the same;
/// an `Uncertain` key marks a value whose meaning the crawler could not pin down (e.g. `5,000 %`
/// without a page language, or a national phone number without a known country): it may be shown
/// as a difference to review, but it must never establish that values are consistent.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize)]
pub enum ValueKey {
    Exact(String),
    Uncertain(String),
}

/// The comparison key of `value` as written, read per `hint`:
/// - Contact: an e-mail → `mail:<lowercase>`; a phone → `tel:+<international digits>` (a national
///   number only with a known `site_country`, see `site_country`, else
///   `Uncertain("tel-national:<digits>")`); a contact with neither → as Text.
/// - Number: see `parse_number`.
/// - Date: `date:YYYY-MM-DD` when the whole value is a date (`D. M. YYYY`, `D.M.YYYY`,
///   `YYYY-MM-DD`, English month names, Czech genitive month names).
/// - Text: `text:` + the value lowercased with whitespace collapsed; digits, operators and
///   punctuation are kept, so `<18` ≠ `>18`.
///
/// A Number or Date value that does not parse as a whole (`cena 1 290 Kč`, `2026/12/31`) is
/// `Uncertain`. `llm_normalized` (the model's machine-readable form of the value) is never used
/// for the key: its digits prove neither the unit, nor a bound, nor the date the value states
/// (`price: 10 EUR` is no `10 USD`, `minimum price: 10 EUR` no `10 EUR`, `2026-01-01 to
/// 2026-12-31` no `2026-01-31`), so it can neither override nor replace the verbatim reading.
pub fn value_key(
    hint: ValueHint,
    value: &str,
    _llm_normalized: &str,
    lang: &str,
    site_country: Option<&str>,
) -> ValueKey {
    let verbatim = match hint {
        ValueHint::Contact => return contact_key(value, site_country),
        ValueHint::Text => return text_key(value),
        ValueHint::Number => parse_number(value, lang),
        ValueHint::Date => parse_date(value).map(date_key),
    };
    if let Some(key) = verbatim {
        return key;
    }
    let prefix = if hint == ValueHint::Number { "num" } else { "date" };
    ValueKey::Uncertain(format!("{prefix}?:{}", collapse_lower(value)))
}

fn text_key(value: &str) -> ValueKey {
    ValueKey::Exact(format!("text:{}", collapse_lower(value)))
}

fn date_key(date: NaiveDate) -> ValueKey {
    ValueKey::Exact(format!("date:{}", date.format("%Y-%m-%d")))
}

fn contact_key(value: &str, site_country: Option<&str>) -> ValueKey {
    if value.contains('@') {
        return email_key(value);
    }
    if digit_string(value).len() >= 3 {
        return phone_key(value, site_country);
    }
    text_key(value)
}

fn email_key(value: &str) -> ValueKey {
    let lower = value.trim().to_lowercase();
    let mut found: Vec<&str> = EMAIL_ADDRESS.find_iter(&lower).map(|m| m.as_str()).collect();
    found.dedup();
    match found.as_slice() {
        [address] => ValueKey::Exact(format!("mail:{address}")),
        _ => ValueKey::Uncertain(format!("mail?:{}", collapse_lower(value))),
    }
}

fn phone_key(value: &str, site_country: Option<&str>) -> ValueKey {
    let cleaned = unify_spaces(value).replace("(0)", " ");
    // The run with the most digits: "Tel. 1: +420 …" must not stop at the "1".
    let run = PHONE_RUN
        .find_iter(&cleaned)
        .map(|m| m.as_str())
        .max_by_key(|run| digit_string(run).len())
        .unwrap_or_default();
    let digits = digit_string(run);
    if run.starts_with('+') || digits.starts_with("00") {
        let mut digits = if run.starts_with('+') {
            digits
        } else {
            digits.get(2..).unwrap_or_default().to_string()
        };
        if !(7..=15).contains(&digits.len()) {
            return ValueKey::Uncertain(format!("tel?:{digits}"));
        }
        // "+49 030 …" repeats the trunk prefix the country drops when dialled from abroad.
        if let Some(country) = COUNTRIES.iter().find(|c| digits.starts_with(c.dial))
            && country.trunk == "0"
            && digits
                .get(country.dial.len()..)
                .is_some_and(|rest| rest.starts_with('0'))
        {
            digits.remove(country.dial.len());
        }
        return ValueKey::Exact(format!("tel:+{digits}"));
    }
    let national = || ValueKey::Uncertain(format!("tel-national:{digits}"));
    let Some(country) = site_country.and_then(country_by_code) else {
        return national();
    };
    let Some(number) = digits.strip_prefix(country.trunk) else {
        return national();
    };
    if (country.trunk.is_empty() && number.starts_with('0')) || !country.national.contains(&number.len()) {
        return national();
    }
    ValueKey::Exact(format!("tel:+{}{number}", country.dial))
}

/// Dialling rules of the countries `site_country` can return: the country calling code, the
/// national trunk prefix (dropped in the international form) and the length of the national
/// significant number.
struct Country {
    code: &'static str,
    dial: &'static str,
    trunk: &'static str,
    national: RangeInclusive<usize>,
}

const COUNTRIES: &[Country] = &[
    Country {
        code: "CZ",
        dial: "420",
        trunk: "",
        national: 9..=9,
    },
    Country {
        code: "SK",
        dial: "421",
        trunk: "0",
        national: 9..=9,
    },
    Country {
        code: "PL",
        dial: "48",
        trunk: "",
        national: 9..=9,
    },
    Country {
        code: "DE",
        dial: "49",
        trunk: "0",
        national: 5..=13,
    },
    Country {
        code: "AT",
        dial: "43",
        trunk: "0",
        national: 4..=13,
    },
    Country {
        code: "HU",
        dial: "36",
        trunk: "06",
        national: 8..=9,
    },
    Country {
        code: "CH",
        dial: "41",
        trunk: "0",
        national: 9..=9,
    },
    Country {
        code: "FR",
        dial: "33",
        trunk: "0",
        national: 9..=9,
    },
    Country {
        code: "NL",
        dial: "31",
        trunk: "0",
        national: 9..=9,
    },
    Country {
        code: "GB",
        dial: "44",
        trunk: "0",
        national: 9..=10,
    },
];

fn country_by_code(code: &str) -> Option<&'static Country> {
    COUNTRIES.iter().find(|c| c.code.eq_ignore_ascii_case(code))
}

/// The country a website most likely addresses, for reading national phone numbers: by the TLD
/// (`.cz` → CZ, `.sk` → SK, `.pl` → PL, `.de` → DE, `.at` → AT, `.hu` → HU), otherwise by the
/// region of `lang` (`cs-CZ` → CZ, `de_AT` → AT), when the crawler knows that country's dialling
/// rules; otherwise `None`. A language without a region is no country.
pub fn site_country(host: &str, lang: &str) -> Option<&'static str> {
    let host = host.trim().to_ascii_lowercase();
    let host = host.split(':').next().unwrap_or_default().trim_end_matches('.');
    let by_tld = match host.rsplit('.').next() {
        Some("cz") => Some("CZ"),
        Some("sk") => Some("SK"),
        Some("pl") => Some("PL"),
        Some("de") => Some("DE"),
        Some("at") => Some("AT"),
        Some("hu") => Some("HU"),
        _ => None,
    };
    if by_tld.is_some() {
        return by_tld;
    }
    let region = lang
        .trim()
        .split(['-', '_'])
        .skip(1)
        .find(|part| part.len() == 2 && part.chars().all(|c| c.is_ascii_alphabetic()))?;
    country_by_code(region).map(|country| country.code)
}

// ---------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------

/// Operators in front of a number, with their canonical form in the key. A longer form comes
/// before the forms it starts with.
const OPERATORS: &[(&str, &str)] = &[
    ("approximately", "~"),
    ("starting at", "from "),
    ("maximálně", "upto "),
    ("minimálně", "from "),
    ("přibližně", "~"),
    ("less than", "<"),
    ("more than", ">"),
    ("méně než", "<"),
    ("více než", ">"),
    ("nejméně", "from "),
    ("nejvýše", "upto "),
    ("approx.", "~"),
    ("at least", "from "),
    ("at most", "upto "),
    ("zhruba", "~"),
    ("around", "~"),
    ("about", "~"),
    ("under", "<"),
    ("up to", "upto "),
    ("až do", "upto "),
    ("from", "from "),
    ("over", ">"),
    ("přes", ">"),
    ("min.", "from "),
    ("max.", "upto "),
    ("cca.", "~"),
    ("cca", "~"),
    ("ca.", "~"),
    ("asi", "~"),
    ("nad", ">"),
    ("pod", "<"),
    ("od", "from "),
    ("do", "upto "),
    ("až", "upto "),
    ("<=", "≤"),
    (">=", "≥"),
    ("≤", "≤"),
    ("≥", "≥"),
    ("<", "<"),
    (">", ">"),
    ("~", "~"),
];

/// Currencies and the percent sign after a number, with their canonical unit. A longer form comes
/// before the forms it starts with.
const UNITS: &[(&str, &str)] = &[
    ("per cent", "%"),
    ("procenta", "%"),
    ("procento", "%"),
    ("procent", "%"),
    ("percent", "%"),
    ("koruny", "CZK"),
    ("koruna", "CZK"),
    ("korun", "CZK"),
    ("euro", "EUR"),
    ("eura", "EUR"),
    ("czk", "CZK"),
    ("eur", "EUR"),
    ("usd", "USD"),
    ("gbp", "GBP"),
    ("chf", "CHF"),
    ("fr.", "CHF"),
    ("pln", "PLN"),
    ("huf", "HUF"),
    ("kč", "CZK"),
    ("zł", "PLN"),
    ("ft", "HUF"),
    ("%", "%"),
    ("€", "EUR"),
    ("$", "USD"),
    ("£", "GBP"),
];

/// Currencies that may stand in front of a number.
const PREFIX_CURRENCIES: &[(&str, &str)] = &[
    ("czk", "CZK"),
    ("eur", "EUR"),
    ("usd", "USD"),
    ("gbp", "GBP"),
    ("chf", "CHF"),
    ("pln", "PLN"),
    ("huf", "HUF"),
    ("kč", "CZK"),
    ("€", "EUR"),
    ("$", "USD"),
    ("£", "GBP"),
];

/// Time bases a value may carry, with their canonical form. A longer form comes before the forms
/// it contains; a token never matches inside a word.
const BASES: &[(&str, &str)] = &[
    ("za měsíc", "month"),
    ("per month", "month"),
    ("a month", "month"),
    ("měsíčně", "month"),
    ("mesicne", "month"),
    ("/měsíc", "month"),
    ("/mesic", "month"),
    ("/month", "month"),
    ("monthly", "month"),
    ("/měs.", "month"),
    ("/měs", "month"),
    ("/mo.", "month"),
    ("/mo", "month"),
    ("per annum", "year"),
    ("per year", "year"),
    ("annually", "year"),
    ("za rok", "year"),
    ("a year", "year"),
    ("yearly", "year"),
    ("ročně", "year"),
    ("rocne", "year"),
    ("/year", "year"),
    ("p. a.", "year"),
    ("/rok", "year"),
    ("p.a.", "year"),
    ("/yr", "year"),
    ("per day", "day"),
    ("za den", "day"),
    ("a day", "day"),
    ("denně", "day"),
    ("daily", "day"),
    ("/den", "day"),
    ("/day", "day"),
    ("za týden", "week"),
    ("per week", "week"),
    ("a week", "week"),
    ("/týden", "week"),
    ("weekly", "week"),
    ("týdně", "week"),
    ("/week", "week"),
    ("za hodinu", "hour"),
    ("per hour", "hour"),
    ("an hour", "hour"),
    ("hourly", "hour"),
    ("/hour", "hour"),
    ("/hod.", "hour"),
    ("/hod", "hour"),
    ("/hr", "hour"),
    ("/h", "hour"),
];

/// Read `value` as a number without changing its meaning. `None` when the value does not start
/// with a number (after an optional operator and currency); otherwise a key
/// `num:{op}{value}:{unit}:{base}` (a range: `range:{op}{low}-{high}:{unit}:{base}`), followed by
/// `:{rest}` when other words remain, so they never merge different claims:
/// - thousands separators: spaces, NBSP, U+202F and `'`; when both `,` and `.` occur, the last one
///   is the decimal point; a single `,`/`.` followed by exactly 3 digits is read by `lang`
///   (decimal comma for cs/sk/de/pl/fr/…, decimal point for en/…) and is ambiguous without it;
/// - operators are kept (`<`, `≤`, `>`, `≥`, `od`/`from`, `do`/`up to`, `cca`/`about`, a trailing
///   `+`), and `od 290 do 390` is the range `290-390`;
/// - the unit: `%`; `Kč`/`CZK`/`,-` → `CZK`; `€`/`EUR`; `$`/`USD`; … otherwise the next word;
/// - the time base (`/měs.`, `měsíčně`, `per month`, `p.a.`, …) → `month`/`year`/`day`/`week`/`hour`.
///
/// An ambiguous or malformed number gives `Uncertain("num?:…")`.
pub fn parse_number(value: &str, lang: &str) -> Option<ValueKey> {
    let text = collapse_lower(&unify_dashes(&unify_spaces(value)));
    let uncertain = || Some(ValueKey::Uncertain(format!("num?:{text}")));
    let decimal = decimal_separator(lang);
    let mut rest = text.as_str();

    let mut op = take_word(&mut rest, OPERATORS);
    let mut unit = take_word(&mut rest, PREFIX_CURRENCIES);
    let negative = rest.starts_with('-') && rest.get(1..).is_some_and(starts_with_digit);
    if negative {
        rest = rest.get(1..).unwrap_or_default();
    }
    if !starts_with_digit(rest) {
        return None;
    }
    let (low, after) = split_numeral(rest);
    rest = after.trim_start();

    let mut high = None;
    for connector in ["-", "až ", "to ", "do "] {
        let Some(after) = rest.strip_prefix(connector) else {
            continue;
        };
        let mut after = after.trim_start();
        let currency = take_word(&mut after, PREFIX_CURRENCIES);
        if !starts_with_digit(after) || (currency.is_some() && unit.is_some() && currency != unit) {
            continue;
        }
        if currency.is_some() {
            unit = currency;
        }
        let (numeral, tail) = split_numeral(after);
        high = Some(numeral);
        rest = tail.trim_start();
        // "od 290 do 390" says the same as "290–390".
        if connector != "-" && op == Some("from ") {
            op = None;
        }
        break;
    }

    if let Some(after) = rest.strip_prefix(",-") {
        unit.get_or_insert("CZK");
        rest = after.trim_start();
    }
    if let Some(found) = take_word(&mut rest, UNITS) {
        match unit {
            None => unit = Some(found),
            Some(known) if known == found => {}
            // A second, different currency ("290 € (7 250 Kč)") is not one value.
            Some(_) => return uncertain(),
        }
    }
    if let Some(after) = rest.strip_prefix('+') {
        op.get_or_insert("≥");
        rest = after.trim_start();
    }
    if starts_with_digit(rest) {
        return uncertain();
    }

    let (base, rest) = take_base(rest);
    let mut rest = rest.trim_matches(|c: char| c.is_whitespace() || matches!(c, '.' | ',' | ';' | ':' | '(' | ')'));
    let mut unit = unit.map(str::to_string).unwrap_or_default();
    if unit.is_empty() && !rest.is_empty() {
        let end = rest.find(|c: char| c.is_whitespace() || c == '/').unwrap_or(rest.len());
        unit = rest
            .get(..end)
            .unwrap_or_default()
            .trim_end_matches(['.', ','])
            .to_string();
        rest = rest
            .get(end..)
            .unwrap_or_default()
            .trim_start_matches(['/', ' '])
            .trim_end();
    }

    let Numeral::Value(low) = parse_numeral(low, decimal) else {
        return uncertain();
    };
    let low = if negative { format!("-{low}") } else { low };
    let op = op.unwrap_or_default();
    let base = base.unwrap_or_default();
    let mut key = match high.map(|high| parse_numeral(high, decimal)) {
        None => format!("num:{op}{low}:{unit}:{base}"),
        Some(Numeral::Value(high)) => format!("range:{op}{low}-{high}:{unit}:{base}"),
        Some(_) => return uncertain(),
    };
    if !rest.is_empty() {
        key.push(':');
        key.push_str(rest);
    }
    Some(ValueKey::Exact(key))
}

/// Every number written in `text`, in order, read with the separators of `lang` (see
/// `parse_number`). An ambiguous number (`1.290` without a language) yields both readings;
/// a malformed one (`25.9.2026`) yields its digit groups. Signs are not read.
pub fn numbers_in(text: &str, lang: &str) -> Vec<f64> {
    let decimal = decimal_separator(lang);
    let mut out = Vec::new();
    for token in numerals(text) {
        match parse_numeral(token, decimal) {
            Numeral::Value(value) => out.extend(value.parse::<f64>().ok()),
            Numeral::Ambiguous(decimal_reading, thousands_reading) => {
                out.extend(decimal_reading.parse::<f64>().ok());
                out.extend(thousands_reading.parse::<f64>().ok());
            }
            Numeral::Invalid => out.extend(
                token
                    .split(|c: char| !c.is_ascii_digit())
                    .filter(|run| !run.is_empty())
                    .filter_map(|run| run.parse::<f64>().ok()),
            ),
        }
    }
    out
}

/// A number as written: its canonical decimal form, two readings, or no valid number at all.
enum Numeral {
    Value(String),
    /// A single `,`/`.` + 3 digits without a language: (decimal reading, thousands reading).
    Ambiguous(String, String),
    Invalid,
}

/// The numerals of `text` (digits with their thousands and decimal separators), in order.
fn numerals(text: &str) -> impl Iterator<Item = &str> {
    numeral_spans(text).filter_map(|span| text.get(span))
}

/// The byte ranges of the numerals of `text`, in order.
fn numeral_spans(text: &str) -> impl Iterator<Item = Range<usize>> + '_ {
    let mut next = 0;
    text.char_indices().filter_map(move |(at, ch)| {
        if at < next || !ch.is_ascii_digit() {
            return None;
        }
        next = scan_numeral(text, at);
        Some(at..next)
    })
}

/// Split `s` (starting with a digit) into its leading numeral and the rest.
fn split_numeral(s: &str) -> (&str, &str) {
    let end = scan_numeral(s, 0);
    (s.get(..end).unwrap_or_default(), s.get(end..).unwrap_or_default())
}

fn is_group_separator(c: char) -> bool {
    matches!(c, ' ' | '\u{a0}' | '\u{202f}' | '\u{2009}' | '\u{2007}' | '\'' | '’')
}

fn digits_at(s: &str, at: usize) -> usize {
    s.get(at..)
        .map_or(0, |rest| rest.bytes().take_while(u8::is_ascii_digit).count())
}

/// The end (byte index) of the numeral starting with the digit at `start`: digits, `.`/`,`
/// followed by a digit, and — before any `.`/`,` — a group separator followed by exactly three
/// digits after a group of at most three.
fn scan_numeral(s: &str, start: usize) -> usize {
    let mut end = start + digits_at(s, start);
    let mut run = end - start;
    let mut point_seen = false;
    while let Some(sep) = s.get(end..).and_then(|rest| rest.chars().next()) {
        let after = end + sep.len_utf8();
        let next = digits_at(s, after);
        if matches!(sep, '.' | ',') && next > 0 {
            point_seen = true;
        } else if !(is_group_separator(sep) && !point_seen && run <= 3 && next == 3) {
            break;
        }
        run = next;
        end = after + next;
    }
    end
}

fn parse_numeral(token: &str, decimal: Option<char>) -> Numeral {
    // The digit runs, and the separator after each run but the last.
    let mut runs: Vec<&str> = Vec::new();
    let mut seps: Vec<char> = Vec::new();
    let mut rest = token;
    loop {
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        let Some((run, tail)) = rest.split_at_checked(digits).filter(|_| digits > 0) else {
            return Numeral::Invalid;
        };
        runs.push(run);
        let Some(sep) = tail.chars().next() else { break };
        seps.push(sep);
        rest = tail.get(sep.len_utf8()..).unwrap_or_default();
    }
    let run = |at: usize| runs.get(at).copied().unwrap_or_default();
    let points: Vec<(usize, char)> = seps
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, sep)| matches!(sep, '.' | ','))
        .collect();
    let decimal_at = match points.as_slice() {
        [] => None,
        [(at, kind)] => {
            let (before, after) = (run(*at), run(at + 1));
            if seps.iter().any(|sep| is_group_separator(*sep))
                || after.len() != 3
                || before.len() > 3
                || before.starts_with('0')
            {
                Some(*at)
            } else {
                match decimal {
                    Some(d) if d == *kind => Some(*at),
                    Some(_) => None,
                    None => {
                        let thousands = canonical(&format!("{before}{after}"), "");
                        return Numeral::Ambiguous(canonical(before, after), thousands);
                    }
                }
            }
        }
        [.., (last, kind)] => {
            // Both kinds: the last one is the decimal point and must occur once.
            let same_kind = points.iter().filter(|(_, sep)| sep == kind).count();
            match (same_kind == points.len(), same_kind) {
                (true, _) => None,
                (false, 1) => Some(*last),
                (false, _) => return Numeral::Invalid,
            }
        }
    };
    if decimal_at.is_some_and(|at| at + 1 != seps.len()) {
        return Numeral::Invalid;
    }
    let integer = runs
        .get(..decimal_at.map_or(runs.len(), |at| at + 1))
        .unwrap_or_default();
    if let [first, groups @ ..] = integer
        && !groups.is_empty()
        && (first.len() > 3 || first.starts_with('0') || groups.iter().any(|group| group.len() != 3))
    {
        return Numeral::Invalid;
    }
    let fraction = decimal_at.map_or("", |at| run(at + 1));
    Numeral::Value(canonical(&integer.concat(), fraction))
}

fn canonical(integer: &str, fraction: &str) -> String {
    let integer = integer.trim_start_matches('0');
    let integer = if integer.is_empty() { "0" } else { integer };
    let fraction = fraction.trim_end_matches('0');
    if fraction.is_empty() {
        integer.to_string()
    } else {
        format!("{integer}.{fraction}")
    }
}

/// The decimal separator of a language tag, when the crawler knows it.
fn decimal_separator(lang: &str) -> Option<char> {
    let lang = lang.trim().to_ascii_lowercase();
    let mut parts = lang.split(['-', '_']);
    let primary = parts.next().unwrap_or_default();
    let region = parts.find(|part| part.len() == 2 && part.chars().all(|c| c.is_ascii_alphabetic()));
    match (primary, region) {
        ("de" | "fr" | "it" | "rm", Some("ch" | "li")) => Some('.'),
        ("es", Some("mx" | "us" | "pr" | "do" | "gt" | "hn" | "ni" | "pa" | "sv")) => Some('.'),
        (
            "cs" | "sk" | "pl" | "de" | "fr" | "es" | "it" | "pt" | "nl" | "hu" | "ru" | "uk" | "be" | "bg" | "sr"
            | "hr" | "bs" | "sl" | "ro" | "sv" | "da" | "nb" | "nn" | "no" | "fi" | "et" | "lv" | "lt" | "el" | "tr"
            | "ca" | "gl" | "eu" | "is" | "id" | "vi",
            _,
        ) => Some(','),
        ("en" | "ja" | "zh" | "ko" | "he" | "th" | "hi" | "ga" | "mt" | "ms" | "tl", _) => Some('.'),
        _ => None,
    }
}

/// Take the first of `words` that `rest` starts with (a word must not run on into a letter) and
/// return its canonical form; `rest` then continues after it, without leading spaces.
fn take_word(rest: &mut &str, words: &[(&str, &'static str)]) -> Option<&'static str> {
    let (word, canonical) = words.iter().find(|(word, _)| {
        rest.strip_prefix(word)
            .is_some_and(|after| !word.ends_with(char::is_alphabetic) || !after.starts_with(char::is_alphabetic))
    })?;
    *rest = rest.get(word.len()..).unwrap_or_default().trim_start();
    Some(canonical)
}

/// Find a time base in `rest` (not inside a word) and return it with the rest without it.
fn take_base(rest: &str) -> (Option<&'static str>, String) {
    for (token, base) in BASES {
        let mut from = 0;
        while let Some(found) = rest.get(from..).and_then(|tail| tail.find(token)) {
            let start = from + found;
            let end = start + token.len();
            let before = rest.get(..start).unwrap_or_default().chars().next_back();
            let after = rest.get(end..).unwrap_or_default().chars().next();
            let free_before = token.starts_with('/') || !before.is_some_and(char::is_alphabetic);
            if free_before && !after.is_some_and(char::is_alphanumeric) {
                let without = format!(
                    "{} {}",
                    rest.get(..start).unwrap_or_default(),
                    rest.get(end..).unwrap_or_default()
                );
                return (Some(base), collapse_lower(&without));
            }
            from = end;
        }
    }
    (None, rest.to_string())
}

// ---------------------------------------------------------------------------
// Dates
// ---------------------------------------------------------------------------

fn month_number(name: &str) -> Option<u32> {
    let month = match name {
        "ledna" => 1,
        "února" => 2,
        "března" => 3,
        "dubna" => 4,
        "května" => 5,
        "června" => 6,
        "července" => 7,
        "srpna" => 8,
        "září" => 9,
        "října" => 10,
        "listopadu" => 11,
        "prosince" => 12,
        _ => match name.get(..3)? {
            "jan" => 1,
            "feb" => 2,
            "mar" => 3,
            "apr" => 4,
            "may" => 5,
            "jun" => 6,
            "jul" => 7,
            "aug" => 8,
            "sep" => 9,
            "oct" => 10,
            "nov" => 11,
            "dec" => 12,
            _ => return None,
        },
    };
    Some(month)
}

/// Every full date (with a year) in normalized text, with its byte range.
fn dates_in(normalized: &str) -> Vec<(Range<usize>, NaiveDate)> {
    let mut out = Vec::new();
    for (re, order) in DATE_PATTERNS.iter() {
        for caps in re.captures_iter(normalized) {
            let (Some(whole), Some(a), Some(b), Some(c)) = (caps.get(0), caps.get(1), caps.get(2), caps.get(3)) else {
                continue;
            };
            let (day, month, year) = match order {
                DateOrder::DayMonthYear => (a.as_str(), b.as_str(), c.as_str()),
                DateOrder::MonthDayYear => (b.as_str(), a.as_str(), c.as_str()),
                DateOrder::YearMonthDay => (c.as_str(), b.as_str(), a.as_str()),
            };
            let month = month.parse::<u32>().ok().or_else(|| month_number(month));
            let (Ok(day), Some(month), Ok(year)) = (day.parse::<u32>(), month, year.parse::<i32>()) else {
                continue;
            };
            if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
                out.push((whole.range(), date));
            }
        }
    }
    out
}

/// The date when the whole value (ignoring a final full stop) is one.
fn parse_date(value: &str) -> Option<NaiveDate> {
    let normalized = normalize_for_match(value).text;
    let normalized = normalized.trim_end_matches('.').trim_end();
    dates_in(normalized)
        .into_iter()
        .find(|(range, _)| range.start == 0 && range.end == normalized.len())
        .map(|(_, date)| date)
}

/// The full dates (with a year) mentioned anywhere in `text`: `D. M. YYYY`, `D.M.YYYY`,
/// `YYYY-MM-DD`, Czech genitive month names (`25. září 2026`) and English month names
/// (`September 25, 2026`, `25 September 2026`). Impossible dates are skipped.
pub fn date_mentions(text: &str) -> HashSet<NaiveDate> {
    dates_in(&normalize_for_match(text).text)
        .into_iter()
        .map(|(_, date)| date)
        .collect()
}

// ---------------------------------------------------------------------------
// Evidence snippets
// ---------------------------------------------------------------------------

/// The crawler's own text around a verified value: the whole block when it has at most
/// `max_chars` characters (callers use 300), otherwise a window of at most `max_chars` characters
/// around `value_span` (a byte range of `block_text`, e.g. from `find_token_bounded`), cut on word
/// boundaries where possible and marked with `…` where text was left out. A span that is not a
/// valid range of `block_text` is treated as the start of the block.
pub fn snippet_of(block_text: &str, value_span: (usize, usize), max_chars: usize) -> String {
    snippet_with_span(block_text, value_span, max_chars).0
}

/// `snippet_of`, together with the byte range of the value within the snippet (`(0, 0)` when the
/// span was not a valid range of `block_text`), so a later, shorter snippet can be cut around the
/// same copy of a value that occurs more than once.
pub fn snippet_with_span(block_text: &str, value_span: (usize, usize), max_chars: usize) -> (String, (usize, usize)) {
    let (start, end) = match value_span {
        (start, end) if start <= end && block_text.get(start..end).is_some() => (start, end),
        _ => (0, 0),
    };
    if block_text.chars().count() <= max_chars {
        return (block_text.to_string(), (start, end));
    }
    let value = block_text.get(start..end).unwrap_or_default();
    let value_chars = value.chars().count();
    // Room for the context and the two ellipses.
    let Some(budget) = max_chars.checked_sub(value_chars + 2) else {
        let kept: String = value.chars().take(max_chars.saturating_sub(1)).collect();
        return if max_chars == 0 {
            (String::new(), (0, 0))
        } else {
            let len = kept.len();
            (format!("{kept}…"), (0, len))
        };
    };
    let before = block_text.get(..start).unwrap_or_default();
    let after = block_text.get(end..).unwrap_or_default();
    let (left_avail, right_avail) = (before.chars().count(), after.chars().count());
    let mut left = left_avail.min(budget / 2);
    let right = right_avail.min(budget - left);
    left = left_avail.min(budget - right);

    let mut from = before.char_indices().nth(left_avail - left).map_or(start, |(at, _)| at);
    let mut to = end + after.char_indices().nth(right).map_or(after.len(), |(at, _)| at);
    let (cut_left, cut_right) = (from > 0, to < block_text.len());
    let split_word = |at: usize| {
        let prev = block_text.get(..at).unwrap_or_default().chars().next_back();
        let next = block_text.get(at..).unwrap_or_default().chars().next();
        prev.is_some_and(|c| !c.is_whitespace()) && next.is_some_and(|c| !c.is_whitespace())
    };
    if cut_left
        && split_word(from)
        && let Some(space) = block_text
            .get(from..start)
            .and_then(|context| context.find(char::is_whitespace))
    {
        from += space;
    }
    if cut_right
        && split_word(to)
        && let Some(space) = block_text
            .get(end..to)
            .and_then(|context| context.rfind(char::is_whitespace))
    {
        to = end + space;
    }
    let window = block_text.get(from..to).unwrap_or_default();
    let kept = window.trim();
    let kept_from = from + (window.len() - window.trim_start().len());
    let prefix = if cut_left { "…" } else { "" };
    let at = prefix.len() + start.saturating_sub(kept_from);
    let snippet = format!("{prefix}{kept}{}", if cut_right { "…" } else { "" });
    (snippet, (at, at + (end - start)))
}

// ---------------------------------------------------------------------------
// Small text helpers
// ---------------------------------------------------------------------------

fn digit_string(s: &str) -> String {
    s.chars().filter(char::is_ascii_digit).collect()
}

fn starts_with_digit(s: &str) -> bool {
    s.starts_with(|c: char| c.is_ascii_digit())
}

/// Lowercase and collapse whitespace (every Unicode space, incl. NBSP) into single spaces.
fn collapse_lower(s: &str) -> String {
    s.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")
}

fn unify_spaces(s: &str) -> String {
    s.chars().map(|c| if c.is_whitespace() { ' ' } else { c }).collect()
}

fn unify_dashes(s: &str) -> String {
    s.chars()
        .map(|c| if matches!(c, '–' | '—' | '−') { '-' } else { c })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn key(hint: ValueHint, value: &str, lang: &str, country: Option<&str>) -> ValueKey {
        value_key(hint, value, "", lang, country)
    }

    fn exact(k: &ValueKey) -> bool {
        matches!(k, ValueKey::Exact(_))
    }

    // --- normalize_for_match / contains_normalized / find_token_bounded ---

    #[test]
    fn normalization_lowercases_unifies_spaces_and_drops_emphasis() {
        let n = normalize_for_match("  Cena:\u{a0}**1\u{202f}290\u{2009}Kč**\t–\r\n_akce_ `A\\B` ");
        assert_eq!(n.text, "cena: 1 290 kč - akce ab");
    }

    #[test]
    fn offsets_map_back_to_the_original_through_nbsp_and_emphasis() {
        let original = "Tarif Basic: **1\u{a0}290\u{a0}Kč** měsíčně";
        let n = normalize_for_match(original);
        let start = n.text.find("1 290 kč").expect("normalized match");
        let (a, b) = n.original_span(start, start + "1 290 kč".len()).expect("a span");
        assert_eq!(&original[a..b], "1\u{a0}290\u{a0}Kč");
        assert_eq!(n.original_span(0, n.text.len() + 1), None, "out of range");
        assert_eq!(n.original_span(3, 3), None, "empty range");

        let span = find_token_bounded(original, "1 290 Kč").expect("found");
        assert_eq!(&original[span.0..span.1], "1\u{a0}290\u{a0}Kč");
        // Uppercase that lowercases into more bytes still maps back to the whole char.
        let upper = "ŽLUŤOUČKÝ KŮŇ";
        let span = find_token_bounded(upper, "kůň").expect("found");
        assert_eq!(&upper[span.0..span.1], "KŮŇ");
    }

    #[test]
    fn contains_normalized_ignores_case_spacing_and_emphasis() {
        assert!(contains_normalized(
            "Zákaznická linka: **800\u{a0}123\u{a0}456** (po–pá)",
            "zákaznická linka: 800 123 456 (po-pá)"
        ));
        assert!(!contains_normalized("Zákaznická linka 800 123 456", "800 123 465"));
        assert!(!contains_normalized("anything", ""));
        assert!(!contains_normalized("anything", " \u{a0} "));
    }

    #[test]
    fn a_value_never_matches_inside_a_longer_number_or_word() {
        assert_eq!(find_token_bounded("Sleva 15 % na vše", "5 %"), None);
        let text = "Sleva 15 % nebo 5 % navíc";
        let (a, b) = find_token_bounded(text, "5 %").expect("the standalone value");
        assert_eq!(&text[a..b], "5 %");
        assert_eq!(a, text.find("5 % navíc").unwrap());
        // Decimal and thousands separators joined to a digit.
        assert_eq!(find_token_bounded("sazba 4,59 %", "59 %"), None);
        assert_eq!(find_token_bounded("sazba 4,59 %", "4"), None);
        assert_eq!(find_token_bounded("cena 1.290 Kč", "290 Kč"), None);
        assert_eq!(find_token_bounded("cena 1 290.50", "1 290"), None);
        // Letters around the match.
        assert_eq!(find_token_bounded("290 Kčs", "290 Kč"), None);
        assert_eq!(find_token_bounded("A290 Kč", "290 Kč"), None);
        // Sentence punctuation that is not joined to a digit is fine.
        assert!(find_token_bounded("Cena je 290 Kč.", "290 Kč").is_some());
        assert!(find_token_bounded("Rok 2025. 5 % sleva", "5 %").is_some());
        assert!(find_token_bounded("(5 %)", "5 %").is_some());
        assert!(find_token_bounded("E-mail:info@example.cz.", "info@example.cz").is_some());
        assert_eq!(find_token_bounded("anything", ""), None);
    }

    #[test]
    fn a_number_never_matches_across_a_thousands_space() {
        // A space (or an apostrophe) between digits may be a thousands separator.
        assert_eq!(find_token_bounded("Premium 1 290 Kč", "290 Kč"), None);
        assert_eq!(find_token_bounded("Premium 1\u{a0}290 Kč", "290 Kč"), None);
        assert_eq!(find_token_bounded("Premium 1\u{202f}290 Kč", "290 Kč"), None);
        assert_eq!(find_token_bounded("Preis 1'290 CHF", "290 CHF"), None);
        assert_eq!(find_token_bounded("Volejte 800 123 456", "123 456"), None);
        assert_eq!(find_token_bounded("Volejte 800 123 456", "800 123"), None);
        assert_eq!(find_token_bounded("Celkem 1 290 Kč", "1"), None);
        // The whole number, and a number next to words, are still tokens.
        assert!(find_token_bounded("Premium 1 290 Kč", "1 290 Kč").is_some());
        assert!(find_token_bounded("Volejte 800 123 456 nebo pište", "800 123 456").is_some());
        assert!(find_token_bounded("Basic 290 Kč, Premium 1 290 Kč", "290 Kč").is_some());
        let text = "Premium 1 290 Kč, Basic 290 Kč";
        let (a, b) = find_token_bounded(text, "290 Kč").expect("the standalone price");
        assert_eq!(a, text.rfind("290 Kč").unwrap());
        assert_eq!(&text[a..b], "290 Kč");
    }

    #[test]
    fn labels_normalize_case_diacritics_and_punctuation() {
        assert_eq!(normalize_label("  Zákaznická  linka! "), "zakaznicka linka");
        assert_eq!(normalize_label("ZÁKAZNICKÁ\u{a0}LINKA"), "zakaznicka linka");
        assert_eq!(normalize_label("Hypotéka – úrok (od)"), "hypoteka urok od");
        assert_eq!(normalize_label("Customer-line / E-mail"), "customer line e mail");
        assert_eq!(normalize_label("Tarif 2 · cena/měs."), "tarif 2 cena mes");
        assert_eq!(normalize_label("Straße Größe"), "straße große");
        assert_eq!(normalize_label(" -– "), "");
    }

    #[test]
    fn labels_keep_a_plus_apart() {
        assert_eq!(normalize_label("Tarif S+"), "tarif s plus");
        assert_ne!(normalize_label("Tarif S+"), normalize_label("Tarif S"));
        assert_eq!(normalize_label("Premium+ plán"), "premium plus plan");
        assert_eq!(normalize_label("Tarif S plus"), normalize_label("Tarif S+"));
    }

    #[test]
    fn an_operator_before_a_value_is_part_of_it() {
        let extended = |text: &str, value: &str| {
            let start = text.find(value).expect("value");
            let (a, b) = extend_over_operator(text, (start, start + value.len()));
            text[a..b].to_string()
        };
        assert_eq!(extended("Půjčka od 18 let.", "18 let"), "od 18 let");
        assert_eq!(extended("Půjčka do 18 let.", "18 let"), "do 18 let");
        assert_eq!(extended("Cena Od 290 Kč", "290 Kč"), "Od 290 Kč");
        assert_eq!(extended("Vyřídíme až do 30 dnů", "30 dnů"), "až do 30 dnů");
        assert_eq!(
            extended("Obsloužili jsme více než 1 000 zákazníků", "1 000"),
            "více než 1 000"
        );
        assert_eq!(extended("Věk < 18", "18"), "< 18");
        assert_eq!(extended("Věk ≥18", "18"), "≥18");
        assert_eq!(extended("Vstup 18+ let", "18"), "18+");
        assert_eq!(extended("Starting at $29", "$29"), "Starting at $29");
        // Not an operator: a word ending like one, or no operator at all.
        assert_eq!(extended("Metod 290 Kč", "290 Kč"), "290 Kč");
        assert_eq!(extended("Kód 290 Kč", "290 Kč"), "290 Kč");
        assert_eq!(extended("290 Kč", "290 Kč"), "290 Kč");
        // The extended value reads with its operator.
        assert_ne!(
            parse_number(&extended("Půjčka od 18 let.", "18 let"), "cs"),
            parse_number(&extended("Půjčka do 18 let.", "18 let"), "cs")
        );
        // An out-of-range span is returned as it is.
        assert_eq!(extend_over_operator("abc", (2, 9)), (2, 9));
    }

    #[test]
    fn a_sign_and_the_other_end_of_a_range_are_part_of_a_number() {
        let extended = |text: &str, value: &str| {
            let start = text.rfind(value).expect("value");
            let (a, b) = extend_number_span(text, (start, start + value.len()));
            text[a..b].to_string()
        };
        // A sign joined to the number.
        assert_eq!(extended("Annual return -5 %", "5 %"), "-5 %");
        assert_eq!(extended("Annual return −5 %", "5 %"), "−5 %");
        assert_eq!(extended("(-5 %)", "5 %"), "-5 %");
        // The other end of a range, with or without spaces and units.
        assert_eq!(extended("Interest 5–10 %", "10 %"), "5–10 %");
        assert_eq!(extended("Interest 1 290 - 1 490 Kč", "1 490 Kč"), "1 290 - 1 490 Kč");
        assert_eq!(extended("Price 10 EUR – 20 EUR", "20 EUR"), "10 EUR – 20 EUR");
        assert_eq!(extended("Interest 5–10 % p.a.", "5"), "5–10 %");
        assert_eq!(extended("Doručení 1–2 dny", "1"), "1–2");
        // An operator before the range still belongs to it.
        assert_eq!(extended("Úrok od 5–10 %", "10 %"), "od 5–10 %");
        // Not a sign or a range: a dash between words, a separating dash, a hyphenated word.
        assert_eq!(extended("Sleva – 5 %", "5 %"), "5 %");
        assert_eq!(extended("COVID-19", "19"), "19");
        assert_eq!(extended("Wi-Fi 6", "6"), "6");
        assert_eq!(extended("Cena 290 Kč", "290 Kč"), "290 Kč");
        // The extended values read with their meaning.
        assert_eq!(
            parse_number(&extended("Annual return -5 %", "5 %"), "en"),
            Some(ValueKey::Exact("num:-5:%:".to_string()))
        );
        assert_eq!(
            parse_number(&extended("Interest 5–10 %", "10 %"), "en"),
            Some(ValueKey::Exact("range:5-10:%:".to_string()))
        );
        // An out-of-range span is returned as it is.
        assert_eq!(extend_number_span("abc", (2, 9)), (2, 9));
    }

    #[test]
    fn a_snippet_reports_where_the_value_is() {
        let block = format!(
            "Basic 290 Kč. {} Premium 290 Kč měsíčně. {}",
            "a ".repeat(200),
            "b ".repeat(200)
        );
        let start = block.rfind("290 Kč").unwrap();
        let (snippet, (a, b)) = snippet_with_span(&block, (start, start + "290 Kč".len()), 60);
        assert_eq!(snippet, snippet_of(&block, (start, start + "290 Kč".len()), 60));
        assert_eq!(&snippet[a..b], "290 Kč");
        assert!(snippet[..a].contains("Premium"), "{snippet}");
        let short = "Basic 290 Kč, Premium 290 Kč";
        let start = short.rfind("290 Kč").unwrap();
        let (whole, span) = snippet_with_span(short, (start, start + "290 Kč".len()), 300);
        assert_eq!((whole.as_str(), span), (short, (start, start + "290 Kč".len())));
    }

    #[test]
    fn a_quoted_value_is_located_inside_its_quote() {
        let block = "Basic 290 Kč měsíčně, Premium 1 290 Kč měsíčně, Business 290 Kč ročně";
        // The value inside the quoted occurrence, not the first one in the block.
        let (a, b) = locate_quoted_value(block, "Business 290 Kč ročně", "290 Kč").expect("located");
        assert_eq!(a, block.rfind("290 Kč").unwrap());
        assert_eq!(&block[a..b], "290 Kč");
        // Token boundaries are checked in the block, not at the quote's edges.
        assert_eq!(locate_quoted_value(block, "290 Kč měsíčně, Business", "290 Kč"), None);
        assert_eq!(locate_quoted_value(block, "Premium 1 290 Kč", "290 Kč"), None);
        // A value elsewhere in the block, a quote that is not in the block, empty inputs.
        assert_eq!(locate_quoted_value(block, "Basic 290 Kč", "1 290 Kč"), None);
        assert_eq!(locate_quoted_value(block, "Basic 390 Kč", "390 Kč"), None);
        assert_eq!(locate_quoted_value(block, "", "290 Kč"), None);
        assert_eq!(locate_quoted_value(block, "Basic 290 Kč", ""), None);
        // Normalization (case, NBSP, emphasis) on both sides; the span is in the original block.
        let block = "Zákaznická linka: **800\u{a0}123\u{a0}456** (po–pá)";
        let (a, b) = locate_quoted_value(block, "zákaznická linka: 800 123 456", "800 123 456").expect("located");
        assert_eq!(&block[a..b], "800\u{a0}123\u{a0}456");
        // The quote occurs twice; the value is a token only in the second occurrence.
        let block = "Sleva 15 % a 5 % navíc";
        let (a, _) = locate_quoted_value(block, "5 %", "5 %").expect("located");
        assert_eq!(a, block.rfind("5 %").unwrap());
    }

    // --- fact_signals ---

    #[test]
    fn fact_signals_detect_the_kinds_of_facts() {
        let s = fact_signals("Cena 1 290 Kč, sleva 15 %");
        assert!(s.money && s.percent && !s.phone && !s.email);
        assert!(fact_signals("od 290,- měsíčně").money);
        assert!(fact_signals("€ 29.90").money);
        assert!(fact_signals("only $5 today").money);
        assert!(fact_signals("Zákaznická linka +420 800 123 456").phone);
        assert!(fact_signals("Volejte 800 123 456").phone);
        assert!(fact_signals("Napište na Info_Desk@example.cz").email);
        assert!(fact_signals("Platnost do 31. 12. 2026").date);
        assert!(fact_signals("Updated on September 25, 2026").date);
        assert!(fact_signals("Akce končí 25. září").date);
        assert!(fact_signals("Example s.r.o., IČO 12345678").ids);
        assert!(fact_signals("IČ: 12345678, DIČ: CZ12345678").ids);
        assert!(fact_signals("Účet 123456789/0100").ids);
        assert!(fact_signals("IBAN CZ65 0800 0000 1920 0014 5399").ids);
        assert!(fact_signals("Po–Pá 8:00–17:00").hours);
        assert!(fact_signals("po-pá 8-18").hours);
        assert_eq!(fact_signals("Tel. 800 123 456").digits, 9);
        assert_eq!(
            fact_signals("We love our customers and our work."),
            Signals::default(),
            "a plain sentence has no signal"
        );
        assert!(!fact_signals("Copyright 2026").phone);
        let dated = fact_signals("31.12.2026 10:00");
        assert!(dated.date && !dated.phone, "{dated:?}");
    }

    // --- value_key: phones and e-mails ---

    #[test]
    fn international_phones_keep_their_country_code() {
        let cz = key(ValueHint::Contact, "+420 800 123 456", "cs", None);
        let sk = key(ValueHint::Contact, "+421 800 123 456", "cs", None);
        assert!(exact(&cz) && exact(&sk));
        assert_ne!(cz, sk, "+420 is not +421");
        assert_eq!(cz, ValueKey::Exact("tel:+420800123456".to_string()));
        assert_eq!(key(ValueHint::Contact, "00420 800 123 456", "", None), cz);
        assert_eq!(key(ValueHint::Contact, "+420800123456", "", None), cz);
        assert_eq!(key(ValueHint::Contact, "tel.: +420 800-123-456 (zdarma)", "", None), cz);
        assert_eq!(
            key(ValueHint::Contact, "+49 (0)30 1234567", "", None),
            key(ValueHint::Contact, "+49 30 1234567", "", None),
            "the trunk (0) after a country code is dropped"
        );
    }

    #[test]
    fn national_phones_need_the_site_country() {
        let intl = key(ValueHint::Contact, "+420 800 123 456", "cs", None);
        assert_eq!(key(ValueHint::Contact, "800 123 456", "cs", Some("CZ")), intl);
        let unknown = key(ValueHint::Contact, "800 123 456", "cs", None);
        assert_eq!(unknown, ValueKey::Uncertain("tel-national:800123456".to_string()));
        assert_ne!(unknown, intl);
        // Trunk prefixes are dropped the way the country dials.
        assert_eq!(
            key(ValueHint::Contact, "0905 123 456", "sk", Some("SK")),
            key(ValueHint::Contact, "+421 905 123 456", "sk", None)
        );
        assert_eq!(
            key(ValueHint::Contact, "030 1234567", "de", Some("DE")),
            key(ValueHint::Contact, "+49 30 1234567", "de", None)
        );
        // A national number that does not fit the country stays unresolved.
        assert!(!exact(&key(ValueHint::Contact, "420 800 123 456", "cs", Some("CZ"))));
        assert!(!exact(&key(ValueHint::Contact, "0800 123 456", "cs", Some("CZ"))));
        assert!(!exact(&key(ValueHint::Contact, "30 1234567", "de", Some("DE"))));
        assert!(!exact(&key(ValueHint::Contact, "800 123 456", "cs", Some("US"))));
    }

    #[test]
    fn emails_compare_case_insensitively() {
        let a = key(ValueHint::Contact, "Info@Example.CZ", "", None);
        assert_eq!(a, ValueKey::Exact("mail:info@example.cz".to_string()));
        assert_eq!(key(ValueHint::Contact, "mailto:info@example.cz", "", None), a);
        assert_ne!(key(ValueHint::Contact, "sales@example.cz", "", None), a);
        assert_eq!(
            key(ValueHint::Contact, "john_doe@example.cz", "", None),
            ValueKey::Exact("mail:john_doe@example.cz".to_string())
        );
        assert!(!exact(&key(ValueHint::Contact, "a@example.cz, b@example.cz", "", None)));
    }

    // --- value_key: numbers ---

    #[test]
    fn operators_and_ranges_are_part_of_the_meaning() {
        assert_ne!(
            key(ValueHint::Text, "<18", "", None),
            key(ValueHint::Text, ">18", "", None)
        );
        assert_ne!(
            key(ValueHint::Number, "<18", "", None),
            key(ValueHint::Number, ">18", "", None)
        );
        let from = key(ValueHint::Number, "od 290 Kč", "cs", None);
        assert!(exact(&from));
        assert_ne!(from, key(ValueHint::Number, "290 Kč", "cs", None));
        assert_eq!(from, key(ValueHint::Number, "from 290 CZK", "en", None));
        let range = key(ValueHint::Number, "290–390 Kč", "cs", None);
        assert_eq!(range, ValueKey::Exact("range:290-390:CZK:".to_string()));
        assert_eq!(key(ValueHint::Number, "290 - 390 Kč", "cs", None), range);
        assert_eq!(key(ValueHint::Number, "od 290 do 390 Kč", "cs", None), range);
        assert_ne!(range, key(ValueHint::Number, "290 Kč", "cs", None));
        assert_ne!(
            key(ValueHint::Number, "cca 1 000", "cs", None),
            key(ValueHint::Number, "1 000", "cs", None)
        );
        assert_eq!(
            key(ValueHint::Number, "1 000+", "cs", None),
            key(ValueHint::Number, "≥ 1 000", "cs", None)
        );
    }

    #[test]
    fn separators_are_read_by_language_and_ambiguity_stays_unresolved() {
        let five = key(ValueHint::Number, "5 %", "cs", None);
        assert_eq!(five, ValueKey::Exact("num:5:%:".to_string()));
        assert_eq!(key(ValueHint::Number, "5,000 %", "cs", None), five);
        assert_eq!(key(ValueHint::Number, "5,000 %", "cs-CZ", None), five);
        assert!(
            !exact(&key(ValueHint::Number, "5,000 %", "", None)),
            "ambiguous without a language"
        );
        assert_eq!(
            key(ValueHint::Number, "5,000 %", "en", None),
            ValueKey::Exact("num:5000:%:".to_string())
        );

        let price = ValueKey::Exact("num:1290:CZK:".to_string());
        assert_eq!(key(ValueHint::Number, "1.290 Kč", "cs", None), price);
        assert_eq!(key(ValueHint::Number, "1 290 Kč", "cs", None), price);
        assert_eq!(key(ValueHint::Number, "1\u{a0}290\u{a0}Kč", "", None), price);
        assert_eq!(key(ValueHint::Number, "1290 CZK", "", None), price);
        assert_eq!(key(ValueHint::Number, "1 290,-", "cs", None), price);
        assert_eq!(key(ValueHint::Number, "1 290,- Kč", "cs", None), price);
        assert_eq!(key(ValueHint::Number, "1,290 CZK", "en", None), price);
        assert!(!exact(&key(ValueHint::Number, "1.290 Kč", "", None)));

        assert_eq!(
            key(ValueHint::Number, "1 290,50 Kč", "cs", None),
            ValueKey::Exact("num:1290.5:CZK:".to_string())
        );
        assert_eq!(
            key(ValueHint::Number, "1.290,50 Kč", "", None),
            key(ValueHint::Number, "1,290.50 CZK", "", None)
        );
        assert_eq!(
            key(ValueHint::Number, "4,59 %", "", None),
            key(ValueHint::Number, "4.59 %", "", None),
            "a separator not followed by 3 digits is a decimal point"
        );
        assert_eq!(
            key(ValueHint::Number, "0,500 %", "", None),
            ValueKey::Exact("num:0.5:%:".to_string())
        );
        assert_eq!(
            key(ValueHint::Number, "€ 29,90", "de", None),
            key(ValueHint::Number, "29,90 €", "de", None)
        );
        assert_eq!(
            key(ValueHint::Number, "CHF 1'290.50", "de-CH", None),
            ValueKey::Exact("num:1290.5:CHF:".to_string())
        );
        assert_ne!(
            key(ValueHint::Number, "-5 %", "cs", None),
            key(ValueHint::Number, "5 %", "cs", None)
        );
        assert!(
            !exact(&key(ValueHint::Number, "1 29 Kč", "cs", None)),
            "broken grouping"
        );
    }

    #[test]
    fn a_time_base_is_part_of_the_key() {
        let monthly = key(ValueHint::Number, "290 Kč/měs.", "cs", None);
        assert_eq!(monthly, ValueKey::Exact("num:290:CZK:month".to_string()));
        assert_eq!(key(ValueHint::Number, "290 Kč měsíčně", "cs", None), monthly);
        assert_eq!(key(ValueHint::Number, "290 CZK per month", "en", None), monthly);
        assert_ne!(monthly, key(ValueHint::Number, "290 Kč", "cs", None));
        assert_ne!(monthly, key(ValueHint::Number, "290 Kč ročně", "cs", None));
        assert_eq!(
            key(ValueHint::Number, "4,59 % p.a.", "cs", None),
            key(ValueHint::Number, "4.59 % p.a.", "en", None)
        );
        // Other words stay in the key, so they never merge different claims.
        assert_ne!(
            key(ValueHint::Number, "290 Kč s DPH", "cs", None),
            key(ValueHint::Number, "290 Kč bez DPH", "cs", None)
        );
        assert_eq!(
            key(ValueHint::Number, "5 let", "cs", None),
            ValueKey::Exact("num:5:let:".to_string())
        );
    }

    #[test]
    fn the_llm_hint_never_overrides_the_verbatim_value() {
        // Valid but wrong hints are ignored when the value itself parses.
        assert_eq!(
            value_key(ValueHint::Number, "4,59 %", "4.69 %", "cs", None),
            key(ValueHint::Number, "4,59 %", "cs", None)
        );
        assert_eq!(
            value_key(ValueHint::Number, "1.290 Kč", "1.29 CZK", "cs", None),
            ValueKey::Exact("num:1290:CZK:".to_string())
        );
        assert_eq!(
            value_key(ValueHint::Contact, "800 123 456", "+420800123456", "cs", None),
            ValueKey::Uncertain("tel-national:800123456".to_string())
        );
        assert_eq!(
            value_key(ValueHint::Date, "25. 9. 2026", "2026-09-26", "cs", None),
            ValueKey::Exact("date:2026-09-25".to_string())
        );
        // An ambiguous value is not resolved by the hint either.
        assert!(!exact(&value_key(ValueHint::Number, "5,000 %", "5 %", "", None)));
        // A value that does not parse stays uncertain whatever the hint says: the words around the
        // number may change its meaning, and the digits of a hint prove neither its unit nor its
        // bound nor its date.
        let promoted: Vec<ValueKey> = [
            (ValueHint::Number, "cena 1 290 Kč", "1290 CZK"),
            (ValueHint::Number, "price: 10 EUR", "10 USD"),
            (ValueHint::Number, "minimum price: 10 EUR", "10 EUR"),
            (ValueHint::Date, "2026/12/31", "2026-12-31"),
            (ValueHint::Date, "2026-01-01 to 2026-12-31", "2026-01-31"),
        ]
        .into_iter()
        .map(|(hint, value, normalized)| value_key(hint, value, normalized, "en", None))
        .filter(exact)
        .collect();
        assert_eq!(promoted, Vec::<ValueKey>::new(), "a hint made these exact");
        assert!(!exact(&value_key(
            ValueHint::Number,
            "cena 1 290 Kč",
            "1390 CZK",
            "cs",
            None
        )));
        assert!(!exact(&value_key(
            ValueHint::Number,
            "cena 1 290 Kč",
            "129 CZK",
            "cs",
            None
        )));
        assert!(!exact(&value_key(ValueHint::Number, "cena 1 290 Kč", "", "cs", None)));
        assert!(!exact(&value_key(ValueHint::Date, "Q4 2026", "2026-12-31", "", None)));
    }

    // --- value_key: dates and text ---

    #[test]
    fn czech_and_english_dates_are_recognized() {
        let expected = ValueKey::Exact("date:2026-12-31".to_string());
        for written in [
            "31. 12. 2026",
            "31.12.2026",
            "31. prosince 2026",
            "31. prosince 2026.",
            "December 31, 2026",
            "Dec 31st, 2026",
            "31 December 2026",
            "31st of December 2026",
            "2026-12-31",
        ] {
            assert_eq!(key(ValueHint::Date, written, "", None), expected, "{written}");
        }
        assert_eq!(
            key(ValueHint::Date, "1.1.2026", "", None),
            ValueKey::Exact("date:2026-01-01".to_string())
        );
        assert!(!exact(&key(ValueHint::Date, "31. 2. 2026", "", None)), "no such day");
        assert!(!exact(&key(ValueHint::Date, "31. 12.", "", None)), "no year");
        assert!(
            !exact(&key(ValueHint::Date, "do 31. 12. 2026", "", None)),
            "a condition"
        );
    }

    #[test]
    fn text_keeps_digits_operators_and_punctuation() {
        assert_eq!(
            key(ValueHint::Text, "Po–Pá  8–18", "", None),
            key(ValueHint::Text, "po–pá 8–18", "", None)
        );
        assert_ne!(
            key(ValueHint::Text, "Po–Pá 8–18", "", None),
            key(ValueHint::Text, "Po–Pá 8–17", "", None)
        );
        assert_eq!(
            key(ValueHint::Text, " Karlova  1,\u{a0}Praha ", "", None),
            ValueKey::Exact("text:karlova 1, praha".to_string())
        );
    }

    // --- numbers_in / date_mentions / snippet_of / site_country ---

    #[test]
    fn numbers_in_reads_numbers_by_language() {
        assert_eq!(numbers_in("1 290,50 Kč a 4,59 %", "cs"), vec![1290.5, 4.59]);
        assert_eq!(numbers_in("1,290.50 USD and 4.59 %", "en"), vec![1290.5, 4.59]);
        assert_eq!(numbers_in("v roce 2026 290 lidí", "cs"), vec![2026.0, 290.0]);
        assert_eq!(numbers_in("25. 9. 2026", "cs"), vec![25.0, 9.0, 2026.0]);
        assert_eq!(numbers_in("25.9.2026", "cs"), vec![25.0, 9.0, 2026.0]);
        assert_eq!(numbers_in("1.290 Kč", "cs"), vec![1290.0]);
        let ambiguous = numbers_in("1.290 Kč", "");
        assert!(
            ambiguous.contains(&1.29) && ambiguous.contains(&1290.0),
            "{ambiguous:?}"
        );
        assert!(numbers_in("no numbers here", "en").is_empty());
    }

    #[test]
    fn date_mentions_find_dates_in_prose() {
        let date = NaiveDate::from_ymd_opt(2026, 9, 25).unwrap();
        assert!(date_mentions("Akce platí do 25. září 2026 včetně.").contains(&date));
        assert!(date_mentions("Published September 25, 2026 by the team").contains(&date));
        assert!(date_mentions("od 25. 9. 2026").contains(&date));
        assert!(date_mentions("on 25 September 2026").contains(&date));
        assert!(date_mentions("ISO 2026-09-25T10:00").contains(&date));
        let both = date_mentions("From 1. 1. 2026 to December 31, 2026.");
        assert_eq!(both.len(), 2);
        assert!(date_mentions("Nothing dated, 31. 2. 2026 is no date").is_empty());
    }

    #[test]
    fn snippet_keeps_short_blocks_whole_and_windows_long_ones() {
        let short = "Zákaznická linka 800 123 456";
        assert_eq!(snippet_of(short, (17, 28), 300), short);

        let long = format!(
            "{} Cena tarifu Basic je 290 Kč měsíčně. {}",
            "úvod ".repeat(80),
            "závěr ".repeat(80)
        );
        let start = long.find("290 Kč").unwrap();
        let snippet = snippet_of(&long, (start, start + "290 Kč".len()), 60);
        assert!(snippet.contains("290 Kč"), "{snippet}");
        assert!(
            snippet.chars().count() <= 60,
            "{} chars: {snippet}",
            snippet.chars().count()
        );
        assert!(snippet.starts_with('…') && snippet.ends_with('…'), "{snippet}");
        let words: std::collections::HashSet<&str> = long.split_whitespace().collect();
        let inner = snippet.trim_start_matches('…').trim_end_matches('…');
        assert!(
            inner.split_whitespace().all(|w| words.contains(w)),
            "cut on words: {snippet}"
        );

        let head = snippet_of(&long, (0, 5), 40);
        assert!(head.starts_with("úvod") && head.ends_with('…'), "{head}");
        // A span that is not on char boundaries falls back to the start of the block.
        let broken = snippet_of(&long, (1, 2), 40);
        assert!(broken.starts_with("úvod"), "{broken}");
        assert!(snippet_of(&long, (start, start + 6), 3).chars().count() <= 3);
    }

    #[test]
    fn helpers_never_panic_on_odd_input() {
        let inputs = [
            "",
            " ",
            "+",
            "00",
            "-",
            ",-",
            "€",
            "%",
            "1",
            "1,",
            ".5",
            "1.",
            "1 ",
            " 1 000 ",
            "1'2'3",
            "1,2.3,4",
            "0,0",
            "000",
            "9".repeat(400).as_str(),
            "+420",
            "++420 800",
            "(0)",
            "-,-",
            "od",
            "od do",
            "290 -",
            "290 - Kč",
            "1 290 -",
            "€ € 5",
            "5 € Kč",
            "2026-13-45",
            "99. 99. 9999",
            "Žluťoučký kůň 🐴 1 290 Kč",
            "\u{a0}\u{202f}",
            "**",
            "a@",
            "@b",
            "a@b@c.cz",
            "1 290 000 000 000 000 000 000 Kč",
            "5 % p.a. /h",
        ]
        .map(str::to_string);
        for input in &inputs {
            for hint in [ValueHint::Contact, ValueHint::Number, ValueHint::Date, ValueHint::Text] {
                for lang in ["", "cs", "en", "de-CH", "x-y-z"] {
                    let _ = value_key(hint, input, input, lang, Some("CZ"));
                    let _ = value_key(hint, input, "", lang, None);
                }
            }
            let _ = numbers_in(input, "cs");
            let _ = numbers_in(input, "");
            let _ = date_mentions(input);
            let _ = fact_signals(input);
            let _ = normalize_for_match(input);
            let _ = site_country(input, input);
            for other in &inputs {
                let _ = find_token_bounded(input, other);
                let _ = contains_normalized(input, other);
            }
            for start in 0..=input.len() + 1 {
                for end in [0, start, start + 1, input.len(), input.len() + 3] {
                    for max in [0, 1, 2, 3, 10] {
                        let _ = snippet_of(input, (start, end), max);
                    }
                }
            }
        }
    }

    #[test]
    fn site_country_comes_from_the_tld_or_the_language_region() {
        assert_eq!(site_country("www.example.cz", ""), Some("CZ"));
        assert_eq!(site_country("Shop.Example.SK.", "en"), Some("SK"));
        assert_eq!(site_country("example.pl:8080", ""), Some("PL"));
        assert_eq!(site_country("example.de", "cs-CZ"), Some("DE"), "the TLD wins");
        assert_eq!(site_country("example.at", ""), Some("AT"));
        assert_eq!(site_country("example.hu", ""), Some("HU"));
        assert_eq!(site_country("example.com", "cs-CZ"), Some("CZ"));
        assert_eq!(site_country("example.com", "de_at"), Some("AT"));
        assert_eq!(site_country("example.com", "zh-Hant-CH"), Some("CH"));
        assert_eq!(
            site_country("example.com", "cs"),
            None,
            "a language alone is no country"
        );
        assert_eq!(site_country("example.com", "en-US"), None, "no dial rules for it");
        assert_eq!(site_country("", ""), None);
    }
}
