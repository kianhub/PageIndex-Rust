//! PageIndex naming v1: legal upload names and folder names. Kept in sync
//! with the shared `naming-v1.json` contract.

use md5::{Digest, Md5};
use unicode_normalization::UnicodeNormalization;

use crate::error::{PageIndexError, Result};

pub const MAX_NAME_BYTES: usize = 180;

fn is_control(c: char) -> bool {
    matches!(c, '\u{00}'..='\u{1f}' | '\u{7f}'..='\u{9f}' | '\u{2028}' | '\u{2029}')
}

fn is_illegal(c: char) -> bool {
    matches!(c, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*')
}

/// JavaScript's whitespace set, narrower than Unicode `White_Space`.
fn is_js_whitespace(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{0b}' | '\u{0c}' | '\r' | ' ' | '\u{a0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}' | '\u{feff}'
    )
}

fn translate_quote(c: char) -> char {
    match c {
        '\u{2018}' | '\u{2019}' | '\u{02bc}' => '\'',
        '\u{201c}' | '\u{201d}' => '"',
        other => other,
    }
}

fn is_reserved(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    let stem = lower.split('.').next().unwrap_or("");
    let bytes = stem.as_bytes();
    matches!(stem, "con" | "prn" | "aux" | "nul")
        || (bytes.len() == 4 && (stem.starts_with("com") || stem.starts_with("lpt")) && bytes[3].is_ascii_digit())
}

/// Normalize Unicode (NFKC), quote variants and whitespace; preserve case.
pub fn normalize_filename(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut in_space = false;
    for c in name.nfkc().map(translate_quote) {
        if is_js_whitespace(c) {
            if !in_space {
                out.push(' ');
            }
            in_space = true;
        } else {
            out.push(c);
            in_space = false;
        }
    }
    out.trim_matches(' ').to_owned()
}

fn utf8_prefix(value: &str, max_bytes: usize) -> &str {
    let mut end = max_bytes.min(value.len());
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

/// `os.path.splitext`: the extension starts at the last dot that has a
/// non-dot character before it.
fn split_ext(name: &str) -> (&str, &str) {
    match name.rfind('.') {
        Some(dot) if name[..dot].chars().any(|c| c != '.') => (&name[..dot], &name[dot..]),
        _ => (name, ""),
    }
}

/// Fit a name, its extension and a collision suffix into the byte budget.
pub fn truncate_filename(name: &str, max_bytes: usize, suffix: &str) -> Result<String> {
    let (base, ext) = split_ext(name);
    let candidate = format!("{base}{suffix}{ext}");
    if candidate.len() <= max_bytes {
        return Ok(candidate);
    }
    let digest = Md5::digest(name.as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    let ending = format!("_{}{suffix}", &hex[..8]);
    let budget = max_bytes.saturating_sub(ending.len());
    if max_bytes < ending.len() + 4 {
        return Err(PageIndexError::invalid("Filename byte limit is too small for its suffix"));
    }
    let first_bytes = base.chars().next().map_or(1, char::len_utf8);
    let ext = utf8_prefix(ext, budget.saturating_sub(first_bytes)).trim_end_matches([' ', '.']);
    let base = utf8_prefix(base, budget - ext.len());
    let base = if base.is_empty() { "_" } else { base };
    Ok(format!("{base}{ending}{ext}"))
}

/// Choose a legal upload name.
pub fn sanitize_filename(name: &str) -> String {
    let replaced: String = name.chars().map(|c| if is_control(c) { '_' } else { c }).collect();
    let normalized: String =
        normalize_filename(&replaced).chars().map(|c| if is_illegal(c) { '_' } else { c }).collect();
    let mut name = normalized.trim_end_matches([' ', '.']).to_owned();
    if name.is_empty() {
        name = "untitled".into();
    }
    if is_reserved(&name) {
        name.insert(0, '_');
    }
    truncate_filename(&name, MAX_NAME_BYTES, "").expect("default budget fits any suffix")
}

/// Normalize a folder name, rejecting characters that would need replacement.
pub fn validate_folder_name(name: &str) -> Result<String> {
    if name.chars().any(is_control) {
        return Err(PageIndexError::invalid("Folder name cannot contain control characters or invalid Unicode"));
    }
    let name = normalize_filename(name);
    if name.is_empty() {
        return Err(PageIndexError::invalid("Folder name cannot be empty"));
    }
    if name.chars().any(is_illegal) {
        return Err(PageIndexError::invalid("Folder name cannot contain / \\ : * ? \" < > |"));
    }
    if name == "." || name == ".." || is_reserved(&name) {
        return Err(PageIndexError::invalid("Folder name is reserved"));
    }
    if name.ends_with('.') {
        return Err(PageIndexError::invalid("Folder name cannot end with a period"));
    }
    if name.len() > MAX_NAME_BYTES {
        return Err(PageIndexError::invalid("Folder name must be at most 180 UTF-8 bytes"));
    }
    Ok(name)
}

/// Read an already assigned name literally, including names predating v1.
pub fn validate_stored_filename(name: &str) -> Result<&str> {
    if name.is_empty() || name == "." || name == ".." || name.contains(['/', '\\']) || name.chars().any(is_control) {
        return Err(PageIndexError::invalid("File name must be a single path component without control characters"));
    }
    Ok(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// Upstream's shared contract. Its one unpaired surrogate (`\ud800`)
    /// cannot exist in a Rust string; paths decode it to U+FFFD at the OS
    /// boundary, which is the input it stands in for here.
    const CONTRACT: &str = include_str!("../tests/fixtures/naming-v1.json");

    fn cases(key: &str) -> Vec<Value> {
        let contract: Value = serde_json::from_str(&CONTRACT.replace("\\ud800", "\\ufffd")).unwrap();
        contract[key].as_array().unwrap().clone()
    }

    #[test]
    fn files_match_the_contract() {
        for case in cases("files") {
            let name = case["name"].as_str().unwrap();
            assert_eq!(sanitize_filename(name), case["expected"].as_str().unwrap(), "{name:?}");
        }
    }

    #[test]
    fn folders_match_the_contract() {
        for case in cases("folders") {
            let name = case["name"].as_str().unwrap();
            if name.contains('\u{FFFD}') {
                continue;
            }
            match (case.get("expected"), case.get("error")) {
                (Some(expected), _) => assert_eq!(validate_folder_name(name).unwrap(), expected.as_str().unwrap()),
                (_, Some(error)) => {
                    let err = validate_folder_name(name).unwrap_err();
                    assert!(err.message.contains(error.as_str().unwrap()), "{name:?}: {}", err.message);
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn suffixes_match_the_contract() {
        for case in cases("suffixes") {
            let name = case["name"].as_str().unwrap();
            let suffix = case["suffix"].as_str().unwrap();
            assert_eq!(truncate_filename(name, MAX_NAME_BYTES, suffix).unwrap(), case["expected"].as_str().unwrap());
        }
    }

    #[test]
    fn stored_names_are_single_components() {
        assert!(validate_stored_filename("a/b.pdf").is_err());
        assert!(validate_stored_filename("..").is_err());
        assert_eq!(validate_stored_filename("CON.pdf").unwrap(), "CON.pdf");
    }
}
