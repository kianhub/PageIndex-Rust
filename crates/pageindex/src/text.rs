//! Text keys used to compare headings across extraction and model output,
//! and token counting.

use std::sync::LazyLock;

use regex::Regex;
use unicode_normalization::UnicodeNormalization;

static NON_WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^\p{L}\p{M}\p{N}]+").unwrap());
static LEADING_NUMBERS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(?:[0-9]+ )+").unwrap());
static HAS_LETTER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^\W\d_]").unwrap());
static HAS_LATIN_LOWER: LazyLock<Regex> = LazyLock::new(|| Regex::new("[a-z]").unwrap());

static BPE: LazyLock<tiktoken_rs::CoreBPE> =
    LazyLock::new(|| tiktoken_rs::cl100k_base().expect("bundled cl100k vocabulary loads"));

/// Han and kana: scripts that put no space between words.
pub fn is_han_kana(c: char) -> bool {
    matches!(c, '\u{3005}'..='\u{3007}' | '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{f900}'..='\u{faff}')
}

/// Remove every space with a Han/kana character on both sides (`both`), or
/// on either side.
fn drop_han_spaces(text: &str, both: bool) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    for (i, &c) in chars.iter().enumerate() {
        if c == ' ' {
            let before = i > 0 && is_han_kana(chars[i - 1]);
            let after = chars.get(i + 1).is_some_and(|&n| is_han_kana(n));
            if (both && before && after) || (!both && (before || after)) {
                continue;
            }
        }
        out.push(c);
    }
    out
}

fn casefold(text: &str) -> String {
    text.to_lowercase().replace('ß', "ss")
}

/// Letters, digits and marks of any script, casefolded; other runs become one
/// space, and none stays between two Han or kana characters.
pub fn normalize(text: &str) -> String {
    let folded = casefold(&text.nfkc().collect::<String>());
    let spaced = NON_WORD.replace_all(&folded, " ");
    drop_han_spaces(spaced.trim(), true)
}

/// [`normalize`], less any space beside a Han or kana character.
pub fn heading_key(text: &str) -> String {
    drop_han_spaces(&normalize(text), false)
}

/// Whether two titles name one heading: equal by key, or equal but for a
/// leading number only one of them prints.
pub fn same_heading(a: &str, b: &str) -> bool {
    let (a, b) = (normalize(a), normalize(b));
    let bare_a = LEADING_NUMBERS.replace(&a, "").into_owned();
    let bare_b = LEADING_NUMBERS.replace(&b, "").into_owned();
    let [a, b, bare_a, bare_b] = [a, b, bare_a, bare_b].map(|t| drop_han_spaces(&t, false));
    !a.is_empty() && (a == b || (bare_a == bare_b && HAS_LETTER.is_match(&bare_a) && (a == bare_a || b == bare_b)))
}

/// Whether a normalized key holds a Latin letter to match lines against.
pub fn has_latin_letter(key: &str) -> bool {
    HAS_LATIN_LOWER.is_match(key)
}

/// Token count under the cl100k vocabulary (LiteLLM's default counter).
pub fn count_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    BPE.encode_ordinary(text).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_spaces_han_but_not_latin() {
        assert_eq!(normalize("4.1. Discriminant Functions"), "4 1 discriminant functions");
        assert_eq!(normalize("第 二 节  定 义"), "第二节定义");
        assert_eq!(heading_key("1、2019 年度"), heading_key("1、2019年度"));
    }

    #[test]
    fn same_heading_tells_numbers_apart() {
        assert!(same_heading("1 总则", "总则"));
        assert!(same_heading("⑥药物工艺", "⑥ 药物工艺"));
        assert!(!same_heading("第一章 总则", "第二章 总则"));
        assert!(!same_heading("1", "2"));
        assert!(same_heading("3 Results", "Results"));
    }

    #[test]
    fn counts_tokens() {
        assert_eq!(count_tokens(""), 0);
        assert!(count_tokens("hello world") >= 2);
    }
}
