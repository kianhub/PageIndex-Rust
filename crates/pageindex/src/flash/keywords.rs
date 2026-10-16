//! Multilingual keyword dictionaries (references, abstract, contents titles,
//! caption labels, chapter and appendix words) bundled with Flash.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use unicode_normalization::UnicodeNormalization;

pub struct Keywords {
    pub references: HashSet<String>,
    pub abstract_words: HashSet<String>,
    pub toc_titles: HashSet<String>,
    /// Figure / table / chart labels that open a caption.
    pub caption_labels: HashSet<String>,
    pub chapter_words: HashSet<String>,
    pub appendix_words: HashSet<String>,
}

pub static KEYWORDS: LazyLock<Keywords> = LazyLock::new(|| {
    let raw: HashMap<String, serde_json::Value> = serde_json::from_str(include_str!("../../data/dictionaries.json"))
        .expect("bundled dictionaries are valid JSON");
    let set = |keys: &[&str]| -> HashSet<String> {
        keys.iter()
            .filter_map(|key| raw.get(*key).and_then(|v| v.as_array()))
            .flatten()
            .filter_map(|v| v.as_str())
            .map(normalize_key)
            .filter(|k| !k.is_empty())
            .collect()
    };
    Keywords {
        references: set(&["references"]),
        abstract_words: set(&["abstract_keywords"]),
        toc_titles: set(&["toc_titles"]),
        caption_labels: set(&["ai_section_keywords", "table_keywords", "chart_keywords"]),
        chapter_words: set(&["chapter_words"]),
        appendix_words: set(&["appendix_keywords", "appendices_dict"]),
    }
});

/// NFKC, single-spaced, lowercased: the key every dictionary lookup uses.
pub fn normalize_key(text: &str) -> String {
    text.nfkc().collect::<String>().split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

impl Keywords {
    /// `text` (or its last word, after a short number) is a references heading.
    pub fn is_references(&self, text: &str) -> bool {
        let key = normalize_key(text);
        if key.chars().count() > 80 {
            return false;
        }
        if self.references.contains(&key) {
            return true;
        }
        let words: Vec<&str> = key.split(' ').collect();
        (1..=4).contains(&words.len()) && self.references.contains(*words.last().unwrap_or(&""))
    }

    pub fn is_abstract(&self, text: &str) -> bool {
        let key = normalize_key(text);
        self.abstract_words.contains(key.trim_end_matches([':', '.']))
    }

    pub fn is_toc_title(&self, text: &str) -> bool {
        let key = normalize_key(text);
        self.toc_titles.contains(key.trim_end_matches([':', '.']))
    }

    /// `Figure 3:`, `Table 2.`, `Fig. 1` and their translations.
    pub fn is_caption(&self, text: &str) -> bool {
        let key = normalize_key(text);
        let mut words = key.split(' ');
        let (Some(first), Some(second)) = (words.next(), words.next()) else {
            return false;
        };
        let label = first.trim_end_matches(['.', ':']);
        let numbered = second.starts_with(|c: char| c.is_ascii_digit())
            || (second.trim_end_matches(['.', ':']).len() <= 4 && second.chars().all(|c| "ivxlc.:".contains(c)));
        numbered && (self.caption_labels.contains(label) || self.caption_labels.contains(first))
    }

    /// `Chapter 3`, `Appendix B`, `Part II` style openers.
    pub fn opens_with_unit_word(&self, text: &str) -> bool {
        let key = normalize_key(text);
        let first = key.split(' ').next().unwrap_or("");
        let has_more = key.split(' ').nth(1).is_some();
        has_more
            && (self.chapter_words.contains(first)
                || self.appendix_words.contains(first)
                || first == "part"
                || first == "section")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionaries_load() {
        let k = &*KEYWORDS;
        assert!(k.is_references("References"));
        assert!(k.is_references("7. References"));
        assert!(k.is_abstract("ABSTRACT"));
        assert!(k.is_toc_title("Table of Contents"));
        assert!(k.is_caption("Figure 3: Results"));
        assert!(k.is_caption("Table 2. Summary"));
        assert!(!k.is_caption("Tables are useful"));
        assert!(k.opens_with_unit_word("Chapter 3 Methods"));
    }
}
