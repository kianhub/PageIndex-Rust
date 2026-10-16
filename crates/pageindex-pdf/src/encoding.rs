//! Glyph names, the standard single-byte encodings, and per-glyph Unicode
//! normalization, all backed by the tables PageIndex Flash bundles.

use std::collections::HashMap;
use std::sync::LazyLock;

struct Tables {
    glyphs: HashMap<String, u32>,
    encodings: HashMap<String, Vec<String>>,
}

static TABLES: LazyLock<Tables> = LazyLock::new(|| {
    #[derive(serde::Deserialize)]
    struct Raw {
        glyphs: HashMap<String, u32>,
        encodings: HashMap<String, Vec<String>>,
    }
    let raw: Raw =
        serde_json::from_str(include_str!("../data/glyph_name_table.json")).expect("bundled glyph table is valid JSON");
    Tables { glyphs: raw.glyphs, encodings: raw.encodings }
});

static NORMALIZED: LazyLock<HashMap<char, String>> = LazyLock::new(|| {
    let raw: HashMap<String, String> = serde_json::from_str(include_str!("../data/normalized_unicodes.json"))
        .expect("bundled normalization table is valid JSON");
    raw.into_iter()
        .filter_map(|(k, v)| {
            let mut chars = k.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => Some((c, v)),
                _ => None,
            }
        })
        .collect()
});

/// The single-byte base encodings a simple font can name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseEncoding {
    Standard,
    WinAnsi,
    MacRoman,
    MacExpert,
    Symbol,
    ZapfDingbats,
}

impl BaseEncoding {
    pub fn from_name(name: &[u8]) -> Option<Self> {
        Some(match name {
            b"StandardEncoding" => Self::Standard,
            b"WinAnsiEncoding" => Self::WinAnsi,
            b"MacRomanEncoding" => Self::MacRoman,
            b"MacExpertEncoding" => Self::MacExpert,
            _ => return None,
        })
    }

    fn table_key(self) -> &'static str {
        match self {
            Self::Standard => "StandardEncoding",
            Self::WinAnsi => "WinAnsiEncoding",
            Self::MacRoman => "MacRomanEncoding",
            Self::MacExpert => "MacExpertEncoding",
            Self::Symbol => "SymbolSetEncoding",
            Self::ZapfDingbats => "ZapfDingbatsEncoding",
        }
    }

    /// The glyph name this encoding assigns to `code`, if any.
    pub fn glyph_name(self, code: u8) -> Option<&'static str> {
        let table = TABLES.encodings.get(self.table_key())?;
        table.get(code as usize).map(String::as_str).filter(|name| !name.is_empty())
    }
}

/// Resolve a glyph name to text: the Adobe glyph list, `uniXXXX` / `uXXXX[XX]`
/// names, and `name.suffix` / `a_b` ligature spellings.
pub fn glyph_name_to_unicode(name: &str) -> Option<String> {
    if let Some(&cp) = TABLES.glyphs.get(name) {
        return char::from_u32(cp).map(String::from);
    }
    if let Some(base) = name.split('.').next().filter(|base| base.len() < name.len() && !base.is_empty()) {
        return glyph_name_to_unicode(base);
    }
    if name.contains('_') {
        let parts: Option<String> = name.split('_').map(glyph_name_to_unicode).collect();
        return parts.filter(|text| !text.is_empty());
    }
    if let Some(hex) = name.strip_prefix("uni")
        && hex.len() >= 4
        && hex.len() % 4 == 0
        && is_upper_hex(hex)
    {
        let units: Option<Vec<u16>> = hex
            .as_bytes()
            .chunks(4)
            .map(|chunk| u16::from_str_radix(std::str::from_utf8(chunk).ok()?, 16).ok())
            .collect();
        return units.and_then(|units| String::from_utf16(&units).ok());
    }
    if let Some(hex) = name.strip_prefix('u')
        && (4..=6).contains(&hex.len())
        && is_upper_hex(hex)
    {
        return u32::from_str_radix(hex, 16).ok().and_then(char::from_u32).map(String::from);
    }
    None
}

fn is_upper_hex(text: &str) -> bool {
    text.bytes().all(|b| b.is_ascii_digit() || (b'A'..=b'F').contains(&b))
}

/// Compatibility-decompose presentation forms (ligatures, full-width forms,
/// spacing accents) the way the Flash parser emits each glyph.
pub fn normalize_glyph_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match NORMALIZED.get(&ch) {
            Some(replacement) => out.push_str(replacement),
            None => out.push(ch),
        }
    }
    out
}
