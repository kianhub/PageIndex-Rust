//! Font dictionaries: code splitting, glyph Unicode, and advance widths for
//! simple (Type1 / TrueType / Type3) and composite (Type0) fonts.

use std::collections::HashMap;
use std::sync::Arc;

use lopdf::{Dictionary, Document, Object};

use crate::cmap::CMap;
use crate::encoding::{BaseEncoding, glyph_name_to_unicode};
use crate::metrics::standard_width;
use crate::objects::{array_f32, dict_get, name_of, number, resolve, stream_bytes};

/// One code of a shown string, decoded against its font.
#[derive(Debug, Clone)]
pub struct DecodedCode {
    pub text: Option<String>,
    /// Advance in text-space units for a font size of 1.
    pub width: f32,
    /// A single-byte code 32, which word spacing (Tw) applies to.
    pub is_word_space: bool,
}

#[derive(Debug)]
enum Encoding {
    Simple(Box<[Option<String>; 256]>),
    Composite { cmap: CMap, ucs2: bool },
}

#[derive(Debug)]
enum Widths {
    Simple { first: u32, widths: Vec<f32>, missing: Option<f32> },
    Composite { explicit: HashMap<u32, f32>, ranges: Vec<(u32, u32, f32)>, default: f32 },
}

#[derive(Debug)]
pub struct Font {
    pub name: Arc<str>,
    pub bold: bool,
    pub italic: bool,
    pub vertical: bool,
    /// Multiplier from the nominal Tf size to the size glyphs render at.
    pub size_scale: f32,
    encoding: Encoding,
    to_unicode: Option<CMap>,
    widths: Widths,
    /// Glyph-space width units to text-space units (1/1000 except Type3).
    width_scale: f32,
    standard_base: Option<String>,
}

impl Font {
    pub fn fallback() -> Self {
        let mut table: Box<[Option<String>; 256]> = Box::new(std::array::from_fn(|_| None));
        for code in 32u32..256 {
            table[code as usize] = char::from_u32(code).map(String::from);
        }
        Font {
            name: Arc::from("Unknown"),
            bold: false,
            italic: false,
            vertical: false,
            size_scale: 1.0,
            encoding: Encoding::Simple(table),
            to_unicode: None,
            widths: Widths::Simple { first: 0, widths: Vec::new(), missing: Some(0.5) },
            width_scale: 0.001,
            standard_base: Some("Helvetica".into()),
        }
    }

    pub fn from_dict(doc: &Document, dict: &Dictionary) -> Self {
        let subtype = dict_get(doc, dict, b"Subtype").and_then(name_of).unwrap_or(b"Type1");
        let base_raw = dict_get(doc, dict, b"BaseFont").and_then(name_of).unwrap_or(b"");
        let base = String::from_utf8_lossy(base_raw).into_owned();
        let name = strip_subset_prefix(&base).to_owned();
        let lower = name.to_ascii_lowercase();
        let bold = lower.contains("bold") || lower.contains("timesb");
        let italic = lower.contains("italic") || lower.contains("oblique");
        let to_unicode = dict_get(doc, dict, b"ToUnicode")
            .and_then(|obj| stream_bytes(doc, obj))
            .map(|data| CMap::parse(&data))
            .filter(|cmap| !cmap.is_empty());

        if subtype == b"Type0" {
            return Self::composite(doc, dict, name, bold, italic, to_unicode);
        }

        let descriptor = dict_get(doc, dict, b"FontDescriptor").and_then(|o| o.as_dict().ok());
        let flags = descriptor.and_then(|d| dict_get(doc, d, b"Flags")).and_then(number).unwrap_or(0.0) as u32;
        let symbolic = flags & 4 != 0 && flags & 32 == 0;
        let is_type3 = subtype == b"Type3";

        let mut table: Box<[Option<String>; 256]> = Box::new(std::array::from_fn(|_| None));
        let builtin = descriptor.and_then(|d| embedded_type1_encoding(doc, d));
        let base_encoding = match dict_get(doc, dict, b"Encoding") {
            Some(Object::Name(n)) => BaseEncoding::from_name(n),
            Some(Object::Dictionary(d)) => {
                dict_get(doc, d, b"BaseEncoding").and_then(name_of).and_then(BaseEncoding::from_name)
            }
            _ => None,
        };
        let default_base = if lower.contains("symbol") {
            BaseEncoding::Symbol
        } else if lower.contains("dingbats") {
            BaseEncoding::ZapfDingbats
        } else if subtype == b"TrueType" && !symbolic {
            BaseEncoding::WinAnsi
        } else {
            BaseEncoding::Standard
        };
        for code in 0..256usize {
            let from_builtin = if base_encoding.is_none() {
                builtin.as_ref().and_then(|b| b.get(&(code as u8))).and_then(|g| glyph_name_to_unicode(g))
            } else {
                None
            };
            table[code] = from_builtin.or_else(|| {
                base_encoding.unwrap_or(default_base).glyph_name(code as u8).and_then(glyph_name_to_unicode)
            });
        }
        if let Some(Object::Dictionary(enc)) = dict_get(doc, dict, b"Encoding")
            && let Some(Object::Array(diffs)) = dict_get(doc, enc, b"Differences")
        {
            let mut code = 0usize;
            for item in diffs {
                match resolve(doc, item) {
                    Object::Integer(n) => code = (*n).clamp(0, 255) as usize,
                    Object::Real(n) => code = (*n as i64).clamp(0, 255) as usize,
                    Object::Name(glyph) => {
                        if code < 256 {
                            let glyph = String::from_utf8_lossy(glyph);
                            table[code] = glyph_name_to_unicode(&glyph).or_else(|| {
                                // Type3 and subset fonts name glyphs freely; keep the code.
                                (is_type3 || symbolic).then(|| char::from_u32(code as u32).map(String::from)).flatten()
                            });
                        }
                        code += 1;
                    }
                    _ => {}
                }
            }
        }
        if symbolic || is_type3 {
            for (code, slot) in table.iter_mut().enumerate().skip(32) {
                if slot.is_none() {
                    *slot = char::from_u32(code as u32).map(String::from);
                }
            }
        }

        let first = dict_get(doc, dict, b"FirstChar").and_then(number).unwrap_or(0.0).max(0.0) as u32;
        let widths: Vec<f32> = dict_get(doc, dict, b"Widths").and_then(|o| array_f32(doc, o)).unwrap_or_default();
        let missing = descriptor.and_then(|d| dict_get(doc, d, b"MissingWidth")).and_then(number);

        let (width_scale, size_scale) = if is_type3 {
            let matrix = dict_get(doc, dict, b"FontMatrix")
                .and_then(|o| array_f32(doc, o))
                .filter(|m| m.len() == 6)
                .unwrap_or_else(|| vec![0.001, 0.0, 0.0, 0.001, 0.0, 0.0]);
            let em = (matrix[3].abs() * 1000.0).max(1e-6);
            let size_scale = if (0.1..=10.0).contains(&em) {
                em
            } else if let Some(height) = charproc_height(doc, dict) {
                (height * matrix[3].abs()).clamp(0.05, 1000.0)
            } else {
                let mut nonzero: Vec<f32> = widths.iter().copied().filter(|w| *w > 0.0).collect();
                nonzero.sort_by(f32::total_cmp);
                nonzero
                    .get(nonzero.len() / 2)
                    .map(|median| (median * matrix[0].abs() / 0.5).clamp(0.05, 1000.0))
                    .unwrap_or(1.0)
            };
            (matrix[0].abs(), size_scale)
        } else {
            (0.001, 1.0)
        };

        Font {
            name: Arc::from(name.as_str()),
            bold,
            italic,
            vertical: false,
            size_scale,
            encoding: Encoding::Simple(table),
            to_unicode,
            widths: Widths::Simple { first, widths, missing: missing.filter(|w| *w > 0.0) },
            width_scale,
            standard_base: (!is_type3).then_some(name),
        }
    }

    fn composite(
        doc: &Document,
        dict: &Dictionary,
        name: String,
        bold: bool,
        italic: bool,
        to_unicode: Option<CMap>,
    ) -> Self {
        let (cmap, ucs2) = match dict_get(doc, dict, b"Encoding") {
            Some(Object::Name(n)) => {
                let vertical = n.ends_with(b"-V");
                let ucs2 = n.starts_with(b"Uni") && (contains(n, b"UCS2") || contains(n, b"UTF16"));
                (CMap::identity(vertical), ucs2)
            }
            Some(obj @ Object::Stream(_)) => {
                let parsed = stream_bytes(doc, obj).map(|d| CMap::parse(&d)).unwrap_or_default();
                if parsed.has_codespace() {
                    (parsed, false)
                } else {
                    let vertical = parsed.vertical;
                    (CMap::identity(vertical), false)
                }
            }
            _ => (CMap::identity(false), false),
        };
        let descendant = dict_get(doc, dict, b"DescendantFonts")
            .and_then(|o| o.as_array().ok())
            .and_then(|a| a.first())
            .map(|o| resolve(doc, o))
            .and_then(|o| o.as_dict().ok());
        let default = descendant.and_then(|d| dict_get(doc, d, b"DW")).and_then(number).unwrap_or(1000.0);
        let mut explicit = HashMap::new();
        let mut ranges = Vec::new();
        if let Some(Object::Array(w)) = descendant.and_then(|d| dict_get(doc, d, b"W")) {
            let items: Vec<&Object> = w.iter().map(|o| resolve(doc, o)).collect();
            let mut i = 0;
            while i < items.len() {
                let Some(first) = number(items[i]) else {
                    i += 1;
                    continue;
                };
                match items.get(i + 1) {
                    Some(Object::Array(list)) => {
                        for (k, width) in list.iter().filter_map(|o| number(resolve(doc, o))).enumerate() {
                            explicit.insert(first as u32 + k as u32, width);
                        }
                        i += 2;
                    }
                    Some(last) => {
                        if let (Some(last), Some(width)) = (number(last), items.get(i + 2).and_then(|o| number(o))) {
                            let lo = first.max(0.0) as u32;
                            let hi = (last.max(0.0) as u64).min(0xFFFF) as u32;
                            if hi >= lo {
                                ranges.push((lo, hi, width));
                            }
                        }
                        i += 3;
                    }
                    None => break,
                }
            }
        }
        let vertical = cmap.vertical;
        Font {
            name: Arc::from(name.as_str()),
            bold,
            italic,
            vertical,
            size_scale: 1.0,
            encoding: Encoding::Composite { cmap, ucs2 },
            to_unicode,
            widths: Widths::Composite { explicit, ranges, default },
            width_scale: 0.001,
            standard_base: None,
        }
    }

    pub fn decode(&self, bytes: &[u8]) -> Vec<DecodedCode> {
        match &self.encoding {
            Encoding::Simple(table) => bytes
                .iter()
                .map(|&b| {
                    let code = b as u32;
                    let text = self
                        .to_unicode
                        .as_ref()
                        .and_then(|m| m.lookup_unicode(code))
                        .or_else(|| table[b as usize].clone());
                    DecodedCode { width: self.simple_width(code, text.as_deref()), text, is_word_space: b == 32 }
                })
                .collect(),
            Encoding::Composite { cmap, ucs2 } => cmap
                .split_codes(bytes)
                .into_iter()
                .map(|(code, len)| {
                    let cid = cmap.lookup_cid(code).unwrap_or(code);
                    let text = self
                        .to_unicode
                        .as_ref()
                        .and_then(|m| m.lookup_unicode(code))
                        .or_else(|| ucs2.then(|| char::from_u32(code).map(String::from)).flatten());
                    DecodedCode { width: self.composite_width(cid), text, is_word_space: len == 1 && code == 32 }
                })
                .collect(),
        }
    }

    fn simple_width(&self, code: u32, text: Option<&str>) -> f32 {
        if let Widths::Simple { first, widths, missing } = &self.widths {
            if code >= *first
                && let Some(&w) = widths.get((code - first) as usize)
            {
                return w * self.width_scale;
            }
            if !widths.is_empty()
                && let Some(m) = missing
            {
                return m * self.width_scale;
            }
            let ch = text.and_then(|t| t.chars().next()).unwrap_or(' ');
            if let Some(w) = self.standard_base.as_deref().and_then(|base| standard_width(base, ch)) {
                return w;
            }
            if let Some(m) = missing {
                return m * self.width_scale;
            }
        }
        0.5
    }

    fn composite_width(&self, cid: u32) -> f32 {
        if let Widths::Composite { explicit, ranges, default } = &self.widths {
            let w = explicit
                .get(&cid)
                .copied()
                .or_else(|| ranges.iter().find(|(lo, hi, _)| *lo <= cid && cid <= *hi).map(|r| r.2))
                .unwrap_or(*default);
            return w * self.width_scale;
        }
        1.0
    }
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

fn strip_subset_prefix(name: &str) -> &str {
    match name.split_once('+') {
        Some((tag, rest)) if tag.len() == 6 && tag.bytes().all(|b| b.is_ascii_uppercase()) => rest,
        _ => name,
    }
}

/// Height in glyph units of the union of a Type3 font's glyph boxes, read
/// from each CharProc's leading `wx wy llx lly urx ury d1`.
fn charproc_height(doc: &Document, dict: &Dictionary) -> Option<f32> {
    let procs = dict_get(doc, dict, b"CharProcs").and_then(|o| crate::objects::as_dict(doc, o))?;
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for (_, proc_obj) in procs.iter() {
        let Some(data) = stream_bytes(doc, proc_obj) else { continue };
        let head = &data[..data.len().min(256)];
        let text = String::from_utf8_lossy(head);
        let tokens: Vec<&str> = text.split_ascii_whitespace().take(7).collect();
        if tokens.len() == 7
            && tokens[6] == "d1"
            && let (Ok(lly), Ok(ury)) = (tokens[3].parse::<f32>(), tokens[5].parse::<f32>())
        {
            lo = lo.min(lly.min(ury));
            hi = hi.max(lly.max(ury));
        }
    }
    (hi > lo).then_some(hi - lo)
}

/// The builtin `/Encoding` of an embedded Type1 program (`dup 65 /A put`).
fn embedded_type1_encoding(doc: &Document, descriptor: &Dictionary) -> Option<HashMap<u8, String>> {
    let program = dict_get(doc, descriptor, b"FontFile")?;
    let data = stream_bytes(doc, program)?;
    let clear_len = match resolve(doc, program) {
        Object::Stream(s) => s.dict.get(b"Length1").ok().and_then(|o| number(resolve(doc, o))),
        _ => None,
    }
    .map(|n| n as usize)
    .unwrap_or(data.len())
    .min(data.len());
    let clear = &data[..clear_len];
    let start = clear.windows(9).position(|w| w == b"/Encoding")?;
    let text = String::from_utf8_lossy(&clear[start..]);
    if text[9..].trim_start().starts_with("StandardEncoding") {
        return None;
    }
    let mut out = HashMap::new();
    let mut tokens = text.split_ascii_whitespace().peekable();
    while let Some(token) = tokens.next() {
        if token == "readonly" || token == "eexec" {
            break;
        }
        if token == "dup" {
            let code = tokens.next().and_then(|t| t.parse::<u16>().ok());
            let glyph = tokens.next().and_then(|t| t.strip_prefix('/'));
            if let (Some(code), Some(glyph)) = (code, glyph)
                && code < 256
            {
                out.insert(code as u8, glyph.to_owned());
            }
        }
    }
    (!out.is_empty()).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::dictionary;

    #[test]
    fn composite_widths_read_both_w_forms() {
        let mut doc = Document::with_version("1.5");
        let descendant = doc.add_object(dictionary! {
            "DW" => 800,
            "W" => vec![10.into(), vec![500.into(), 600.into()].into(), 20.into(), Object::Integer(1 << 31), 700.into()],
        });
        let font = dictionary! {
            "Subtype" => "Type0",
            "BaseFont" => "ABCDEF+Noto",
            "Encoding" => "Identity-H",
            "DescendantFonts" => vec![descendant.into()],
        };
        let font = Font::from_dict(&doc, &font);
        assert_eq!(&*font.name, "Noto");
        let widths: Vec<f32> = [10u32, 11, 12, 20, 0xFFFF, 5].iter().map(|&c| font.composite_width(c)).collect();
        for (got, want) in widths.iter().zip([0.5, 0.6, 0.8, 0.7, 0.7, 0.8]) {
            assert!((got - want).abs() < 1e-6, "{widths:?}");
        }
    }

    #[test]
    fn differences_override_the_base_encoding() {
        let doc = Document::with_version("1.5");
        let font = dictionary! {
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
            "Encoding" => dictionary! {
                "BaseEncoding" => "WinAnsiEncoding",
                "Differences" => vec![97.into(), Object::Name(b"afii57664".to_vec()), Object::Name(b"fi".to_vec())],
            },
        };
        let font = Font::from_dict(&doc, &font);
        let decoded = font.decode(b"abc");
        let texts: Vec<_> = decoded.iter().map(|d| d.text.clone().unwrap_or_default()).collect();
        assert_eq!(texts, vec!["\u{05d0}", "\u{fb01}", "c"]);
        assert!((decoded[2].width - 0.5).abs() < 1e-6);
    }
}
