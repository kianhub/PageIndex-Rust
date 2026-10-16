//! Reference-following accessors over lopdf objects.

use lopdf::{Dictionary, Document, Object};

const MAX_DEPTH: usize = 32;

pub fn resolve<'a>(doc: &'a Document, mut obj: &'a Object) -> &'a Object {
    for _ in 0..MAX_DEPTH {
        match obj {
            Object::Reference(id) => match doc.get_object(*id) {
                Ok(target) => obj = target,
                Err(_) => return &Object::Null,
            },
            _ => return obj,
        }
    }
    &Object::Null
}

pub fn dict_get<'a>(doc: &'a Document, dict: &'a Dictionary, key: &[u8]) -> Option<&'a Object> {
    dict.get(key).ok().map(|obj| resolve(doc, obj)).filter(|obj| !matches!(obj, Object::Null))
}

pub fn as_dict<'a>(doc: &'a Document, obj: &'a Object) -> Option<&'a Dictionary> {
    match resolve(doc, obj) {
        Object::Dictionary(d) => Some(d),
        Object::Stream(s) => Some(&s.dict),
        _ => None,
    }
}

pub fn name_of(obj: &Object) -> Option<&[u8]> {
    match obj {
        Object::Name(n) => Some(n),
        _ => None,
    }
}

pub fn number(obj: &Object) -> Option<f32> {
    match obj {
        Object::Integer(n) => Some(*n as f32),
        Object::Real(n) => Some(*n),
        _ => None,
    }
}

pub fn array_f32(doc: &Document, obj: &Object) -> Option<Vec<f32>> {
    match resolve(doc, obj) {
        Object::Array(items) => Some(items.iter().map(|o| number(resolve(doc, o)).unwrap_or(0.0)).collect()),
        _ => None,
    }
}

pub fn stream_bytes(doc: &Document, obj: &Object) -> Option<Vec<u8>> {
    match resolve(doc, obj) {
        Object::Stream(stream) => stream
            .decompressed_content()
            .ok()
            .or_else(|| stream.filters().map_or(true, |f| f.is_empty()).then(|| stream.content.clone())),
        _ => None,
    }
}

/// Decode a PDF text string: UTF-16 with a BOM, else PDFDocEncoding.
pub fn text_string(bytes: &[u8]) -> String {
    if bytes.len() >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF {
        return crate::cmap::utf16be_to_string(&bytes[2..]);
    }
    if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE {
        let units: Vec<u16> = bytes[2..]
            .chunks(2)
            .map(|p| if p.len() == 2 { u16::from_le_bytes([p[0], p[1]]) } else { p[0] as u16 })
            .collect();
        return char::decode_utf16(units).map(|r| r.unwrap_or('\u{FFFD}')).collect();
    }
    if bytes.len() >= 3 && bytes[..3] == [0xEF, 0xBB, 0xBF] {
        return String::from_utf8_lossy(&bytes[3..]).into_owned();
    }
    bytes.iter().map(|&b| pdf_doc_char(b)).collect()
}

fn pdf_doc_char(b: u8) -> char {
    const HIGH: [u16; 32] = [
        0x2022, 0x2020, 0x2021, 0x2026, 0x2014, 0x2013, 0x0192, 0x2044, 0x2039, 0x203A, 0x2212, 0x2030, 0x201E, 0x201C,
        0x201D, 0x2018, 0x2019, 0x201A, 0x2122, 0xFB01, 0xFB02, 0x0141, 0x0152, 0x0160, 0x0178, 0x017D, 0x0131, 0x0142,
        0x0153, 0x0161, 0x017E, 0xFFFD,
    ];
    match b {
        0x80..=0x9F => char::from_u32(HIGH[(b - 0x80) as usize] as u32).unwrap_or('\u{FFFD}'),
        0x18..=0x1F => ['\u{02D8}', '\u{02C7}', '\u{02C6}', '\u{02D9}', '\u{02DD}', '\u{02DB}', '\u{02DA}', '\u{02DC}']
            [(b - 0x18) as usize],
        _ => b as char,
    }
}
