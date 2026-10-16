//! CMap parsing: ToUnicode maps (`bfchar` / `bfrange`) and encoding CMaps
//! (`codespacerange`, `cidrange` / `cidchar`).

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Hex(Vec<u8>),
    Int(i64),
    Name(Vec<u8>),
    Keyword(Vec<u8>),
    ArrayOpen,
    ArrayClose,
    Other,
}

struct Lexer<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.data.get(self.pos).copied()
    }

    fn skip_space(&mut self) {
        while let Some(b) = self.peek() {
            if b == b'%' {
                while let Some(c) = self.peek() {
                    if c == b'\n' || c == b'\r' {
                        break;
                    }
                    self.pos += 1;
                }
            } else if is_space(b) {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn next_token(&mut self) -> Option<Token> {
        self.skip_space();
        let b = self.peek()?;
        match b {
            b'<' if self.data.get(self.pos + 1) == Some(&b'<') => {
                self.pos += 2;
                Some(Token::Other)
            }
            b'>' if self.data.get(self.pos + 1) == Some(&b'>') => {
                self.pos += 2;
                Some(Token::Other)
            }
            b'<' => {
                self.pos += 1;
                let mut digits = Vec::new();
                while let Some(c) = self.peek() {
                    self.pos += 1;
                    if c == b'>' {
                        break;
                    }
                    if c.is_ascii_hexdigit() {
                        digits.push(c);
                    }
                }
                if digits.len() % 2 == 1 {
                    digits.push(b'0');
                }
                let bytes = digits.chunks(2).map(|pair| (hex_val(pair[0]) << 4) | hex_val(pair[1])).collect();
                Some(Token::Hex(bytes))
            }
            b'[' => {
                self.pos += 1;
                Some(Token::ArrayOpen)
            }
            b']' => {
                self.pos += 1;
                Some(Token::ArrayClose)
            }
            b'(' => {
                let mut depth = 0usize;
                while let Some(c) = self.peek() {
                    self.pos += 1;
                    match c {
                        b'\\' => self.pos += 1,
                        b'(' => depth += 1,
                        b')' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                Some(Token::Other)
            }
            b'/' => {
                self.pos += 1;
                let start = self.pos;
                while self.peek().is_some_and(|c| !is_space(c) && !is_delim(c)) {
                    self.pos += 1;
                }
                Some(Token::Name(self.data[start..self.pos].to_vec()))
            }
            _ if is_delim(b) => {
                self.pos += 1;
                Some(Token::Other)
            }
            _ => {
                let start = self.pos;
                while self.peek().is_some_and(|c| !is_space(c) && !is_delim(c)) {
                    self.pos += 1;
                }
                let word = &self.data[start..self.pos];
                match std::str::from_utf8(word).ok().and_then(|w| w.parse::<i64>().ok()) {
                    Some(n) => Some(Token::Int(n)),
                    None => Some(Token::Keyword(word.to_vec())),
                }
            }
        }
    }
}

fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n' | b'\x0c' | b'\0')
}

fn is_delim(b: u8) -> bool {
    matches!(b, b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%')
}

fn hex_val(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => 0,
    }
}

fn bytes_to_code(bytes: &[u8]) -> u32 {
    bytes.iter().take(4).fold(0u32, |acc, &b| (acc << 8) | b as u32)
}

/// Decode a ToUnicode destination: UTF-16BE, lone surrogates become U+FFFD.
pub fn utf16be_to_string(bytes: &[u8]) -> String {
    let units: Vec<u16> = bytes
        .chunks(2)
        .map(|pair| if pair.len() == 2 { u16::from_be_bytes([pair[0], pair[1]]) } else { pair[0] as u16 })
        .collect();
    char::decode_utf16(units).map(|r| r.unwrap_or('\u{FFFD}')).collect()
}

#[derive(Debug, Clone, Copy)]
struct CodeRange {
    len: usize,
    lo: u32,
    hi: u32,
}

#[derive(Debug, Clone)]
struct UnicodeRange {
    lo: u32,
    hi: u32,
    base: Vec<u16>,
}

#[derive(Debug, Clone, Copy)]
struct CidRange {
    lo: u32,
    hi: u32,
    cid: u32,
}

/// A parsed CMap. One structure serves both roles: ToUnicode (code -> text)
/// and encoding (byte string -> codes -> CIDs).
#[derive(Debug, Clone, Default)]
pub struct CMap {
    codespace: Vec<CodeRange>,
    unicode_chars: HashMap<u32, String>,
    unicode_ranges: Vec<UnicodeRange>,
    cid_chars: HashMap<u32, u32>,
    cid_ranges: Vec<CidRange>,
    /// The predefined CMap this one extends via `usecmap`.
    pub use_cmap: Option<Vec<u8>>,
    pub vertical: bool,
}

impl CMap {
    pub fn parse(data: &[u8]) -> Self {
        let mut cmap = CMap::default();
        let mut lexer = Lexer::new(data);
        let mut operands: Vec<Token> = Vec::new();
        while let Some(token) = lexer.next_token() {
            match &token {
                Token::Keyword(word) => {
                    match word.as_slice() {
                        b"begincodespacerange" => cmap.read_codespace(&mut lexer),
                        b"beginbfchar" => cmap.read_bfchar(&mut lexer),
                        b"beginbfrange" => cmap.read_bfrange(&mut lexer),
                        b"begincidchar" => cmap.read_cidchar(&mut lexer),
                        b"begincidrange" => cmap.read_cidrange(&mut lexer),
                        b"usecmap" => {
                            if let Some(Token::Name(name)) = operands.last() {
                                cmap.use_cmap = Some(name.clone());
                            }
                        }
                        b"def" => {
                            if let [.., Token::Name(key), Token::Int(value)] = operands.as_slice()
                                && key == b"WMode"
                            {
                                cmap.vertical = *value == 1;
                            }
                        }
                        _ => {}
                    }
                    operands.clear();
                }
                _ => {
                    operands.push(token);
                    if operands.len() > 8 {
                        operands.remove(0);
                    }
                }
            }
        }
        cmap
    }

    /// A two-byte identity CMap (`Identity-H` / `Identity-V`).
    pub fn identity(vertical: bool) -> Self {
        CMap {
            codespace: vec![CodeRange { len: 2, lo: 0, hi: 0xFFFF }],
            cid_ranges: vec![CidRange { lo: 0, hi: 0xFFFF, cid: 0 }],
            vertical,
            ..CMap::default()
        }
    }

    fn read_codespace(&mut self, lexer: &mut Lexer) {
        loop {
            match lexer.next_token() {
                Some(Token::Hex(lo)) => {
                    if let Some(Token::Hex(hi)) = lexer.next_token() {
                        self.codespace.push(CodeRange {
                            len: lo.len().clamp(1, 4),
                            lo: bytes_to_code(&lo),
                            hi: bytes_to_code(&hi),
                        });
                    }
                }
                Some(Token::Keyword(_)) | None => return,
                _ => {}
            }
        }
    }

    fn read_bfchar(&mut self, lexer: &mut Lexer) {
        loop {
            match lexer.next_token() {
                Some(Token::Hex(src)) => match lexer.next_token() {
                    Some(Token::Hex(dst)) => {
                        self.unicode_chars.insert(bytes_to_code(&src), utf16be_to_string(&dst));
                    }
                    Some(Token::Name(name)) => {
                        if let Some(text) = crate::encoding::glyph_name_to_unicode(&String::from_utf8_lossy(&name)) {
                            self.unicode_chars.insert(bytes_to_code(&src), text);
                        }
                    }
                    Some(Token::Keyword(_)) | None => return,
                    _ => {}
                },
                Some(Token::Keyword(_)) | None => return,
                _ => {}
            }
        }
    }

    fn read_bfrange(&mut self, lexer: &mut Lexer) {
        loop {
            let lo = match lexer.next_token() {
                Some(Token::Hex(lo)) => bytes_to_code(&lo),
                Some(Token::Keyword(_)) | None => return,
                _ => continue,
            };
            let hi = match lexer.next_token() {
                Some(Token::Hex(hi)) => bytes_to_code(&hi),
                Some(Token::Keyword(_)) | None => return,
                _ => continue,
            };
            if hi < lo {
                continue;
            }
            match lexer.next_token() {
                Some(Token::Hex(dst)) => {
                    let units: Vec<u16> = dst
                        .chunks(2)
                        .map(|p| if p.len() == 2 { u16::from_be_bytes([p[0], p[1]]) } else { p[0] as u16 })
                        .collect();
                    if !units.is_empty() {
                        self.unicode_ranges.push(UnicodeRange { lo, hi, base: units });
                    }
                }
                Some(Token::ArrayOpen) => {
                    let mut code = lo;
                    loop {
                        match lexer.next_token() {
                            Some(Token::Hex(dst)) => {
                                if code <= hi {
                                    self.unicode_chars.insert(code, utf16be_to_string(&dst));
                                }
                                code = code.saturating_add(1);
                            }
                            Some(Token::ArrayClose) | None => break,
                            Some(Token::Keyword(_)) => return,
                            _ => {}
                        }
                    }
                }
                Some(Token::Keyword(_)) | None => return,
                _ => {}
            }
        }
    }

    fn read_cidchar(&mut self, lexer: &mut Lexer) {
        loop {
            match lexer.next_token() {
                Some(Token::Hex(src)) => {
                    if let Some(Token::Int(cid)) = lexer.next_token() {
                        self.cid_chars.insert(bytes_to_code(&src), cid.max(0) as u32);
                    }
                    self.note_codespace(src.len());
                }
                Some(Token::Keyword(_)) | None => return,
                _ => {}
            }
        }
    }

    fn read_cidrange(&mut self, lexer: &mut Lexer) {
        loop {
            match lexer.next_token() {
                Some(Token::Hex(lo)) => {
                    let len = lo.len();
                    if let (Some(Token::Hex(hi)), Some(Token::Int(cid))) = (lexer.next_token(), lexer.next_token()) {
                        self.cid_ranges.push(CidRange {
                            lo: bytes_to_code(&lo),
                            hi: bytes_to_code(&hi),
                            cid: cid.max(0) as u32,
                        });
                    }
                    self.note_codespace(len);
                }
                Some(Token::Keyword(_)) | None => return,
                _ => {}
            }
        }
    }

    fn note_codespace(&mut self, len: usize) {
        if self.codespace.is_empty() && (1..=4).contains(&len) {
            let hi = if len >= 4 { u32::MAX } else { (1u32 << (8 * len)) - 1 };
            self.codespace.push(CodeRange { len, lo: 0, hi });
        }
    }

    pub fn is_empty(&self) -> bool {
        self.unicode_chars.is_empty() && self.unicode_ranges.is_empty()
    }

    /// The text a ToUnicode CMap assigns to `code`.
    pub fn lookup_unicode(&self, code: u32) -> Option<String> {
        if let Some(text) = self.unicode_chars.get(&code) {
            return Some(text.clone());
        }
        self.unicode_ranges.iter().rev().find(|r| r.lo <= code && code <= r.hi).map(|r| {
            let mut units = r.base.clone();
            let offset = code - r.lo;
            if let Some(last) = units.last_mut() {
                *last = last.wrapping_add(offset as u16);
            }
            char::decode_utf16(units).map(|c| c.unwrap_or('\u{FFFD}')).collect()
        })
    }

    pub fn lookup_cid(&self, code: u32) -> Option<u32> {
        if let Some(&cid) = self.cid_chars.get(&code) {
            return Some(cid);
        }
        self.cid_ranges.iter().find(|r| r.lo <= code && code <= r.hi).map(|r| r.cid + (code - r.lo))
    }

    pub fn has_codespace(&self) -> bool {
        !self.codespace.is_empty()
    }

    /// Split a shown string into `(code, byte length)` pairs by codespace.
    pub fn split_codes(&self, bytes: &[u8]) -> Vec<(u32, usize)> {
        let mut out = Vec::with_capacity(bytes.len());
        let mut pos = 0;
        let shortest = self.codespace.iter().map(|r| r.len).min().unwrap_or(1);
        while pos < bytes.len() {
            let mut matched = None;
            for len in 1..=4usize {
                if pos + len > bytes.len() {
                    break;
                }
                let code = bytes_to_code(&bytes[pos..pos + len]);
                if self.codespace.iter().any(|r| r.len == len && r.lo <= code && code <= r.hi) {
                    matched = Some((code, len));
                    break;
                }
            }
            let (code, len) = matched.unwrap_or_else(|| {
                let len = shortest.min(bytes.len() - pos);
                (bytes_to_code(&bytes[pos..pos + len]), len)
            });
            out.push((code, len));
            pos += len;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bfchar_and_both_bfrange_forms() {
        let data = b"/CIDInit /ProcSet findresource begin
            1 begincodespacerange <0000> <FFFF> endcodespacerange
            2 beginbfchar <0003> <0020> <0011> <00660069> endbfchar
            2 beginbfrange <0024> <0026> <0041> <0030> <0031> [<0078> <D835DC4E>] endbfrange
            endcmap";
        let cmap = CMap::parse(data);
        assert_eq!(cmap.lookup_unicode(3).as_deref(), Some(" "));
        assert_eq!(cmap.lookup_unicode(0x11).as_deref(), Some("fi"));
        assert_eq!(cmap.lookup_unicode(0x26).as_deref(), Some("C"));
        assert_eq!(cmap.lookup_unicode(0x30).as_deref(), Some("x"));
        assert_eq!(cmap.lookup_unicode(0x31).as_deref(), Some("\u{1d44e}"));
        assert_eq!(cmap.split_codes(&[0, 3, 0, 0x11]), vec![(3, 2), (0x11, 2)]);
    }

    #[test]
    fn lone_surrogates_become_replacement_chars() {
        assert_eq!(utf16be_to_string(&[0xD8, 0x3D]), "\u{FFFD}");
    }

    #[test]
    fn mixed_width_codespace_splits_by_range() {
        let data = b"2 begincodespacerange <00> <80> <8140> <9FFC> endcodespacerange
            1 begincidrange <8140> <817E> 633 endcidrange";
        let cmap = CMap::parse(data);
        assert_eq!(cmap.split_codes(&[0x41, 0x81, 0x40]), vec![(0x41, 1), (0x8140, 2)]);
        assert_eq!(cmap.lookup_cid(0x8141), Some(634));
    }
}
