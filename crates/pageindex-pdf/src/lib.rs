//! Character-level PDF text extraction: every shown glyph with its Unicode
//! text, font, effective size and box in displayed page coordinates
//! (origin top-left, y down, after `/Rotate`), plus the embedded outline.
//!
//! Pages are interpreted in parallel; fonts are parsed once per document.

mod cmap;
mod content;
mod encoding;
mod font;
mod metrics;
mod objects;
mod outline;

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use lopdf::{Dictionary, Document, Object, ObjectId};
use rayon::prelude::*;

pub use outline::Bookmark;

use crate::font::Font;
use crate::objects::{array_f32, as_dict, dict_get, number};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("PDF file not found: {0}")]
    NotFound(String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("File does not look like a PDF")]
    NotPdf,
    #[error("PDF is encrypted or password-protected")]
    Encrypted,
    #[error("PDF contains no pages")]
    NoPages,
    #[error("Could not open PDF: {0}")]
    Malformed(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// One shown glyph. Coordinates are in points on the displayed page.
#[derive(Debug, Clone)]
pub struct Glyph {
    pub text: String,
    pub x0: f32,
    pub x1: f32,
    pub top: f32,
    pub bottom: f32,
    /// Display y of the glyph origin (its baseline for horizontal text).
    pub baseline: f32,
    /// Effective rendered font size.
    pub size: f32,
    pub font: Arc<str>,
    pub bold: bool,
    pub italic: bool,
    /// Left-to-right horizontal text on the displayed page.
    pub upright: bool,
    pub vertical: bool,
}

impl Glyph {
    pub fn is_space(&self) -> bool {
        self.text.chars().all(char::is_whitespace)
    }
}

/// A page's glyphs in paint (content-stream) order.
#[derive(Debug, Clone)]
pub struct Page {
    /// 0-based page index.
    pub index: usize,
    pub width: f32,
    pub height: f32,
    pub rotation: i32,
    pub glyphs: Vec<Glyph>,
}

/// The visible box of a page and the transform onto the displayed page.
#[derive(Debug, Clone, Copy)]
pub struct PageGeometry {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub rotation: i32,
    pub width: f32,
    pub height: f32,
}

impl PageGeometry {
    fn new(view: [f32; 4], rotation: i32) -> Self {
        let [x0, y0, x1, y1] = view;
        let (w, h) = ((x1 - x0).abs(), (y1 - y0).abs());
        let (width, height) = if rotation % 180 == 90 { (h, w) } else { (w, h) };
        Self { x0: x0.min(x1), y0: y0.min(y1), x1: x0.max(x1), y1: y0.max(y1), rotation, width, height }
    }

    /// User space to displayed coordinates (origin top-left, y down).
    pub fn to_display(&self, x: f32, y: f32) -> (f32, f32) {
        match self.rotation {
            90 => (y - self.y0, x - self.x0),
            180 => (self.x1 - x, y - self.y0),
            270 => (self.y1 - y, self.x1 - x),
            _ => (x - self.x0, self.y1 - y),
        }
    }
}

pub(crate) struct FontCache {
    fonts: Mutex<HashMap<ObjectId, Arc<Font>>>,
    fallback: Arc<Font>,
}

impl FontCache {
    fn new() -> Self {
        Self { fonts: Mutex::new(HashMap::new()), fallback: Arc::new(Font::fallback()) }
    }

    pub(crate) fn get_or_load(&self, doc: &Document, id: ObjectId) -> Arc<Font> {
        if let Some(font) = self.fonts.lock().expect("font cache lock").get(&id) {
            return font.clone();
        }
        let font = match as_dict(doc, &Object::Reference(id)) {
            Some(dict) => Arc::new(Font::from_dict(doc, dict)),
            None => self.fallback.clone(),
        };
        self.fonts.lock().expect("font cache lock").entry(id).or_insert(font).clone()
    }

    pub(crate) fn fallback(&self) -> Arc<Font> {
        self.fallback.clone()
    }
}

pub struct PdfDocument {
    doc: Document,
    pages: Vec<ObjectId>,
    fonts: FontCache,
}

impl PdfDocument {
    /// Open a PDF file, checking that it exists and starts with `%PDF-`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(Error::NotFound(path.display().to_string()));
        }
        Self::from_bytes(&std::fs::read(path)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if !bytes.starts_with(b"%PDF-") {
            return Err(Error::NotPdf);
        }
        let doc = Document::load_mem(bytes).map_err(|err| {
            let message = err.to_string();
            let lower = message.to_ascii_lowercase();
            if lower.contains("password") || lower.contains("decrypt") || lower.contains("encrypt") {
                Error::Encrypted
            } else {
                Error::Malformed(message)
            }
        })?;
        if doc.is_encrypted() {
            return Err(Error::Encrypted);
        }
        let pages: Vec<ObjectId> = doc.get_pages().into_values().collect();
        if pages.is_empty() {
            return Err(Error::NoPages);
        }
        Ok(Self { doc, pages, fonts: FontCache::new() })
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// The document's `/Info` title, if set.
    pub fn info_title(&self) -> Option<String> {
        let info = self.doc.trailer.get(b"Info").ok().and_then(|o| as_dict(&self.doc, o))?;
        match dict_get(&self.doc, info, b"Title") {
            Some(Object::String(bytes, _)) => Some(objects::text_string(bytes)).filter(|t| !t.trim().is_empty()),
            _ => None,
        }
    }

    pub fn page(&self, index: usize) -> Page {
        let id = self.pages[index];
        let page_dict = self.doc.get_dictionary(id).ok();
        let media = page_dict
            .and_then(|d| self.inherited(d, b"MediaBox"))
            .and_then(|o| array_f32(&self.doc, o))
            .filter(|b| b.len() == 4)
            .map(|b| [b[0], b[1], b[2], b[3]])
            .unwrap_or([0.0, 0.0, 612.0, 792.0]);
        let crop = page_dict
            .and_then(|d| self.inherited(d, b"CropBox"))
            .and_then(|o| array_f32(&self.doc, o))
            .filter(|b| b.len() == 4)
            .map(|b| [b[0], b[1], b[2], b[3]]);
        let view = match crop {
            Some(c) => intersect(normalize_box(c), normalize_box(media)),
            None => normalize_box(media),
        };
        let rotation = page_dict
            .and_then(|d| self.inherited(d, b"Rotate"))
            .and_then(number)
            .map(|r| (r as i32).rem_euclid(360) / 90 * 90)
            .unwrap_or(0);
        let geometry = PageGeometry::new(view, rotation);
        let resources = page_dict.and_then(|d| self.inherited(d, b"Resources")).and_then(|o| as_dict(&self.doc, o));
        let content = self.doc.get_page_content(id);
        let glyphs = content::Interpreter::new(&self.doc, &self.fonts, &geometry).run(&content, resources);
        Page { index, width: geometry.width, height: geometry.height, rotation, glyphs }
    }

    /// Every page, interpreted in parallel.
    pub fn pages(&self) -> Vec<Page> {
        (0..self.pages.len()).into_par_iter().map(|i| self.page(i)).collect()
    }

    pub fn bookmarks(&self) -> Vec<Bookmark> {
        let numbers: HashMap<ObjectId, usize> = self.pages.iter().enumerate().map(|(i, id)| (*id, i + 1)).collect();
        outline::read(&self.doc, &numbers)
    }

    fn inherited<'a>(&'a self, page: &'a Dictionary, key: &[u8]) -> Option<&'a Object> {
        let mut node = page;
        for _ in 0..64 {
            if let Some(value) = dict_get(&self.doc, node, key) {
                return Some(value);
            }
            node = dict_get(&self.doc, node, b"Parent").and_then(|p| as_dict(&self.doc, p))?;
        }
        None
    }
}

fn normalize_box(b: [f32; 4]) -> [f32; 4] {
    [b[0].min(b[2]), b[1].min(b[3]), b[0].max(b[2]), b[1].max(b[3])]
}

fn intersect(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let out = [a[0].max(b[0]), a[1].max(b[1]), a[2].min(b[2]), a[3].min(b[3])];
    if out[2] > out[0] && out[3] > out[1] { out } else { b }
}
