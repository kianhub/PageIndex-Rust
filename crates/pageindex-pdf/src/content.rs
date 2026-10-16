//! Content-stream interpreter: walks text operators (and Form XObjects) and
//! emits one positioned [`Glyph`] per shown code, in paint order.

use std::collections::HashSet;
use std::sync::Arc;

use lopdf::content::Content;
use lopdf::{Dictionary, Document, Object, ObjectId};

use crate::font::Font;
use crate::objects::{array_f32, as_dict, dict_get, name_of, number, resolve, stream_bytes};
use crate::{FontCache, Glyph, PageGeometry};

pub type Matrix = [f32; 6];

pub const IDENTITY: Matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// `m` then `n`, in PDF's row-vector convention.
pub fn concat(m: &Matrix, n: &Matrix) -> Matrix {
    [
        m[0] * n[0] + m[1] * n[2],
        m[0] * n[1] + m[1] * n[3],
        m[2] * n[0] + m[3] * n[2],
        m[2] * n[1] + m[3] * n[3],
        m[4] * n[0] + m[5] * n[2] + n[4],
        m[4] * n[1] + m[5] * n[3] + n[5],
    ]
}

pub fn apply(m: &Matrix, x: f32, y: f32) -> (f32, f32) {
    (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])
}

const MAX_FORM_DEPTH: usize = 12;
const ASCENT: f32 = 0.8;
const DESCENT: f32 = -0.2;

#[derive(Clone)]
struct GraphicsState {
    ctm: Matrix,
    font: Option<Arc<Font>>,
    font_size: f32,
    char_spacing: f32,
    word_spacing: f32,
    h_scale: f32,
    leading: f32,
    rise: f32,
}

impl GraphicsState {
    fn new(ctm: Matrix) -> Self {
        Self {
            ctm,
            font: None,
            font_size: 0.0,
            char_spacing: 0.0,
            word_spacing: 0.0,
            h_scale: 1.0,
            leading: 0.0,
            rise: 0.0,
        }
    }
}

pub struct Interpreter<'a> {
    doc: &'a Document,
    fonts: &'a FontCache,
    geometry: &'a PageGeometry,
    glyphs: Vec<Glyph>,
    forms: HashSet<ObjectId>,
}

impl<'a> Interpreter<'a> {
    pub fn new(doc: &'a Document, fonts: &'a FontCache, geometry: &'a PageGeometry) -> Self {
        Self { doc, fonts, geometry, glyphs: Vec::new(), forms: HashSet::new() }
    }

    pub fn run(mut self, content: &[u8], resources: Option<&Dictionary>) -> Vec<Glyph> {
        self.walk(content, resources, GraphicsState::new(IDENTITY), 0);
        self.glyphs
    }

    fn walk(&mut self, content: &[u8], resources: Option<&Dictionary>, initial: GraphicsState, depth: usize) {
        let Ok(content) = Content::decode(content) else {
            return;
        };
        let mut gs = initial;
        let mut stack: Vec<GraphicsState> = Vec::new();
        let mut tm = IDENTITY;
        let mut tlm = IDENTITY;
        for op in &content.operations {
            let args = &op.operands;
            let num = |i: usize| args.get(i).and_then(number).unwrap_or(0.0);
            match op.operator.as_str() {
                "q" => stack.push(gs.clone()),
                "Q" => {
                    if let Some(saved) = stack.pop() {
                        gs = saved;
                    }
                }
                "cm" if args.len() == 6 => {
                    let m = [num(0), num(1), num(2), num(3), num(4), num(5)];
                    gs.ctm = concat(&m, &gs.ctm);
                }
                "BT" => {
                    tm = IDENTITY;
                    tlm = IDENTITY;
                }
                "Tf" if args.len() == 2 => {
                    gs.font = args.first().and_then(name_of).map(|name| self.font(resources, name));
                    gs.font_size = num(1);
                }
                "Tc" => gs.char_spacing = num(0),
                "Tw" => gs.word_spacing = num(0),
                "Tz" => gs.h_scale = num(0) / 100.0,
                "TL" => gs.leading = num(0),
                "Ts" => gs.rise = num(0),
                "Td" => {
                    tlm = concat(&[1.0, 0.0, 0.0, 1.0, num(0), num(1)], &tlm);
                    tm = tlm;
                }
                "TD" => {
                    gs.leading = -num(1);
                    tlm = concat(&[1.0, 0.0, 0.0, 1.0, num(0), num(1)], &tlm);
                    tm = tlm;
                }
                "Tm" if args.len() == 6 => {
                    tlm = [num(0), num(1), num(2), num(3), num(4), num(5)];
                    tm = tlm;
                }
                "T*" => {
                    tlm = concat(&[1.0, 0.0, 0.0, 1.0, 0.0, -gs.leading], &tlm);
                    tm = tlm;
                }
                "Tj" => {
                    if let Some(Object::String(bytes, _)) = args.first() {
                        self.show(&gs, &mut tm, bytes);
                    }
                }
                "'" => {
                    tlm = concat(&[1.0, 0.0, 0.0, 1.0, 0.0, -gs.leading], &tlm);
                    tm = tlm;
                    if let Some(Object::String(bytes, _)) = args.first() {
                        self.show(&gs, &mut tm, bytes);
                    }
                }
                "\"" => {
                    gs.word_spacing = num(0);
                    gs.char_spacing = num(1);
                    tlm = concat(&[1.0, 0.0, 0.0, 1.0, 0.0, -gs.leading], &tlm);
                    tm = tlm;
                    if let Some(Object::String(bytes, _)) = args.get(2) {
                        self.show(&gs, &mut tm, bytes);
                    }
                }
                "TJ" => {
                    if let Some(Object::Array(items)) = args.first() {
                        for item in items {
                            match item {
                                Object::String(bytes, _) => self.show(&gs, &mut tm, bytes),
                                other => {
                                    if let Some(adjust) = number(other) {
                                        let shift = -adjust / 1000.0 * gs.font_size;
                                        let vertical = gs.font.as_ref().is_some_and(|f| f.vertical);
                                        let step = if vertical {
                                            [1.0, 0.0, 0.0, 1.0, 0.0, shift]
                                        } else {
                                            [1.0, 0.0, 0.0, 1.0, shift * gs.h_scale, 0.0]
                                        };
                                        tm = concat(&step, &tm);
                                    }
                                }
                            }
                        }
                    }
                }
                "Do" if depth < MAX_FORM_DEPTH => {
                    if let Some(name) = args.first().and_then(name_of) {
                        self.form(resources, name, &gs, depth);
                    }
                }
                _ => {}
            }
        }
    }

    fn font(&self, resources: Option<&Dictionary>, name: &[u8]) -> Arc<Font> {
        let entry = resources
            .and_then(|r| dict_get(self.doc, r, b"Font"))
            .and_then(|fonts| as_dict(self.doc, fonts))
            .and_then(|fonts| fonts.get(name).ok());
        match entry {
            Some(Object::Reference(id)) => self.fonts.get_or_load(self.doc, *id),
            Some(obj) => match as_dict(self.doc, obj) {
                Some(dict) => Arc::new(Font::from_dict(self.doc, dict)),
                None => self.fonts.fallback(),
            },
            None => self.fonts.fallback(),
        }
    }

    fn form(&mut self, resources: Option<&Dictionary>, name: &[u8], gs: &GraphicsState, depth: usize) {
        let Some(entry) = resources
            .and_then(|r| dict_get(self.doc, r, b"XObject"))
            .and_then(|x| as_dict(self.doc, x))
            .and_then(|x| x.get(name).ok())
        else {
            return;
        };
        let id = match entry {
            Object::Reference(id) => Some(*id),
            _ => None,
        };
        if let Some(id) = id
            && !self.forms.insert(id)
        {
            return;
        }
        let doc = self.doc;
        let Object::Stream(stream) = resolve(doc, entry) else {
            return;
        };
        if dict_get(doc, &stream.dict, b"Subtype").and_then(name_of) != Some(b"Form") {
            if let Some(id) = id {
                self.forms.remove(&id);
            }
            return;
        }
        let matrix = dict_get(doc, &stream.dict, b"Matrix")
            .and_then(|m| array_f32(doc, m))
            .filter(|m| m.len() == 6)
            .map(|m| [m[0], m[1], m[2], m[3], m[4], m[5]])
            .unwrap_or(IDENTITY);
        let own = dict_get(doc, &stream.dict, b"Resources").and_then(|r| as_dict(doc, r));
        if let Some(data) = stream_bytes(doc, resolve(doc, entry)) {
            let mut inner = gs.clone();
            inner.ctm = concat(&matrix, &gs.ctm);
            self.walk(&data, own.or(resources), inner, depth + 1);
        }
        if let Some(id) = id {
            self.forms.remove(&id);
        }
    }

    fn show(&mut self, gs: &GraphicsState, tm: &mut Matrix, bytes: &[u8]) {
        let Some(font) = gs.font.clone() else {
            return;
        };
        let size = gs.font_size;
        for code in font.decode(bytes) {
            let text_matrix = concat(tm, &gs.ctm);
            let trm = concat(&[size * gs.h_scale, 0.0, 0.0, size, 0.0, gs.rise], &text_matrix);
            let effective = size.abs() * font.size_scale * text_matrix[2].hypot(text_matrix[3]);
            let advance = code.width * size + gs.char_spacing + if code.is_word_space { gs.word_spacing } else { 0.0 };
            if let Some(text) = code.text.as_deref() {
                self.emit(&font, &trm, effective, code.width, text);
            }
            let step = if font.vertical {
                [1.0, 0.0, 0.0, 1.0, 0.0, -advance.abs().max(size.abs())]
            } else {
                [1.0, 0.0, 0.0, 1.0, advance * gs.h_scale, 0.0]
            };
            *tm = concat(&step, tm);
        }
    }

    fn emit(&mut self, font: &Font, trm: &Matrix, size: f32, width: f32, raw: &str) {
        let text = crate::encoding::normalize_glyph_text(raw);
        let text: String = text.chars().filter(|c| !c.is_control() || *c == '\t').collect();
        if text.is_empty() {
            return;
        }
        let scale = font.size_scale;
        let corners = if font.vertical {
            [(-0.5, -1.0), (0.5, -1.0), (-0.5, 0.0), (0.5, 0.0)]
        } else {
            let w = width.max(0.0);
            [(0.0, DESCENT * scale), (w, DESCENT * scale), (0.0, ASCENT * scale), (w, ASCENT * scale)]
        };
        let geom = self.geometry;
        let mut x0 = f32::INFINITY;
        let mut x1 = f32::NEG_INFINITY;
        let mut top = f32::INFINITY;
        let mut bottom = f32::NEG_INFINITY;
        for (gx, gy) in corners {
            let (ux, uy) = apply(trm, gx, gy);
            let (dx, dy) = geom.to_display(ux, uy);
            x0 = x0.min(dx);
            x1 = x1.max(dx);
            top = top.min(dy);
            bottom = bottom.max(dy);
        }
        if !(x0.is_finite() && top.is_finite()) {
            return;
        }
        let (ox, oy) = apply(trm, 0.0, 0.0);
        let (ex, ey) = apply(trm, 1.0, 0.0);
        let (dox, doy) = geom.to_display(ox, oy);
        let (dex, dey) = geom.to_display(ex, ey);
        let (bx, by) = (dex - dox, dey - doy);
        let upright = !font.vertical && bx > 0.0 && by.abs() <= 0.1 * bx.abs();
        let cx = (x0 + x1) / 2.0;
        let cy = (top + bottom) / 2.0;
        if cx < -1.0 || cy < -1.0 || cx > geom.width + 1.0 || cy > geom.height + 1.0 {
            return;
        }
        self.glyphs.push(Glyph {
            text,
            x0,
            x1,
            top,
            bottom,
            baseline: doy,
            size,
            font: font.name.clone(),
            bold: font.bold,
            italic: font.italic,
            upright,
            vertical: font.vertical,
        });
    }
}
