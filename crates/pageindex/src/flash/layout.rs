//! Glyphs to lines to blocks, in paint order.
//!
//! A run is a stretch of glyphs on one baseline; runs that overlap
//! vertically and sit side by side form a line (so superscripts and
//! separately painted words join their line); lines close in size, weight
//! and spacing form a block.

use std::collections::HashMap;
use std::sync::Arc;

use pageindex_pdf::{Glyph, Page};

/// Gap (in font sizes) above which a space is inserted between glyphs.
const SPACE_GAP: f32 = 0.18;
/// Gap (in font sizes) above which a run breaks: a column gutter or table cell.
const RUN_BREAK_GAP: f32 = 1.6;

#[derive(Debug, Clone)]
pub struct Line {
    pub text: String,
    pub x0: f32,
    pub x1: f32,
    pub top: f32,
    pub bottom: f32,
    /// Dominant font size by character count.
    pub size: f32,
    pub bold_chars: usize,
    pub chars: usize,
    pub font: Arc<str>,
    pub upright: bool,
}

impl Line {
    pub fn height(&self) -> f32 {
        (self.bottom - self.top).max(0.1)
    }

    pub fn is_bold(&self) -> bool {
        self.chars > 0 && self.bold_chars * 2 > self.chars
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    Body,
    Header,
    Footer,
    Title,
    /// A block on a table-of-contents page.
    Toc,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub lines: Vec<Line>,
    pub x0: f32,
    pub x1: f32,
    pub top: f32,
    pub bottom: f32,
    pub size: f32,
    pub bold: bool,
    pub font: Arc<str>,
    pub upright: bool,
    pub kind: BlockKind,
}

impl Block {
    fn from_line(line: Line) -> Self {
        Self {
            x0: line.x0,
            x1: line.x1,
            top: line.top,
            bottom: line.bottom,
            size: line.size,
            bold: line.is_bold(),
            font: line.font.clone(),
            upright: line.upright,
            kind: BlockKind::Body,
            lines: vec![line],
        }
    }

    fn push(&mut self, line: Line) {
        self.x0 = self.x0.min(line.x0);
        self.x1 = self.x1.max(line.x1);
        self.top = self.top.min(line.top);
        self.bottom = self.bottom.max(line.bottom);
        self.lines.push(line);
        let (mut sizes, mut bold, mut chars) = (HashMap::new(), 0usize, 0usize);
        for line in &self.lines {
            *sizes.entry(size_key(line.size)).or_insert(0usize) += line.chars;
            bold += line.bold_chars;
            chars += line.chars;
        }
        self.size = sizes.into_iter().max_by_key(|(k, n)| (*n, *k)).map(|(k, _)| k as f32 / 10.0).unwrap_or(self.size);
        self.bold = chars > 0 && bold * 2 > chars;
    }

    /// Lines joined by single spaces.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for (i, line) in self.lines.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(&line.text);
        }
        out
    }

    pub fn chars(&self) -> usize {
        self.lines.iter().map(|l| l.chars).sum()
    }

    pub fn width(&self) -> f32 {
        self.x1 - self.x0
    }
}

/// A page laid out into blocks, in reading (paint) order.
#[derive(Debug, Clone)]
pub struct PageLayout {
    /// 1-based page number.
    pub number: usize,
    pub width: f32,
    pub height: f32,
    pub blocks: Vec<Block>,
}

impl PageLayout {
    pub fn text(&self) -> String {
        self.blocks.iter().map(Block::text).collect::<Vec<_>>().join("\n")
    }

    pub fn has_text(&self) -> bool {
        self.blocks.iter().any(|b| b.chars() > 0)
    }
}

/// Font sizes are compared at 0.1pt resolution.
pub fn size_key(size: f32) -> i32 {
    (size * 10.0).round() as i32
}

struct Run {
    text: String,
    x0: f32,
    x1: f32,
    top: f32,
    bottom: f32,
    baseline: f32,
    size: f32,
    sizes: HashMap<i32, usize>,
    bold_chars: usize,
    chars: usize,
    font: Arc<str>,
    upright: bool,
    pending_space: bool,
}

impl Run {
    fn start(glyph: &Glyph) -> Self {
        let mut run = Run {
            text: String::new(),
            x0: glyph.x0,
            x1: glyph.x1,
            top: glyph.top,
            bottom: glyph.bottom,
            baseline: glyph.baseline,
            size: glyph.size,
            sizes: HashMap::new(),
            bold_chars: 0,
            chars: 0,
            font: glyph.font.clone(),
            upright: glyph.upright,
            pending_space: false,
        };
        run.add(glyph);
        run
    }

    fn add(&mut self, glyph: &Glyph) {
        if glyph.is_space() {
            self.pending_space = !self.text.is_empty();
            self.x1 = self.x1.max(glyph.x1);
            return;
        }
        if self.pending_space && !self.text.ends_with(' ') {
            self.text.push(' ');
        }
        self.pending_space = false;
        self.text.push_str(&glyph.text);
        let n = glyph.text.chars().count();
        self.chars += n;
        if glyph.bold {
            self.bold_chars += n;
        }
        *self.sizes.entry(size_key(glyph.size)).or_insert(0) += n;
        self.x0 = self.x0.min(glyph.x0);
        self.x1 = self.x1.max(glyph.x1);
        self.top = self.top.min(glyph.top);
        self.bottom = self.bottom.max(glyph.bottom);
    }

    fn accepts(&self, glyph: &Glyph) -> Option<bool> {
        if glyph.upright != self.upright {
            return None;
        }
        let size = self.size.max(glyph.size).max(0.1);
        if !self.upright {
            let dx = (glyph.x0 - self.x1).max(self.x0 - glyph.x1).max(0.0);
            let dy = (glyph.top - self.bottom).max(self.top - glyph.bottom).max(0.0);
            return (dx <= RUN_BREAK_GAP * size && dy <= RUN_BREAK_GAP * size).then_some(false);
        }
        if (glyph.baseline - self.baseline).abs() > 0.3 * size {
            return None;
        }
        let gap = glyph.x0 - self.x1;
        if gap < -0.6 * size || gap > RUN_BREAK_GAP * size {
            return None;
        }
        Some(gap > SPACE_GAP * glyph.size.max(0.1))
    }

    fn finish(mut self) -> Option<Line> {
        let text = std::mem::take(&mut self.text).trim().to_owned();
        if text.is_empty() {
            return None;
        }
        let size =
            self.sizes.iter().max_by_key(|(k, n)| (**n, **k)).map(|(k, _)| *k as f32 / 10.0).unwrap_or(self.size);
        Some(Line {
            text,
            x0: self.x0,
            x1: self.x1,
            top: self.top,
            bottom: self.bottom,
            size,
            bold_chars: self.bold_chars,
            chars: self.chars,
            font: self.font,
            upright: self.upright,
        })
    }
}

fn runs(glyphs: &[Glyph]) -> Vec<Line> {
    let mut out = Vec::new();
    let mut current: Option<Run> = None;
    for glyph in glyphs {
        if glyph.size <= 0.0 || glyph.text.is_empty() {
            continue;
        }
        match current.as_mut().map(|run| run.accepts(glyph)) {
            Some(Some(space)) => {
                let run = current.as_mut().expect("run present");
                if space && !glyph.is_space() {
                    run.pending_space = !run.text.is_empty();
                }
                run.add(glyph);
            }
            _ => {
                if let Some(line) = current.take().and_then(Run::finish) {
                    out.push(line);
                }
                if !glyph.is_space() {
                    current = Some(Run::start(glyph));
                }
            }
        }
    }
    if let Some(line) = current.and_then(Run::finish) {
        out.push(line);
    }
    out
}

fn vertical_overlap(a: &Line, b: &Line) -> f32 {
    let overlap = a.bottom.min(b.bottom) - a.top.max(b.top);
    overlap / a.height().min(b.height())
}

fn merge_into_lines(pieces: Vec<Line>) -> Vec<Line> {
    let mut lines: Vec<Line> = Vec::with_capacity(pieces.len());
    for piece in pieces {
        if let Some(last) = lines.last_mut() {
            let size = last.size.max(piece.size);
            let gap = piece.x0 - last.x1;
            if last.upright
                && piece.upright
                && vertical_overlap(last, &piece) >= 0.5
                && gap > -0.6 * size
                && gap <= RUN_BREAK_GAP * size
            {
                if gap > SPACE_GAP * piece.size.min(last.size) && !last.text.ends_with(' ') {
                    last.text.push(' ');
                }
                last.text.push_str(&piece.text);
                let (main_chars, main_size) = (last.chars, last.size);
                if piece.chars > main_chars {
                    last.size = piece.size;
                    last.font = piece.font.clone();
                } else {
                    last.size = main_size;
                }
                last.chars += piece.chars;
                last.bold_chars += piece.bold_chars;
                last.x1 = last.x1.max(piece.x1);
                last.x0 = last.x0.min(piece.x0);
                if piece.size >= 0.8 * main_size {
                    last.top = last.top.min(piece.top);
                    last.bottom = last.bottom.max(piece.bottom);
                }
                continue;
            }
        }
        lines.push(piece);
    }
    lines
}

fn joins_block(block: &Block, line: &Line) -> bool {
    let prev = block.lines.last().expect("blocks hold a line");
    if !(prev.upright && line.upright) {
        return false;
    }
    let ratio = line.size / prev.size.max(0.1);
    if !(0.9..=1.1).contains(&ratio) || prev.is_bold() != line.is_bold() {
        return false;
    }
    let height = prev.height().max(line.height());
    let gap = line.top - prev.bottom;
    if gap < -0.5 * height || gap > 0.75 * height {
        return false;
    }
    line.x0 < block.x1 && line.x1 > block.x0
}

fn is_rtl(c: char) -> bool {
    matches!(c, '\u{0590}'..='\u{08FF}' | '\u{FB1D}'..='\u{FDFF}' | '\u{FE70}'..='\u{FEFF}')
}

/// Glyphs arrive in visual (left-to-right) order; a right-to-left run reads
/// reversed. Runs bounded by RTL letters are reversed in place and set off
/// from neighbouring left-to-right text by a space.
fn reorder_rtl(text: &str) -> String {
    if !text.chars().any(is_rtl) {
        return text.to_owned();
    }
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<char> = Vec::with_capacity(chars.len() + 2);
    let mut i = 0;
    while i < chars.len() {
        if !is_rtl(chars[i]) {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        let start = i;
        let mut end = i;
        let mut j = i;
        while j < chars.len() && (is_rtl(chars[j]) || !chars[j].is_alphanumeric()) {
            if is_rtl(chars[j]) {
                end = j;
            }
            j += 1;
        }
        if out.last().is_some_and(|c| !c.is_whitespace()) {
            out.push(' ');
        }
        out.extend(chars[start..=end].iter().rev());
        if chars.get(end + 1).is_some_and(|c| !c.is_whitespace()) {
            out.push(' ');
        }
        i = end + 1;
    }
    out.into_iter().collect()
}

/// Lay out one parsed page.
pub fn layout_page(page: &Page) -> PageLayout {
    let mut lines = merge_into_lines(runs(&page.glyphs));
    for line in lines.iter_mut() {
        line.text = reorder_rtl(&line.text);
    }
    let mut blocks: Vec<Block> = Vec::new();
    for line in lines {
        match blocks.last_mut() {
            Some(block) if joins_block(block, &line) => block.push(line),
            _ => blocks.push(Block::from_line(line)),
        }
    }
    PageLayout { number: page.index + 1, width: page.width, height: page.height, blocks }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glyph(text: &str, x0: f32, width: f32, baseline: f32, size: f32) -> Glyph {
        Glyph {
            text: text.into(),
            x0,
            x1: x0 + width,
            top: baseline - 0.8 * size,
            bottom: baseline + 0.2 * size,
            baseline,
            size,
            font: Arc::from("Helvetica"),
            bold: false,
            italic: false,
            upright: true,
            vertical: false,
        }
    }

    fn word(word: &str, x: f32, baseline: f32, size: f32) -> Vec<Glyph> {
        word.chars()
            .enumerate()
            .map(|(i, c)| glyph(&c.to_string(), x + i as f32 * 0.5 * size, 0.5 * size, baseline, size))
            .collect()
    }

    #[test]
    fn rtl_runs_read_in_logical_order() {
        assert_eq!(reorder_rtl("xy\u{05d1}\u{05d0} z"), "xy \u{05d0}\u{05d1} z");
        assert_eq!(reorder_rtl("plain"), "plain");
    }

    #[test]
    fn gaps_become_spaces_and_lines_become_blocks() {
        let mut glyphs = word("Heading", 72.0, 100.0, 20.0);
        glyphs.extend(word("body", 72.0, 140.0, 11.0));
        glyphs.extend(word("text", 72.0 + 4.0 * 5.5 + 3.0, 140.0, 11.0));
        glyphs.extend(word("more", 72.0, 154.0, 11.0));
        let page = Page { index: 0, width: 595.0, height: 842.0, rotation: 0, glyphs };
        let layout = layout_page(&page);
        let texts: Vec<_> = layout.blocks.iter().map(Block::text).collect();
        assert_eq!(texts, ["Heading", "body text more"]);
        assert_eq!(layout.text(), "Heading\nbody text more");
    }
}
