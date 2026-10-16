//! Layout statistics to outline: document body style, recurring
//! headers/footers, contents pages, the document title, heading
//! candidates, level assignment and the dict tree.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use regex::Regex;

use super::keywords::{KEYWORDS, normalize_key};
use super::layout::{Block, BlockKind, PageLayout, size_key};
use crate::tree::Node;

static DIGIT_CHAIN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(\d{1,3}(?:\.\d{1,3})*)\.?(?:\s|$)").unwrap());
static ROMAN_OR_LETTER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(?:[IVXLC]{1,6}|[A-Z])[.)]\s+\S").unwrap());
static PAGE_NUMBER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^[\s\-–—(\[|]*(?:page\s+)?(?:\d{1,4}|[ivxlcdm]{1,7})(?:\s*(?:of|/)\s*\d{1,4})?[\s\-–—)\]|]*$")
        .unwrap()
});
static TOC_ENTRY_END: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:\.{2,}|…|\s|·)\s*(?:\d{1,4}|[ivxlc]{1,6})\s*$").unwrap());
static DIGITS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").unwrap());
static SUBFIGURE_LABEL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\([a-z]\)\s").unwrap());

/// Document-wide body text statistics.
#[derive(Debug, Clone, Copy)]
pub struct DocStats {
    pub body_size: f32,
    pub body_bold: bool,
}

pub fn doc_stats(pages: &[PageLayout]) -> DocStats {
    let mut sizes: HashMap<(i32, bool), usize> = HashMap::new();
    for line in pages.iter().flat_map(|p| &p.blocks).flat_map(|b| &b.lines).filter(|l| l.upright) {
        *sizes.entry((size_key(line.size), line.is_bold())).or_insert(0) += line.chars;
    }
    let ((size, bold), _) = sizes.into_iter().max_by_key(|(k, n)| (*n, -k.0)).unwrap_or(((100, false), 0));
    DocStats { body_size: size as f32 / 10.0, body_bold: bold }
}

fn recurring_key(text: &str) -> String {
    DIGITS.replace_all(&normalize_key(text), "#").into_owned()
}

/// Mark page numbers and text repeated in the top/bottom margins.
pub fn mark_headers_footers(pages: &mut [PageLayout]) {
    let with_text = pages.iter().filter(|p| p.has_text()).count();
    let mut seen: HashMap<String, HashSet<usize>> = HashMap::new();
    for page in pages.iter() {
        for block in &page.blocks {
            if margin_band(page.height, block).is_some() {
                seen.entry(recurring_key(&block.text())).or_default().insert(page.number);
            }
        }
    }
    let min_pages = ((with_text as f32) * 0.3).ceil().max(2.0) as usize;
    for page in pages.iter_mut() {
        let height = page.height;
        for block in page.blocks.iter_mut() {
            let Some(kind) = margin_band(height, block) else { continue };
            let text = block.text();
            let repeated = with_text >= 3 && seen.get(&recurring_key(&text)).is_some_and(|p| p.len() >= min_pages);
            if repeated || PAGE_NUMBER.is_match(&text) {
                block.kind = kind;
            }
        }
    }
}

fn margin_band(page_height: f32, block: &Block) -> Option<BlockKind> {
    if block.bottom <= 0.12 * page_height {
        Some(BlockKind::Header)
    } else if block.top >= 0.88 * page_height {
        Some(BlockKind::Footer)
    } else {
        None
    }
}

/// Mark contents pages: a contents title over entries, or mostly lines
/// that end in a page number.
pub fn mark_toc_pages(pages: &mut [PageLayout]) {
    for page in pages.iter_mut() {
        let lines: Vec<&str> = page
            .blocks
            .iter()
            .filter(|b| b.kind == BlockKind::Body)
            .flat_map(|b| b.lines.iter().map(|l| l.text.as_str()))
            .collect();
        if lines.is_empty() {
            continue;
        }
        let titled = page
            .blocks
            .iter()
            .any(|b| b.kind == BlockKind::Body && b.top < 0.35 * page.height && KEYWORDS.is_toc_title(&b.text()));
        let entries = lines.iter().filter(|l| TOC_ENTRY_END.is_match(l) && l.chars().any(char::is_alphabetic)).count();
        let ratio = entries as f32 / lines.len() as f32;
        if (titled && entries >= 3 && ratio >= 0.25) || (lines.len() >= 6 && ratio >= 0.6) {
            for block in page.blocks.iter_mut().filter(|b| b.kind == BlockKind::Body) {
                block.kind = BlockKind::Toc;
            }
        }
    }
}

fn letters(text: &str) -> usize {
    text.chars().filter(|c| c.is_alphabetic()).count()
}

fn is_caps_heavy(text: &str) -> bool {
    let upper = text.chars().filter(|c| c.is_uppercase()).count();
    let lower = text.chars().filter(|c| c.is_lowercase()).count();
    upper >= 2 && upper > 3 * lower
}

/// Size plus bonuses for caps-heavy text (2) and bold (1).
fn heading_score(block: &Block) -> f32 {
    let text = block.text();
    block.size + if is_caps_heavy(&text) { 2.0 } else { 0.0 } + if block.bold { 1.0 } else { 0.0 }
}

fn numbering_depth(text: &str) -> Option<usize> {
    DIGIT_CHAIN.captures(text).map(|c| c[1].split('.').count())
}

fn is_numbered(text: &str) -> bool {
    let has_rest = |len: usize| text[len..].trim().chars().any(char::is_alphabetic);
    DIGIT_CHAIN.find(text).is_some_and(|m| has_rest(m.end()))
        || ROMAN_OR_LETTER.is_match(text)
        || KEYWORDS.opens_with_unit_word(text)
}

/// Select the document title on the opening page(s); its blocks become
/// [`BlockKind::Title`] and echoes of it elsewhere become headers.
pub fn detect_title(pages: &mut [PageLayout], stats: &DocStats) -> Option<String> {
    let mut found: Option<(usize, Vec<usize>)> = None;
    let mut checked = 0;
    for (page_idx, page) in pages.iter().enumerate() {
        if !page.has_text() {
            continue;
        }
        checked += 1;
        let mut best: Option<(f32, usize)> = None;
        for (idx, block) in page.blocks.iter().enumerate() {
            let text = block.text();
            if block.kind != BlockKind::Body
                || !block.upright
                || block.top > 0.66 * page.height
                || block.lines.len() > 4
                || letters(&text) < 2
                || text.chars().count() > 300
                || numbering_depth(&text).is_some()
            {
                continue;
            }
            let score = heading_score(block);
            let larger = block.size >= 1.15 * stats.body_size || score >= stats.body_size + 3.0;
            if larger && best.is_none_or(|(s, _)| score > s + 0.05) {
                best = Some((score, idx));
            }
        }
        if let Some((_, idx)) = best {
            let lead = &page.blocks[idx];
            let mut members = vec![idx];
            let mut bottom = lead.bottom;
            for (next_idx, next) in page.blocks.iter().enumerate().skip(idx + 1) {
                let gap = next.top - bottom;
                if next.kind == BlockKind::Body
                    && size_key(next.size) == size_key(lead.size)
                    && next.bold == lead.bold
                    && gap >= -0.5 * lead.size
                    && gap <= 1.2 * lead.size
                    && members.len() < 3
                {
                    members.push(next_idx);
                    bottom = next.bottom;
                } else {
                    break;
                }
            }
            found = Some((page_idx, members));
            break;
        }
        let chars: usize = page.blocks.iter().map(Block::chars).sum();
        if checked >= 2 || chars >= 400 {
            break;
        }
    }
    let (page_idx, members) = found?;
    let title = members.iter().map(|&i| pages[page_idx].blocks[i].text()).collect::<Vec<_>>().join(" ");
    for &i in &members {
        pages[page_idx].blocks[i].kind = BlockKind::Title;
    }
    let title_key = normalize_key(&title);
    for page in pages.iter_mut() {
        for block in page.blocks.iter_mut().filter(|b| b.kind == BlockKind::Body) {
            let key = normalize_key(&block.text());
            let long = key.chars().count() > 20 && title_key.chars().count() > 20;
            let echo = (long
                && (key.starts_with(&title_key) || title_key.starts_with(&key) || title_key.ends_with(&key)))
                || key == title_key;
            if echo {
                block.kind = BlockKind::Header;
            }
        }
    }
    Some(title.split_whitespace().collect::<Vec<_>>().join(" "))
}

#[derive(Debug, Clone)]
pub struct Heading {
    pub title: String,
    pub page: usize,
    pub size: f32,
    pub bold: bool,
    pub depth: Option<usize>,
    /// Nothing but margins precedes the heading on its page.
    pub opens_page: bool,
}

/// Whether a body block reads as a heading.
fn is_heading_block(block: &Block, stats: &DocStats) -> bool {
    if block.kind != BlockKind::Body || !block.upright {
        return false;
    }
    let text = block.text();
    let chars = text.chars().count();
    let words = text.split_whitespace().count();
    let letter_count = letters(&text);
    let visible = text.chars().filter(|c| !c.is_whitespace()).count().max(1);
    if letter_count < 2 || chars > 200 || words > 24 || letter_count * 5 < visible * 2 {
        return false;
    }
    let ratio = block.size / stats.body_size.max(0.1);
    let max_lines = if ratio >= 1.3 { 4 } else { 3 };
    if block.lines.len() > max_lines || KEYWORDS.is_caption(&text) || SUBFIGURE_LABEL.is_match(&text) {
        return false;
    }
    let sentence = text.ends_with('.') && words > 8;
    let larger = ratio >= 1.12;
    let bold = block.bold && !stats.body_bold && ratio >= 0.9;
    let caps = is_caps_heavy(&text) && ratio >= 0.95 && words <= 10 && block.lines.len() == 1;
    let numbered = is_numbered(&text) && block.lines.len() <= 2 && words <= 14;
    if sentence && !larger {
        return false;
    }
    if larger || bold {
        return true;
    }
    caps && (numbered || letter_count >= 4) && !text.ends_with([',', ';'])
}

fn is_body_paragraph(block: &Block, stats: &DocStats) -> bool {
    let ratio = block.size / stats.body_size.max(0.1);
    block.kind == BlockKind::Body
        && (0.9..=1.1).contains(&ratio)
        && !block.bold
        && (block.lines.len() >= 2 || block.text().chars().count() >= 80)
}

/// On the title page, the blocks between the title and the first body
/// paragraph (authors, affiliations, dates) are front matter.
fn front_matter(page: &PageLayout, stats: &DocStats) -> std::ops::Range<usize> {
    let Some(last_title) = page.blocks.iter().rposition(|b| b.kind == BlockKind::Title) else {
        return 0..0;
    };
    let end = page.blocks[last_title..]
        .iter()
        .position(|b| is_body_paragraph(b, stats))
        .map_or(page.blocks.len(), |i| last_title + i);
    last_title + 1..end
}

pub fn collect_headings(pages: &[PageLayout], stats: &DocStats) -> Vec<Heading> {
    let mut headings = Vec::new();
    for page in pages {
        let first_content =
            page.blocks.iter().position(|b| !matches!(b.kind, BlockKind::Header | BlockKind::Footer) && b.chars() > 0);
        let front = front_matter(page, stats);
        for (idx, block) in page.blocks.iter().enumerate() {
            if !is_heading_block(block, stats) {
                continue;
            }
            let title = heading_title(block);
            if front.contains(&idx) && !(KEYWORDS.is_abstract(&title) || is_numbered(&title) || is_caps_heavy(&title)) {
                continue;
            }
            headings.push(Heading {
                depth: numbering_depth(&title),
                title,
                page: page.number,
                size: block.size,
                bold: block.bold,
                opens_page: first_content == Some(idx),
            });
        }
    }
    let with_text = pages.iter().filter(|p| p.has_text()).count();
    let mut pages_of: HashMap<String, HashSet<usize>> = HashMap::new();
    let mut per_page: HashMap<(String, usize), usize> = HashMap::new();
    for heading in &headings {
        let key = normalize_key(&heading.title);
        pages_of.entry(key.clone()).or_default().insert(heading.page);
        *per_page.entry((key, heading.page)).or_insert(0) += 1;
    }
    headings.retain(|h| {
        let key = normalize_key(&h.title);
        let n = pages_of[&key].len();
        let repeated_on_page = per_page[&(key, h.page)] > 1;
        !(repeated_on_page || (n >= 3 && n as f32 > 0.25 * with_text as f32))
    });
    headings
}

/// Lines joined into one title; a word hyphenated across a line break is rejoined.
fn heading_title(block: &Block) -> String {
    let mut title = String::new();
    for line in &block.lines {
        let text = line.text.split_whitespace().collect::<Vec<_>>().join(" ");
        if title.ends_with('-') && text.starts_with(|c: char| c.is_lowercase()) {
            title.pop();
        } else if !title.is_empty() {
            title.push(' ');
        }
        title.push_str(&text);
    }
    title
}

fn style_of(h: &Heading) -> (i32, bool) {
    (size_key(h.size) / 5, h.bold)
}

/// Digit numbering decides the level of a numbered heading. An unnumbered
/// heading takes the depth its style carries among numbered headings, else
/// sits one below the deepest numbered style ranked above it (size desc,
/// bold first); without numbering at all, the style rank is the level.
pub fn assign_levels(headings: &[Heading]) -> Vec<usize> {
    let mut styles: Vec<(i32, bool)> = headings.iter().map(style_of).collect();
    styles.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
    styles.dedup();
    let mut depth_votes: HashMap<(i32, bool), HashMap<usize, usize>> = HashMap::new();
    for h in headings {
        if let Some(depth) = h.depth {
            *depth_votes.entry(style_of(h)).or_default().entry(depth).or_insert(0) += 1;
        }
    }
    let style_depth: HashMap<(i32, bool), usize> = depth_votes
        .into_iter()
        .map(|(style, votes)| {
            (style, votes.into_iter().max_by_key(|(d, n)| (*n, std::cmp::Reverse(*d))).map(|(d, _)| d).unwrap_or(1))
        })
        .collect();
    headings
        .iter()
        .map(|h| {
            if let Some(depth) = h.depth {
                return depth.clamp(1, 6);
            }
            let style = style_of(h);
            if let Some(&depth) = style_depth.get(&style) {
                return depth.clamp(1, 6);
            }
            let rank = styles.iter().position(|s| *s == style).unwrap_or(styles.len());
            if style_depth.is_empty() {
                return (rank + 1).min(6);
            }
            let above = styles[..rank].iter().filter_map(|s| style_depth.get(s)).max().copied().unwrap_or(0);
            (above + 1).min(6)
        })
        .collect()
}

/// Reject outlines too sparse to navigate: fewer than two headings, or a
/// gap between headings spanning most of a longer document.
pub fn outline_is_usable(headings: &[Heading], page_count: usize) -> bool {
    if headings.len() < 2 {
        return false;
    }
    if page_count >= 3 {
        let mut starts: Vec<usize> = headings.iter().map(|h| h.page).collect();
        starts.push(page_count);
        let max_gap = starts.windows(2).map(|w| w[1].saturating_sub(w[0])).max().unwrap_or(0);
        if max_gap as f32 > 0.85 * page_count as f32 {
            return false;
        }
    }
    true
}

/// Nest headings by level and fill page ranges: a section ends the page
/// before the next heading when that heading opens its page, else on it;
/// a parent's range covers its subtree.
pub fn build_tree(headings: &[Heading], levels: &[usize], total_pages: usize) -> Vec<Node> {
    struct Pending {
        node: Node,
        level: usize,
        children: Vec<usize>,
    }
    let mut arena: Vec<Pending> = Vec::new();
    let mut roots: Vec<usize> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for (heading, &level) in headings.iter().zip(levels) {
        let idx = arena.len();
        arena.push(Pending {
            node: Node::new(heading.title.clone(), heading.page, heading.page),
            level,
            children: Vec::new(),
        });
        while stack.last().is_some_and(|&top| arena[top].level >= level) {
            stack.pop();
        }
        match stack.last() {
            Some(&parent) => arena[parent].children.push(idx),
            None => roots.push(idx),
        }
        stack.push(idx);
    }
    // Pre-order equals heading order, so ends come straight from the list.
    for (i, pending) in arena.iter_mut().enumerate() {
        let start = pending.node.start_index;
        let boundary = match headings.get(i + 1) {
            Some(next) if next.opens_page => next.page.saturating_sub(1),
            Some(next) => next.page,
            None => total_pages,
        };
        pending.node.end_index = boundary.max(start);
    }
    if let Some(last) = arena.last_mut() {
        last.node.end_index = last.node.start_index.max(total_pages);
    }
    fn assemble(idx: usize, arena: &mut Vec<Pending>) -> Node {
        let children = std::mem::take(&mut arena[idx].children);
        let mut node = std::mem::take(&mut arena[idx].node);
        node.nodes = children.into_iter().map(|c| assemble(c, arena)).collect();
        if let Some(end) = node.nodes.iter().map(|c| c.end_index).max() {
            node.end_index = node.end_index.max(end);
        }
        node
    }
    let mut tree: Vec<Node> = roots.into_iter().map(|r| assemble(r, &mut arena)).collect();
    crate::tree::write_node_ids(&mut tree);
    tree
}

/// A top-level entry is an abstract or references heading.
pub fn has_abstract_or_references(structure: &[Node]) -> bool {
    structure.iter().any(|n| KEYWORDS.is_abstract(&n.title) || KEYWORDS.is_references(&n.title))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heading(title: &str, page: usize, size: f32, opens_page: bool) -> Heading {
        Heading { title: title.into(), page, size, bold: true, depth: numbering_depth(title), opens_page }
    }

    #[test]
    fn levels_follow_style_rank_capped_by_numbering() {
        let hs = [
            heading("1 Intro", 1, 14.0, true),
            heading("1.1 Scope", 1, 12.0, false),
            heading("2 Methods", 2, 14.0, true),
            heading("Notes", 3, 12.0, false),
        ];
        assert_eq!(assign_levels(&hs), vec![1, 2, 1, 2]);
    }

    #[test]
    fn tree_shares_boundary_pages_unless_the_next_heading_opens_its_page() {
        let hs = [heading("A", 1, 14.0, true), heading("B", 2, 14.0, false), heading("C", 4, 14.0, true)];
        let tree = build_tree(&hs, &[1, 1, 1], 5);
        let ranges: Vec<_> = tree.iter().map(|n| (n.title.as_str(), n.start_index, n.end_index)).collect();
        assert_eq!(ranges, [("A", 1, 2), ("B", 2, 3), ("C", 4, 5)]);
    }

    #[test]
    fn numbering_and_page_numbers() {
        assert_eq!(numbering_depth("2.1 Computing the EMD"), Some(2));
        assert_eq!(numbering_depth("2023 Annual Report"), None);
        assert!(is_numbered("IV. Results"));
        assert!(PAGE_NUMBER.is_match("- 12 -"));
        assert!(PAGE_NUMBER.is_match("Page 3 of 10"));
        assert!(!PAGE_NUMBER.is_match("Results"));
    }
}
