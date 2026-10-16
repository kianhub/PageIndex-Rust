//! Embedded PDF bookmark consumption.
//!
//! Bookmarks are validated, then classified into tiers:
//!
//! * FULL: deep and dense enough to be the frame. The bookmark tree replaces
//!   the detected hierarchy and detected sections it lacks are grafted back
//!   in by page range, after dropping recurring-label and overlong-title noise.
//! * SKELETON: coarse but reliable (typically chapters only). Top-level
//!   entries pin the chapter frame and detected nodes are re-hung under
//!   them. With page text, garbled detected titles are repaired from
//!   same-page bookmark strings and deeper entries confirmed by the text are
//!   filled in.
//! * IGNORE: absent or untrustworthy (non-monotonic, generic "Page 12"
//!   titles, near-empty, or an enumeration of one template plus a counter).
//!
//! A bookmark target has no reliable on-page position, so a section runs
//! onto the page where the next entry starts; parents cover their subtree.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

use crate::tree::Node;
use pageindex_pdf::Bookmark;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Ignore,
    Skeleton,
    Full,
}

const REPAIR_SIMILARITY: f64 = 0.7;
const BACKFILL_DUP_MIN: usize = 3;
const BACKFILL_MAX_TITLE: usize = 100;

static GENERIC_TITLE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(?:(?:page|slide|folie|document\s+page)\s*)?\d+$").unwrap());
static LATEX_SPAN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\$.*?\$").unwrap());
static TOKEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^\W_]+").unwrap());
static ROMAN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$").unwrap());
static DIGIT_RUN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[0-9]+").unwrap());

const ROMAN_EXCLUDED: [&str; 7] = ["di", "div", "li", "liv", "mi", "mix", "xi"];
const CONTENT_UNITS: [&str; 22] = [
    "chapter", "chap", "ch", "part", "pt", "section", "sec", "book", "appendix", "unit", "lesson", "章", "部", "节",
    "篇", "卷", "第章", "第部", "第节", "第篇", "第卷", "",
];

/// A validated bookmark: levels re-stacked, pages 1-based and in range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub title: String,
    pub level: usize,
    pub page: usize,
}

fn is_roman(token: &str) -> bool {
    ROMAN.is_match(token) && !ROMAN_EXCLUDED.contains(&token)
}

fn roman_to_arabic(token: &str) -> String {
    let value = |c: char| match c {
        'i' => 1,
        'v' => 5,
        'x' => 10,
        'l' => 50,
        'c' => 100,
        'd' => 500,
        _ => 1000,
    };
    let (mut total, mut prev) = (0i64, 0i64);
    for c in token.chars().rev() {
        let v = value(c);
        total += if v < prev { -v } else { v };
        prev = prev.max(v);
    }
    total.to_string()
}

fn tokens(title: &str) -> Vec<String> {
    let lower = title.to_lowercase();
    let stripped = LATEX_SPAN.replace_all(&lower, "");
    TOKEN.find_iter(&stripped).map(|m| m.as_str().to_owned()).collect()
}

/// Lowercased, LaTeX dropped, punctuation stripped, roman numerals as digits.
pub fn normalize_title(title: &str) -> String {
    tokens(title).iter().map(|t| if is_roman(t) { roman_to_arabic(t) } else { t.clone() }).collect()
}

/// Every counter (digit run or roman token) collapsed to `#`.
fn title_template(title: &str) -> String {
    tokens(title)
        .iter()
        .map(|t| if is_roman(t) { "#".to_owned() } else { DIGIT_RUN.replace_all(t, "#").into_owned() })
        .collect()
}

/// `difflib.SequenceMatcher(None, a, b).ratio()`.
pub fn similarity(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    fn matches(a: &[char], b: &[char]) -> usize {
        if a.is_empty() || b.is_empty() {
            return 0;
        }
        let (mut best, mut best_i, mut best_j) = (0usize, 0usize, 0usize);
        let mut prev = vec![0usize; b.len() + 1];
        for (i, ai) in a.iter().enumerate() {
            let mut cur = vec![0usize; b.len() + 1];
            for (j, bj) in b.iter().enumerate() {
                if ai == bj {
                    cur[j + 1] = prev[j] + 1;
                    let len = cur[j + 1];
                    let (si, sj) = (i + 1 - len, j + 1 - len);
                    if len > best || (len == best && (si < best_i || (si == best_i && sj < best_j))) {
                        best = len;
                        best_i = si;
                        best_j = sj;
                    }
                }
            }
            prev = cur;
        }
        if best == 0 {
            return 0;
        }
        best + matches(&a[..best_i], &b[..best_j]) + matches(&a[best_i + best..], &b[best_j + best..])
    }
    2.0 * matches(&a, &b) as f64 / (a.len() + b.len()) as f64
}

/// Drop entries outside the document or walking backwards, then re-stack levels.
pub fn validate_bookmarks(bookmarks: &[Bookmark], n_pages: usize) -> Vec<Entry> {
    let mut kept = Vec::new();
    let mut last_page = 0;
    for b in bookmarks {
        if !(1..=n_pages).contains(&b.page) || b.page < last_page {
            continue;
        }
        kept.push(b);
        last_page = b.page;
    }
    let mut stack: Vec<usize> = Vec::new();
    kept.into_iter()
        .map(|b| {
            while stack.last().is_some_and(|&l| l >= b.level) {
                stack.pop();
            }
            stack.push(b.level);
            Entry { title: b.title.clone(), level: stack.len(), page: b.page }
        })
        .collect()
}

pub fn classify_bookmarks(entries: &[Entry], n_pages: usize) -> Tier {
    if entries.len() < 3 {
        return Tier::Ignore;
    }
    let generic = entries.iter().filter(|e| GENERIC_TITLE.is_match(&e.title)).count();
    if 2 * generic >= entries.len() {
        return Tier::Ignore;
    }
    let mut counts: Vec<(String, usize)> = Vec::new();
    for entry in entries {
        let template = title_template(&entry.title);
        match counts.iter_mut().find(|(t, _)| *t == template) {
            Some((_, n)) => *n += 1,
            None => counts.push((template, 1)),
        }
    }
    let max = counts.iter().map(|(_, n)| *n).max().unwrap_or(0);
    let (top_template, top_count) = counts.iter().find(|(_, n)| *n == max).cloned().unwrap_or_default();
    let unit = top_template.replace('#', "");
    if top_template.contains('#')
        && 2 * top_count >= entries.len()
        && (top_template.matches('#').count() >= 2 || !CONTENT_UNITS[..21].contains(&unit.as_str()))
    {
        return Tier::Ignore;
    }
    let mut distinct: Vec<String> = entries.iter().map(|e| normalize_title(&e.title)).collect();
    distinct.sort_unstable();
    distinct.dedup();
    if 4 * distinct.len() <= entries.len() {
        return Tier::Ignore;
    }
    let max_level = entries.iter().map(|e| e.level).max().unwrap_or(1);
    if max_level >= 2 && entries.len() >= 6.max(n_pages / 20) { Tier::Full } else { Tier::Skeleton }
}

/// Arena of nodes so placement can target any node, bookmark or detected.
struct Arena {
    nodes: Vec<(Node, Vec<usize>)>,
}

impl Arena {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn leaf(&mut self, mut node: Node) -> usize {
        node.nodes.clear();
        self.nodes.push((node, Vec::new()));
        self.nodes.len() - 1
    }

    fn add_tree(&mut self, mut node: Node) -> usize {
        let children = std::mem::take(&mut node.nodes);
        let idx = self.leaf(node);
        let kids: Vec<usize> = children.into_iter().map(|c| self.add_tree(c)).collect();
        self.nodes[idx].1 = kids;
        idx
    }

    fn start(&self, idx: usize) -> usize {
        self.nodes[idx].0.start_index
    }

    fn title(&self, idx: usize) -> &str {
        &self.nodes[idx].0.title
    }

    fn kids(&self, idx: usize) -> &[usize] {
        &self.nodes[idx].1
    }

    fn subtree_max_start(&self, idx: usize) -> usize {
        self.kids(idx).iter().map(|&k| self.subtree_max_start(k)).fold(self.start(idx), usize::max)
    }

    fn insert_by_page(&mut self, parent: usize, child: usize) {
        let start = self.start(child);
        let pos = self.nodes[parent].1.iter().position(|&s| self.nodes[s].0.start_index > start);
        let kids = &mut self.nodes[parent].1;
        match pos {
            Some(p) => kids.insert(p, child),
            None => kids.push(child),
        }
    }

    fn build(&mut self, idx: usize) -> Node {
        let kids = std::mem::take(&mut self.nodes[idx].1);
        let mut node = std::mem::take(&mut self.nodes[idx].0);
        node.nodes = kids.into_iter().map(|k| self.build(k)).collect();
        node
    }
}

fn same_heading(arena: &Arena, node: usize, target: usize) -> bool {
    if arena.start(node) != arena.start(target) {
        return false;
    }
    let (a, b) = (normalize_title(arena.title(node)), normalize_title(arena.title(target)));
    !a.is_empty()
        && !b.is_empty()
        && (a == b || a.ends_with(&b) || b.ends_with(&a) || similarity(&a, &b) >= REPAIR_SIMILARITY)
}

/// End fill with a shared boundary page, union promotion, node ids.
fn finalize(mut roots: Vec<Node>, n_pages: usize) -> Vec<Node> {
    let mut starts = Vec::new();
    crate::tree::for_each_mut(&mut roots, &mut |n| starts.push(n.start_index));
    let mut i = 0;
    crate::tree::for_each_mut(&mut roots, &mut |n| {
        let boundary = starts.get(i + 1).copied().unwrap_or(n_pages);
        n.end_index = n.start_index.max(boundary);
        i += 1;
    });
    fn promote(nodes: &mut [Node]) -> usize {
        let mut end = 0;
        for node in nodes {
            if !node.nodes.is_empty() {
                node.end_index = node.end_index.max(promote(&mut node.nodes));
            }
            end = end.max(node.end_index);
        }
        end
    }
    promote(&mut roots);
    crate::tree::write_node_ids(&mut roots);
    roots
}

fn chapter_of(starts: &[usize], page: usize) -> Option<usize> {
    let mut idx = None;
    for (i, &s) in starts.iter().enumerate() {
        if page >= s {
            idx = Some(i);
        } else {
            break;
        }
    }
    idx
}

fn title_in_text(entry: &Entry, page_norms: &[String]) -> bool {
    let norm = normalize_title(&entry.title);
    if norm.is_empty() {
        return false;
    }
    let lo = entry.page.saturating_sub(2);
    let hi = (entry.page + 1).min(page_norms.len());
    (lo..hi).any(|p| page_norms[p].contains(&norm))
}

fn repair_titles(structure: &mut [Node], entries: &[Entry]) {
    let mut by_page: HashMap<usize, Vec<&Entry>> = HashMap::new();
    for entry in entries {
        by_page.entry(entry.page).or_default().push(entry);
    }
    crate::tree::for_each_mut(structure, &mut |node| {
        let node_norm = normalize_title(&node.title);
        let mut best: Option<&Entry> = None;
        let mut best_ratio = REPAIR_SIMILARITY;
        for entry in by_page.get(&node.start_index).into_iter().flatten() {
            let entry_norm = normalize_title(&entry.title);
            if node_norm.is_empty()
                || entry_norm.is_empty()
                || node_norm.ends_with(&entry_norm)
                || entry_norm.ends_with(&node_norm)
            {
                continue;
            }
            let ratio = similarity(&node_norm, &entry_norm);
            if ratio >= best_ratio {
                best = Some(entry);
                best_ratio = ratio;
            }
        }
        if let Some(entry) = best {
            node.title = entry.title.clone();
        }
    });
}

fn find_entry_node(arena: &Arena, root: usize, entry: &Entry) -> Option<usize> {
    let entry_norm = normalize_title(&entry.title);
    if arena.start(root).abs_diff(entry.page) <= 1 {
        let node_norm = normalize_title(arena.title(root));
        if !entry_norm.is_empty()
            && !node_norm.is_empty()
            && (node_norm.ends_with(&entry_norm)
                || entry_norm.ends_with(&node_norm)
                || similarity(&node_norm, &entry_norm) >= REPAIR_SIMILARITY)
        {
            return Some(root);
        }
    }
    arena.kids(root).to_vec().into_iter().find_map(|k| find_entry_node(arena, k, entry))
}

fn bookmark_node(entry: &Entry) -> Node {
    Node::new(entry.title.clone(), entry.page, entry.page)
}

/// SKELETON tier: re-hang the detected tree under the bookmark chapters.
pub fn merge_bookmark_skeleton(
    mut structure: Vec<Node>,
    entries: &[Entry],
    n_pages: usize,
    page_texts: Option<&[String]>,
) -> Vec<Node> {
    let chapters: Vec<&Entry> = entries.iter().filter(|e| e.level == 1).collect();
    if chapters.is_empty() {
        return structure;
    }
    if page_texts.is_some() {
        repair_titles(&mut structure, entries);
    }
    let mut arena = Arena::new();
    let chapter_ids: Vec<usize> = chapters.iter().map(|e| arena.leaf(bookmark_node(e))).collect();
    let starts: Vec<usize> = chapters.iter().map(|e| e.page).collect();
    let range_end: Vec<usize> =
        (0..chapters.len()).map(|i| chapters.get(i + 1).map_or(n_pages + 1, |c| c.page)).collect();
    let mut front: Vec<usize> = Vec::new();

    fn place(arena: &mut Arena, node: usize, ctx: &Ctx, front: &mut Vec<usize>) {
        let children = arena.nodes[node].1.clone();
        match chapter_of(&ctx.starts, arena.start(node)) {
            None => {
                if arena.subtree_max_start(node) >= ctx.starts[0] {
                    arena.nodes[node].1.clear();
                    front.push(node);
                    for child in children {
                        place(arena, child, ctx, front);
                    }
                } else {
                    front.push(node);
                }
            }
            Some(idx) => {
                let target = ctx.targets[idx];
                if arena.subtree_max_start(node) < ctx.range_end[idx] {
                    if same_heading(arena, node, target) {
                        arena.nodes[target].1.extend(children);
                    } else {
                        arena.nodes[target].1.push(node);
                    }
                    return;
                }
                arena.nodes[node].1.clear();
                if !same_heading(arena, node, target) {
                    arena.nodes[target].1.push(node);
                }
                for child in children {
                    place(arena, child, ctx, front);
                }
            }
        }
    }

    let ctx = Ctx { starts, range_end, targets: chapter_ids.clone() };
    for root in structure {
        let idx = arena.add_tree(root);
        place(&mut arena, idx, &ctx, &mut front);
    }

    if let Some(texts) = page_texts {
        let page_norms: Vec<String> = texts.iter().map(|t| normalize_title(t)).collect();
        let mut node_for_entry: HashMap<usize, usize> = HashMap::new();
        let mut chapter_pos = 0;
        let mut stack: Vec<(usize, usize)> = Vec::new();
        for (index, entry) in entries.iter().enumerate() {
            while stack.last().is_some_and(|&(level, _)| level >= entry.level) {
                stack.pop();
            }
            let parent_index = stack.last().map(|&(_, i)| i);
            stack.push((entry.level, index));
            if entry.level == 1 {
                node_for_entry.insert(index, chapter_ids[chapter_pos]);
                chapter_pos += 1;
                continue;
            }
            let Some(idx) = chapter_of(&ctx.starts, entry.page) else { continue };
            if let Some(existing) = find_entry_node(&arena, chapter_ids[idx], entry) {
                node_for_entry.insert(index, existing);
                continue;
            }
            if GENERIC_TITLE.is_match(&entry.title) || !title_in_text(entry, &page_norms) {
                continue;
            }
            let node = arena.leaf(bookmark_node(entry));
            let parent = parent_index.and_then(|p| node_for_entry.get(&p).copied()).unwrap_or(chapter_ids[idx]);
            arena.insert_by_page(parent, node);
            node_for_entry.insert(index, node);
        }
    }

    let roots: Vec<Node> = front.into_iter().chain(chapter_ids).map(|i| arena.build(i)).collect();
    finalize(roots, n_pages)
}

struct Ctx {
    starts: Vec<usize>,
    range_end: Vec<usize>,
    targets: Vec<usize>,
}

/// FULL tier: graft filtered detected sections into the bookmark tree.
pub fn merge_bookmark_tree(structure: Vec<Node>, entries: &[Entry], n_pages: usize) -> Vec<Node> {
    let mut arena = Arena::new();
    let mut roots: Vec<usize> = Vec::new();
    let mut flat: Vec<usize> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for entry in entries {
        let idx = arena.leaf(bookmark_node(entry));
        while stack.len() >= entry.level {
            stack.pop();
        }
        match stack.last() {
            Some(&parent) => arena.nodes[parent].1.push(idx),
            None => roots.push(idx),
        }
        stack.push(idx);
        flat.push(idx);
    }
    if flat.is_empty() {
        return structure;
    }
    let starts: Vec<usize> = flat.iter().map(|&i| arena.start(i)).collect();
    let range_end: Vec<usize> = (0..flat.len()).map(|p| starts.get(p + 1).copied().unwrap_or(n_pages + 1)).collect();

    let mut title_counts: HashMap<String, usize> = HashMap::new();
    for node in crate::tree::subtree(&structure) {
        *title_counts.entry(normalize_title(&node.title)).or_insert(0) += 1;
    }
    let filtered_out = |node: &Node| {
        let norm = normalize_title(&node.title);
        (!norm.is_empty() && title_counts.get(&norm).copied().unwrap_or(0) >= BACKFILL_DUP_MIN)
            || node.title.chars().count() > BACKFILL_MAX_TITLE
    };
    fn prune(nodes: Vec<Node>, filtered_out: &dyn Fn(&Node) -> bool) -> Vec<Node> {
        let mut kept = Vec::new();
        for mut node in nodes {
            let children = prune(std::mem::take(&mut node.nodes), filtered_out);
            if filtered_out(&node) {
                kept.extend(children);
            } else {
                node.nodes = children;
                kept.push(node);
            }
        }
        kept
    }

    let ctx = Ctx { starts, range_end, targets: flat };
    let mut front = Vec::new();
    fn place(arena: &mut Arena, node: usize, ctx: &Ctx, front: &mut Vec<usize>) {
        let children = arena.nodes[node].1.clone();
        match chapter_of(&ctx.starts, arena.start(node)) {
            None => {
                if arena.subtree_max_start(node) >= ctx.starts[0] {
                    arena.nodes[node].1.clear();
                    front.push(node);
                    for child in children {
                        place(arena, child, ctx, front);
                    }
                } else {
                    front.push(node);
                }
            }
            Some(idx) => {
                let target = ctx.targets[idx];
                if arena.subtree_max_start(node) < ctx.range_end[idx] {
                    if same_heading(arena, node, target) {
                        for child in children {
                            arena.insert_by_page(target, child);
                        }
                    } else {
                        arena.insert_by_page(target, node);
                    }
                    return;
                }
                arena.nodes[node].1.clear();
                if !same_heading(arena, node, target) {
                    arena.insert_by_page(target, node);
                }
                for child in children {
                    place(arena, child, ctx, front);
                }
            }
        }
    }
    for root in prune(structure, &filtered_out) {
        let idx = arena.add_tree(root);
        place(&mut arena, idx, &ctx, &mut front);
    }
    let built: Vec<Node> = front.into_iter().chain(roots).map(|i| arena.build(i)).collect();
    finalize(built, n_pages)
}

/// Classify the document's bookmarks and apply the matching tier. Returns
/// the structure and its `toc_source`.
pub fn apply_embedded_toc(
    structure: Vec<Node>,
    bookmarks: &[Bookmark],
    n_pages: usize,
    page_texts: Option<&[String]>,
) -> (Vec<Node>, &'static str) {
    let entries = validate_bookmarks(bookmarks, n_pages);
    match classify_bookmarks(&entries, n_pages) {
        Tier::Full => (merge_bookmark_tree(structure, &entries, n_pages), "bookmarks"),
        Tier::Skeleton => (merge_bookmark_skeleton(structure, &entries, n_pages, page_texts), "hybrid"),
        Tier::Ignore => (structure, "detected"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bm(title: &str, level: usize, page: usize) -> Bookmark {
        Bookmark { title: title.into(), level, page }
    }

    #[test]
    fn normalization_and_templates() {
        assert_eq!(normalize_title("II. The General Assembly"), normalize_title("2. The General Assembly"));
        assert_eq!(title_template("Chapter 12"), "chapter#");
        assert!((similarity("abcd", "bcde") - 0.75).abs() < 1e-9);
    }

    #[test]
    fn tiers() {
        let pages: Vec<Bookmark> = (1..=10).map(|p| bm(&format!("Page {p}"), 1, p)).collect();
        assert_eq!(classify_bookmarks(&validate_bookmarks(&pages, 10), 10), Tier::Ignore);
        let chapters: Vec<Bookmark> = (1..=4).map(|p| bm(&format!("Chapter {p}"), 1, p * 2)).collect();
        assert_eq!(classify_bookmarks(&validate_bookmarks(&chapters, 10), 10), Tier::Skeleton);
        let deep = vec![
            bm("Intro", 1, 1),
            bm("Background", 2, 2),
            bm("Method", 1, 3),
            bm("Data", 2, 4),
            bm("Model", 2, 5),
            bm("Results", 1, 6),
        ];
        assert_eq!(classify_bookmarks(&validate_bookmarks(&deep, 10), 10), Tier::Full);
    }

    #[test]
    fn backwards_entries_are_dropped_and_levels_restacked() {
        let entries = validate_bookmarks(&[bm("A", 1, 3), bm("B", 3, 2), bm("C", 3, 4)], 5);
        assert_eq!(
            entries,
            vec![Entry { title: "A".into(), level: 1, page: 3 }, Entry { title: "C".into(), level: 2, page: 4 }]
        );
    }

    #[test]
    fn skeleton_rehangs_detected_nodes_under_chapters() {
        let detected = vec![Node::new("1 Intro", 1, 2), Node::new("1.1 Scope", 2, 3), Node::new("2 Methods", 4, 6)];
        let entries = validate_bookmarks(&[bm("Intro", 1, 1), bm("Methods", 1, 4), bm("End", 1, 6)], 6);
        let tree = merge_bookmark_skeleton(detected, &entries, 6, None);
        let shape: Vec<_> =
            crate::tree::subtree(&tree).iter().map(|n| (n.title.clone(), n.start_index, n.end_index)).collect();
        assert_eq!(
            shape,
            [
                ("Intro".to_string(), 1, 4),
                ("1.1 Scope".to_string(), 2, 4),
                ("Methods".to_string(), 4, 6),
                ("End".to_string(), 6, 6)
            ]
        );
    }
}
