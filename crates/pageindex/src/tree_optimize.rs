//! Tree optimization: merge and expand driven by worst-case search cost.
//!
//! Search cost is measured in pages, routing cost R(v) = 1 page:
//!
//! * S(v): pages to scan linearly if v is collapsed (its whole subtree span)
//! * S_residual(v): pages of v covered by no child
//! * tree_cost(v) = S(v) for a frontier node, else R + max(S_residual, max tree_cost(c))
//!
//! merge (deterministic, bottom-up) collapses a subtree when S(v) <= tree_cost(v),
//! keeping removed titles as `key_items`. expand asks a model for one level of
//! subsections of a collapsed node over the trigger and keeps them when
//! R + max S(c) < S(v) and the tokens read to reach a page go down.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::Serialize;
use serde_json::{Value, json};
use tokio::sync::Semaphore;

use crate::error::{PageIndexError, Result};
use crate::llm::{CompletionParams, LlmClient, complete_prompt};
use crate::summary::{SUMMARY_MAX_WORDS, TOKENS_PER_WORD};
use crate::text::{count_tokens, has_latin_letter, heading_key, normalize, same_heading};
use crate::tree::{Node, intro_title};

pub const TRIGGER_PAGES: usize = 5;
pub const ROUTING_COST: usize = 1;
pub const EXPAND_CONCURRENCY: usize = 32;
pub const PAGE_CHARS: usize = 6000;
pub const TITLE_MAX_CHARS: usize = 200;
pub const ENTRY_TOKENS: f64 = 45.0;

const EXPAND_PROMPT: &str = r#"You are splitting an over-long section of a PDF into its subsections.

Section title: {title}
Pages: {start}-{end}

{pages}

List the subsection headings that BEGIN within these pages, in document order,
each with the page number it begins on. Rules:

- Use only headings printed in the document. Never invent or paraphrase one.
- A running header, a table column label, a table row label, or a cross-reference
  is not a subsection heading.
- If this section is continuous prose, or a single table spanning the pages,
  return an empty list. That is a valid and expected answer.
- Do not include the section's own title.

Reply with JSON only:
{"subsections": [{"title": "<verbatim heading>", "page": <int>}]}"#;

// ---------------------------------------------------------------------------
// geometry
// ---------------------------------------------------------------------------

/// Per-page non-empty lines, as Flash reads them.
pub fn page_lines(page_texts: &[String]) -> Vec<Vec<String>> {
    page_texts
        .iter()
        .map(|text| text.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_owned).collect())
        .collect()
}

/// Is the heading the first line on its page? No when that cannot be told
/// (no Latin letter to match, or a running header repeats it below).
pub fn heading_at_page_start(lines: &[Vec<String>], page_no: usize, heading: &str) -> bool {
    let Some(page) = page_no.checked_sub(1).and_then(|i| lines.get(i)) else {
        return false;
    };
    let key = normalize(heading);
    if page.is_empty() || !has_latin_letter(&key) {
        return false;
    }
    let prefix = format!("{key} ");
    let starts: Vec<bool> = page.iter().map(|line| format!("{} ", normalize(line)).starts_with(&prefix)).collect();
    starts[0] && !starts[1..].iter().any(|&s| s)
}

/// end_index for a candidate level: next.start - 1 when the next heading
/// opens its page, else next.start; the last child ends with the node.
pub fn assign_ends(node: &Node, children: &[Node], lines: &[Vec<String>]) -> Vec<Node> {
    let old_end = node.subtree_end();
    let mut sized: Vec<Node> = children.to_vec();
    for (i, child) in sized.iter_mut().enumerate() {
        child.end_index = match children.get(i + 1) {
            Some(next) if heading_at_page_start(lines, next.start_index, &next.title) => {
                child.start_index.max(next.start_index.saturating_sub(1))
            }
            Some(next) => next.start_index,
            None => old_end,
        };
    }
    sized
}

/// Give every parent whose first child starts on a later page an intro
/// child for the pages before it.
pub fn add_intro_nodes(structure: &mut [Node], lines: Option<&[Vec<String>]>) {
    for node in structure.iter_mut() {
        add_intro(node, lines);
    }
}

fn add_intro(node: &mut Node, lines: Option<&[Vec<String>]>) {
    if let Some(first) = node.nodes.first() {
        let first_start = first.start_index;
        if first_start > node.start_index {
            let opens =
                lines.is_some_and(|l| first_start <= l.len() && heading_at_page_start(l, first_start, &first.title));
            let end = node.start_index.max(node.end_index.min(if opens { first_start - 1 } else { first_start }));
            node.nodes.insert(0, Node::new(intro_title(&node.title), node.start_index, end));
        }
    }
    for child in node.nodes.iter_mut() {
        add_intro(child, lines);
    }
}

fn attach_children(node: &mut Node, children: Vec<Node>, lines: &[Vec<String>]) {
    let count = children.len();
    node.nodes = children;
    let mut wrapper = [std::mem::take(node)];
    add_intro_nodes(&mut wrapper, Some(lines));
    *node = std::mem::take(&mut wrapper[0]);
    if node.nodes.len() > count {
        let id = node.node_id.clone().unwrap_or_default();
        node.nodes[0].node_id = Some(format!("{id}.0"));
    }
}

/// Renumber every node_id in document order; returns the old -> new map.
pub fn relabel(structure: &mut [Node]) -> HashMap<String, String> {
    let mut mapping = HashMap::new();
    let mut counter = 0usize;
    crate::tree::for_each_mut(structure, &mut |node| {
        let new = format!("{counter:04}");
        if let Some(old) = node.node_id.replace(new.clone()) {
            mapping.insert(old, new);
        }
        counter += 1;
    });
    mapping
}

// ---------------------------------------------------------------------------
// cost model
// ---------------------------------------------------------------------------

/// Pages to scan linearly if this node were collapsed.
pub fn span(node: &Node) -> usize {
    (node.subtree_end() + 1).saturating_sub(node.start_index)
}

/// Pages of the node covered by no child.
pub fn residual(node: &Node) -> usize {
    if node.nodes.is_empty() {
        return span(node);
    }
    let (start, end) = (node.start_index, node.subtree_end());
    let mut covered = vec![false; end + 1 - start.min(end + 1)];
    for child in &node.nodes {
        for page in child.start_index.max(start)..=child.subtree_end().min(end) {
            covered[page - start] = true;
        }
    }
    covered.iter().filter(|c| !**c).count()
}

pub fn tree_cost(node: &Node, routing: usize) -> usize {
    if node.nodes.is_empty() {
        return span(node);
    }
    let mut worst = node.nodes.iter().map(|c| tree_cost(c, routing)).max().unwrap_or(0);
    worst = worst.max(residual(node));
    routing + worst
}

/// Every branch as (routing distance, pages to scan).
fn frontier_costs(node: &Node, distance: usize, out: &mut Vec<(usize, usize)>) {
    if node.nodes.is_empty() {
        out.push((distance, span(node)));
        return;
    }
    let r = residual(node);
    if r > 0 {
        out.push((distance + 1, r));
    }
    for child in &node.nodes {
        frontier_costs(child, distance + 1, out);
    }
}

pub fn tree_cost_via_frontier(node: &Node, routing: usize) -> usize {
    let mut entries = Vec::new();
    frontier_costs(node, 0, &mut entries);
    entries.iter().map(|(d, s)| d * routing + s).max().unwrap_or(0)
}

fn expand_cost(node: &Node, children: &[Node], lines: &[Vec<String>], routing: usize) -> (usize, usize) {
    let mut trial = Node { nodes: children.to_vec(), ..node.clone() };
    let r = residual(&trial);
    add_intro(&mut trial, Some(lines));
    (tree_cost(&trial, routing), r)
}

fn expand_read_tokens(
    node: &Node,
    children: &[Node],
    lines: &[Vec<String>],
    page_tokens: &[usize],
    summary_words: usize,
) -> (usize, f64) {
    let mut trial = Node { nodes: children.to_vec(), ..node.clone() };
    add_intro(&mut trial, Some(lines));
    let text = |n: &Node| -> usize {
        let start = n.start_index.saturating_sub(1).min(page_tokens.len());
        let end = n.subtree_end().min(page_tokens.len()).max(start);
        page_tokens[start..end].iter().sum()
    };
    let summary = TOKENS_PER_WORD * summary_words as f64;
    let entries: f64 = trial.nodes.iter().map(|c| ENTRY_TOKENS + (text(c) as f64).min(summary)).sum();
    let largest = trial.nodes.iter().map(text).max().unwrap_or(0);
    (text(node), entries + largest as f64)
}

// ---------------------------------------------------------------------------
// whole-tree metrics
// ---------------------------------------------------------------------------

fn frontier_nodes(structure: &[Node], root_depth: usize) -> Vec<(&Node, usize)> {
    fn visit<'a>(node: &'a Node, depth: usize, out: &mut Vec<(&'a Node, usize)>) {
        if node.nodes.is_empty() {
            out.push((node, depth));
            return;
        }
        for child in &node.nodes {
            visit(child, depth + 1, out);
        }
        if residual(node) > 0 {
            out.push((node, depth + 1));
        }
    }
    let mut out = Vec::new();
    for root in structure {
        visit(root, root_depth, &mut out);
    }
    out
}

fn frontier_pages(node: &Node) -> usize {
    if node.nodes.is_empty() { span(node) } else { residual(node) }
}

/// Search-complexity metrics over a whole tree.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Complexity {
    pub total_pages: usize,
    pub frontier_nodes: usize,
    pub worst_case_search_complexity: usize,
    pub average_search_complexity: f64,
    pub normalized_worst_case_complexity: f64,
    pub max_depth: usize,
    pub mean_depth: f64,
    pub probability_mass: f64,
}

fn round_to(value: f64, digits: i32) -> f64 {
    let scale = 10f64.powi(digits);
    (value * scale).round() / scale
}

pub fn complexity(structure: &[Node], total_pages: usize) -> Complexity {
    let entries = frontier_nodes(structure, 1);
    let worst = entries.iter().map(|(n, d)| d * ROUTING_COST + frontier_pages(n)).max().unwrap_or(0);
    let (mut average, mut weight) = (0.0, 0.0);
    if total_pages > 0 {
        for (node, depth) in &entries {
            let n = frontier_pages(node) as f64;
            let p = n / total_pages as f64;
            weight += p;
            average += p * ((depth * ROUTING_COST) as f64 + (n + 1.0) / 2.0);
        }
    }
    let depths: Vec<usize> = entries.iter().map(|(_, d)| *d).collect();
    Complexity {
        total_pages,
        frontier_nodes: entries.len(),
        worst_case_search_complexity: worst,
        average_search_complexity: round_to(average, 3),
        normalized_worst_case_complexity: if total_pages > 0 {
            round_to(worst as f64 / total_pages as f64, 4)
        } else {
            0.0
        },
        max_depth: depths.iter().copied().max().unwrap_or(0),
        mean_depth: if depths.is_empty() {
            0.0
        } else {
            round_to(depths.iter().sum::<usize>() as f64 / depths.len() as f64, 2)
        },
        probability_mass: round_to(weight, 4),
    }
}

/// Structural problems: ranges outside the document, inverted ranges,
/// children before their parent, misordered siblings, uncovered pages.
pub fn validate(structure: &[Node], page_count: usize) -> Vec<String> {
    let mut issues = Vec::new();
    fn walk(nodes: &[Node], parent: Option<&Node>, page_count: usize, issues: &mut Vec<String>) {
        let mut previous: Option<usize> = None;
        for node in nodes {
            let id = node.node_id.clone().unwrap_or_default();
            let (start, end) = (node.start_index, node.end_index);
            if start < 1 || start > page_count {
                issues.push(format!("[{id}] start_index {start} outside the PDF (1-{page_count})"));
            }
            if end < 1 || end > page_count {
                issues.push(format!("[{id}] end_index {end} outside the PDF (1-{page_count})"));
            }
            if start > end {
                issues.push(format!("[{id}] start_index {start} is after end_index {end}"));
            }
            if let Some(parent) = parent
                && start < parent.start_index
            {
                issues.push(format!(
                    "[{id}] start_index {start} precedes parent {} (starts {})",
                    parent.node_id.clone().unwrap_or_default(),
                    parent.start_index
                ));
            }
            if let Some(prev) = previous
                && start < prev
            {
                issues.push(format!("[{id}] start_index {start} is before the preceding sibling's {prev}"));
            }
            previous = Some(start);
            walk(&node.nodes, Some(node), page_count, issues);
        }
    }
    walk(structure, None, page_count, &mut issues);
    let mut covered = vec![false; page_count + 1];
    for node in crate::tree::subtree(structure) {
        let end = node.end_index.min(page_count);
        if node.start_index <= end {
            covered[node.start_index..=end].fill(true);
        }
    }
    let gaps: Vec<usize> = (1..=page_count).filter(|p| !covered[*p]).collect();
    if !gaps.is_empty() {
        issues.push(format!("pages covered by no node: {gaps:?}"));
    }
    issues
}

// ---------------------------------------------------------------------------
// merge
// ---------------------------------------------------------------------------

fn page_label(node: &Node) -> String {
    let (start, end) = (node.start_index, node.subtree_end());
    if start == end { format!("p.{start}") } else { format!("p.{start}-{end}") }
}

/// The titles of merged same-page siblings joined, or a page label when
/// that is empty or too long to serve as a title.
fn union_title(titles: &[String], node: &Node) -> String {
    let joined = titles.iter().filter(|t| !t.is_empty()).cloned().collect::<Vec<_>>().join("; ");
    if joined.is_empty() || joined.chars().count() > TITLE_MAX_CHARS { page_label(node) } else { joined }
}

/// Collapse frontier siblings that cover exactly the same pages.
pub fn merge_same_page(structure: &mut Vec<Node>, log: &mut Vec<Value>) -> bool {
    let mut changed = false;
    for node in structure.iter_mut() {
        changed |= merge_same_page(&mut node.nodes, log);
    }
    let mut groups: Vec<((usize, usize), Vec<usize>)> = Vec::new();
    for (i, node) in structure.iter().enumerate() {
        if node.nodes.is_empty() {
            let key = (node.start_index, node.subtree_end());
            match groups.iter_mut().find(|(k, _)| *k == key) {
                Some((_, members)) => members.push(i),
                None => groups.push((key, vec![i])),
            }
        }
    }
    let mut drop: HashSet<usize> = HashSet::new();
    for (span, members) in groups.into_iter().filter(|(_, m)| m.len() >= 2) {
        let mut titles = Vec::new();
        for &i in &members {
            titles.push(structure[i].title.clone());
            titles.extend(structure[i].key_items.iter().cloned());
        }
        let keeper = members[0];
        log.push(json!({
            "op": "merge_same_page", "node_id": structure[keeper].node_id, "pages": [span.0, span.1],
            "dropped": members.len() - 1,
            "dropped_ids": members[1..].iter().map(|&i| structure[i].node_id.clone()).collect::<Vec<_>>(),
            "key_items": titles,
        }));
        let title = union_title(&titles, &structure[keeper]);
        let node = &mut structure[keeper];
        node.key_items = titles;
        node.title = title;
        node.same_page = true;
        drop.extend(members[1..].iter().copied());
        changed = true;
    }
    if !drop.is_empty() {
        let mut i = 0;
        structure.retain(|_| {
            let keep = !drop.contains(&i);
            i += 1;
            keep
        });
    }
    changed
}

fn frozen_key(node: &Node) -> String {
    node.node_id.clone().unwrap_or_default()
}

/// Collapse any subtree whose structure does not beat a linear scan.
pub fn merge(structure: &mut [Node], routing: usize, log: &mut Vec<Value>, frozen: &mut HashSet<String>) -> bool {
    fn visit(node: &mut Node, routing: usize, log: &mut Vec<Value>, frozen: &mut HashSet<String>) -> bool {
        if node.nodes.is_empty() {
            return false;
        }
        let mut changed = false;
        for child in node.nodes.iter_mut() {
            changed |= visit(child, routing, log, frozen);
        }
        let cost = tree_cost(node, routing);
        let checked = tree_cost_via_frontier(node, routing);
        let s = span(node);
        if s <= cost {
            let mut titles = Vec::new();
            fn collect(parent: &Node, nodes: &[Node], titles: &mut Vec<String>, removed: &mut Vec<Option<String>>) {
                for child in nodes {
                    removed.push(child.node_id.clone());
                    if !parent.is_intro(child) {
                        titles.push(child.title.clone());
                    }
                    titles.extend(child.key_items.iter().cloned());
                    collect(child, &child.nodes, titles, removed);
                }
            }
            let mut removed = Vec::new();
            collect(node, &node.nodes, &mut titles, &mut removed);
            log.push(json!({
                "op": "merge", "node_id": node.node_id, "S": s, "tree_cost": cost, "frontier_cost": checked,
                "merge_gain": cost - s, "removed": removed.len(), "removed_ids": removed, "key_items": titles,
            }));
            node.end_index = node.subtree_end();
            node.nodes.clear();
            if !titles.is_empty() {
                node.key_items = titles;
            }
            frozen.insert(frozen_key(node));
            changed = true;
        }
        changed
    }
    let mut changed = false;
    for root in structure.iter_mut() {
        changed |= visit(root, routing, log, frozen);
    }
    changed
}

/// Deterministic merge over a structure: the no-LLM path.
pub fn merge_tree(structure: &mut [Node]) {
    merge(structure, ROUTING_COST, &mut Vec::new(), &mut HashSet::new());
}

// ---------------------------------------------------------------------------
// expand
// ---------------------------------------------------------------------------

/// Pull a JSON object out of a model reply, fenced or not.
pub fn extract_json_object(content: &str) -> Result<Value> {
    let mut text = content.trim();
    if text.is_empty() {
        return Err(PageIndexError::invalid("model returned no content"));
    }
    let owned;
    if text.contains("```") {
        let after = text.split_once("```").map_or(text, |(_, rest)| rest);
        let after = after.strip_prefix("json").unwrap_or(after).trim_start();
        owned = after.split("```").next().unwrap_or("").to_owned();
        text = &owned;
    }
    let (Some(start), Some(end)) = (text.find('{'), text.rfind('}')) else {
        return Err(PageIndexError::invalid(format!(
            "no JSON object in reply: {:?}",
            content.chars().take(200).collect::<String>()
        )));
    };
    if end < start {
        return Err(PageIndexError::invalid("no JSON object in reply"));
    }
    serde_json::from_str(&text[start..=end]).map_err(|e| PageIndexError::invalid(format!("invalid JSON in reply: {e}")))
}

async fn ask_model(llm: &dyn LlmClient, model: &str, prompt: &str) -> Result<Value> {
    let params = CompletionParams::default();
    let reply = complete_prompt(llm, model, prompt, &params).await?;
    match extract_json_object(&reply) {
        Ok(value) => Ok(value),
        Err(_) => extract_json_object(&complete_prompt(llm, model, prompt, &params).await?),
    }
}

/// Generate one temporary level of children via the model; validated, not committed.
pub async fn propose_children(llm: &dyn LlmClient, model: &str, node: &Node, pages: &[String]) -> Result<Vec<Node>> {
    let start = node.start_index;
    let end = node.subtree_end().min(pages.len());
    if end < start {
        return Ok(Vec::new());
    }
    let block: Vec<String> = (start..=end)
        .map(|n| {
            let text: String = pages[n - 1].chars().take(PAGE_CHARS).collect();
            format!("<page_{n}>\n{text}\n</page_{n}>")
        })
        .collect();
    let prompt = EXPAND_PROMPT
        .replace("{title}", &node.title)
        .replace("{start}", &start.to_string())
        .replace("{end}", &end.to_string())
        .replace("{pages}", &block.join("\n"));
    let answer = ask_model(llm, model, &prompt).await?;
    let own_key = heading_key(&node.title);
    let mut accepted: Vec<Node> = Vec::new();
    let mut seen = HashSet::new();
    let id = node.node_id.clone().unwrap_or_default();
    for item in answer.get("subsections").and_then(Value::as_array).into_iter().flatten() {
        let title = item.get("title").and_then(Value::as_str).unwrap_or("");
        let Some(page) = item.get("page").and_then(Value::as_u64).map(|p| p as usize) else { continue };
        if title.is_empty() || !(start..=end).contains(&page) {
            continue;
        }
        let key = heading_key(title);
        if !heading_key(&pages[page - 1]).contains(&key) || seen.contains(&key) || key == own_key {
            continue;
        }
        if accepted.last().is_some_and(|last| page < last.start_index) {
            continue;
        }
        seen.insert(key);
        let mut child = Node::new(title.trim(), page, end);
        child.node_id = Some(format!("{id}.{}", accepted.len() + 1));
        accepted.push(child);
    }
    Ok(accepted)
}

/// The headings printed for a node: a same-page fusion keeps them in key_items.
fn headings(node: &Node) -> Vec<String> {
    if node.same_page { node.key_items.clone() } else { vec![node.title.clone()] }
}

/// A node's start page and the headings printed for it.
pub type Printed = (usize, Vec<String>);

#[derive(Debug, Clone, Default)]
pub struct AncestorView {
    pub printed: Printed,
    pub children: Vec<Printed>,
}

/// What [`own_children`] needs to know about a node's surroundings.
#[derive(Debug, Clone, Default)]
pub struct Surroundings {
    /// Ancestors, nearest first.
    pub ancestors: Vec<AncestorView>,
    /// The next node and its first descendants.
    pub next_chain: Vec<Printed>,
}

impl Surroundings {
    fn of(ancestors: &[&Node], next: Option<&Node>, end: usize) -> Self {
        let ancestors = ancestors
            .iter()
            .map(|a| AncestorView {
                printed: (a.start_index, headings(a)),
                children: a.nodes.iter().map(|c| (c.start_index, headings(c))).collect(),
            })
            .collect();
        let mut next_chain = Vec::new();
        let mut cur = next;
        while let Some(n) = cur.filter(|n| n.start_index == end) {
            next_chain.push((n.start_index, headings(n)));
            cur = n.nodes.first();
        }
        Self { ancestors, next_chain }
    }
}

/// The proposed children printed inside the node's own text. Dropped: a
/// heading that already is a node (`known`, or the node's lineage); on a
/// shared first page anything above the node's heading; on a shared last
/// page anything at or below the next node's heading.
pub fn own_children(
    node: &Node,
    children: Vec<Node>,
    lines: &[Vec<String>],
    known: &HashMap<usize, Vec<String>>,
    around: &Surroundings,
) -> Vec<Node> {
    let (start, end) = (node.start_index, node.subtree_end());
    let mut lineage: Vec<Printed> = Vec::new();
    for ancestor in &around.ancestors {
        lineage.push(ancestor.printed.clone());
        lineage.extend(ancestor.children.iter().cloned());
    }
    let found = |page: usize, matches: &dyn Fn(&str) -> bool| -> Vec<usize> {
        lines.get(page.wrapping_sub(1)).map_or_else(Vec::new, |page_lines| {
            page_lines.iter().enumerate().filter(|(_, l)| matches(l)).map(|(i, _)| i).collect()
        })
    };
    let is_node = |page: usize, title: &str| {
        known.get(&page).into_iter().flatten().any(|t| same_heading(title, t))
            || lineage.iter().filter(|(s, _)| *s == page).flat_map(|(_, h)| h).any(|t| same_heading(title, t))
    };
    let own = headings(node);
    let mut tops_sources: Vec<&Vec<String>> = vec![&own];
    tops_sources.extend(around.ancestors.iter().filter(|a| a.printed.0 == start).map(|a| &a.printed.1));
    let top = tops_sources
        .into_iter()
        .map(|hs| found(start, &|line| hs.iter().any(|t| same_heading(line, t))))
        .find(|hits| !hits.is_empty())
        .unwrap_or_default();
    let bottom = around
        .next_chain
        .iter()
        .map(|(_, hs)| found(end, &|line| hs.iter().any(|t| same_heading(line, t))))
        .find(|hits| !hits.is_empty())
        .unwrap_or_default();
    children
        .into_iter()
        .filter(|child| {
            let page = child.start_index;
            let key = heading_key(&child.title);
            let printed =
                if key.is_empty() { Vec::new() } else { found(page, &|line| heading_key(line).contains(&key)) };
            let above = page == start && !top.is_empty() && printed.last().is_some_and(|&p| p < top[0]);
            let below =
                page == end && !bottom.is_empty() && printed.first().is_some_and(|&p| p >= *bottom.last().unwrap());
            !(is_node(page, &child.title) || above || below)
        })
        .collect()
}

/// Options for [`optimize`].
#[derive(Clone)]
pub struct OptimizeOptions {
    pub model: String,
    pub do_merge: bool,
    pub do_expand: bool,
    pub max_rounds: usize,
    pub trigger_pages: usize,
    pub empty_retries: usize,
    pub concurrency: usize,
    pub summary_words: usize,
    pub do_relabel: bool,
    pub llm: Option<Arc<dyn LlmClient>>,
}

impl OptimizeOptions {
    pub fn merge_only() -> Self {
        Self {
            model: String::new(),
            do_merge: true,
            do_expand: false,
            max_rounds: 3,
            trigger_pages: TRIGGER_PAGES,
            empty_retries: 0,
            concurrency: EXPAND_CONCURRENCY,
            summary_words: SUMMARY_MAX_WORDS,
            do_relabel: true,
            llm: None,
        }
    }

    pub fn full(llm: Arc<dyn LlmClient>, model: impl Into<String>) -> Self {
        Self { model: model.into(), do_expand: true, llm: Some(llm), ..Self::merge_only() }
    }
}

/// What an optimize run did.
#[derive(Debug, Clone, Default, Serialize)]
pub struct OptimizeReport {
    pub merges: usize,
    pub expands: usize,
    pub same_page_merges: usize,
    pub same_page_dropped: usize,
    pub kept_collapsed: usize,
    pub before: Complexity,
    pub after: Complexity,
    #[serde(skip)]
    pub log: Vec<Value>,
    #[serde(skip)]
    pub rounds: usize,
    #[serde(skip)]
    pub new_issues: Vec<String>,
}

fn node_at<'a>(structure: &'a [Node], path: &[usize]) -> &'a Node {
    let mut node = &structure[path[0]];
    for &i in &path[1..] {
        node = &node.nodes[i];
    }
    node
}

fn node_at_mut<'a>(structure: &'a mut [Node], path: &[usize]) -> &'a mut Node {
    let mut node = &mut structure[path[0]];
    for &i in &path[1..] {
        node = &mut node.nodes[i];
    }
    node
}

struct WorkItem {
    path: Vec<usize>,
    ancestors: Vec<Vec<usize>>,
    next: Option<Vec<usize>>,
}

fn preorder_paths(
    nodes: &[Node],
    prefix: &mut Vec<usize>,
    ancestors: &mut Vec<Vec<usize>>,
    out: &mut Vec<(Vec<usize>, Vec<Vec<usize>>)>,
) {
    for (i, node) in nodes.iter().enumerate() {
        prefix.push(i);
        out.push((prefix.clone(), ancestors.clone()));
        ancestors.insert(0, prefix.clone());
        preorder_paths(&node.nodes, prefix, ancestors, out);
        ancestors.remove(0);
        prefix.pop();
    }
}

async fn expand(
    structure: &mut [Node],
    pages: &[String],
    lines: &[Vec<String>],
    page_tokens: &[usize],
    opts: &OptimizeOptions,
    log: &mut Vec<Value>,
    frozen: &mut HashSet<String>,
) -> Result<bool> {
    let Some(llm) = opts.llm.clone() else {
        return Err(PageIndexError::invalid("expand needs a model client"));
    };
    let mut known: HashMap<usize, Vec<String>> = HashMap::new();
    for node in crate::tree::subtree(structure) {
        known.entry(node.start_index).or_default().extend(headings(node));
    }
    let mut flat = Vec::new();
    preorder_paths(structure, &mut Vec::new(), &mut Vec::new(), &mut flat);
    let mut wave: Vec<WorkItem> = flat
        .iter()
        .enumerate()
        .map(|(i, (path, ancestors))| WorkItem {
            path: path.clone(),
            ancestors: ancestors.clone(),
            next: flat.get(i + 1).map(|(p, _)| p.clone()),
        })
        .collect();
    let semaphore = Arc::new(Semaphore::new(opts.concurrency.clamp(1, EXPAND_CONCURRENCY)));
    let mut changed = false;

    while !wave.is_empty() {
        let mut jobs = Vec::new();
        for item in wave.drain(..) {
            let node = node_at(structure, &item.path);
            if !node.nodes.is_empty() || frozen.contains(&frozen_key(node)) || span(node) <= opts.trigger_pages {
                continue;
            }
            let ancestors: Vec<&Node> = item.ancestors.iter().map(|p| node_at(structure, p)).collect();
            let next = item.next.as_ref().map(|p| node_at(structure, p));
            let around = Surroundings::of(&ancestors, next, node.subtree_end());
            jobs.push((item, node.clone(), around));
        }
        let futures = jobs.iter().map(|(_, node, around)| {
            let (llm, semaphore, known) = (llm.clone(), semaphore.clone(), &known);
            async move {
                let mut entries = Vec::new();
                let mut attempts = 0;
                while attempts <= opts.empty_retries {
                    attempts += 1;
                    let proposed = {
                        let _permit = semaphore.acquire().await.expect("semaphore open");
                        propose_children(llm.as_ref(), &opts.model, node, pages).await
                    };
                    match proposed {
                        Err(err) if err.is_unrecoverable() => return Err(err),
                        Err(err) => entries.push(json!({
                            "op": "expand", "node_id": node.node_id, "decision": "error", "attempt": attempts,
                            "detail": err.message,
                        })),
                        Ok(children) => {
                            let kept = own_children(node, children, lines, known, around);
                            if !kept.is_empty() {
                                return Ok((Some(kept), attempts, entries));
                            }
                        }
                    }
                }
                Ok((None, attempts, entries))
            }
        });
        let results = futures::future::join_all(futures).await;
        let mut next_wave = Vec::new();
        for ((item, node, _), result) in jobs.into_iter().zip(results) {
            let (children, attempts, entries) = result?;
            log.extend(entries);
            let s = span(&node);
            frozen.insert(frozen_key(&node));
            let Some(children) = children else {
                log.push(json!({"op": "expand", "node_id": node.node_id, "decision": "no_children", "S": s, "attempts": attempts}));
                continue;
            };
            let sized = assign_ends(&node, &children, lines);
            let (cost, res) = expand_cost(&node, &sized, lines, ROUTING_COST);
            let gain = s as i64 - cost as i64;
            let (read_before, read_after) = expand_read_tokens(&node, &sized, lines, page_tokens, opts.summary_words);
            let keep = cost < s && (read_after) < read_before as f64;
            log.push(json!({
                "op": "expand", "node_id": node.node_id, "decision": if keep { "expand" } else { "keep_collapsed" },
                "S": s, "expand_cost": cost, "expand_gain": gain, "S_residual": res,
                "read_before": read_before, "read_after": read_after.round(), "source": format!("llm:{attempts}"),
                "children": sized.iter().map(|c| json!({"node_id": c.node_id, "title": c.title,
                    "start_index": c.start_index, "end_index": c.end_index})).collect::<Vec<_>>(),
            }));
            if !keep {
                continue;
            }
            changed = true;
            let target = node_at_mut(structure, &item.path);
            attach_children(target, sized, lines);
            if opts.do_merge {
                merge_same_page(&mut target.nodes, log);
            }
            let count = target.nodes.len();
            let mut ancestors = vec![item.path.clone()];
            ancestors.extend(item.ancestors.iter().cloned());
            for i in 0..count {
                let mut path = item.path.clone();
                path.push(i);
                let next = if i + 1 < count {
                    let mut p = item.path.clone();
                    p.push(i + 1);
                    Some(p)
                } else {
                    item.next.clone()
                };
                next_wave.push(WorkItem { path, ancestors: ancestors.clone(), next });
            }
        }
        wave = next_wave;
    }
    Ok(changed)
}

/// Run merge and expand until neither changes anything (at most
/// `max_rounds`), then renumber ids. Mutates `structure` in place.
pub async fn optimize(
    structure: &mut Vec<Node>,
    pages: &[String],
    page_count: usize,
    opts: &OptimizeOptions,
) -> Result<OptimizeReport> {
    let lines = page_lines(pages);
    let page_tokens: Vec<usize> =
        if opts.do_expand { pages.iter().map(|p| count_tokens(p)).collect() } else { Vec::new() };
    let baseline: HashSet<String> =
        if page_count > 0 { validate(structure, page_count).into_iter().collect() } else { HashSet::new() };
    let before = if page_count > 0 { complexity(structure, page_count) } else { Complexity::default() };
    let mut log = Vec::new();
    let mut frozen = HashSet::new();
    let mut rounds = 0;
    for round in 1..=opts.max_rounds.max(1) {
        rounds = round;
        let start = log.len();
        if opts.do_merge {
            merge_same_page(structure, &mut log);
        }
        let merged = opts.do_merge && merge(structure, ROUTING_COST, &mut log, &mut frozen);
        if merged {
            merge_same_page(structure, &mut log);
        }
        let expanded =
            opts.do_expand && expand(structure, pages, &lines, &page_tokens, opts, &mut log, &mut frozen).await?;
        let same_page = log[start..].iter().any(|e| e["op"] == "merge_same_page");
        log.push(
            json!({"op": "round", "round": round, "same_page": same_page, "merged": merged, "expanded": expanded}),
        );
        if !(same_page || merged || expanded) {
            break;
        }
    }
    if opts.do_relabel {
        relabel(structure);
    }
    let after = if page_count > 0 { complexity(structure, page_count) } else { Complexity::default() };
    let new_issues = if page_count > 0 {
        validate(structure, page_count).into_iter().filter(|i| !baseline.contains(i)).collect()
    } else {
        Vec::new()
    };
    let count = |op: &str| log.iter().filter(|e| e["op"] == op).count();
    Ok(OptimizeReport {
        merges: count("merge"),
        same_page_merges: count("merge_same_page"),
        same_page_dropped: log
            .iter()
            .filter(|e| e["op"] == "merge_same_page")
            .filter_map(|e| e["dropped"].as_u64())
            .sum::<u64>() as usize,
        expands: log.iter().filter(|e| e["decision"] == "expand").count(),
        kept_collapsed: log.iter().filter(|e| e["decision"] == "keep_collapsed").count(),
        before,
        after,
        rounds,
        new_issues,
        log,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::testing::ScriptedLlm;

    fn shape(nodes: &[Node]) -> Vec<(String, usize, usize)> {
        crate::tree::subtree(nodes).iter().map(|n| (n.title.clone(), n.start_index, n.end_index)).collect()
    }

    fn lines_of(pages: &[String]) -> Vec<Vec<String>> {
        page_lines(pages)
    }

    fn s(t: &str, a: usize, b: usize) -> (String, usize, usize) {
        (t.to_owned(), a, b)
    }

    #[test]
    fn intro_node_holds_the_pages_a_parent_opens_with() {
        let mut lines: Vec<Vec<String>> = (0..40).map(|_| vec!["body".to_owned()]).collect();
        lines[11] = vec!["3.1 Scope".into(), "body".into()];
        let mut tree = vec![
            Node::new("Ch 3", 10, 30).with_children(vec![
                Node::new("3.1 Scope", 12, 20),
                Node::new("3.2", 21, 30).with_children(vec![Node::new("3.2.1", 21, 30)]),
            ]),
            Node::new("", 31, 40).with_children(vec![Node::new("4.1", 33, 40)]),
        ];
        add_intro_nodes(&mut tree, Some(&lines));
        assert_eq!(
            shape(&tree),
            [
                s("Ch 3", 10, 30),
                s("Ch 3 (intro)", 10, 11),
                s("3.1 Scope", 12, 20),
                s("3.2", 21, 30),
                s("3.2.1", 21, 30),
                s("", 31, 40),
                s("Intro", 31, 33),
                s("4.1", 33, 40)
            ]
        );
        let before = shape(&tree);
        add_intro_nodes(&mut tree, Some(&lines));
        assert_eq!(shape(&tree), before);
        assert!(!heading_at_page_start(&[vec!["第一章 总则".into()]], 1, "第一章 总则"));
        assert!(!heading_at_page_start(&[vec!["2".into(), "Body text".into()]], 1, "2"));
    }

    #[test]
    fn a_running_header_or_a_word_prefix_does_not_open_the_page() {
        let page: Vec<String> = ["4.1. Discriminant Functions 181", "opening words", "4.1. Discriminant Functions"]
            .map(String::from)
            .to_vec();
        assert!(!heading_at_page_start(std::slice::from_ref(&page), 1, "4.1. Discriminant Functions"));
        assert!(heading_at_page_start(&[page[2..].to_vec()], 1, "4.1. Discriminant Functions"));
        assert!(!heading_at_page_start(&[vec!["Filed 03/04/24".into(), "I. Background".into()]], 1, "I."));
    }

    #[test]
    fn merge_folds_an_intro_without_listing_its_title() {
        let mut tree =
            vec![Node::new("P", 1, 4).with_children(vec![Node::new("P (intro)", 1, 1), Node::new("C", 2, 4)])];
        merge_tree(&mut tree);
        assert!(tree[0].nodes.is_empty());
        assert_eq!(tree[0].key_items, ["C"]);
    }

    #[test]
    fn same_page_siblings_fuse() {
        let mut tree = vec![Node::new("A", 1, 1), Node::new("B", 1, 1), Node::new("C", 2, 3)];
        let mut log = Vec::new();
        assert!(merge_same_page(&mut tree, &mut log));
        assert_eq!(shape(&tree), [s("A; B", 1, 1), s("C", 2, 3)]);
        assert!(tree[0].same_page);
    }

    #[test]
    fn complexity_metrics() {
        let tree = vec![Node::new("A", 1, 10).with_children(vec![Node::new("A1", 1, 4), Node::new("A2", 5, 10)])];
        let c = complexity(&tree, 10);
        assert_eq!(c.worst_case_search_complexity, 2 + 6);
        assert_eq!(c.frontier_nodes, 2);
        assert!(validate(&tree, 10).is_empty());
        assert_eq!(validate(&[Node::new("A", 2, 3)], 3), vec!["pages covered by no node: [1]".to_string()]);
    }

    fn body_pages(n: usize, words: usize) -> Vec<String> {
        (0..n).map(|_| "body ".repeat(words)).collect()
    }

    fn root(end: usize) -> Vec<Node> {
        let mut r = Node::new("R", 1, end);
        r.node_id = Some("0000".into());
        vec![r]
    }

    #[tokio::test]
    async fn expand_asks_once_when_the_model_finds_no_subsections() {
        let pages = body_pages(20, 250);
        let llm = Arc::new(ScriptedLlm::new(vec![r#"{"subsections": []}"#.into()]));
        let mut tree = root(20);
        optimize(&mut tree, &pages, 20, &OptimizeOptions::full(llm.clone(), "m")).await.unwrap();
        assert_eq!(llm.calls(), 1);
        assert!(tree[0].nodes.is_empty());
    }

    #[tokio::test]
    async fn expand_asks_again_after_a_broken_reply() {
        let pages: Vec<String> = (1..=20)
            .map(|p| {
                format!(
                    "{}{}",
                    if [6, 11, 16].contains(&p) { format!("Part {p}\n") } else { String::new() },
                    "body ".repeat(250)
                )
            })
            .collect();
        let llm = Arc::new(ScriptedLlm::new(vec![
            "The section has three parts.".into(),
            r#"{"subsections": [{"title": "Part 6", "page": 6}, {"title": "Part 11", "page": 11}, {"title": "Part 16", "page": 16}]}"#.into(),
        ]));
        let mut tree = root(20);
        optimize(&mut tree, &pages, 20, &OptimizeOptions::full(llm.clone(), "m")).await.unwrap();
        assert_eq!(llm.calls(), 2);
        assert_eq!(
            shape(&tree[0].nodes),
            [s("R (intro)", 1, 5), s("Part 6", 6, 10), s("Part 11", 11, 15), s("Part 16", 16, 20)]
        );
        let ids: Vec<_> = crate::tree::subtree(&tree).iter().map(|n| n.node_id.clone().unwrap()).collect();
        assert_eq!(ids, ["0000", "0001", "0002", "0003", "0004"]);
    }

    #[tokio::test]
    async fn expand_weighs_what_the_pieces_cost_to_read() {
        for (headings, summary_words, splits) in [
            ((2..=20).map(|p| (p, format!("Part {p}"))).collect::<Vec<_>>(), 150, false),
            ((2..=20).map(|p| (p, format!("Part {p}"))).collect::<Vec<_>>(), 0, true),
            (vec![(6, "Part A".to_string()), (11, "Part B".into()), (16, "Part C".into())], 150, true),
        ] {
            let pages: Vec<String> = (1..=20)
                .map(|p| {
                    let head =
                        headings.iter().find(|(hp, _)| *hp == p).map(|(_, t)| format!("{t}\n")).unwrap_or_default();
                    format!("{head}{}", "body ".repeat(150))
                })
                .collect();
            let reply = json!({"subsections": headings.iter().map(|(p, t)| json!({"title": t, "page": p})).collect::<Vec<_>>()});
            let llm = Arc::new(ScriptedLlm::new(vec![reply.to_string()]));
            let mut tree = root(20);
            let opts = OptimizeOptions { summary_words, ..OptimizeOptions::full(llm, "m") };
            let report = optimize(&mut tree, &pages, 20, &opts).await.unwrap();
            let decision = report.log.iter().find(|e| e["op"] == "expand").unwrap();
            assert!(decision["expand_cost"].as_u64().unwrap() < decision["S"].as_u64().unwrap());
            assert_eq!(!tree[0].nodes.is_empty(), splits, "{summary_words} {}", headings.len());
        }
    }

    #[test]
    fn a_proposed_child_must_be_printed_inside_its_nodes_own_text() {
        let mut lines: Vec<Vec<String>> = (0..20).map(|_| vec!["body".to_owned()]).collect();
        lines[11] = ["2.3 Limits", "3 Results", "Scope", "3.1 Data", "3.2 Results"].map(String::from).to_vec();
        let methods = Node::new("2 Methods", 4, 12);
        let results = Node::new("3 Results", 12, 20);
        let root = Node::new("R", 1, 20).with_children(vec![methods.clone(), results.clone()]);
        let mut known: HashMap<usize, Vec<String>> = HashMap::new();
        for n in crate::tree::subtree(std::slice::from_ref(&root)) {
            known.entry(n.start_index).or_default().push(n.title.clone());
        }
        let kept = |node: &Node,
                    ancestors: &[&Node],
                    next: Option<&Node>,
                    titles: &[&str],
                    known: &HashMap<usize, Vec<String>>| {
            let children = titles.iter().map(|t| Node::new(*t, 12, 0)).collect();
            let around = Surroundings::of(ancestors, next, node.subtree_end());
            own_children(node, children, &lines, known, &around).into_iter().map(|c| c.title).collect::<Vec<_>>()
        };
        assert_eq!(
            kept(&methods, &[&root], Some(&results), &["2.3 Limits", "Results", "3.1 Data"], &known),
            ["2.3 Limits"]
        );
        assert_eq!(kept(&methods, &[&root], None, &["2.3 Limits", "Results"], &known), ["2.3 Limits"]);
        assert_eq!(
            kept(&results, &[&root], None, &["2.3 Limits", "Results", "3.1 Data", "3.2 Results"], &known),
            ["3.1 Data", "3.2 Results"]
        );
        let findings = Node { title: "Findings".into(), ..results.clone() };
        assert_eq!(
            kept(&methods, &[&root], Some(&findings), &["2.3 Limits", "3.2 Results"], &known),
            ["2.3 Limits", "3.2 Results"]
        );
    }

    #[test]
    fn extracts_fenced_json() {
        let v = extract_json_object("```json\n{\"a\": 1}\n```").unwrap();
        assert_eq!(v["a"], 1);
        assert!(extract_json_object("no json").is_err());
    }

    #[test]
    fn lines_helper() {
        assert_eq!(lines_of(&["a\n\n b ".into()]), vec![vec!["a".to_string(), "b".to_string()]]);
    }
}
