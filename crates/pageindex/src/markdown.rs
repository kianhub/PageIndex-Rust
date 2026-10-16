//! Markdown to tree: `#` headings (and whole-line bold headings) outside
//! code fences become nodes holding the text up to the next heading.

use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;
use serde::Serialize;

use crate::error::{ErrorKind, PageIndexError, Result};
use crate::llm::{CompletionParams, LlmClient, complete_prompt};
use crate::summary::generate_doc_description_from;
use crate::text::count_tokens;
use crate::tree::{Node, for_each_mut, write_node_ids};

static HEADER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(#{1,6})\s+(.+)$").unwrap());
static BOLD_HEADING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\*\*(.+?)\*\*\s*$").unwrap());

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MdHeading {
    pub title: String,
    pub line_num: usize,
    pub level: usize,
}

#[derive(Debug, Clone)]
struct MdNode {
    title: String,
    line_num: usize,
    level: usize,
    text: String,
    tokens: usize,
}

#[derive(Debug, Clone)]
pub struct MdOptions {
    pub if_thinning: bool,
    pub min_token_threshold: usize,
    pub if_add_node_summary: bool,
    pub summary_token_threshold: usize,
    pub summary_model: String,
    pub if_add_doc_description: bool,
    pub if_add_node_text: bool,
    pub if_add_node_id: bool,
}

impl Default for MdOptions {
    fn default() -> Self {
        Self {
            if_thinning: false,
            min_token_threshold: 5000,
            if_add_node_summary: false,
            summary_token_threshold: 200,
            summary_model: crate::config::DEFAULT_INDEX_MODEL.into(),
            if_add_doc_description: false,
            if_add_node_text: false,
            if_add_node_id: true,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MdResult {
    pub doc_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_description: Option<String>,
    pub line_count: usize,
    pub structure: Vec<Node>,
}

pub fn extract_nodes_from_markdown(content: &str) -> Vec<MdHeading> {
    let mut nodes = Vec::new();
    let mut in_code = false;
    for (i, line) in content.split('\n').enumerate() {
        let stripped = line.trim();
        if stripped.starts_with("```") {
            in_code = !in_code;
            continue;
        }
        if stripped.is_empty() || in_code {
            continue;
        }
        if let Some(caps) = HEADER.captures(stripped) {
            nodes.push(MdHeading { title: caps[2].trim().to_owned(), line_num: i + 1, level: caps[1].len() });
            continue;
        }
        if let Some(caps) = BOLD_HEADING.captures(stripped) {
            let title = caps[1].trim();
            if !title.is_empty() {
                nodes.push(MdHeading { title: title.to_owned(), line_num: i + 1, level: 1 });
            }
        }
    }
    nodes
}

fn with_text(headings: Vec<MdHeading>, lines: &[&str]) -> Vec<MdNode> {
    let starts: Vec<usize> = headings.iter().map(|h| h.line_num).collect();
    headings
        .into_iter()
        .enumerate()
        .map(|(i, h)| {
            let end = starts.get(i + 1).map_or(lines.len(), |&next| next - 1);
            let text = lines[h.line_num - 1..end].join("\n").trim().to_owned();
            MdNode { title: h.title, line_num: h.line_num, level: h.level, text, tokens: 0 }
        })
        .collect()
}

fn descendants(nodes: &[MdNode], parent: usize) -> Vec<usize> {
    let level = nodes[parent].level;
    (parent + 1..nodes.len()).take_while(|&i| nodes[i].level > level).collect()
}

fn count_subtree_tokens(nodes: &mut [MdNode]) {
    for i in (0..nodes.len()).rev() {
        let mut total = nodes[i].text.clone();
        for child in descendants(nodes, i) {
            if !nodes[child].text.is_empty() {
                total.push('\n');
                total.push_str(&nodes[child].text);
            }
        }
        nodes[i].tokens = count_tokens(&total);
    }
}

/// Fold every node whose subtree is under `min_tokens` into one node.
fn thin(mut nodes: Vec<MdNode>, min_tokens: usize) -> Vec<MdNode> {
    let mut removed = vec![false; nodes.len()];
    for i in (0..nodes.len()).rev() {
        if removed[i] || nodes[i].tokens >= min_tokens {
            continue;
        }
        let mut texts = Vec::new();
        for child in descendants(&nodes, i) {
            if !removed[child] {
                if !nodes[child].text.trim().is_empty() {
                    texts.push(nodes[child].text.clone());
                }
                removed[child] = true;
            }
        }
        if !texts.is_empty() {
            let mut merged = nodes[i].text.clone();
            for text in texts {
                if !merged.is_empty() && !merged.ends_with('\n') {
                    merged.push_str("\n\n");
                }
                merged.push_str(&text);
            }
            nodes[i].tokens = count_tokens(&merged);
            nodes[i].text = merged;
        }
    }
    let mut i = 0;
    nodes.retain(|_| {
        let keep = !removed[i];
        i += 1;
        keep
    });
    nodes
}

fn build_tree(nodes: Vec<MdNode>) -> Vec<Node> {
    struct Pending {
        node: Node,
        level: usize,
        children: Vec<usize>,
    }
    let mut arena: Vec<Pending> = Vec::new();
    let mut roots = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for (counter, n) in nodes.into_iter().enumerate() {
        let node = Node {
            title: n.title,
            node_id: Some(format!("{:04}", counter + 1)),
            line_num: Some(n.line_num),
            text: Some(n.text),
            ..Node::default()
        };
        let idx = arena.len();
        arena.push(Pending { node, level: n.level, children: Vec::new() });
        while stack.last().is_some_and(|&top| arena[top].level >= n.level) {
            stack.pop();
        }
        match stack.last() {
            Some(&parent) => arena[parent].children.push(idx),
            None => roots.push(idx),
        }
        stack.push(idx);
    }
    fn assemble(idx: usize, arena: &mut Vec<Pending>) -> Node {
        let children = std::mem::take(&mut arena[idx].children);
        let mut node = std::mem::take(&mut arena[idx].node);
        node.nodes = children.into_iter().map(|c| assemble(c, arena)).collect();
        node
    }
    roots.into_iter().map(|r| assemble(r, &mut arena)).collect()
}

async fn node_summary(text: &str, threshold: usize, llm: &dyn LlmClient, model: &str) -> Result<String> {
    if count_tokens(text) < threshold {
        return Ok(text.to_owned());
    }
    let prompt = format!(
        "You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

    Partial Document Text: {text}
    
    Directly return the description, do not include any other text.
    "
    );
    complete_prompt(llm, model, &prompt, &CompletionParams::default()).await
}

/// Build the tree for a Markdown file. `llm` is only called when summaries
/// (and a description) are requested.
pub async fn md_to_tree(path: impl AsRef<Path>, opts: &MdOptions, llm: Option<&dyn LlmClient>) -> Result<MdResult> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|e| {
        PageIndexError::new(ErrorKind::NotFound, format!("Markdown file not found: {}: {e}", path.display()))
    })?;
    md_to_tree_from_str(&content, &doc_stem(path), opts, llm).await
}

fn doc_stem(path: &Path) -> String {
    path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

pub async fn md_to_tree_from_str(
    content: &str,
    doc_name: &str,
    opts: &MdOptions,
    llm: Option<&dyn LlmClient>,
) -> Result<MdResult> {
    let line_count = content.matches('\n').count() + 1;
    let lines: Vec<&str> = content.split('\n').collect();
    let mut nodes = with_text(extract_nodes_from_markdown(content), &lines);
    if opts.if_thinning {
        count_subtree_tokens(&mut nodes);
        nodes = thin(nodes, opts.min_token_threshold);
    }
    let mut structure = build_tree(nodes);
    if opts.if_add_node_id {
        write_node_ids(&mut structure);
    }
    let mut doc_description = None;
    if opts.if_add_node_summary {
        let llm = llm.ok_or_else(|| PageIndexError::invalid("node summaries need a model client"))?;
        let flat: Vec<(String, bool)> = crate::tree::subtree(&structure)
            .iter()
            .map(|n| (n.text.clone().unwrap_or_default(), n.nodes.is_empty()))
            .collect();
        let summaries = futures::future::join_all(
            flat.iter().map(|(text, _)| node_summary(text, opts.summary_token_threshold, llm, &opts.summary_model)),
        )
        .await;
        let mut i = 0;
        let mut failure = None;
        for_each_mut(&mut structure, &mut |node| {
            match &summaries[i] {
                Ok(summary) if flat[i].1 => node.summary = Some(summary.clone()),
                Ok(summary) => node.prefix_summary = Some(summary.clone()),
                Err(err) => failure = failure.take().or_else(|| Some(err.clone())),
            }
            i += 1;
        });
        if let Some(err) = failure {
            return Err(err);
        }
        if !opts.if_add_node_text {
            crate::tree::remove_text(&mut structure);
        }
        if opts.if_add_doc_description {
            let clean = clean_for_description(&structure);
            doc_description = Some(generate_doc_description_from(&clean, llm, &opts.summary_model).await?);
        }
    } else if !opts.if_add_node_text {
        crate::tree::remove_text(&mut structure);
    }
    Ok(MdResult { doc_name: doc_name.to_owned(), doc_description, line_count, structure })
}

fn clean_for_description(nodes: &[Node]) -> serde_json::Value {
    serde_json::Value::Array(
        nodes
            .iter()
            .map(|n| {
                let mut obj = serde_json::Map::new();
                obj.insert("title".into(), n.title.clone().into());
                if let Some(id) = &n.node_id {
                    obj.insert("node_id".into(), id.clone().into());
                }
                if let Some(s) = &n.summary {
                    obj.insert("summary".into(), s.clone().into());
                }
                if let Some(s) = &n.prefix_summary {
                    obj.insert("prefix_summary".into(), s.clone().into());
                }
                if !n.nodes.is_empty() {
                    obj.insert("nodes".into(), clean_for_description(&n.nodes));
                }
                serde_json::Value::Object(obj)
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::testing::ScriptedLlm;

    #[test]
    fn skips_bold_heading_with_only_whitespace() {
        let nodes = extract_nodes_from_markdown("**   **\n**Valid heading**");
        assert_eq!(nodes, vec![MdHeading { title: "Valid heading".into(), line_num: 2, level: 1 }]);
    }

    #[test]
    fn headings_inside_code_fences_are_ignored() {
        let nodes = extract_nodes_from_markdown("# A\n```\n# not\n```\n## B");
        assert_eq!(nodes.iter().map(|n| n.title.as_str()).collect::<Vec<_>>(), ["A", "B"]);
    }

    #[tokio::test]
    async fn builds_a_nested_tree_without_a_model() {
        let md = "# Title\n\nIntro.\n\n## Section\n\nBody.\n";
        let out = md_to_tree_from_str(md, "notes", &MdOptions { if_add_node_id: false, ..Default::default() }, None)
            .await
            .unwrap();
        assert_eq!(out.line_count, 8);
        let json = serde_json::to_value(&out).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"doc_name": "notes", "line_count": 8, "structure": [
                {"title": "Title", "node_id": "0001", "line_num": 1,
                 "nodes": [{"title": "Section", "node_id": "0002", "line_num": 5}]}]})
        );
    }

    #[tokio::test]
    async fn thinning_folds_small_subtrees() {
        let md = "# A\nalpha\n## B\nbeta\n# C\ngamma";
        let opts =
            MdOptions { if_thinning: true, min_token_threshold: 1000, if_add_node_text: true, ..Default::default() };
        let out = md_to_tree_from_str(md, "d", &opts, None).await.unwrap();
        let titles: Vec<_> = out.structure.iter().map(|n| n.title.as_str()).collect();
        assert_eq!(titles, ["A", "C"]);
        assert_eq!(out.structure[0].text.as_deref(), Some("# A\nalpha\n\n## B\nbeta"));
    }

    #[tokio::test]
    async fn summary_model_drives_summary_and_description_calls() {
        let llm = ScriptedLlm::new(vec!["node summary".into()]);
        let opts = MdOptions {
            if_add_node_summary: true,
            if_add_doc_description: true,
            summary_token_threshold: 1,
            summary_model: "SUMMARY-SENTINEL".into(),
            ..Default::default()
        };
        let out =
            md_to_tree_from_str("# Title\n\nIntro.\n\n## Section\n\nBody.\n", "n", &opts, Some(&llm)).await.unwrap();
        let mut models = llm.models.lock().unwrap().clone();
        models.dedup();
        assert_eq!(models, ["SUMMARY-SENTINEL"]);
        assert_eq!(out.structure[0].prefix_summary.as_deref(), Some("node summary"));
        assert_eq!(out.structure[0].nodes[0].summary.as_deref(), Some("node summary"));
        assert!(out.doc_description.is_some());
    }
}
