//! Bottom-up node summaries: leaves from their own pages, parents from
//! their children's summaries plus the pages no child covers. A parent's
//! summary describes its whole subtree.

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, Ordering};

use regex::Regex;
use serde_json::{Value, json};
use tokio::sync::Semaphore;

use crate::error::{ErrorKind, PageIndexError, Result};
use crate::llm::{CompletionParams, LlmClient, complete_prompt};
use crate::text::{count_tokens, has_latin_letter, normalize};
use crate::tree::{Node, pages_text};

pub const SUMMARY_CONCURRENCY: usize = 64;
pub const SUMMARY_INTRO_MAX_PAGES: usize = 3;
pub const SUMMARY_MAX_WORDS: usize = 150;
pub const TOKENS_PER_WORD: f64 = 4.0 / 3.0;
pub const SUMMARY_REASONING_EFFORT: &str = "none";
pub const FALLBACK_SUMMARY_CHARS: usize = 600;

static LEADING_NUMBERS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(\d+ )+").unwrap());
static FENCE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)^.*?```(?:json)?\s*").unwrap());

#[derive(Debug, Clone)]
pub struct SummaryOptions {
    pub model: String,
    pub concurrency: usize,
    pub max_words: usize,
    /// Leaves under this many tokens keep their raw text (default: the summary length).
    pub small_node_tokens: Option<f64>,
    pub max_intro_pages: usize,
    /// A leaf reads only its side of a page it shares with a neighbour.
    pub split_shared_pages: bool,
    pub reasoning_effort: Option<String>,
    /// Stop sending the effort once the server refused a prompt with it (HTTP 400).
    pub drop_refused_effort: bool,
}

impl SummaryOptions {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            concurrency: SUMMARY_CONCURRENCY,
            max_words: SUMMARY_MAX_WORDS,
            small_node_tokens: None,
            max_intro_pages: SUMMARY_INTRO_MAX_PAGES,
            split_shared_pages: false,
            reasoning_effort: None,
            drop_refused_effort: false,
        }
    }
}

/// The JSON object in a model reply, trying light repairs.
fn reply_json(reply: &str) -> Option<Value> {
    let text = reply.trim();
    if text.is_empty() {
        return None;
    }
    let unfenced;
    let text = if text.contains("```") {
        unfenced = FENCE.replace(text, "").split("```").next().unwrap_or("").to_owned();
        unfenced.as_str()
    } else {
        text
    };
    let (start, end) = (text.find('{')?, text.rfind('}')?);
    if end <= start {
        return None;
    }
    let obj = &text[start..=end];
    let collapsed = obj.split_whitespace().collect::<Vec<_>>().join(" ");
    let repaired = collapsed.replace(",]", "]").replace(",}", "}");
    [obj, collapsed.as_str(), repaired.as_str()].iter().find_map(|c| serde_json::from_str(c).ok())
}

fn field_text(value: &Value) -> String {
    match value {
        Value::Array(items) => items
            .iter()
            .map(|i| i.as_str().map(str::to_owned).unwrap_or_else(|| i.to_string()).trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" "),
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// The `summary` field of a reply, or the reply itself when there is none.
pub fn parse_summary(reply: &str) -> String {
    if reply.trim().is_empty() {
        return String::new();
    }
    match reply_json(reply) {
        Some(Value::Object(map)) if map.contains_key("summary") => field_text(&map["summary"]).trim().to_owned(),
        _ => reply.trim().to_owned(),
    }
}

/// The `title` field of a reply, or empty.
pub fn parse_title(reply: &str) -> String {
    match reply_json(reply) {
        Some(Value::Object(map)) => {
            map.get("title").map(field_text).unwrap_or_default().split_whitespace().collect::<Vec<_>>().join(" ")
        }
        _ => String::new(),
    }
}

/// A summary for a node the model left unsummarized.
pub fn fallback_summary(node: &Node, text: &str) -> String {
    let summary = if node.nodes.is_empty() {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    } else {
        node.nodes.iter().filter(|c| !c.title.is_empty()).map(|c| c.title.as_str()).collect::<Vec<_>>().join("; ")
    };
    let summary: String = summary.chars().take(FALLBACK_SUMMARY_CHARS).collect();
    if summary.is_empty() { node.title.clone() } else { summary }
}

fn intro_text(node: &Node, pages: &[String], max_pages: usize) -> String {
    match node.nodes.first() {
        Some(first) if first.start_index > node.start_index => {
            let end = (first.start_index - 1).min(node.start_index + max_pages - 1);
            pages_text(pages, node.start_index, end)
        }
        _ => String::new(),
    }
}

/// Byte offset of the one line opening with `node`'s heading, its section
/// number aside; `None` when that cannot be told.
fn heading_offset(page_text: &str, node: Option<&Node>) -> Option<usize> {
    let node = node.filter(|n| !n.same_page)?;
    let key = normalize(&node.title);
    if !has_latin_letter(&key) {
        return None;
    }
    let prefix = format!("{key} ");
    let mut hits = Vec::new();
    let mut offset = 0;
    for line in page_text.split_inclusive('\n') {
        let norm = normalize(line);
        let bare = LEADING_NUMBERS.replace(&norm, "").into_owned();
        if [norm, bare].iter().any(|n| format!("{n} ").starts_with(&prefix)) {
            hits.push(offset);
        }
        offset += line.len();
    }
    (hits.len() == 1).then(|| hits[0])
}

/// A leaf's pages less what its neighbouring leaf reads of a page they share.
pub fn leaf_own_text(flat: &[&Node], leaf: usize, pages: &[String]) -> String {
    let node = flat[leaf];
    let (start, end) = (node.start_index, node.end_index);
    let text = pages_text(pages, start, end);
    if start == 0 || end < start || end > pages.len() {
        return text;
    }
    let mut top = leaf;
    while top > 0 && !flat[top - 1].nodes.is_empty() {
        top -= 1;
    }
    let head = (top > 0 && flat[top - 1].end_index == start && flat[top].start_index == start).then(|| flat[top]);
    let tail = flat.get(leaf + 1).copied().filter(|tail| {
        let mut next_leaf = *tail;
        while let Some(first) = next_leaf.nodes.first() {
            next_leaf = first;
        }
        tail.start_index == end && next_leaf.start_index == end
    });
    let (first, last) = (&pages[start - 1], &pages[end - 1]);
    let mut cut_start = heading_offset(first, head);
    let mut cut_end = heading_offset(last, tail);
    if start == end
        && let (Some(s), Some(e)) = (cut_start, cut_end)
        && e <= s
    {
        cut_start = None;
        cut_end = None;
    }
    if cut_start.is_none() && cut_end.is_none() {
        return text;
    }
    let cs = cut_start.unwrap_or(0);
    if start == end {
        return first[cs..cut_end.unwrap_or(first.len())].to_owned();
    }
    format!("{}{}{}", &first[cs..], pages_text(pages, start + 1, end - 1), &last[..cut_end.unwrap_or(last.len())])
}

fn leaf_prompt(text: &str, max_words: usize, retitle_items: Option<&str>) -> String {
    let ask_title = retitle_items.map_or(String::new(), |titles| {
        format!(
            "\n    The text is one page holding several short sections: {titles}. Also return a short title, at most 12 words, naming what the whole page covers."
        )
    });
    let title_field = if retitle_items.is_some() {
        "\n        \"title\": <a short title naming what the whole page covers>,"
    } else {
        ""
    };
    format!(
        "You are given a text chunk from a document.
    Your task is to generate a concise description of everything that is covered in the text, summarizing all its points without omitting any type of content.
    Keep the description concise and to the point, avoiding unnecessary details, within {max_words} words.{ask_title}

    Given Text: {text}

    Reply strictly in the following JSON format:
    {{{title_field}
        \"summary\": <a concise description of everything that is covered in the text, summarizing all its points without omitting any type of content>
    }}

    Follow strictly the above JSON return format. Do not include any other text!
    "
    )
}

fn parent_prompt(node: &Node, intro: &str, max_words: usize) -> String {
    let listing: Vec<Value> = node
        .nodes
        .iter()
        .map(|c| json!({"title": c.title, "summary": c.summary.clone().unwrap_or_default()}))
        .collect();
    let listing = serde_json::to_string(&listing).unwrap_or_default();
    format!(
        "You are given a section of a document: the text that opens the section (possibly empty) and the titles and summaries of its subsections.
    Your task is to generate a concise description of everything that is covered in the whole section, summarizing all its points without omitting any type of content.
    Keep the description concise and to the point, avoiding unnecessary details, within {max_words} words.

    Section Title: {}

    Opening Text: {intro}

    Subsection Titles and Summaries: {listing}

    Reply strictly in the following JSON format:
    {{
        \"summary\": <a concise description of everything that is covered in the section, summarizing all its points without omitting any type of content>
    }}

    Follow strictly the above JSON return format. Do not include any other text!
    ",
        node.title
    )
}

struct Caller<'a> {
    llm: &'a dyn LlmClient,
    model: &'a str,
    effort: Option<String>,
    drop_refused: bool,
    effort_dropped: AtomicBool,
    gate: Arc<Semaphore>,
    asked: AtomicBool,
    answered: AtomicBool,
}

impl Caller<'_> {
    async fn ask(&self, prompt: &str) -> Result<String> {
        self.asked.store(true, Ordering::SeqCst);
        let _permit = self.gate.acquire().await.expect("summary gate open");
        let send_effort = !self.effort_dropped.load(Ordering::SeqCst);
        let params =
            CompletionParams { reasoning_effort: self.effort.clone().filter(|_| send_effort), ..Default::default() };
        let reply = match complete_prompt(self.llm, self.model, prompt, &params).await {
            Ok(reply) => reply,
            Err(err) if params.reasoning_effort.is_some() && self.drop_refused && err.status_code == Some(400) => {
                let reply = complete_prompt(self.llm, self.model, prompt, &CompletionParams::default()).await?;
                self.effort_dropped.store(true, Ordering::SeqCst);
                reply
            }
            Err(err) => return Err(err),
        };
        if !parse_summary(&reply).is_empty() {
            self.answered.store(true, Ordering::SeqCst);
        }
        Ok(reply)
    }
}

fn paths(nodes: &[Node], prefix: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    for (i, node) in nodes.iter().enumerate() {
        prefix.push(i);
        out.push(prefix.clone());
        paths(&node.nodes, prefix, out);
        prefix.pop();
    }
}

fn at_mut<'a>(nodes: &'a mut [Node], path: &[usize]) -> &'a mut Node {
    let mut node = &mut nodes[path[0]];
    for &i in &path[1..] {
        node = &mut node.nodes[i];
    }
    node
}

/// Summarize every node that has no summary yet. Fails when the model never
/// answered (every call failed or came back empty).
pub async fn summarize_tree(
    structure: &mut [Node],
    pages: &[String],
    llm: &dyn LlmClient,
    opts: &SummaryOptions,
) -> Result<()> {
    let mut all = Vec::new();
    paths(structure, &mut Vec::new(), &mut all);
    let small = opts.small_node_tokens.unwrap_or(TOKENS_PER_WORD * opts.max_words as f64);
    let caller = Caller {
        llm,
        model: &opts.model,
        effort: opts.reasoning_effort.clone(),
        drop_refused: opts.drop_refused_effort,
        effort_dropped: AtomicBool::new(false),
        gate: Arc::new(Semaphore::new(opts.concurrency.max(1))),
        asked: AtomicBool::new(false),
        answered: AtomicBool::new(false),
    };

    // Leaves, all at once.
    let leaf_jobs: Vec<(Vec<usize>, String, Option<String>)> = {
        let flat = crate::tree::subtree(structure);
        all.iter()
            .enumerate()
            .filter(|(i, _)| flat[*i].nodes.is_empty() && flat[*i].summary.is_none())
            .map(|(i, path)| {
                let text = if opts.split_shared_pages {
                    leaf_own_text(&flat, i, pages)
                } else {
                    pages_text(pages, flat[i].start_index, flat[i].end_index)
                };
                let retitle = flat[i].same_page.then(|| flat[i].key_items.join("; "));
                (path.clone(), text, retitle)
            })
            .collect()
    };
    let leaf_results = futures::future::join_all(leaf_jobs.iter().map(|(_, text, retitle)| async {
        if (count_tokens(text) as f64) < small {
            return Ok((text.trim().to_owned(), None));
        }
        let reply = caller.ask(&leaf_prompt(text, opts.max_words, retitle.as_deref())).await?;
        let title = retitle.is_some().then(|| parse_title(&reply)).filter(|t| !t.is_empty());
        Ok::<_, PageIndexError>((parse_summary(&reply), title))
    }))
    .await;
    for ((path, text, _), result) in leaf_jobs.iter().zip(leaf_results) {
        let node = at_mut(structure, path);
        match result {
            Ok((summary, title)) => {
                if let Some(title) = title {
                    node.title = title;
                }
                node.summary = Some(summary);
            }
            Err(err) if err.is_unrecoverable() => return Err(err),
            Err(_) => node.summary = Some(String::new()),
        }
        if node.summary.as_deref().is_some_and(str::is_empty) {
            node.summary = Some(fallback_summary(node, text));
        }
    }

    // Parents, deepest first, one depth at a time.
    let max_depth = all.iter().map(Vec::len).max().unwrap_or(0);
    for depth in (1..=max_depth).rev() {
        let jobs: Vec<(Vec<usize>, String)> = all
            .iter()
            .filter(|p| p.len() == depth)
            .filter_map(|p| {
                let node = at_mut(structure, p);
                (!node.nodes.is_empty() && node.summary.is_none()).then(|| {
                    (p.clone(), parent_prompt(node, &intro_text(node, pages, opts.max_intro_pages), opts.max_words))
                })
            })
            .collect();
        let results = futures::future::join_all(jobs.iter().map(|(_, prompt)| caller.ask(prompt))).await;
        for ((path, _), result) in jobs.iter().zip(results) {
            let node = at_mut(structure, path);
            node.summary = Some(match result {
                Ok(reply) => parse_summary(&reply),
                Err(err) if err.is_unrecoverable() => return Err(err),
                Err(_) => String::new(),
            });
            if node.summary.as_deref().is_some_and(str::is_empty) {
                node.summary = Some(fallback_summary(node, ""));
            }
        }
    }

    let any_summary =
        crate::tree::subtree(structure).iter().any(|n| n.summary.as_deref().is_some_and(|s| !s.is_empty()));
    if (caller.asked.load(Ordering::SeqCst) && !caller.answered.load(Ordering::SeqCst))
        || (!all.is_empty() && !any_summary)
    {
        return Err(PageIndexError::new(
            ErrorKind::Llm,
            "Summary generation failed for all nodes (every summary call failed or returned empty; check the model and its context limits)",
        ));
    }
    crate::tree::for_each_mut(structure, &mut |n| n.same_page = false);
    Ok(())
}

/// A one-sentence description of the document from its structure; empty
/// when the prompt overran the context (HTTP 400).
pub async fn generate_doc_description(structure: &[Node], llm: &dyn LlmClient, model: &str) -> Result<String> {
    fn clean(nodes: &[Node], top_level_only: bool) -> Vec<Value> {
        nodes
            .iter()
            .map(|n| {
                let mut obj = serde_json::Map::new();
                obj.insert("title".into(), json!(n.title));
                if let Some(id) = &n.node_id {
                    obj.insert("node_id".into(), json!(id));
                }
                if let Some(s) = &n.summary {
                    obj.insert("summary".into(), json!(s));
                }
                if let Some(s) = &n.prefix_summary {
                    obj.insert("prefix_summary".into(), json!(s));
                }
                if !top_level_only && !n.nodes.is_empty() {
                    obj.insert("nodes".into(), Value::Array(clean(&n.nodes, false)));
                }
                Value::Object(obj)
            })
            .collect()
    }
    generate_doc_description_from(&Value::Array(clean(structure, true)), llm, model).await
}

pub async fn generate_doc_description_from(structure: &Value, llm: &dyn LlmClient, model: &str) -> Result<String> {
    let prompt = format!(
        "Your are an expert in generating descriptions for a document.
    You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.
        
    Document Structure: {structure}
    
    Directly return the description, do not include any other text.
    "
    );
    match complete_prompt(llm, model, &prompt, &CompletionParams::default()).await {
        Ok(text) => Ok(text),
        Err(err) if err.status_code == Some(400) => Ok(String::new()),
        Err(err) => Err(err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::testing::ScriptedLlm;

    #[test]
    fn parses_summary_replies() {
        assert_eq!(parse_summary(r#"{"summary": "ok"}"#), "ok");
        assert_eq!(parse_summary("```json\n{\"summary\": [\"a\", \"b\"]}\n```"), "a b");
        assert_eq!(parse_summary("plain"), "plain");
        assert_eq!(parse_title(r#"{"title": " A  page ", "summary": "x"}"#), "A page");
        assert_eq!(parse_title("plain"), "");
    }

    #[tokio::test]
    async fn parents_fall_back_to_their_subsection_titles() {
        // Leaves answer, the parent's reply is empty.
        let llm = ScriptedLlm::new(vec![r#"{"summary": "ok"}"#.into(), r#"{"summary": "ok"}"#.into(), String::new()]);
        let mut tree = vec![Node::new("R", 1, 2).with_children(vec![Node::new("A", 1, 1), Node::new("B", 2, 2)])];
        let pages = vec!["alpha ".repeat(300), "beta ".repeat(300)];
        summarize_tree(&mut tree, &pages, &llm, &SummaryOptions::new("m")).await.unwrap();
        let summaries: Vec<_> = crate::tree::subtree(&tree).iter().map(|n| n.summary.clone().unwrap()).collect();
        assert_eq!(summaries, ["A; B", "ok", "ok"]);
    }

    #[tokio::test]
    async fn short_leaves_keep_their_text_and_fail_loud_when_the_model_never_answers() {
        let llm = ScriptedLlm::new(vec![String::new()]);
        let mut tree = vec![Node::new("A", 1, 1), Node::new("B", 2, 2)];
        let pages = vec!["short".to_string(), "beta ".repeat(300)];
        let err = summarize_tree(&mut tree, &pages, &llm, &SummaryOptions::new("m")).await.unwrap_err();
        assert!(err.message.contains("Summary generation failed"));
        assert_eq!(tree[0].summary.as_deref(), Some("short"));
    }

    #[test]
    fn a_leaf_reads_only_its_side_of_a_shared_page() {
        let pages = vec!["Intro text\n2 Methods\nmethod body".to_string(), "more".to_string()];
        let tree = vec![Node::new("1 Intro", 1, 1), Node::new("2 Methods", 1, 2)];
        let flat = crate::tree::subtree(&tree);
        assert_eq!(leaf_own_text(&flat, 0, &pages), "Intro text\n");
        assert_eq!(leaf_own_text(&flat, 1, &pages), "2 Methods\nmethod bodymore");
    }
}
