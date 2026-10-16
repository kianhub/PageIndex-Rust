//! The tree node every indexer emits, and the structural helpers shared by
//! Flash, standard, Markdown, optimize and summaries.

use serde::{Deserialize, Serialize};

pub const INTRO_SUFFIX: &str = " (intro)";

/// One section of a document tree. PDF nodes carry 1-based inclusive page
/// ranges (`end_index` covers the whole subtree); Markdown nodes carry
/// `line_num` instead.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub start_index: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub end_index: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_num: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_summary: Option<String>,
    /// Titles of subsections a merge folded into this node.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub key_items: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<Node>,
    /// A title synthesized from same-page siblings, which a summary call may rewrite.
    #[serde(skip)]
    pub same_page: bool,
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

impl Node {
    pub fn new(title: impl Into<String>, start_index: usize, end_index: usize) -> Self {
        Self { title: title.into(), start_index, end_index, ..Self::default() }
    }

    pub fn with_children(mut self, nodes: Vec<Node>) -> Self {
        self.nodes = nodes;
        self
    }

    pub fn is_leaf(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Last page covered by this node or any descendant.
    pub fn subtree_end(&self) -> usize {
        self.nodes.iter().map(Node::subtree_end).fold(self.end_index, usize::max)
    }

    /// Whether `child` is the intro node holding this node's opening pages.
    pub fn is_intro(&self, child: &Node) -> bool {
        child.start_index == self.start_index && child.title == intro_title(&self.title)
    }

    /// First and last page of the node's own text. A parent's runs onto the
    /// page its first child starts on; `None` when its intro holds them.
    pub fn own_pages(&self) -> (usize, Option<usize>) {
        match self.nodes.first() {
            Some(first) if self.is_intro(first) => (self.start_index, None),
            Some(first) => (self.start_index, Some(self.end_index.min(first.start_index))),
            None => (self.start_index, Some(self.end_index)),
        }
    }
}

/// The title of a parent's intro node; a split intro keeps its own.
pub fn intro_title(title: &str) -> String {
    let title = title.trim();
    if title.is_empty() {
        "Intro".to_owned()
    } else if title.ends_with(INTRO_SUFFIX) {
        title.to_owned()
    } else {
        format!("{title}{INTRO_SUFFIX}")
    }
}

/// Pre-order (document-order) references to every node.
pub fn subtree(nodes: &[Node]) -> Vec<&Node> {
    let mut out = Vec::new();
    fn walk<'a>(nodes: &'a [Node], out: &mut Vec<&'a Node>) {
        for node in nodes {
            out.push(node);
            walk(&node.nodes, out);
        }
    }
    walk(nodes, &mut out);
    out
}

/// Visit every node mutably in pre-order.
pub fn for_each_mut(nodes: &mut [Node], f: &mut impl FnMut(&mut Node)) {
    for node in nodes {
        f(node);
        for_each_mut(&mut node.nodes, f);
    }
}

pub fn count_nodes(nodes: &[Node]) -> usize {
    nodes.iter().map(|n| 1 + count_nodes(&n.nodes)).sum()
}

/// Number every node `0000`, `0001`, ... in pre-order.
pub fn write_node_ids(nodes: &mut [Node]) {
    let mut counter = 0usize;
    for_each_mut(nodes, &mut |node| {
        node.node_id = Some(format!("{counter:04}"));
        counter += 1;
    });
}

/// Leaf nodes, in document order.
pub fn leaf_nodes(nodes: &[Node]) -> Vec<&Node> {
    subtree(nodes).into_iter().filter(|n| n.is_leaf()).collect()
}

/// `[top-level ancestor, ..., node]` for `node_id`; empty if absent.
pub fn node_path<'a>(nodes: &'a [Node], node_id: &str) -> Vec<&'a Node> {
    for node in nodes {
        if node.node_id.as_deref() == Some(node_id) {
            return vec![node];
        }
        let mut path = node_path(&node.nodes, node_id);
        if !path.is_empty() {
            path.insert(0, node);
            return path;
        }
    }
    Vec::new()
}

pub fn find_node<'a>(nodes: &'a [Node], node_id: &str) -> Option<&'a Node> {
    node_path(nodes, node_id).pop()
}

pub fn node_parent<'a>(nodes: &'a [Node], node_id: &str) -> Option<&'a Node> {
    let path = node_path(nodes, node_id);
    (path.len() > 1).then(|| path[path.len() - 2])
}

/// Widen every parent's range to cover its whole subtree.
pub fn cover_subtree_ranges(nodes: &mut [Node]) {
    for node in nodes {
        if !node.nodes.is_empty() {
            cover_subtree_ranges(&mut node.nodes);
            node.start_index = node.nodes.iter().map(|c| c.start_index).fold(node.start_index, usize::min);
            node.end_index = node.nodes.iter().map(|c| c.end_index).fold(node.end_index, usize::max);
        }
    }
}

/// Concatenated text of pages `start..=end` (1-based).
pub fn pages_text(pages: &[String], start: usize, end: usize) -> String {
    if start == 0 || end < start {
        return String::new();
    }
    pages[(start - 1).min(pages.len())..end.min(pages.len())].concat()
}

/// Fill each node's `text` with its own pages (see [`Node::own_pages`]).
pub fn add_node_text(nodes: &mut [Node], pages: &[String]) {
    for_each_mut(nodes, &mut |node| {
        let (start, end) = node.own_pages();
        node.text = Some(end.map(|end| pages_text(pages, start, end)).unwrap_or_default());
    });
}

pub fn remove_text(nodes: &mut [Node]) {
    for_each_mut(nodes, &mut |node| node.text = None);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> Vec<Node> {
        vec![Node::new("Report", 1, 3).with_children(vec![
            Node::new("Report (intro)", 1, 1),
            Node::new("Alpha", 2, 3).with_children(vec![Node::new("Alpha one", 2, 2), Node::new("Alpha two", 3, 3)]),
        ])]
    }

    #[test]
    fn node_ids_are_preorder() {
        let mut nodes = tree();
        write_node_ids(&mut nodes);
        let ids: Vec<_> = subtree(&nodes).iter().map(|n| n.node_id.clone().unwrap()).collect();
        assert_eq!(ids, ["0000", "0001", "0002", "0003", "0004"]);
        assert_eq!(node_parent(&nodes, "0003").unwrap().title, "Alpha");
        assert_eq!(find_node(&nodes, "0004").unwrap().title, "Alpha two");
        assert!(find_node(&nodes, "9999").is_none());
    }

    #[test]
    fn a_parent_shares_its_first_childs_page_unless_an_intro_holds_it() {
        let mut nodes = tree();
        let pages: Vec<String> = ["Opening words", "Alpha body", "Beta body"].map(String::from).to_vec();
        add_node_text(&mut nodes, &pages);
        let texts: Vec<_> = subtree(&nodes).iter().map(|n| n.text.clone().unwrap()).collect();
        assert_eq!(texts, ["", "Opening words", "Alpha body", "Alpha body", "Beta body"]);
    }

    #[test]
    fn intro_titles() {
        assert_eq!(intro_title("Ch 3"), "Ch 3 (intro)");
        assert_eq!(intro_title("Ch 3 (intro)"), "Ch 3 (intro)");
        assert_eq!(intro_title("  "), "Intro");
    }

    #[test]
    fn serializes_without_empty_fields() {
        let mut node = Node::new("A", 1, 2);
        node.node_id = Some("0000".into());
        assert_eq!(
            serde_json::to_string(&node).unwrap(),
            r#"{"title":"A","node_id":"0000","start_index":1,"end_index":2}"#
        );
    }
}
