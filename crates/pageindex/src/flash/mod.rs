//! PageIndex Flash: the tree structure of a PDF from layout statistics,
//! without an LLM. Summaries and the optimize expand pass add model calls.

mod detect;
pub mod embedded_toc;
mod keywords;
pub mod layout;

use std::path::Path;
use std::sync::Arc;

use rayon::prelude::*;
use serde::Serialize;

use crate::error::{ErrorKind, PageIndexError, Result};
use crate::llm::LlmClient;
use crate::naming::sanitize_filename;
use crate::summary::{SUMMARY_MAX_WORDS, SUMMARY_REASONING_EFFORT, SummaryOptions, summarize_tree};
use crate::tree::{Node, write_node_ids};
use crate::tree_optimize::{
    OptimizeOptions, OptimizeReport, add_intro_nodes, heading_at_page_start, optimize, page_lines,
};
use crate::types::OptimizeMode;
use pageindex_pdf::PdfDocument;

/// Largest page-node fallback the managed pipelines accept as an index.
pub const FLAT_TREE_MAX_NODES: usize = 10;

/// Where a structure came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TocSource {
    /// From the layout.
    Detected,
    /// From the embedded outline.
    Bookmarks,
    /// Bookmarks framing the detected sections.
    Hybrid,
    /// No hierarchy: one node per page.
    Pages,
    /// No page carries text.
    Unreadable,
}

impl TocSource {
    fn from_label(label: &str) -> Self {
        match label {
            "bookmarks" => Self::Bookmarks,
            "hybrid" => Self::Hybrid,
            _ => Self::Detected,
        }
    }
}

/// The raw extraction, before page fallback, intros and model passes.
#[derive(Debug, Clone, Serialize)]
pub struct Extraction {
    pub doc_name: String,
    pub doc_title: Option<String>,
    pub structure: Vec<Node>,
    pub has_abstract_or_references_section: bool,
    pub page_texts: Vec<String>,
    pub toc_source: TocSource,
}

/// The Flash result, shaped like `*_structure.json`.
#[derive(Debug, Clone, Serialize)]
pub struct FlashResult {
    pub doc_name: String,
    pub doc_title: Option<String>,
    pub structure: Vec<Node>,
    pub has_abstract_or_references_section: bool,
    pub toc_source: TocSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimize: Option<OptimizeReport>,
    /// Per-page text the tree was read from (not serialized).
    #[serde(skip)]
    pub page_texts: Vec<String>,
}

#[derive(Clone)]
pub struct FlashOptions {
    /// LLM node summaries (needs a model client).
    pub summary: bool,
    pub summary_model: Option<String>,
    pub optimize: OptimizeMode,
    /// Model for expand; defaults to the summary model.
    pub optimize_model: Option<String>,
    pub summary_concurrency: Option<usize>,
    pub use_embedded_toc: bool,
    pub summary_max_words: Option<usize>,
    pub summary_reasoning_effort: Option<String>,
    /// The model client; [`crate::llm::LiteLlm`] when unset and needed.
    pub llm: Option<Arc<dyn LlmClient>>,
}

impl Default for FlashOptions {
    fn default() -> Self {
        Self {
            summary: true,
            summary_model: None,
            optimize: OptimizeMode::Full,
            optimize_model: None,
            summary_concurrency: None,
            use_embedded_toc: true,
            summary_max_words: None,
            summary_reasoning_effort: None,
            llm: None,
        }
    }
}

impl FlashOptions {
    /// The raw tree only: no summaries, no optimize, no model calls.
    pub fn structure_only() -> Self {
        Self { summary: false, optimize: OptimizeMode::Off, ..Self::default() }
    }
}

fn validate_path(path: &Path) -> Result<()> {
    if !path.exists() {
        return Err(PageIndexError::new(ErrorKind::NotFound, format!("PDF file not found: {}", path.display())));
    }
    if !path.is_file() {
        return Err(PageIndexError::invalid(format!("PDF path is not a file: {}", path.display())));
    }
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if !sanitize_filename(&name).to_lowercase().ends_with(".pdf") {
        return Err(PageIndexError::invalid(format!("PDF file must have a .pdf extension: {}", path.display())));
    }
    Ok(())
}

/// Parse and lay out a PDF, then detect its outline (and apply embedded
/// bookmarks when `use_embedded_toc`). No model calls.
pub fn extract_toc(path: impl AsRef<Path>, use_embedded_toc: bool) -> Result<Extraction> {
    let path = path.as_ref();
    validate_path(path)?;
    let doc = PdfDocument::open(path)?;
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "document.pdf".into());
    Ok(extract_from_document(&doc, &name, use_embedded_toc))
}

/// [`extract_toc`] over an in-memory PDF.
pub fn extract_toc_from_bytes(bytes: &[u8], use_embedded_toc: bool) -> Result<Extraction> {
    let doc = PdfDocument::from_bytes(bytes)?;
    Ok(extract_from_document(&doc, "document.pdf", use_embedded_toc))
}

fn extract_from_document(doc: &PdfDocument, doc_name: &str, use_embedded_toc: bool) -> Extraction {
    let mut pages: Vec<layout::PageLayout> = doc.pages().par_iter().map(layout::layout_page).collect();
    let page_count = pages.len();
    let stats = detect::doc_stats(&pages);
    detect::mark_headers_footers(&mut pages);
    detect::mark_toc_pages(&mut pages);
    let doc_title = detect::detect_title(&mut pages, &stats);
    let headings = detect::collect_headings(&pages, &stats);
    let structure = if detect::outline_is_usable(&headings, page_count) {
        let levels = detect::assign_levels(&headings);
        detect::build_tree(&headings, &levels, page_count)
    } else {
        Vec::new()
    };
    let has_abstract_or_references_section = detect::has_abstract_or_references(&structure);
    let page_texts: Vec<String> = pages.iter().map(layout::PageLayout::text).collect();
    let (structure, toc_source) = if use_embedded_toc {
        let (structure, label) =
            embedded_toc::apply_embedded_toc(structure, &doc.bookmarks(), page_count, Some(&page_texts));
        (structure, TocSource::from_label(label))
    } else {
        (structure, TocSource::Detected)
    };
    Extraction {
        doc_name: doc_name.to_owned(),
        doc_title,
        structure,
        has_abstract_or_references_section,
        page_texts,
        toc_source,
    }
}

/// Why a managed pipeline should refuse this result, or `None` to accept
/// it. `standard_hint` is how the caller spells the standard-mode switch.
pub fn flash_rejection_reason(structure: &[Node], toc_source: TocSource, standard_hint: &str) -> Option<String> {
    if toc_source == TocSource::Unreadable {
        return Some(
            "PageIndex Flash found no text layer in this PDF (scanned or image-only); run OCR before indexing it."
                .to_owned(),
        );
    }
    if toc_source == TocSource::Pages && structure.len() > FLAT_TREE_MAX_NODES {
        return Some(format!(
            "PageIndex Flash found no layout structure in this document ({} pages); try {standard_hint}, which builds the structure with the model.",
            structure.len()
        ));
    }
    if structure.is_empty() {
        return Some(format!(
            "PageIndex Flash could not extract a structure from this PDF; try {standard_hint}, which builds the structure with the model."
        ));
    }
    None
}

/// One node per page, so every page is reachable.
fn page_nodes(page_texts: &[String]) -> Vec<Node> {
    if !page_texts.iter().any(|t| !t.trim().is_empty()) {
        return Vec::new();
    }
    let mut nodes: Vec<Node> = (1..=page_texts.len()).map(|i| Node::new(format!("Page {i}"), i, i)).collect();
    write_node_ids(&mut nodes);
    nodes
}

/// The pages before a late-starting hierarchy become a Preface node; it
/// runs onto the first section's page unless that heading opens it.
fn add_preface(structure: &mut Vec<Node>, lines: &[Vec<String>]) {
    let first = &structure[0];
    let opens = first.start_index <= lines.len() && heading_at_page_start(lines, first.start_index, &first.title);
    let end = first.start_index - usize::from(opens);
    structure.insert(0, Node::new("Preface", 1, end));
}

/// Finish an extraction: page fallback, preface, intro nodes and ids.
pub fn finish_structure(extraction: &mut Extraction) {
    let lines = page_lines(&extraction.page_texts);
    if extraction.structure.is_empty() {
        extraction.structure = page_nodes(&extraction.page_texts);
        extraction.toc_source = if extraction.structure.is_empty() { TocSource::Unreadable } else { TocSource::Pages };
    } else if extraction.structure[0].start_index > 1 {
        add_preface(&mut extraction.structure, &lines);
    }
    if !extraction.structure.is_empty() {
        add_intro_nodes(&mut extraction.structure, Some(&lines));
        write_node_ids(&mut extraction.structure);
    }
}

fn check_options(opts: &FlashOptions) -> Result<()> {
    for (name, value) in
        [("summary_concurrency", opts.summary_concurrency), ("summary_max_words", opts.summary_max_words)]
    {
        if value == Some(0) {
            return Err(PageIndexError::invalid(format!("{name} must be a positive int, got 0")));
        }
    }
    if opts.summary_reasoning_effort.as_deref().is_some_and(|e| e.trim().is_empty()) {
        return Err(PageIndexError::invalid("summary_reasoning_effort must be a non-empty str"));
    }
    Ok(())
}

/// Build the PageIndex tree of a PDF. With [`FlashOptions::structure_only`]
/// it runs fully LLM-free.
pub async fn page_index_flash(path: impl AsRef<Path>, opts: &FlashOptions) -> Result<FlashResult> {
    check_options(opts)?;
    let path = path.as_ref().to_path_buf();
    let use_toc = opts.use_embedded_toc;
    let extraction = tokio::task::spawn_blocking(move || extract_toc(&path, use_toc))
        .await
        .map_err(|e| PageIndexError::new(ErrorKind::Document, format!("extraction task failed: {e}")))??;
    finish_flash(extraction, opts).await
}

/// The model passes over an extraction (shared by path and bytes inputs).
pub async fn finish_flash(mut extraction: Extraction, opts: &FlashOptions) -> Result<FlashResult> {
    check_options(opts)?;
    finish_structure(&mut extraction);
    let Extraction { doc_name, doc_title, mut structure, has_abstract_or_references_section, page_texts, toc_source } =
        extraction;
    let refused_flat = toc_source == TocSource::Pages && structure.len() > FLAT_TREE_MAX_NODES;
    let mut report = None;
    if !refused_flat && !structure.is_empty() {
        let config = crate::config::ConfigLoader::new().load(&Default::default());
        let summary_model = opts.summary_model.clone().unwrap_or(config.summary_model);
        let needs_model =
            opts.summary || (opts.optimize == OptimizeMode::Full && page_texts.iter().any(|t| !t.is_empty()));
        let llm: Option<Arc<dyn LlmClient>> = needs_model
            .then(|| opts.llm.clone().unwrap_or_else(|| Arc::new(crate::llm::LiteLlm::new()) as Arc<dyn LlmClient>));
        let do_expand = opts.optimize == OptimizeMode::Full && page_texts.iter().any(|t| !t.is_empty());
        let max_words = opts.summary_max_words.unwrap_or(SUMMARY_MAX_WORDS);
        if opts.optimize != OptimizeMode::Off {
            let mut optimize_opts = match (&llm, do_expand) {
                (Some(llm), true) => OptimizeOptions::full(
                    llm.clone(),
                    opts.optimize_model.clone().unwrap_or_else(|| summary_model.clone()),
                ),
                _ => OptimizeOptions::merge_only(),
            };
            optimize_opts.summary_words = if opts.summary { max_words } else { 0 };
            if let Some(c) = opts.summary_concurrency {
                optimize_opts.concurrency = c;
            }
            report = Some(optimize(&mut structure, &page_texts, page_texts.len(), &optimize_opts).await?);
        }
        if opts.summary {
            let llm = llm.expect("a summary run has a client");
            let effort = opts.summary_reasoning_effort.as_deref().map(str::trim).map(str::to_owned);
            let defaulted = effort.is_none();
            let summary_opts = SummaryOptions {
                concurrency: opts.summary_concurrency.unwrap_or(crate::summary::SUMMARY_CONCURRENCY),
                max_words,
                split_shared_pages: true,
                reasoning_effort: effort.or_else(|| Some(SUMMARY_REASONING_EFFORT.to_owned())),
                drop_refused_effort: defaulted,
                ..SummaryOptions::new(summary_model)
            };
            summarize_tree(&mut structure, &page_texts, llm.as_ref(), &summary_opts).await?;
        }
        crate::tree::for_each_mut(&mut structure, &mut |n| n.same_page = false);
    }
    Ok(FlashResult {
        doc_name,
        doc_title,
        structure,
        has_abstract_or_references_section,
        toc_source,
        optimize: report,
        page_texts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extraction(structure: Vec<Node>, pages: &[&str]) -> Extraction {
        Extraction {
            doc_name: "doc.pdf".into(),
            doc_title: None,
            structure,
            has_abstract_or_references_section: false,
            page_texts: pages.iter().map(|p| p.to_string()).collect(),
            toc_source: TocSource::Detected,
        }
    }

    fn shape(nodes: &[Node]) -> Vec<(String, String, usize, usize)> {
        crate::tree::subtree(nodes)
            .iter()
            .map(|n| (n.title.clone(), n.node_id.clone().unwrap_or_default(), n.start_index, n.end_index))
            .collect()
    }

    #[test]
    fn parents_get_their_intro_nodes() {
        let tree = vec![Node::new("R", 1, 6).with_children(vec![Node::new("A", 3, 4), Node::new("X", 5, 6)])];
        let mut ex = extraction(tree, &["Cover", "Foreword", "A\nalpha", "alpha", "X\nbody", "body"]);
        finish_structure(&mut ex);
        assert_eq!(
            shape(&ex.structure),
            [
                ("R".into(), "0000".into(), 1, 6),
                ("R (intro)".into(), "0001".into(), 1, 2),
                ("A".into(), "0002".into(), 3, 4),
                ("X".into(), "0003".into(), 5, 6)
            ]
        );
    }

    #[test]
    fn preface_runs_onto_a_page_its_first_heading_does_not_open() {
        let mut ends = Vec::new();
        for third in ["A\nalpha", "Contents, continued\nA\nalpha"] {
            let mut ex = extraction(vec![Node::new("A", 3, 4)], &["Cover", "Contents", third, "alpha"]);
            finish_structure(&mut ex);
            let preface = &ex.structure[0];
            ends.push((preface.title.clone(), preface.node_id.clone().unwrap(), preface.end_index));
        }
        assert_eq!(
            ends,
            [("Preface".to_string(), "0000".to_string(), 2), ("Preface".to_string(), "0000".to_string(), 3)]
        );
    }

    #[test]
    fn rejection_reasons() {
        let node = Node::new("T", 1, 1);
        assert!(flash_rejection_reason(std::slice::from_ref(&node), TocSource::Detected, "mode='standard'").is_none());
        assert!(flash_rejection_reason(&[node.clone(), node.clone()], TocSource::Pages, "x").is_none());
        let unreadable = flash_rejection_reason(&[], TocSource::Unreadable, "mode='standard'").unwrap();
        assert!(unreadable.contains("no text layer") && !unreadable.contains("standard"));
        let many = vec![node; FLAT_TREE_MAX_NODES + 1];
        let flat = flash_rejection_reason(&many, TocSource::Pages, "mode='standard'").unwrap();
        assert!(flat.contains("no layout structure") && flat.contains("mode='standard'"));
        assert!(
            flash_rejection_reason(&many, TocSource::Pages, "--mode standard").unwrap().contains("--mode standard")
        );
        assert!(flash_rejection_reason(&[], TocSource::Detected, "x").unwrap().contains("could not extract"));
    }

    #[tokio::test]
    async fn options_are_validated() {
        let opts = FlashOptions { summary_concurrency: Some(0), ..FlashOptions::structure_only() };
        assert!(page_index_flash("missing.pdf", &opts).await.unwrap_err().message.contains("positive"));
        let err = page_index_flash("missing.pdf", &FlashOptions::structure_only()).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::NotFound);
    }
}
