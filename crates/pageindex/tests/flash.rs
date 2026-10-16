mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::*;
use pageindex::flash::{FLAT_TREE_MAX_NODES, extract_toc, flash_rejection_reason, page_index_flash};
use pageindex::llm::{ChatMessage, CompletionParams};
use pageindex::{FlashOptions, LlmClient, Node, OptimizeMode, TocSource};

fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)
}

fn titles(nodes: &[Node]) -> Vec<String> {
    nodes.iter().map(|n| n.title.clone()).collect()
}

async fn flash(path: &Path) -> pageindex::FlashResult {
    page_index_flash(path, &FlashOptions::structure_only()).await.unwrap()
}

#[test]
fn short_document_headings_are_detected() {
    let dir = tempfile::tempdir().unwrap();
    let pdf = write(dir.path(), "memo.pdf", &memo_pdf());
    let result = extract_toc(&pdf, true).unwrap();
    assert_eq!(titles(&result.structure), ["Budget", "Team"]);
    assert_eq!(result.doc_title.as_deref(), Some("Mission"));
}

#[test]
fn landscape_deck_is_structured() {
    let dir = tempfile::tempdir().unwrap();
    let pdf = write(dir.path(), "deck.pdf", &deck_pdf());
    let result = extract_toc(&pdf, true).unwrap();
    assert_eq!(titles(&result.structure), DECK_TITLES[1..]);
    assert_eq!(result.toc_source, TocSource::Detected);
    assert_eq!(result.doc_title.as_deref(), Some(DECK_TITLES[0]));
}

#[tokio::test]
async fn toc_source_is_always_present_and_a_preface_covers_the_title_page() {
    let dir = tempfile::tempdir().unwrap();
    let pdf = write(dir.path(), "memo.pdf", &memo_pdf());
    let opts = FlashOptions { use_embedded_toc: false, ..FlashOptions::structure_only() };
    let result = page_index_flash(&pdf, &opts).await.unwrap();
    assert_eq!(result.toc_source, TocSource::Detected);
    assert_eq!(titles(&result.structure), ["Preface", "Budget", "Team"]);
    let json = serde_json::to_value(&result).unwrap();
    assert_eq!(json["toc_source"], "detected");
    assert!(json.get("page_texts").is_none());
}

#[tokio::test]
async fn no_hierarchy_falls_back_to_page_nodes() {
    let dir = tempfile::tempdir().unwrap();
    let pdf = layout_pdf(
        &[
            ("Mission", vec!["The launch is named Skylark.".into()]),
            ("Budget", vec!["The budget is 420 euros.".into()]),
        ],
        false,
    );
    let result = flash(&write(dir.path(), "two.pdf", &pdf)).await;
    assert_eq!(result.toc_source, TocSource::Pages);
    assert_eq!(
        serde_json::to_value(&result.structure).unwrap(),
        serde_json::json!([
            {"title": "Page 1", "node_id": "0000", "start_index": 1, "end_index": 1},
            {"title": "Page 2", "node_id": "0001", "start_index": 2, "end_index": 2}
        ])
    );
}

struct CountingLlm(AtomicUsize);

#[async_trait::async_trait]
impl LlmClient for CountingLlm {
    async fn complete(&self, _: &str, _: &[ChatMessage], _: &CompletionParams) -> pageindex::Result<String> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(r#"{"summary": "ok", "subsections": []}"#.into())
    }
}

#[tokio::test]
async fn flat_fallback_over_limit_skips_model_passes() {
    let dir = tempfile::tempdir().unwrap();
    let pages: Vec<PageSpec> = (0..=FLAT_TREE_MAX_NODES).map(|i| PageSpec::Line(format!("Body number {i}"))).collect();
    let pdf = write(dir.path(), "letter.pdf", &build_pdf(&pages, HELVETICA));
    let llm = Arc::new(CountingLlm(AtomicUsize::new(0)));
    let opts = FlashOptions { summary_model: Some("m".into()), llm: Some(llm.clone()), ..FlashOptions::default() };
    let result = page_index_flash(&pdf, &opts).await.unwrap();
    assert_eq!(result.toc_source, TocSource::Pages);
    assert_eq!(result.structure.len(), FLAT_TREE_MAX_NODES + 1);
    assert_eq!(llm.0.load(Ordering::SeqCst), 0);
    assert!(result.optimize.is_none());
    let reason = flash_rejection_reason(&result.structure, result.toc_source, "--mode standard").unwrap();
    assert!(reason.contains("no layout structure") && reason.contains("--mode standard"));
}

#[tokio::test]
async fn summaries_and_merge_run_through_the_model_client() {
    let dir = tempfile::tempdir().unwrap();
    let pdf = write(dir.path(), "deck.pdf", &deck_pdf());
    let llm = Arc::new(CountingLlm(AtomicUsize::new(0)));
    let opts = FlashOptions {
        summary_model: Some("m".into()),
        optimize: OptimizeMode::Merge,
        llm: Some(llm.clone()),
        ..FlashOptions::default()
    };
    let result = page_index_flash(&pdf, &opts).await.unwrap();
    assert!(result.optimize.is_some());
    assert!(pageindex::tree::subtree(&result.structure).iter().all(|n| n.summary.is_some()));
}

#[tokio::test]
async fn textless_pdf_is_unreadable() {
    let dir = tempfile::tempdir().unwrap();
    let pdf = write(
        dir.path(),
        "scan.pdf",
        &build_pdf(&[PageSpec::Line(String::new()), PageSpec::Line(String::new())], HELVETICA),
    );
    let result = flash(&pdf).await;
    assert!(result.structure.is_empty());
    assert_eq!(result.toc_source, TocSource::Unreadable);
    assert!(flash_rejection_reason(&result.structure, result.toc_source, "x").unwrap().contains("no text layer"));
}

#[tokio::test]
async fn page_fallback_covers_every_page() {
    let dir = tempfile::tempdir().unwrap();
    let pages = ["", "Alpha body", "", "Delta body"].map(|t| PageSpec::Line(t.into()));
    let result = flash(&write(dir.path(), "sparse.pdf", &build_pdf(&pages, HELVETICA))).await;
    assert_eq!(result.toc_source, TocSource::Pages);
    let shape: Vec<_> =
        result.structure.iter().map(|n| (n.title.as_str(), n.start_index, n.end_index, n.nodes.len())).collect();
    assert_eq!(shape, [("Page 1", 1, 1, 0), ("Page 2", 2, 2, 0), ("Page 3", 3, 3, 0), ("Page 4", 4, 4, 0)]);
}

#[test]
fn no_heading_result_carries_page_texts() {
    let dir = tempfile::tempdir().unwrap();
    let pages = ["Alpha body", "Beta body"].map(|t| PageSpec::Line(t.into()));
    let result = extract_toc(write(dir.path(), "doc.pdf", &build_pdf(&pages, HELVETICA)), true).unwrap();
    assert!(result.structure.is_empty());
    assert_eq!(result.page_texts, ["Alpha body", "Beta body"]);
}

#[tokio::test]
async fn every_page_is_in_a_node() {
    let dir = tempfile::tempdir().unwrap();
    let mut pdfs = vec![write(dir.path(), "memo.pdf", &memo_pdf()), write(dir.path(), "deck.pdf", &deck_pdf())];
    pdfs.extend(["ar_report.pdf", "hi_report.pdf", "ja_report.pdf", "zh_body_en_headings.pdf"].map(fixture));
    for pdf in pdfs {
        let pages = pageindex_pdf::PdfDocument::open(&pdf).unwrap().page_count();
        let structure = flash(&pdf).await.structure;
        let covered: std::collections::BTreeSet<usize> =
            structure.iter().flat_map(|n| n.start_index..=n.end_index).collect();
        assert_eq!(covered, (1..=pages).collect(), "{}", pdf.display());
        let (preface, first) = (&structure[0], &structure[1]);
        assert_eq!(
            (preface.title.as_str(), preface.node_id.as_deref(), preface.start_index),
            ("Preface", Some("0000"), 1)
        );
        assert!([first.start_index - 1, first.start_index].contains(&preface.end_index));
        assert_eq!(first.node_id.as_deref(), Some("0001"));
    }
}

#[test]
fn ink_past_the_advance_does_not_split_the_word() {
    let dir = tempfile::tempdir().unwrap();
    let mut widths = vec![556; 95];
    widths[(b'f' - 32) as usize] = 50;
    let font = helvetica_with_widths(&widths, "/WinAnsiEncoding");
    let result =
        extract_toc(write(dir.path(), "doc.pdf", &build_pdf(&[PageSpec::Line("inference".into())], &font)), true)
            .unwrap();
    assert_eq!(result.page_texts, ["inference"]);
}

#[test]
fn rotated_text_ignores_code_widths() {
    let dir = tempfile::tempdir().unwrap();
    let mut widths = vec![556; 95];
    widths[(b'\'' - 32) as usize] = 191;
    let font = helvetica_with_widths(&widths, "/WinAnsiEncoding");
    let raw = PageSpec::Raw(b"BT /F1 12 Tf 0 -1 1 0 300 700 Tm (O'Brien rock'n'roll) Tj ET".to_vec());
    let result = extract_toc(write(dir.path(), "doc.pdf", &build_pdf(&[raw], &font)), true).unwrap();
    assert_eq!(result.page_texts, ["O'Brien rock'n'roll"]);
}

#[test]
fn rtl_chars_read_in_logical_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut widths = vec![556; 95];
    widths[(b'a' - 32) as usize] = 550;
    widths[(b'b' - 32) as usize] = 150;
    let hebrew = "<< /Type /Encoding /BaseEncoding /WinAnsiEncoding /Differences [97 /afii57664 /afii57665] >>";
    let font = helvetica_with_widths(&widths, hebrew);
    let result =
        extract_toc(write(dir.path(), "doc.pdf", &build_pdf(&[PageSpec::Line("xyba z".into())], &font)), true).unwrap();
    assert_eq!(result.page_texts, ["xy \u{05d0}\u{05d1} z"]);
}

#[test]
fn japanese_and_cross_script_headings_are_kept() {
    let ja = extract_toc(fixture("ja_report.pdf"), false).unwrap();
    assert_eq!(titles(&ja.structure), ["財務ハイライト", "リスク要因", "今後の見通し"]);
    let zh = extract_toc(fixture("zh_body_en_headings.pdf"), false).unwrap();
    assert_eq!(titles(&zh.structure), ["Financial Review", "Risk Factors", "Business Outlook"]);
}

#[tokio::test]
async fn non_latin_documents_are_indexed() {
    for name in ["hi_report.pdf", "ar_report.pdf"] {
        let raw = extract_toc(fixture(name), false).unwrap();
        assert_eq!(raw.toc_source, TocSource::Detected);
        assert_eq!(raw.structure.len(), 3, "{name}");
        let result = flash(&fixture(name)).await;
        assert!(flash_rejection_reason(&result.structure, result.toc_source, "x").is_none());
    }
}

#[test]
fn deep_bookmarks_become_the_frame() {
    let dir = tempfile::tempdir().unwrap();
    let pages: Vec<PageSpec> = (1..=8).map(|p| PageSpec::Line(format!("Body of page {p}"))).collect();
    let outline = [
        ("Introduction", 1, 1),
        ("Background", 2, 2),
        ("Method", 1, 3),
        ("Data", 2, 4),
        ("Model", 2, 5),
        ("Results", 1, 6),
        ("Discussion", 1, 8),
    ];
    let pdf = write(dir.path(), "book.pdf", &build_pdf_with_outline(&pages, HELVETICA, &outline, (612.0, 792.0)));
    let bookmarks = pageindex_pdf::PdfDocument::open(&pdf).unwrap().bookmarks();
    assert_eq!(bookmarks.len(), 7);
    assert_eq!(bookmarks[1], pageindex_pdf::Bookmark { title: "Background".into(), level: 2, page: 2 });
    let result = extract_toc(&pdf, true).unwrap();
    assert_eq!(result.toc_source, TocSource::Bookmarks);
    let shape: Vec<_> = pageindex::tree::subtree(&result.structure)
        .iter()
        .map(|n| (n.title.clone(), n.start_index, n.end_index))
        .collect();
    assert_eq!(shape[0], ("Introduction".to_string(), 1, 3));
    assert_eq!(shape[2], ("Method".to_string(), 3, 6));
    assert_eq!(result.structure.len(), 4);
}

/// The full JSON for a generated report: title page, numbered sections with
/// subsections, footer page numbers. Regenerate with `UPDATE_GOLDEN=1`.
#[tokio::test]
async fn golden_report_structure() {
    let dir = tempfile::tempdir().unwrap();
    let pdf = write(dir.path(), "report.pdf", &report_pdf());
    let result = flash(&pdf).await;
    let actual = serde_json::to_string_pretty(&result).unwrap();
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/report_structure.json");
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::create_dir_all(golden.parent().unwrap()).unwrap();
        std::fs::write(&golden, &actual).unwrap();
    }
    let expected = std::fs::read_to_string(&golden).expect("golden file (run with UPDATE_GOLDEN=1 to create)");
    assert_eq!(actual.trim(), expected.trim());
}
