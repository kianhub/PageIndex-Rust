//! Python PageIndex vs this Rust port on identical PDFs.
//!
//! Run: `cargo bench -p pageindex --bench compare`
//! Token counts use cl100k on the same extracted text. Tree and e2e use the
//! same generated PDFs. The e2e pass uses a deterministic stub LLM on both
//! sides (`{"subsections": []}` / `{"summary": "..."}`).

#[path = "../tests/common/mod.rs"]
mod common;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use pageindex::OptimizeMode;
use pageindex::flash::{FlashOptions, extract_toc, page_index_flash};
use pageindex::llm::{ChatMessage, CompletionParams, LlmClient};
use pageindex::text::count_tokens;
use pageindex_pdf::PdfDocument;

struct BenchLlm;

#[async_trait]
impl LlmClient for BenchLlm {
    async fn complete(&self, _: &str, messages: &[ChatMessage], _: &CompletionParams) -> pageindex::Result<String> {
        let prompt = messages.last().map(|m| m.content.as_str()).unwrap_or("");
        if prompt.to_ascii_lowercase().contains("subsection") {
            Ok(r#"{"subsections": []}"#.into())
        } else {
            Ok(r#"{"summary": "Deterministic bench summary."}"#.into())
        }
    }
}

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples[samples.len() / 2]
}

fn time_ms(repeats: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        f();
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    median_ms(samples)
}

fn book_pdf() -> Vec<u8> {
    let chapters = ["1 Introduction", "2 Methods", "3 Results", "4 Discussion", "5 Related Work", "6 Conclusion"];
    let subs = ["Scope", "Data", "Notes"];
    let mut pages: Vec<(String, Vec<String>)> = vec![
        ("Annual Systems Report".into(), vec!["This volume records the year's findings in six chapters.".into()]),
        ("Contents".into(), chapters.iter().map(|c| (*c).to_string()).collect()),
    ];
    for (i, title) in chapters.iter().enumerate() {
        pages.push(((*title).into(), vec![format!("Opening paragraphs of {title} describe the chapter plan.")]));
        for (s, sub) in subs.iter().enumerate() {
            let heading = format!("{}.{} {sub}", i + 1, s + 1);
            pages.push((
                heading.clone(),
                (0..8).map(|k| format!("Body paragraph {k} of {heading} with extra words for token weight.")).collect(),
            ));
        }
    }
    let borrowed: Vec<(&str, Vec<String>)> = pages.iter().map(|(t, b)| (t.as_str(), b.clone())).collect();
    common::layout_pdf(&borrowed, false)
}

struct Doc {
    name: &'static str,
    path: PathBuf,
}

fn write_docs(dir: &Path) -> Vec<Doc> {
    let generated = [
        ("memo", common::memo_pdf()),
        ("report", common::report_pdf()),
        ("deck", common::deck_pdf()),
        ("book", book_pdf()),
    ];
    let mut docs = Vec::new();
    for (name, bytes) in generated {
        let path = dir.join(format!("{name}.pdf"));
        std::fs::write(&path, bytes).expect("write generated pdf");
        docs.push(Doc { name, path });
    }
    let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    for name in ["ja_report", "ar_report"] {
        let src = data.join(format!("{name}.pdf"));
        if src.is_file() {
            let path = dir.join(format!("{name}.pdf"));
            std::fs::copy(&src, &path).expect("copy fixture pdf");
            docs.push(Doc { name, path });
        }
    }
    docs
}

fn rust_parse(path: &Path) {
    let doc = PdfDocument::open(path).expect("open pdf");
    let _ = doc.pages();
}

fn rust_tree(path: &Path) {
    let mut extraction = extract_toc(path, true).expect("extract toc");
    pageindex::flash::finish_structure(&mut extraction);
}

fn rust_e2e(rt: &tokio::runtime::Runtime, path: &Path) {
    let llm = Arc::new(BenchLlm);
    let opts = FlashOptions {
        summary: true,
        optimize: OptimizeMode::Full,
        llm: Some(llm),
        summary_model: Some("gpt-bench".into()),
        ..FlashOptions::default()
    };
    rt.block_on(page_index_flash(path, &opts)).expect("e2e flash");
}

fn run_python(dir: &Path, docs: &[Doc], token_text: &str, repeats: usize) -> Option<serde_json::Value> {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/compare.py");
    if !script.is_file() {
        return None;
    }
    let docs_json: Vec<serde_json::Value> =
        docs.iter().map(|d| serde_json::json!({"name": d.name, "path": d.path})).collect();
    let docs_path = dir.join("docs.json");
    let tokens_path = dir.join("tokens.txt");
    let out_path = dir.join("python.json");
    std::fs::write(&docs_path, serde_json::to_string(&docs_json).unwrap()).ok()?;
    std::fs::write(&tokens_path, token_text).ok()?;
    let upstream = std::env::var("PAGEINDEX_PYTHON").unwrap_or_else(|_| "/tmp/pageindex-upstream".into());
    let status = Command::new("python3")
        .arg(&script)
        .arg("--docs")
        .arg(&docs_path)
        .arg("--tokens")
        .arg(&tokens_path)
        .arg("--out")
        .arg(&out_path)
        .arg("--repeats")
        .arg(repeats.to_string())
        .env("PYTHONPATH", &upstream)
        .env("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    serde_json::from_str(&std::fs::read_to_string(out_path).ok()?).ok()
}

fn cell(ms: f64) -> String {
    if ms >= 100.0 { format!("{ms:.1}") } else { format!("{ms:.2}") }
}

fn speedup(py: f64, rs: f64) -> String {
    if rs == 0.0 { "n/a".into() } else { format!("{:.1}×", py / rs) }
}

fn main() {
    let repeats: usize = std::env::var("BENCH_REPEATS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let dir = tempfile::tempdir().expect("temp dir");
    let docs = write_docs(dir.path());
    let rt = tokio::runtime::Runtime::new().expect("tokio");

    let token_corpus: String = docs
        .iter()
        .flat_map(|d| extract_toc(&d.path, true).ok())
        .flat_map(|e| e.page_texts)
        .collect::<Vec<_>>()
        .join("\n");
    let rust_tokens = count_tokens(&token_corpus);
    let rust_token_ms = time_ms(repeats * 4, || {
        let _ = count_tokens(&token_corpus);
    });

    println!("# PageIndex Python vs Rust (median of {repeats} timed runs, warmup discarded)");
    println!("Identical generated/fixture PDFs. Token counting is cl100k on the same extracted text.");
    println!("E2E is Flash + merge + expand + summaries with a deterministic stub LLM.\n");

    let mut rust_rows = Vec::new();
    for doc in &docs {
        let parse_ms = time_ms(repeats, || rust_parse(&doc.path));
        let tree_ms = time_ms(repeats, || rust_tree(&doc.path));
        let e2e_ms = time_ms(repeats.max(3), || rust_e2e(&rt, &doc.path));
        let mut extraction = extract_toc(&doc.path, true).expect("toc");
        pageindex::flash::finish_structure(&mut extraction);
        rust_rows.push((doc.name, parse_ms, tree_ms, e2e_ms, extraction.page_texts.len(), extraction.structure.len()));
        println!(
            "rust {:>12}  parse {:>8} ms  tree {:>8} ms  e2e {:>8} ms  pages {} nodes {}",
            doc.name,
            cell(parse_ms),
            cell(tree_ms),
            cell(e2e_ms),
            extraction.page_texts.len(),
            extraction.structure.len()
        );
    }
    println!("rust tokens {rust_tokens} in {} ms", cell(rust_token_ms));

    let python = run_python(dir.path(), &docs, &token_corpus, repeats);
    if python.is_none() {
        eprintln!("python bench skipped (set PAGEINDEX_PYTHON to the upstream checkout and install pypdfium2)");
    }

    println!("\n| Document | Pages | Parse py / rs (ms) | Tree py / rs (ms) | E2E py / rs (ms) | Speedup (tree) |");
    println!("|---|---:|---:|---:|---:|---:|");
    if let Some(py) = &python {
        let py_docs = py.get("docs").cloned().unwrap_or(serde_json::json!({}));
        for (name, parse_ms, tree_ms, e2e_ms, pages, _nodes) in &rust_rows {
            let row = py_docs.get(*name);
            let py_parse = row.and_then(|r| r.get("parse_ms")).and_then(serde_json::Value::as_f64).unwrap_or(f64::NAN);
            let py_tree = row.and_then(|r| r.get("tree_ms")).and_then(serde_json::Value::as_f64).unwrap_or(f64::NAN);
            let py_e2e = row.and_then(|r| r.get("e2e_ms")).and_then(serde_json::Value::as_f64).unwrap_or(f64::NAN);
            println!(
                "| {name} | {pages} | {} / {} | {} / {} | {} / {} | {} |",
                cell(py_parse),
                cell(*parse_ms),
                cell(py_tree),
                cell(*tree_ms),
                cell(py_e2e),
                cell(*e2e_ms),
                speedup(py_tree, *tree_ms)
            );
        }
        let py_tokens = py.get("token_count").and_then(serde_json::Value::as_u64).unwrap_or(0);
        let py_token_ms = py.get("token_ms").and_then(serde_json::Value::as_f64).unwrap_or(f64::NAN);
        println!(
            "\nToken counting (cl100k, {} chars of extracted text): Python {py_tokens} tokens in {} ms, Rust {rust_tokens} tokens in {} ms ({}).",
            token_corpus.len(),
            cell(py_token_ms),
            cell(rust_token_ms),
            speedup(py_token_ms, rust_token_ms)
        );
    } else {
        for (name, parse_ms, tree_ms, e2e_ms, pages, _nodes) in &rust_rows {
            println!(
                "| {name} | {pages} | n/a / {} | n/a / {} | n/a / {} | n/a |",
                cell(*parse_ms),
                cell(*tree_ms),
                cell(*e2e_ms)
            );
        }
        println!("\nToken counting: Rust {rust_tokens} tokens in {} ms.", cell(rust_token_ms));
    }
}
