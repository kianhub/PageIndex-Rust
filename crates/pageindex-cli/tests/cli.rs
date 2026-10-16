#[path = "../../pageindex/tests/common/mod.rs"]
mod common;

use std::path::Path;
use std::process::{Command, Output};

fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_pageindex"))
        .args(args)
        .current_dir(dir)
        .env("OPENAI_API_KEY", "")
        .env("CHATGPT_API_KEY", "")
        .env("PAGEINDEX_CONFIG_DIR", dir.join("cfg"))
        .output()
        .unwrap()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn md_cli_runs_without_llm_or_key() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("notes.md"), "# Title\n\nIntro.\n\n## Section\n\nBody.\n").unwrap();
    let out = run(dir.path(), &["--md_path", "notes.md"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("results/notes_structure.json")).unwrap())
            .unwrap();
    assert_eq!(json["doc_name"], "notes");
    assert_eq!(json["structure"][0]["nodes"][0]["title"], "Section");
}

#[test]
fn flash_cli_writes_the_structure_without_a_key() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("memo.pdf"), common::memo_pdf()).unwrap();
    let out = run(dir.path(), &["--pdf_path", "memo.pdf", "--no-summary", "--optimize", "merge"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(String::from_utf8_lossy(&out.stdout).contains("Optimize: merges="));
    let text = std::fs::read_to_string(dir.path().join("results/memo_structure.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["doc_title"], "Mission");
    let titles: Vec<_> = json["structure"].as_array().unwrap().iter().map(|n| n["title"].as_str().unwrap()).collect();
    assert_eq!(titles, ["Preface", "Budget", "Team"]);
    assert!(text.starts_with("{\n  \"doc_name\""), "pretty-printed with two-space indent");
}

#[test]
fn flash_cli_summaries_need_a_key() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("report.pdf"), common::report_pdf()).unwrap();
    let out = run(dir.path(), &["--pdf_path", "report.pdf", "--optimize", "off"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("OPENAI_API_KEY"), "{}", stderr(&out));
}

#[test]
fn flash_cli_rejects_unreadable_and_oversized_flat_trees() {
    let dir = tempfile::tempdir().unwrap();
    let blank = common::build_pdf(&[common::PageSpec::Line(String::new())], common::HELVETICA);
    std::fs::write(dir.path().join("scan.pdf"), blank).unwrap();
    let out = run(dir.path(), &["--pdf_path", "scan.pdf", "--no-summary", "--optimize", "off"]);
    assert!(!out.status.success() && stderr(&out).contains("no text layer"));
    assert!(!dir.path().join("results").exists());

    let pages: Vec<_> = (0..12).map(|i| common::PageSpec::Line(format!("Body number {i}"))).collect();
    std::fs::write(dir.path().join("flat.pdf"), common::build_pdf(&pages, common::HELVETICA)).unwrap();
    let out = run(dir.path(), &["--pdf_path", "flat.pdf", "--no-summary"]);
    let err = stderr(&out);
    assert!(!out.status.success() && err.contains("no layout structure") && err.contains("--mode standard"), "{err}");
}

#[test]
fn flag_combinations_are_validated() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("memo.pdf"), common::memo_pdf()).unwrap();
    std::fs::write(dir.path().join("notes.md"), "# A\n").unwrap();
    let cases: [(&[&str], &str); 6] = [
        (&[], "Either --pdf_path or --md_path must be specified"),
        (&["--pdf_path", "memo.pdf", "--md_path", "notes.md"], "Only one of --pdf_path or --md_path"),
        (
            &["--pdf_path", "memo.pdf", "--mode", "standard", "--summary-concurrency", "8"],
            "--summary-concurrency requires Flash mode",
        ),
        (&["--md_path", "notes.md", "--optimize"], "--optimize requires Flash mode"),
        (&["--pdf_path", "memo.pdf", "--if-add-node-id", "yes"], "--if-add-node-id is not supported in flash mode"),
        (&["--pdf_path", "missing.pdf"], "PDF file not found"),
    ];
    for (args, message) in cases {
        let out = run(dir.path(), args);
        assert!(!out.status.success());
        assert!(stderr(&out).contains(message), "{args:?}: {}", stderr(&out));
    }
}

#[test]
fn chatgpt_status_and_help() {
    let dir = tempfile::tempdir().unwrap();
    let out = run(dir.path(), &["chatgpt", "status"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Not signed in with ChatGPT"), "{stdout}");
    let out = run(dir.path(), &["chatgpt", "logout"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let out = run(dir.path(), &["--help"]);
    let help = String::from_utf8_lossy(&out.stdout);
    assert!(help.contains("chatgpt"), "{help}");
}

#[test]
fn chatgpt_flag_needs_login_not_an_api_key() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("report.pdf"), common::report_pdf()).unwrap();
    let out = run(dir.path(), &["--pdf_path", "report.pdf", "--chatgpt", "--optimize", "off"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("chatgpt login") || err.contains("Not signed in"), "{err}");
    assert!(!err.contains("OPENAI_API_KEY"), "{err}");
}
