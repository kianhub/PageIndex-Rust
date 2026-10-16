//! Chat, agent tools, classic mode, imaging, and the unified client.

#[path = "common/mod.rs"]
mod common;

use std::sync::Arc;

use pageindex::agent::{call_tool, parse_citations, tool_names};
use pageindex::chat::ChatRequest;
use pageindex::classic::{ClassicOptions, page_index};
use pageindex::client::{OcrFormat, PageIndexClient, PageIndexLocalClient, SubmitOptions};
use pageindex::imaging::highlight_region;
use pageindex::llm::testing::ScriptedLlm;
use pageindex::llm::{ChatMessage, LiteLlm};
use pageindex::{ChatProcessOptions, LocalIndexConfig, chat};

#[tokio::test]
async fn local_client_indexes_and_tools_read_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let pdf = dir.path().join("memo.pdf");
    std::fs::write(&pdf, common::memo_pdf()).unwrap();
    let llm = Arc::new(ScriptedLlm::new(vec![
        r#"{"summary": "A short memo about the mission."}"#.into(),
        "A memo on mission, budget and team.".into(),
    ]));
    let client = PageIndexLocalClient::with_config(LocalIndexConfig {
        storage_path: Some(dir.path().join("store")),
        optimize: Some(pageindex::OptimizeMode::Off),
        ..Default::default()
    })
    .with_llm(llm);
    let submitted =
        client.submit_document(&pdf, SubmitOptions { mode: Some("flash".into()), metadata: None }).await.unwrap();
    assert!(submitted.doc_id.starts_with("pi-"));
    assert_eq!(submitted.name, "memo.pdf");

    let listed = client.list_documents(10, 0, None).unwrap();
    assert_eq!(listed.total, 1);
    let tree = client.get_tree(&submitted.doc_id, true, false).unwrap();
    assert_eq!(tree.status, "completed");
    assert!(tree.retrieval_ready);
    assert!(!tree.result.is_empty());

    let (browse, err) = call_tool(&client, "browse_documents", serde_json::json!({}), None).await;
    assert!(!err, "{browse}");
    let browse: serde_json::Value = serde_json::from_str(&browse).unwrap();
    assert_eq!(browse["success"], true);
    assert_eq!(browse["documents"][0]["name"], "memo.pdf");

    let (structure, err) =
        call_tool(&client, "get_document_structure", serde_json::json!({"doc_name": "memo.pdf"}), None).await;
    assert!(!err, "{structure}");
    let structure: serde_json::Value = serde_json::from_str(&structure).unwrap();
    assert!(structure["structure"].as_array().is_some_and(|a| !a.is_empty()));

    let (pages, err) =
        call_tool(&client, "get_page_content", serde_json::json!({"doc_name": "memo.pdf", "pages": "1"}), None).await;
    assert!(!err, "{pages}");
    let pages: serde_json::Value = serde_json::from_str(&pages).unwrap();
    assert!(pages["total_pages"].as_u64().unwrap() >= 1);
    assert!(!pages["content"][0]["text"].as_str().unwrap().is_empty());

    let ocr = client.get_ocr(&submitted.doc_id, OcrFormat::Page).unwrap();
    match ocr.result {
        pageindex::client::OcrResult::Pages(pages) => assert!(!pages.is_empty()),
        _ => panic!("expected page OCR"),
    }
}

#[tokio::test]
async fn chat_tool_loop_answers_from_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let pdf = dir.path().join("memo.pdf");
    std::fs::write(&pdf, common::memo_pdf()).unwrap();
    let index_llm = Arc::new(ScriptedLlm::new(vec![r#"{"summary": "memo"}"#.into(), "A memo.".into()]));
    let client = PageIndexLocalClient::with_config(LocalIndexConfig {
        storage_path: Some(dir.path().join("store")),
        optimize: Some(pageindex::OptimizeMode::Off),
        ..Default::default()
    })
    .with_llm(index_llm);
    client.submit_document(&pdf, SubmitOptions::default()).await.unwrap();

    let chat_llm = ScriptedLlm::new(vec![r#"TOOL browse_documents {}"#.into(), "The library holds memo.pdf.".into()]);
    let mut req = ChatRequest::new("gpt-test", vec![ChatMessage::user("What documents do I have?")]);
    req.show_process = ChatProcessOptions { thinking: false, tool_call: false, tool_result: false, max_chars: 200 };
    let value = chat::run_chat_completions(&client, &chat_llm, &req, false).await.unwrap();
    assert!(value["choices"][0]["message"]["content"].as_str().unwrap().contains("memo.pdf"));
}

#[tokio::test]
async fn classic_mode_builds_a_tree_from_scripted_llm() {
    let dir = tempfile::tempdir().unwrap();
    let pdf = dir.path().join("memo.pdf");
    std::fs::write(&pdf, common::memo_pdf()).unwrap();
    // Detector: no TOC; generate_toc_init returns two sections; later prompts unused.
    let llm = ScriptedLlm::new(vec![
        r#"{"toc_detected": "no"}"#.into(),
        r#"{"toc_detected": "no"}"#.into(),
        r#"{"toc_detected": "no"}"#.into(),
        r#"[{"structure":"1","title":"Preface","physical_index":"<physical_index_1>"}]"#.into(),
        r#"{"answer": "yes"}"#.into(),
        r#"{"start_begin": "yes"}"#.into(),
    ]);
    let result = page_index(
        &pdf,
        ClassicOptions {
            if_add_node_id: Some(true),
            if_add_node_summary: Some(false),
            if_add_node_text: Some(false),
            if_add_doc_description: Some(false),
            ..Default::default()
        },
        &llm,
    )
    .await
    .unwrap();
    assert_eq!(result.doc_name, "memo.pdf");
    assert!(!result.structure.is_empty());
    assert!(result.structure.iter().any(|n| n.title.contains("Preface") || n.start_index >= 1));
}

#[test]
fn imaging_highlights_and_citations_parse() {
    let mut img = image::RgbImage::new(40, 40);
    for p in img.pixels_mut() {
        *p = image::Rgb([20, 20, 20]);
    }
    let mut png = Vec::new();
    image::DynamicImage::ImageRgb8(img).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
    let out = highlight_region(&png, &[100.0, 100.0, 400.0, 500.0], 1000.0).unwrap();
    assert!(!out.is_empty());
    let cites = parse_citations(r#"x <cite doc="A.pdf" page="2"/>"#);
    assert_eq!(cites[0]["page"], 2);
}

#[test]
fn tool_schemas_and_unified_client_construct() {
    assert!(tool_names(false).contains(&"browse_documents"));
    assert!(tool_names(true).contains(&"remove_document"));
    let _ = PageIndexClient::new();
    let _ = LiteLlm::new();
    let openai = pageindex::as_openai_tools(false, false, None, "https://api.pageindex.ai");
    assert_eq!(openai[0]["type"], "function");
    let anthropic = pageindex::as_anthropic_tools(false);
    assert_eq!(anthropic[0]["name"], "browse_documents");
    let claude = pageindex::as_claude_mcp(false, Some("k"), "https://api.pageindex.ai");
    assert_eq!(claude["type"], "http");
}

#[test]
fn mcp_sse_and_json_extract() {
    let messages = pageindex::mcp::parse_sse("data: {\"id\": 3}\n\n");
    assert_eq!(messages[0]["id"], 3);
    let value = pageindex::jsonx::extract_json("```json\n{\"a\": 1}\n```");
    assert_eq!(value["a"], 1);
}
