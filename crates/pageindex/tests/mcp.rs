//! Streamable HTTP MCP server: official-runner handshake + McpBridge client.

use std::sync::Arc;

use async_trait::async_trait;
use pageindex::agent::AgentBackend;
use pageindex::error::{PageIndexError, Result};
use pageindex::mcp::{McpBridge, McpServer, handle_mcp_message};
use pageindex::{as_mcp_tools, tool_names};
use serde_json::{Value, json};

struct StubBackend {
    docs: Vec<Value>,
}

impl StubBackend {
    fn one_doc() -> Self {
        Self {
            docs: vec![json!({
                "id": "pi-stub",
                "name": "memo.pdf",
                "description": "A short memo",
                "status": "completed",
                "createdAt": "2026-10-16T00:00:00.000",
                "pageNum": 1,
                "folderId": null,
                "metadata": null
            })],
        }
    }
}

#[async_trait]
impl AgentBackend for StubBackend {
    fn api_key(&self) -> Option<&str> {
        None
    }

    async fn list_documents(&self, limit: usize, offset: usize) -> Result<Value> {
        let total = self.docs.len();
        let documents: Vec<Value> = self.docs.iter().skip(offset).take(limit).cloned().collect();
        Ok(json!({"documents": documents, "total": total, "limit": limit, "offset": offset}))
    }

    async fn get_document(&self, doc_id: &str) -> Result<Value> {
        self.docs
            .iter()
            .find(|d| d.get("id").and_then(Value::as_str) == Some(doc_id))
            .cloned()
            .ok_or_else(|| PageIndexError::api("Document not found.").with_status(Some(404)))
    }

    async fn get_tree(&self, doc_id: &str, _node_summary: bool, _include_text: bool) -> Result<Value> {
        Ok(json!({
            "doc_id": doc_id,
            "status": "completed",
            "retrieval_ready": true,
            "result": [{"title": "Mission", "node_id": "0001", "start_index": 1, "end_index": 1, "summary": "A memo"}]
        }))
    }

    async fn get_ocr_pages(&self, doc_id: &str) -> Result<Value> {
        Ok(json!({
            "doc_id": doc_id,
            "status": "completed",
            "retrieval_ready": true,
            "result": [{"page_index": 1, "markdown": "The launch is named Skylark."}]
        }))
    }

    async fn delete_document(&self, _doc_id: &str) -> Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn handle_mcp_message_initialize_tools_and_ping() {
    let backend = StubBackend::one_doc();
    let init = handle_mcp_message(
        &backend,
        false,
        &json!({
            "jsonrpc": "2.0",
            "id": "init-1",
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test"}}
        }),
    )
    .await
    .unwrap();
    assert_eq!(init["id"], "init-1");
    assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(init["result"]["serverInfo"]["name"], "pageindex-rust");
    assert!(init["result"]["capabilities"].get("tools").is_some());
    assert!(init["result"]["capabilities"].get("prompts").is_some());
    assert!(init["result"]["instructions"].as_str().unwrap().contains("PageIndex"));

    assert!(
        handle_mcp_message(&backend, false, &json!({"jsonrpc": "2.0", "method": "notifications/initialized"}))
            .await
            .is_none()
    );

    let ping =
        handle_mcp_message(&backend, false, &json!({"jsonrpc": "2.0", "id": 2, "method": "ping"})).await.unwrap();
    assert_eq!(ping["result"], json!({}));

    let listed =
        handle_mcp_message(&backend, false, &json!({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})).await.unwrap();
    let tools = listed["result"]["tools"].as_array().unwrap();
    let names: Vec<&str> = tools.iter().filter_map(|t| t.get("name").and_then(Value::as_str)).collect();
    assert_eq!(names, tool_names(false));
    assert!(tools[0].get("inputSchema").is_some());
    assert!(tools.iter().all(|t| t.get("name") != Some(&json!("remove_document"))));

    let managed =
        handle_mcp_message(&backend, true, &json!({"jsonrpc": "2.0", "id": 4, "method": "tools/list"})).await.unwrap();
    assert!(
        managed["result"]["tools"].as_array().unwrap().iter().any(|t| t.get("name") == Some(&json!("remove_document")))
    );

    let unknown =
        handle_mcp_message(&backend, false, &json!({"jsonrpc": "2.0", "id": 5, "method": "nope/list"})).await.unwrap();
    assert_eq!(unknown["error"]["code"], -32601);
}

#[tokio::test]
async fn handle_mcp_message_calls_tools_and_prompts() {
    let backend = StubBackend::one_doc();
    let browse = handle_mcp_message(
        &backend,
        false,
        &json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "browse_documents", "arguments": {}}
        }),
    )
    .await
    .unwrap();
    assert_eq!(browse["result"]["isError"], false);
    let text = browse["result"]["content"][0]["text"].as_str().unwrap();
    let envelope: Value = serde_json::from_str(text).unwrap();
    assert_eq!(envelope["documents"][0]["name"], "memo.pdf");

    let pages = handle_mcp_message(
        &backend,
        false,
        &json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "get_page_content", "arguments": {"doc_name": "memo.pdf", "pages": "1"}}
        }),
    )
    .await
    .unwrap();
    let text = pages["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("Skylark"));

    let gated = handle_mcp_message(
        &backend,
        false,
        &json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "remove_document", "arguments": {"doc_names": ["memo.pdf"]}}
        }),
    )
    .await
    .unwrap();
    assert_eq!(gated["result"]["isError"], true);

    let prompt = handle_mcp_message(
        &backend,
        false,
        &json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "prompts/get",
            "params": {"name": "cited_answer", "arguments": {"format": "cite"}}
        }),
    )
    .await
    .unwrap();
    let messages = prompt["result"]["messages"].as_array().unwrap();
    assert!(!messages.is_empty());
}

#[tokio::test]
async fn mcp_server_speaks_official_streamable_http() {
    let server = McpServer::new(Arc::new(StubBackend::one_doc()), false).start().await.unwrap();
    let url = server.url();
    let http = reqwest::Client::new();

    let init = http
        .post(&url)
        .header("Accept", "application/json, text/event-stream")
        .header("Content-Type", "application/json")
        .json(&json!({
            "jsonrpc": "2.0",
            "id": "init-1",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "openai-agents", "version": "test"}
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(init.status(), 200);
    let session = init.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()).unwrap().to_owned();
    assert!(!session.is_empty());
    let init_body: Value = init.json().await.unwrap();
    assert_eq!(init_body["id"], "init-1");
    assert_eq!(init_body["result"]["protocolVersion"], "2025-06-18");

    let notified = http
        .post(&url)
        .header("Accept", "application/json, text/event-stream")
        .header("Content-Type", "application/json")
        .header("Mcp-Session-Id", &session)
        .header("MCP-Protocol-Version", "2025-06-18")
        .json(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        .send()
        .await
        .unwrap();
    assert_eq!(notified.status(), 202);

    let tools = http
        .post(&url)
        .header("Accept", "application/json, text/event-stream")
        .header("Content-Type", "application/json")
        .header("Mcp-Session-Id", &session)
        .header("MCP-Protocol-Version", "2025-06-18")
        .json(&json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
        .send()
        .await
        .unwrap();
    let tools_body: Value = tools.json().await.unwrap();
    assert_eq!(tools_body["id"], 2);
    assert!(tools_body["result"]["tools"].as_array().unwrap().iter().any(|t| t["name"] == "browse_documents"));

    let missing = http
        .post(&url)
        .header("Accept", "application/json")
        .header("Content-Type", "application/json")
        .json(&json!({"jsonrpc": "2.0", "id": 3, "method": "ping"}))
        .send()
        .await
        .unwrap();
    assert_eq!(missing.status(), 400);

    let unknown = http
        .post(&url)
        .header("Accept", "application/json")
        .header("Content-Type", "application/json")
        .header("Mcp-Session-Id", "missing-session")
        .json(&json!({"jsonrpc": "2.0", "id": 4, "method": "ping"}))
        .send()
        .await
        .unwrap();
    assert_eq!(unknown.status(), 404);

    let get = http.get(&url).header("Accept", "text/event-stream").send().await.unwrap();
    assert_eq!(get.status(), 405);

    let deleted = http.delete(&url).header("Mcp-Session-Id", &session).send().await.unwrap();
    assert_eq!(deleted.status(), 200);
    server.shutdown().await;
}

#[tokio::test]
async fn mcp_bridge_talks_to_local_server() {
    let server = McpServer::new(Arc::new(StubBackend::one_doc()), false).start().await.unwrap();
    let bridge = McpBridge::new(server.url(), Vec::new());
    let instructions = bridge.instructions().await.unwrap();
    assert!(instructions.as_deref().unwrap_or("").contains("PageIndex"));

    let tools = bridge.list_tools().await.unwrap();
    assert_eq!(tools.len(), as_mcp_tools(false).as_array().unwrap().len());
    assert_eq!(tools[0]["name"], "browse_documents");
    assert!(tools[0].get("inputSchema").is_some());

    let (blocks, is_error) = bridge.call_tool("browse_documents", &json!({})).await.unwrap();
    assert!(!is_error);
    let text = pageindex::mcp::render_text(&blocks);
    assert!(text.contains("memo.pdf"));

    let (desc, messages) = bridge.get_prompt("cited_answer", Some(&json!({"format": "cite"}))).await.unwrap();
    assert!(desc.is_some());
    assert!(!pageindex::mcp::render_prompt_text(&messages).is_empty());
    server.shutdown().await;
}

#[tokio::test]
async fn mcp_server_rejects_non_localhost_bind() {
    let err =
        McpServer::new(Arc::new(StubBackend::one_doc()), false).bind("0.0.0.0:0".parse().unwrap()).await.unwrap_err();
    assert!(err.message.contains("localhost"));
}
