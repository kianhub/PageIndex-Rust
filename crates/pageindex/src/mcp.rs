//! Minimal MCP client (streamable HTTP) for the PageIndex cloud MCP server.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::{Value, json};

use crate::error::{PageIndexError, Result};

const PROTOCOL_VERSION: &str = "2025-06-18";
const SDK_NAME: &str = "pageindex-rust-sdk";
const SDK_VERSION: &str = env!("CARGO_PKG_VERSION");

/// JSON-RPC messages out of a `text/event-stream` body.
pub fn parse_sse(text: &str) -> Vec<Value> {
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut messages = Vec::new();
    for block in text.split("\n\n") {
        let data: Vec<&str> = block
            .lines()
            .filter_map(|line| line.strip_prefix("data:").map(|rest| rest.strip_prefix(' ').unwrap_or(rest)))
            .collect();
        if data.is_empty() {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<Value>(&data.join("\n")) {
            messages.push(value);
        }
    }
    messages
}

/// The text-only rendering of MCP content: text verbatim, base64 payloads
/// as a size stub.
pub fn render_text(blocks: &[Value]) -> String {
    let mut texts = Vec::new();
    for block in blocks {
        if block.get("type").and_then(Value::as_str) == Some("text") {
            texts.push(block.get("text").and_then(Value::as_str).unwrap_or("").to_owned());
        } else if block.get("data").and_then(Value::as_str).is_some() {
            let kind = block.get("mimeType").or_else(|| block.get("type")).and_then(Value::as_str).unwrap_or("binary");
            let size = block.get("data").and_then(Value::as_str).map_or(0, str::len);
            let size_kb = (size * 3 / 4096).max(1);
            texts.push(format!("[{kind} content omitted: ~{size_kb} KB]"));
        } else if let Some(blob) = block.pointer("/resource/blob").and_then(Value::as_str) {
            let kind = block.pointer("/resource/mimeType").and_then(Value::as_str).unwrap_or("binary");
            let size_kb = (blob.len() * 3 / 4096).max(1);
            texts.push(format!("[{kind} content omitted: ~{size_kb} KB]"));
        } else {
            texts.push(block.to_string());
        }
    }
    texts.join("\n")
}

/// The text-only rendering of prompt messages, roles dropped.
pub fn render_prompt_text(messages: &[Value]) -> String {
    let blocks: Vec<Value> = messages.iter().filter_map(|m| m.get("content").cloned()).collect();
    render_text(&blocks)
}

/// Streamable-HTTP MCP client against the PageIndex cloud MCP server.
pub struct McpBridge {
    url: String,
    auth_headers: Vec<(String, String)>,
    http: reqwest::Client,
    session_id: Mutex<Option<String>>,
    protocol_version: Mutex<Option<String>>,
    instructions: Mutex<Option<String>>,
    capabilities: Mutex<Value>,
    initialized: Mutex<bool>,
    next_id: AtomicU64,
}

impl McpBridge {
    pub fn new(url: impl Into<String>, headers: Vec<(String, String)>) -> Self {
        Self {
            url: url.into(),
            auth_headers: headers,
            http: reqwest::Client::builder().timeout(Duration::from_secs(240)).build().unwrap_or_default(),
            session_id: Mutex::new(None),
            protocol_version: Mutex::new(None),
            instructions: Mutex::new(None),
            capabilities: Mutex::new(json!({})),
            initialized: Mutex::new(false),
            next_id: AtomicU64::new(0),
        }
    }

    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst) + 1
    }

    async fn post(
        &self,
        payload: &Value,
        session_id: Option<&str>,
        protocol_version: Option<&str>,
    ) -> Result<reqwest::Response> {
        let mut req = self
            .http
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .json(payload);
        for (k, v) in &self.auth_headers {
            req = req.header(k, v);
        }
        if let Some(id) = session_id {
            req = req.header("Mcp-Session-Id", id);
        }
        if let Some(ver) = protocol_version {
            req = req.header("MCP-Protocol-Version", ver);
        }
        req.send().await.map_err(|e| PageIndexError::api(format!("Could not reach the PageIndex MCP server: {e}")))
    }

    fn extract_result(content_type: &str, status: u16, body: &[u8], request_id: u64) -> Result<Value> {
        let messages = if content_type.contains("text/event-stream") {
            parse_sse(&String::from_utf8_lossy(body))
        } else {
            vec![serde_json::from_slice(body).map_err(|_| {
                PageIndexError::api(format!("MCP server returned a non-JSON response (HTTP {status})."))
                    .with_status(Some(status))
            })?]
        };
        let reply = messages
            .into_iter()
            .find(|m| m.get("id").and_then(Value::as_u64) == Some(request_id))
            .ok_or_else(|| PageIndexError::api("MCP server response contained no reply matching the request."))?;
        if let Some(error) = reply.get("error") {
            return Err(PageIndexError::api(format!(
                "MCP error {}: {}",
                error.get("code").map(|c| c.to_string()).unwrap_or_else(|| "unknown".into()),
                error.get("message").and_then(Value::as_str).unwrap_or("")
            )));
        }
        Ok(reply.get("result").cloned().unwrap_or(Value::Null))
    }

    async fn request(&self, method: &str, params: Option<Value>, retry: bool) -> Result<Value> {
        self.ensure_initialized().await?;
        let request_id = self.next_id();
        let session_id = self.session_id.lock().expect("mcp lock").clone();
        let protocol_version = self.protocol_version.lock().expect("mcp lock").clone();
        let mut payload = json!({"jsonrpc": "2.0", "id": request_id, "method": method});
        if let Some(params) = params.clone() {
            payload["params"] = params;
        }
        let response = self.post(&payload, session_id.as_deref(), protocol_version.as_deref()).await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND && session_id.is_some() && retry {
            {
                let mut sid = self.session_id.lock().expect("mcp lock");
                if sid.as_deref() == session_id.as_deref() {
                    *self.initialized.lock().expect("mcp lock") = false;
                    *sid = None;
                    *self.protocol_version.lock().expect("mcp lock") = None;
                }
            }
            return Box::pin(self.request(method, params, false)).await;
        }
        let status = response.status().as_u16();
        let content_type =
            response.headers().get("Content-Type").and_then(|v| v.to_str().ok()).unwrap_or("").to_owned();
        if status >= 400 {
            let text = response.text().await.unwrap_or_default();
            return Err(PageIndexError::api(format!(
                "MCP request failed: HTTP {status} ({})",
                text.chars().take(200).collect::<String>()
            ))
            .with_status(Some(status)));
        }
        let body = response.bytes().await.map_err(|e| PageIndexError::api(e.to_string()))?;
        Self::extract_result(&content_type, status, &body, request_id)
    }

    async fn ensure_initialized(&self) -> Result<()> {
        if *self.initialized.lock().expect("mcp lock") {
            return Ok(());
        }
        let request_id = self.next_id();
        let payload = json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": SDK_NAME, "version": SDK_VERSION},
            }
        });
        let response = self.post(&payload, None, None).await?;
        let status = response.status().as_u16();
        if status >= 400 {
            let text = response.text().await.unwrap_or_default();
            let hint = if matches!(status, 401 | 403) { " Check your API key." } else { "" };
            return Err(PageIndexError::api(format!(
                "Could not connect to the PageIndex MCP server: HTTP {status} ({}).{hint}",
                text.chars().take(200).collect::<String>()
            ))
            .with_status(Some(status)));
        }
        let session = response.headers().get("Mcp-Session-Id").and_then(|v| v.to_str().ok()).map(str::to_owned);
        let content_type =
            response.headers().get("Content-Type").and_then(|v| v.to_str().ok()).unwrap_or("").to_owned();
        let body = response.bytes().await.map_err(|e| PageIndexError::api(e.to_string()))?;
        let result = Self::extract_result(&content_type, status, &body, request_id)?;
        *self.session_id.lock().expect("mcp lock") = session.clone();
        *self.protocol_version.lock().expect("mcp lock") =
            Some(result.get("protocolVersion").and_then(Value::as_str).unwrap_or(PROTOCOL_VERSION).to_owned());
        *self.instructions.lock().expect("mcp lock") =
            result.get("instructions").and_then(Value::as_str).map(str::to_owned);
        *self.capabilities.lock().expect("mcp lock") = result.get("capabilities").cloned().unwrap_or(json!({}));
        *self.initialized.lock().expect("mcp lock") = true;
        let notify = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
        let session = self.session_id.lock().expect("mcp lock").clone();
        let protocol = self.protocol_version.lock().expect("mcp lock").clone();
        let _ = self.post(&notify, session.as_deref(), protocol.as_deref()).await;
        Ok(())
    }

    pub async fn instructions(&self) -> Result<Option<String>> {
        self.ensure_initialized().await?;
        Ok(self.instructions.lock().expect("mcp lock").clone())
    }

    async fn list_paginated(&self, method: &str, key: &str) -> Result<Vec<Value>> {
        let mut items = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..50 {
            let params = cursor.as_ref().map(|c| json!({"cursor": c})).unwrap_or(json!({}));
            let result = self.request(method, Some(params), true).await?;
            if let Some(batch) = result.get(key).and_then(Value::as_array) {
                items.extend(batch.iter().cloned());
            }
            let next = result.get("nextCursor").and_then(Value::as_str).map(str::to_owned);
            if next.is_none() || next == cursor {
                return Ok(items);
            }
            cursor = next;
        }
        Err(PageIndexError::api(format!("MCP {method} pagination did not terminate within 50 pages.")))
    }

    pub async fn list_tools(&self) -> Result<Vec<Value>> {
        self.list_paginated("tools/list", "tools").await
    }

    fn require_prompts(&self) -> Result<()> {
        let caps = self.capabilities.lock().expect("mcp lock");
        if caps.get("prompts").is_none() {
            return Err(PageIndexError::api(format!("The MCP server at {} does not serve prompts.", self.url)));
        }
        Ok(())
    }

    pub async fn list_prompts(&self) -> Result<Vec<Value>> {
        self.ensure_initialized().await?;
        self.require_prompts()?;
        self.list_paginated("prompts/list", "prompts").await
    }

    pub async fn get_prompt(&self, name: &str, arguments: Option<&Value>) -> Result<(Option<String>, Vec<Value>)> {
        self.ensure_initialized().await?;
        self.require_prompts()?;
        let mut params = json!({"name": name});
        if let Some(args) = arguments {
            let mapped: serde_json::Map<String, Value> = args
                .as_object()
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .map(|(k, v)| (k, Value::String(v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string()))))
                .collect();
            params["arguments"] = Value::Object(mapped);
        }
        let result = self.request("prompts/get", Some(params), true).await?;
        let description = result.get("description").and_then(Value::as_str).map(str::to_owned);
        let messages = result.get("messages").and_then(Value::as_array).cloned().unwrap_or_default();
        Ok((description, messages))
    }

    pub async fn call_tool(&self, name: &str, arguments: &Value) -> Result<(Vec<Value>, bool)> {
        let result = self.request("tools/call", Some(json!({"name": name, "arguments": arguments})), true).await?;
        let content = result.get("content").and_then(Value::as_array).cloned().unwrap_or_default();
        let is_error = result.get("isError").and_then(Value::as_bool).unwrap_or(false);
        Ok((content, is_error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sse_and_renders_stubs() {
        let messages = parse_sse("data: {\"id\": 1, \"result\": {\"ok\": true}}\n\n");
        assert_eq!(messages[0]["id"], 1);
        let text = render_text(&[json!({"type": "text", "text": "hi"}), json!({"type": "image", "data": "abcd"})]);
        assert!(text.contains("hi"));
        assert!(text.contains("omitted"));
    }
}
