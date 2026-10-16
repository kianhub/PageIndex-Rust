//! Streamable-HTTP MCP client (`McpBridge`) and local MCP server (`McpServer`).
//!
//! Official runners attach to [`McpServer`] as an MCP server:
//! `openai-agents` `MCPServerStreamableHttp`, and Anthropic's MCP connector.

use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::agent::{AGENT_INSTRUCTIONS, AgentBackend, as_mcp_tools, call_tool, fetch_citation_prompt};
use crate::error::{PageIndexError, Result};

const PROTOCOL_VERSION: &str = "2025-06-18";
const SUPPORTED_PROTOCOLS: &[&str] = &["2025-03-26", "2025-06-18"];
const SDK_NAME: &str = "pageindex-rust-sdk";
const SERVER_NAME: &str = "pageindex-rust";
const SDK_VERSION: &str = env!("CARGO_PKG_VERSION");

/// JSON-RPC ids match as JSON values or as their decimal text (`1` vs `"1"`).
pub fn jsonrpc_ids_equal(left: &Value, right: &Value) -> bool {
    if left == right {
        return true;
    }
    jsonrpc_id_text(left).zip(jsonrpc_id_text(right)).is_some_and(|(a, b)| a == b)
}

fn jsonrpc_id_text(value: &Value) -> Option<String> {
    match value {
        Value::Number(n) => Some(n.to_string()),
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

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

/// One JSON-RPC object as a Streamable HTTP SSE event.
pub fn encode_sse(value: &Value) -> String {
    format!("event: message\ndata: {value}\n\n")
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
    let mut blocks = Vec::new();
    for message in messages {
        match message.get("content") {
            Some(Value::Array(items)) => blocks.extend(items.iter().cloned()),
            Some(Value::String(text)) => blocks.push(json!({"type": "text", "text": text})),
            Some(other) => blocks.push(other.clone()),
            None => {}
        }
    }
    render_text(&blocks)
}

fn rpc_result(id: Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

fn rpc_error(id: Value, code: i64, message: impl Into<String>) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message.into()}})
}

fn cited_answer_prompt() -> Value {
    json!({
        "name": "cited_answer",
        "description": "Citation instructions for grounded answers from PageIndex documents.",
        "arguments": [{
            "name": "format",
            "description": "cite or markdown",
            "required": false
        }]
    })
}

fn negotiate_protocol(requested: Option<&str>) -> &'static str {
    match requested {
        Some(v) if SUPPORTED_PROTOCOLS.contains(&v) => {
            if v == "2025-03-26" {
                "2025-03-26"
            } else {
                PROTOCOL_VERSION
            }
        }
        _ => PROTOCOL_VERSION,
    }
}

async fn initialize_result(backend: &dyn AgentBackend, include_management: bool, params: Option<&Value>) -> Value {
    let requested = params.and_then(|p| p.get("protocolVersion")).and_then(Value::as_str);
    let protocol = negotiate_protocol(requested);
    let instructions = crate::agent::base_instructions(backend, include_management)
        .await
        .unwrap_or_else(|_| AGENT_INSTRUCTIONS.to_owned());
    json!({
        "protocolVersion": protocol,
        "capabilities": {
            "tools": {"listChanged": false},
            "prompts": {"listChanged": false}
        },
        "serverInfo": {"name": SERVER_NAME, "version": SDK_VERSION},
        "instructions": instructions
    })
}

async fn call_mcp_tool(backend: &dyn AgentBackend, include_management: bool, params: Option<&Value>) -> Value {
    let name = params.and_then(|p| p.get("name")).and_then(Value::as_str).unwrap_or("");
    if name == "remove_document" && !include_management {
        return json!({
            "content": [{"type": "text", "text": "remove_document is not enabled on this server"}],
            "isError": true
        });
    }
    let args = params.and_then(|p| p.get("arguments")).cloned().unwrap_or(json!({}));
    let (text, is_error) = call_tool(backend, name, args, None).await;
    json!({"content": [{"type": "text", "text": text}], "isError": is_error})
}

async fn get_mcp_prompt(backend: &dyn AgentBackend, params: Option<&Value>) -> std::result::Result<Value, String> {
    let name = params.and_then(|p| p.get("name")).and_then(Value::as_str).unwrap_or("");
    if name != "cited_answer" {
        return Err(format!("No prompt named {name:?}"));
    }
    let format =
        params.and_then(|p| p.get("arguments")).and_then(|a| a.get("format")).and_then(Value::as_str).unwrap_or("cite");
    let text = fetch_citation_prompt(backend, format).await.map_err(|e| e.to_string())?;
    Ok(json!({
        "description": "Citation instructions for grounded answers from PageIndex documents.",
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]
    }))
}

/// One JSON-RPC MCP message. Notifications return `None` (HTTP 202, no body).
pub async fn handle_mcp_message(
    backend: &dyn AgentBackend,
    include_management: bool,
    message: &Value,
) -> Option<Value> {
    let method = match message.get("method").and_then(Value::as_str) {
        Some(method) => method,
        None => return message.get("id").cloned().map(|id| rpc_error(id, -32600, "Invalid Request")),
    };
    let id = message.get("id").cloned()?;
    let params = message.get("params");
    match method {
        "initialize" => Some(rpc_result(id, initialize_result(backend, include_management, params).await)),
        "ping" => Some(rpc_result(id, json!({}))),
        "tools/list" => Some(rpc_result(id, json!({"tools": as_mcp_tools(include_management)}))),
        "tools/call" => Some(rpc_result(id, call_mcp_tool(backend, include_management, params).await)),
        "prompts/list" => Some(rpc_result(id, json!({"prompts": [cited_answer_prompt()]}))),
        "prompts/get" => match get_mcp_prompt(backend, params).await {
            Ok(result) => Some(rpc_result(id, result)),
            Err(message) => Some(rpc_error(id, -32602, message)),
        },
        other => Some(rpc_error(id, -32601, format!("Method not found: {other}"))),
    }
}

/// Streamable-HTTP MCP client against a PageIndex (or other) MCP server.
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

    fn next_id(&self) -> Value {
        json!(self.next_id.fetch_add(1, Ordering::SeqCst) + 1)
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

    fn extract_result(content_type: &str, status: u16, body: &[u8], request_id: &Value) -> Result<Value> {
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
            .find(|m| m.get("id").is_some_and(|id| jsonrpc_ids_equal(id, request_id)))
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
        Self::extract_result(&content_type, status, &body, &request_id)
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
        let response = self.post(&payload, None, Some(PROTOCOL_VERSION)).await?;
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
        let result = Self::extract_result(&content_type, status, &body, &request_id)?;
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
        if let Ok(response) = self.post(&notify, session.as_deref(), protocol.as_deref()).await {
            let status = response.status();
            if !(status.is_success() || status == reqwest::StatusCode::ACCEPTED) {
                return Err(PageIndexError::api(format!(
                    "MCP initialized notification failed: HTTP {}",
                    status.as_u16()
                ))
                .with_status(Some(status.as_u16())));
            }
        }
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

struct ServerState {
    backend: Arc<dyn AgentBackend>,
    include_management: bool,
    sessions: Mutex<HashMap<String, String>>,
}

/// Local Streamable HTTP MCP server official SDK runners attach to.
pub struct McpServer {
    backend: Arc<dyn AgentBackend>,
    include_management: bool,
}

/// Running [`McpServer`]. Drop (or [`shutdown`](Self::shutdown)) stops accept.
pub struct McpServerHandle {
    pub addr: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    join: Option<JoinHandle<()>>,
}

impl McpServer {
    pub fn new(backend: Arc<dyn AgentBackend>, include_management: bool) -> Self {
        Self { backend, include_management }
    }

    /// Bind `127.0.0.1:0` (localhost, ephemeral port).
    pub async fn start(self) -> Result<McpServerHandle> {
        self.bind("127.0.0.1:0".parse().expect("localhost bind")).await
    }

    pub async fn bind(self, addr: SocketAddr) -> Result<McpServerHandle> {
        if !addr.ip().is_loopback() {
            return Err(PageIndexError::invalid(
                "McpServer binds to localhost only — official runners attach over loopback Streamable HTTP.",
            ));
        }
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| PageIndexError::api(format!("Could not bind the PageIndex MCP server: {e}")))?;
        let addr = listener.local_addr().map_err(|e| PageIndexError::api(e.to_string()))?;
        let state = Arc::new(ServerState {
            backend: self.backend,
            include_management: self.include_management,
            sessions: Mutex::new(HashMap::new()),
        });
        let (tx, rx) = oneshot::channel();
        let join = tokio::spawn(accept_loop(listener, state, rx));
        Ok(McpServerHandle { addr, shutdown: Some(tx), join: Some(join) })
    }
}

impl McpServerHandle {
    /// Endpoint official runners pass as `url` / `MCPServerStreamableHttp` params.
    pub fn url(&self) -> String {
        format!("http://{}/mcp", self.addr)
    }

    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            let _ = join.await;
        }
    }
}

impl Drop for McpServerHandle {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            join.abort();
        }
    }
}

async fn accept_loop(listener: TcpListener, state: Arc<ServerState>, mut shutdown: oneshot::Receiver<()>) {
    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            accepted = listener.accept() => {
                let Ok((stream, _)) = accepted else { continue };
                let state = state.clone();
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let service = service_fn(move |req| {
                        let state = state.clone();
                        async move { Ok::<_, Infallible>(dispatch_http(state, req).await) }
                    });
                    let _ = http1::Builder::new().serve_connection(io, service).await;
                });
            }
        }
    }
}

fn header_str<'a>(req: &'a Request<Incoming>, name: &str) -> Option<&'a str> {
    req.headers().get(name).and_then(|v| v.to_str().ok())
}

fn origin_allowed(origin: Option<&str>) -> bool {
    let Some(origin) = origin else {
        return true;
    };
    if origin.eq_ignore_ascii_case("null") {
        return true;
    }
    let Ok(uri) = origin.parse::<http::Uri>() else {
        return false;
    };
    matches!(uri.host(), Some("127.0.0.1" | "localhost" | "[::1]" | "::1"))
}

fn protocol_supported(version: Option<&str>) -> bool {
    match version {
        None => true,
        Some(v) => SUPPORTED_PROTOCOLS.contains(&v),
    }
}

fn wants_sse(accept: &str) -> bool {
    let accept = accept.to_ascii_lowercase();
    accept.contains("text/event-stream") && !accept.contains("application/json")
}

fn empty_response(status: StatusCode, session: Option<&str>, protocol: Option<&str>) -> Response<Full<Bytes>> {
    let mut builder = Response::builder().status(status);
    if let Some(session) = session {
        builder = builder.header("mcp-session-id", session);
    }
    if let Some(protocol) = protocol {
        builder = builder.header("mcp-protocol-version", protocol);
    }
    builder.body(Full::new(Bytes::new())).expect("empty response")
}

fn message_response(
    accept: &str,
    status: StatusCode,
    body: Value,
    session: Option<&str>,
    protocol: Option<&str>,
) -> Response<Full<Bytes>> {
    let (content_type, bytes) = if wants_sse(accept) {
        ("text/event-stream", Bytes::from(encode_sse(&body)))
    } else {
        ("application/json", Bytes::from(body.to_string()))
    };
    let mut builder = Response::builder().status(status).header("content-type", content_type);
    if let Some(session) = session {
        builder = builder.header("mcp-session-id", session);
    }
    if let Some(protocol) = protocol {
        builder = builder.header("mcp-protocol-version", protocol);
    }
    builder.body(Full::new(bytes)).expect("mcp response")
}

async fn dispatch_http(state: Arc<ServerState>, req: Request<Incoming>) -> Response<Full<Bytes>> {
    if !origin_allowed(header_str(&req, "origin")) {
        return empty_response(StatusCode::FORBIDDEN, None, None);
    }
    match *req.method() {
        Method::OPTIONS => Response::builder()
            .status(StatusCode::NO_CONTENT)
            .header("allow", "GET, POST, DELETE, OPTIONS")
            .body(Full::new(Bytes::new()))
            .expect("options"),
        Method::GET => empty_response(StatusCode::METHOD_NOT_ALLOWED, None, None),
        Method::DELETE => {
            if let Some(session) = header_str(&req, "mcp-session-id") {
                state.sessions.lock().expect("mcp lock").remove(session);
            }
            empty_response(StatusCode::OK, None, None)
        }
        Method::POST => dispatch_post(state, req).await,
        _ => empty_response(StatusCode::METHOD_NOT_ALLOWED, None, None),
    }
}

async fn dispatch_post(state: Arc<ServerState>, req: Request<Incoming>) -> Response<Full<Bytes>> {
    let accept = header_str(&req, "accept").unwrap_or("application/json").to_owned();
    let session_header = header_str(&req, "mcp-session-id").map(str::to_owned);
    let protocol_header = header_str(&req, "mcp-protocol-version").map(str::to_owned);
    if !protocol_supported(protocol_header.as_deref()) {
        return empty_response(StatusCode::BAD_REQUEST, session_header.as_deref(), None);
    }
    let body = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => return empty_response(StatusCode::BAD_REQUEST, None, None),
    };
    let message: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(_) => {
            return message_response(
                &accept,
                StatusCode::OK,
                rpc_error(Value::Null, -32700, "Parse error"),
                None,
                None,
            );
        }
    };
    let method = message.get("method").and_then(Value::as_str);
    let has_id = message.get("id").is_some();
    if method.is_none() {
        return empty_response(StatusCode::ACCEPTED, session_header.as_deref(), protocol_header.as_deref());
    }
    let method = method.unwrap();
    if !has_id {
        return empty_response(StatusCode::ACCEPTED, session_header.as_deref(), protocol_header.as_deref());
    }
    if method == "initialize" {
        let reply = handle_mcp_message(state.backend.as_ref(), state.include_management, &message).await;
        let Some(reply) = reply else {
            return empty_response(StatusCode::ACCEPTED, None, None);
        };
        let protocol =
            reply.pointer("/result/protocolVersion").and_then(Value::as_str).unwrap_or(PROTOCOL_VERSION).to_owned();
        let session = uuid::Uuid::new_v4().simple().to_string();
        state.sessions.lock().expect("mcp lock").insert(session.clone(), protocol.clone());
        return message_response(&accept, StatusCode::OK, reply, Some(&session), Some(&protocol));
    }
    let Some(session) = session_header else {
        return empty_response(StatusCode::BAD_REQUEST, None, None);
    };
    let stored = state.sessions.lock().expect("mcp lock").get(&session).cloned();
    let Some(stored_protocol) = stored else {
        return empty_response(StatusCode::NOT_FOUND, Some(&session), None);
    };
    let protocol = protocol_header.unwrap_or(stored_protocol);
    let reply = handle_mcp_message(state.backend.as_ref(), state.include_management, &message).await;
    match reply {
        Some(reply) => message_response(&accept, StatusCode::OK, reply, Some(&session), Some(&protocol)),
        None => empty_response(StatusCode::ACCEPTED, Some(&session), Some(&protocol)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sse_and_renders_stubs() {
        let messages = parse_sse("event: message\ndata: {\"id\": 1, \"result\": {\"ok\": true}}\n\n");
        assert_eq!(messages[0]["id"], 1);
        let text = render_text(&[json!({"type": "text", "text": "hi"}), json!({"type": "image", "data": "abcd"})]);
        assert!(text.contains("hi"));
        assert!(text.contains("omitted"));
        let prompt = render_prompt_text(&[json!({"role": "user", "content": [{"type": "text", "text": "cite"}]})]);
        assert_eq!(prompt, "cite");
    }

    #[test]
    fn jsonrpc_ids_match_number_and_string() {
        assert!(jsonrpc_ids_equal(&json!(1), &json!("1")));
        assert!(jsonrpc_ids_equal(&json!("init-1"), &json!("init-1")));
        assert!(!jsonrpc_ids_equal(&json!(1), &json!("2")));
        assert_eq!(encode_sse(&json!({"id": 1})), "event: message\ndata: {\"id\":1}\n\n");
    }
}
