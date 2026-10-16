//! Streaming Responses API calls for ChatGPT plan usage.

use futures::StreamExt;
use serde_json::{Value, json};

use crate::error::{ErrorKind, PageIndexError, Result};
use crate::llm::{ChatMessage, Completion, CompletionParams, ToolSpec};

/// Fields the preview route accepts on `POST /v1/responses`.
/// `store` and `stream` are required; `instructions` holds system text.
/// Temperature, max tokens, tools, and the rest of the omitted list are not sent.
pub async fn complete_responses(
    http: &reqwest::Client,
    api_base: &str,
    access_token: &str,
    model: &str,
    messages: &[ChatMessage],
    _params: &CompletionParams,
    tools: &[ToolSpec],
) -> Result<Completion> {
    if !tools.is_empty() {
        return Err(PageIndexError::new(
            ErrorKind::Llm,
            "Sign in with ChatGPT preview does not accept function tools on POST /v1/responses without namespaced additional_tools. Use an API key for tool-calling chat.",
        )
        .with_status(Some(400)));
    }
    let (instructions, input) = split_instructions(messages);
    let mut body = json!({
        "model": model,
        "input": input,
        "store": false,
        "stream": true,
    });
    if let Some(instructions) = instructions {
        body["instructions"] = json!(instructions);
    }
    let url = format!("{}/responses", api_base.trim_end_matches('/'));
    let response = http
        .post(url)
        .bearer_auth(access_token)
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| PageIndexError::new(ErrorKind::Llm, format!("ChatGPT Responses request failed: {e}")))?;
    let status = response.status();
    let request_id = response
        .headers()
        .get("openai-request-id")
        .or_else(|| response.headers().get("x-request-id"))
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-")
        .to_owned();
    if !status.is_success() {
        let payload: Value = response.json().await.unwrap_or(Value::Null);
        return Err(responses_error(status.as_u16(), &payload, &request_id));
    }
    let mut stream = response.bytes_stream();
    let mut parser = SseParser::default();
    let mut text = String::new();
    let mut completed = false;
    let mut failed: Option<PageIndexError> = None;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| PageIndexError::new(ErrorKind::Llm, format!("Responses stream error: {e}")))?;
        for event in parser.push(&String::from_utf8_lossy(&chunk)) {
            match event.get("type").and_then(Value::as_str) {
                Some("response.output_text.delta") => {
                    if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                        text.push_str(delta);
                    }
                }
                Some("response.output_text.done") => {
                    if text.is_empty()
                        && let Some(full) = event.get("text").and_then(Value::as_str)
                    {
                        text.push_str(full);
                    }
                }
                Some("response.failed") => {
                    failed = Some(failed_event(&event, &request_id));
                }
                Some("response.incomplete") => {
                    failed = Some(PageIndexError::new(
                        ErrorKind::Llm,
                        format!("Responses stream incomplete (request {request_id})"),
                    ));
                }
                Some("response.completed") => {
                    completed = true;
                    if text.is_empty() {
                        text = completed_text(&event);
                    }
                }
                _ => {}
            }
        }
    }
    if let Some(err) = failed {
        return Err(err);
    }
    if !completed {
        return Err(PageIndexError::new(ErrorKind::Llm, "Stream ended without response.completed."));
    }
    Ok(Completion { text, tool_calls: Vec::new(), finished: true })
}

fn split_instructions(messages: &[ChatMessage]) -> (Option<String>, Vec<Value>) {
    let instructions =
        messages.iter().filter(|m| m.role == "system").map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n\n");
    let input: Vec<Value> = messages
        .iter()
        .filter(|m| m.role != "system")
        .map(|m| match m.role.as_str() {
            "tool" => json!({
                "role": "user",
                "content": format!("Tool {} result: {}", m.name.as_deref().unwrap_or("tool"), m.content),
            }),
            "developer" => json!({"role": "developer", "content": m.content}),
            role => json!({"role": role, "content": m.content}),
        })
        .collect();
    let instructions = if instructions.is_empty() { None } else { Some(instructions) };
    (instructions, input)
}

fn responses_error(status: u16, payload: &Value, request_id: &str) -> PageIndexError {
    let code = payload.pointer("/error/code").and_then(Value::as_str);
    let message = payload
        .pointer("/error/message")
        .and_then(Value::as_str)
        .or_else(|| payload.get("detail").and_then(Value::as_str))
        .unwrap_or("ChatGPT plan usage request failed");
    let extra = match code {
        Some("subscription_sharing_usage_limit_exceeded") => {
            " Pause ChatGPT plan requests and review ChatGPT settings → Usage."
        }
        Some("subscription_sharing_user_not_eligible") => {
            " ChatGPT plan usage is unavailable for this account or workspace."
        }
        Some("subscription_sharing_usage_unavailable") => " Usage availability could not be checked; retry later.",
        _ => "",
    };
    let code_bit = code.map(|c| format!(" [{c}]")).unwrap_or_default();
    PageIndexError::new(
        ErrorKind::Llm,
        format!("ChatGPT Responses {status}{code_bit} (request {request_id}): {message}{extra}"),
    )
    .with_status(Some(status))
}

fn failed_event(event: &Value, request_id: &str) -> PageIndexError {
    let payload = event.get("response").unwrap_or(event);
    let code = payload.pointer("/error/code").and_then(Value::as_str);
    let status = match code {
        Some("subscription_sharing_usage_limit_exceeded") => 429,
        Some("subscription_sharing_user_not_eligible") => 403,
        Some("subscription_sharing_usage_unavailable") => 503,
        _ => 400,
    };
    responses_error(status, payload, request_id)
}

fn completed_text(event: &Value) -> String {
    let output = event.pointer("/response/output").or_else(|| event.get("output")).and_then(Value::as_array);
    let Some(output) = output else {
        return String::new();
    };
    let mut text = String::new();
    for item in output {
        if let Some(content) = item.get("content").and_then(Value::as_array) {
            for block in content {
                if let Some(t) = block.get("text").and_then(Value::as_str) {
                    text.push_str(t);
                }
            }
        }
    }
    text
}

#[derive(Default)]
struct SseParser {
    buf: String,
}

impl SseParser {
    fn push(&mut self, chunk: &str) -> Vec<Value> {
        self.buf.push_str(&chunk.replace("\r\n", "\n"));
        let mut events = Vec::new();
        while let Some(idx) = self.buf.find("\n\n") {
            let raw = self.buf[..idx].to_owned();
            self.buf = self.buf[idx + 2..].to_owned();
            if let Some(value) = parse_sse_block(&raw) {
                events.push(value);
            }
        }
        events
    }
}

fn parse_sse_block(block: &str) -> Option<Value> {
    let mut data = String::new();
    for line in block.lines() {
        let Some(rest) = line.strip_prefix("data:") else {
            continue;
        };
        if !data.is_empty() {
            data.push('\n');
        }
        data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
    }
    if data.is_empty() || data == "[DONE]" {
        return None;
    }
    serde_json::from_str(&data).ok()
}
