//! Own-model chat: a tool-calling agent over the local or cloud document store.

use serde_json::{Value, json};

use crate::agent::{
    AGENT_INSTRUCTIONS, AgentBackend, CHAT_HEADER, as_openai_tools, base_instructions, call_tool,
    fetch_citation_prompt, local_tool_spec, targeting_block, tool_names,
};
use crate::chat_stream::{ChatEvent, ChatStream};
use crate::error::{PageIndexError, Result};
use crate::llm::{ChatMessage, CompletionParams, LlmClient, ToolSpec};
use crate::types::ChatProcessOptions;

const DEFAULT_MAX_TURNS: usize = 10;

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub doc_id: Vec<String>,
    pub folder_id: Option<String>,
    pub model: String,
    pub reasoning_effort: Option<String>,
    pub show_process: ChatProcessOptions,
    pub citations: bool,
    pub max_turns: usize,
    pub include_management: bool,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            doc_id: Vec::new(),
            folder_id: None,
            model: model.into(),
            reasoning_effort: None,
            show_process: ChatProcessOptions::default(),
            citations: false,
            max_turns: DEFAULT_MAX_TURNS,
            include_management: false,
            temperature: None,
            top_p: None,
            max_tokens: None,
        }
    }
}

fn weave(events: &[ChatEvent], opts: ChatProcessOptions) -> Vec<String> {
    let mut out = Vec::new();
    for event in events {
        match event.kind.as_str() {
            "thinking" if opts.thinking => {
                if let Some(delta) = &event.delta {
                    out.push(clip(&format!("[thinking] {delta}"), opts.max_chars));
                }
            }
            "tool_call" if opts.tool_call => {
                out.push(clip(
                    &format!(
                        "[tool_call] {} {}",
                        event.name.as_deref().unwrap_or("tool"),
                        event.arguments.clone().unwrap_or(json!({}))
                    ),
                    opts.max_chars,
                ));
            }
            "tool_result" if opts.tool_result => {
                let output = event.output.clone().unwrap_or(Value::Null);
                let text = output.as_str().map(str::to_owned).unwrap_or_else(|| output.to_string());
                out.push(clip(
                    &format!("[tool_result] {}: {text}", event.name.as_deref().unwrap_or("tool")),
                    opts.max_chars,
                ));
            }
            "answer" => {
                if let Some(delta) = &event.delta {
                    out.push(delta.clone());
                }
            }
            _ => {}
        }
    }
    out
}

fn clip(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return format!("{text}\n");
    }
    let clipped: String = text.chars().take(max_chars).collect();
    format!("{clipped}…\n")
}

async fn system_prompt(client: &dyn AgentBackend, req: &ChatRequest) -> Result<String> {
    let mut instructions = format!("{CHAT_HEADER}\n\n{}", base_instructions(client, req.include_management).await?);
    if req.citations {
        instructions.push_str("\n\n");
        instructions.push_str(&fetch_citation_prompt(client, "cite").await?);
    }
    let extra: Vec<&str> = req.messages.iter().filter(|m| m.role == "system").map(|m| m.content.as_str()).collect();
    if !extra.is_empty() {
        instructions.push_str("\n\n");
        instructions.push_str(&extra.join("\n\n"));
    }
    Ok(instructions)
}

fn history(req: &ChatRequest) -> Vec<ChatMessage> {
    req.messages.iter().filter(|m| m.role == "user" || m.role == "assistant").cloned().collect()
}

fn tools(include_management: bool) -> Vec<ToolSpec> {
    tool_names(include_management).into_iter().map(local_tool_spec).collect()
}

/// Own-model chat completions: OpenAI-shaped `{choices, usage}` or a stream.
pub async fn run_chat_completions(
    client: &dyn AgentBackend,
    llm: &dyn LlmClient,
    req: &ChatRequest,
    stream: bool,
) -> Result<Value> {
    let (answer, events) = run_agent(client, llm, req).await?;
    if stream {
        return Ok(json!({"stream": weave(&events, req.show_process), "events": events, "answer": answer}));
    }
    Ok(json!({
        "id": "chatcmpl-pageindex",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }))
}

/// Streamed own-model chat: text pieces and typed events.
pub async fn run_chat_stream(client: &dyn AgentBackend, llm: &dyn LlmClient, req: &ChatRequest) -> Result<ChatStream> {
    let (answer, events) = run_agent(client, llm, req).await?;
    let text = if req.show_process.thinking || req.show_process.tool_call || req.show_process.tool_result {
        weave(&events, req.show_process)
    } else if answer.is_empty() {
        Vec::new()
    } else {
        vec![answer]
    };
    Ok(ChatStream::from_parts(text, events))
}

/// OpenAI Responses-shaped envelope.
pub async fn run_responses(client: &dyn AgentBackend, llm: &dyn LlmClient, req: &ChatRequest) -> Result<Value> {
    let (answer, events) = run_agent(client, llm, req).await?;
    Ok(json!({
        "id": "resp-pageindex",
        "object": "response",
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": answer}]}],
        "tools": as_openai_tools(req.include_management, false, None, client.base_url()),
        "events": events
    }))
}

/// Anthropic Messages-shaped envelope.
pub async fn run_messages(client: &dyn AgentBackend, llm: &dyn LlmClient, req: &ChatRequest) -> Result<Value> {
    let (answer, events) = run_agent(client, llm, req).await?;
    Ok(json!({
        "id": "msg-pageindex",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": answer}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "events": events
    }))
}

/// Weave a managed cloud chunk stream into a [`ChatStream`].
pub fn run_cloud_chat_stream(chunks: &[Value], show_process: ChatProcessOptions) -> ChatStream {
    let mut events = Vec::new();
    let mut text = Vec::new();
    for chunk in chunks {
        if let Some(content) = chunk.pointer("/choices/0/delta/content").and_then(Value::as_str) {
            events.push(ChatEvent::answer(content));
            text.push(content.to_owned());
        }
        if let Some(meta) = chunk.get("block_metadata")
            && let Some(name) = meta.get("name").and_then(Value::as_str)
        {
            events.push(ChatEvent::tool_call(
                meta.get("id").and_then(Value::as_str).unwrap_or(""),
                name,
                meta.get("input").cloned().unwrap_or(json!({})),
            ));
        }
    }
    if show_process.tool_call || show_process.thinking {
        ChatStream::from_parts(weave(&events, show_process), events)
    } else {
        ChatStream::from_parts(text, events)
    }
}

async fn run_agent(
    client: &dyn AgentBackend,
    llm: &dyn LlmClient,
    req: &ChatRequest,
) -> Result<(String, Vec<ChatEvent>)> {
    let system = system_prompt(client, req).await.unwrap_or_else(|_| format!("{CHAT_HEADER}\n\n{AGENT_INSTRUCTIONS}"));
    let mut messages = vec![ChatMessage::system(system)];
    if let Some(block) =
        targeting_block(client, (!req.doc_id.is_empty()).then_some(req.doc_id.as_slice()), req.folder_id.as_deref())
            .await?
    {
        messages.push(ChatMessage::user(block));
    }
    messages.extend(history(req));
    if messages.iter().all(|m| m.role != "user") {
        return Err(PageIndexError::invalid("chat requires at least one user message"));
    }
    let tools = tools(req.include_management);
    let params = CompletionParams {
        reasoning_effort: req.reasoning_effort.clone(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
    };
    let mut events = Vec::new();
    let scope = if req.doc_id.is_empty() { None } else { Some(req.doc_id.as_slice()) };
    for turn in 0..req.max_turns {
        let completion = llm.complete_turn(&req.model, &messages, &params, &tools).await?;
        if !completion.tool_calls.is_empty() {
            messages.push(ChatMessage::assistant_tools(completion.tool_calls.clone()));
            for call in completion.tool_calls {
                let args: Value = serde_json::from_str(&call.arguments).unwrap_or(json!({}));
                events.push(ChatEvent::tool_call(&call.id, &call.name, args.clone()));
                let ids = scope.map(|s| s.to_vec());
                let (output, _) = call_tool(client, &call.name, args, ids.as_deref()).await;
                events.push(ChatEvent::tool_result(&call.id, &call.name, json!(output)));
                messages.push(ChatMessage::tool(&call.id, &call.name, output));
            }
            continue;
        }
        if !completion.text.is_empty() {
            events.push(ChatEvent::answer(&completion.text));
        }
        let _ = turn;
        return Ok((completion.text, events));
    }
    Err(PageIndexError::api(format!(
        "Chat exceeded max_turns={} without a final answer. Raise max_turns or narrow the question.",
        req.max_turns
    )))
}
