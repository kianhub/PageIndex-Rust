//! Model calls behind one async trait, with a LiteLLM-style router.
//!
//! Model names follow LiteLLM: a bare name is an OpenAI model; other
//! providers are spelled `provider/model` (`anthropic/claude-sonnet-5-5`,
//! `openrouter/...`, `ollama/llama3`). Credentials come only from the
//! environment (or `.env`), or from a per-lane [`Backend`].

use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::config::{env_var, openai_api_key};
use crate::error::{ErrorKind, PageIndexError, Result};
use crate::types::Backend;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompletionParams {
    /// Sent where the provider supports it, dropped elsewhere.
    pub reasoning_effort: Option<String>,
    pub max_tokens: Option<u32>,
}

/// One chat completion. Errors carry the HTTP status when there was one.
#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, model: &str, messages: &[ChatMessage], params: &CompletionParams) -> Result<String>;
}

pub const MAX_RETRIES: usize = 10;

/// Statuses a retry cannot fix; 400 is per-prompt (the caller absorbs it).
fn no_retry(status: Option<u16>) -> bool {
    matches!(status, Some(400 | 401 | 403 | 404))
}

/// One user prompt through the retry ladder: up to [`MAX_RETRIES`] tries a
/// second apart; 400/401/403/404 are raised at once.
pub async fn complete_prompt(
    llm: &dyn LlmClient,
    model: &str,
    prompt: &str,
    params: &CompletionParams,
) -> Result<String> {
    let messages = [ChatMessage::user(prompt)];
    complete_messages(llm, model, &messages, params).await
}

pub async fn complete_messages(
    llm: &dyn LlmClient,
    model: &str,
    messages: &[ChatMessage],
    params: &CompletionParams,
) -> Result<String> {
    let mut last: Option<PageIndexError> = None;
    for attempt in 0..MAX_RETRIES {
        match llm.complete(model, messages, params).await {
            Ok(text) => return Ok(text),
            Err(err) if no_retry(err.status_code) => return Err(err),
            Err(err) => {
                last = Some(err);
                if attempt + 1 < MAX_RETRIES {
                    tokio::time::sleep(retry_delay()).await;
                }
            }
        }
    }
    let last = last.expect("at least one attempt ran");
    Err(PageIndexError::new(
        ErrorKind::Llm,
        format!("LLM completion failed after {MAX_RETRIES} retries: {}", last.message),
    )
    .with_status(last.status_code))
}

fn retry_delay() -> Duration {
    if cfg!(test) { Duration::from_millis(1) } else { Duration::from_secs(1) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Wire {
    OpenAi,
    Anthropic,
}

struct Provider {
    name: &'static str,
    wire: Wire,
    base: &'static str,
    base_env: &'static [&'static str],
    key_env: &'static [&'static str],
    needs_key: bool,
    reasoning_effort: bool,
}

const PROVIDERS: &[Provider] = &[
    Provider {
        name: "openai",
        wire: Wire::OpenAi,
        base: "https://api.openai.com/v1",
        base_env: &["OPENAI_BASE_URL", "OPENAI_API_BASE"],
        key_env: &[],
        needs_key: true,
        reasoning_effort: true,
    },
    Provider {
        name: "anthropic",
        wire: Wire::Anthropic,
        base: "https://api.anthropic.com/v1",
        base_env: &["ANTHROPIC_BASE_URL", "ANTHROPIC_API_BASE"],
        key_env: &["ANTHROPIC_API_KEY"],
        needs_key: true,
        reasoning_effort: false,
    },
    Provider {
        name: "openrouter",
        wire: Wire::OpenAi,
        base: "https://openrouter.ai/api/v1",
        base_env: &["OPENROUTER_API_BASE"],
        key_env: &["OPENROUTER_API_KEY"],
        needs_key: true,
        reasoning_effort: true,
    },
    Provider {
        name: "groq",
        wire: Wire::OpenAi,
        base: "https://api.groq.com/openai/v1",
        base_env: &["GROQ_API_BASE"],
        key_env: &["GROQ_API_KEY"],
        needs_key: true,
        reasoning_effort: false,
    },
    Provider {
        name: "deepseek",
        wire: Wire::OpenAi,
        base: "https://api.deepseek.com/v1",
        base_env: &["DEEPSEEK_API_BASE"],
        key_env: &["DEEPSEEK_API_KEY"],
        needs_key: true,
        reasoning_effort: false,
    },
    Provider {
        name: "mistral",
        wire: Wire::OpenAi,
        base: "https://api.mistral.ai/v1",
        base_env: &["MISTRAL_API_BASE"],
        key_env: &["MISTRAL_API_KEY"],
        needs_key: true,
        reasoning_effort: false,
    },
    Provider {
        name: "together_ai",
        wire: Wire::OpenAi,
        base: "https://api.together.xyz/v1",
        base_env: &["TOGETHERAI_API_BASE"],
        key_env: &["TOGETHERAI_API_KEY", "TOGETHER_API_KEY"],
        needs_key: true,
        reasoning_effort: false,
    },
    Provider {
        name: "fireworks_ai",
        wire: Wire::OpenAi,
        base: "https://api.fireworks.ai/inference/v1",
        base_env: &["FIREWORKS_API_BASE"],
        key_env: &["FIREWORKS_AI_API_KEY", "FIREWORKS_API_KEY"],
        needs_key: true,
        reasoning_effort: false,
    },
    Provider {
        name: "xai",
        wire: Wire::OpenAi,
        base: "https://api.x.ai/v1",
        base_env: &["XAI_API_BASE"],
        key_env: &["XAI_API_KEY"],
        needs_key: true,
        reasoning_effort: true,
    },
    Provider {
        name: "gemini",
        wire: Wire::OpenAi,
        base: "https://generativelanguage.googleapis.com/v1beta/openai",
        base_env: &["GEMINI_API_BASE"],
        key_env: &["GEMINI_API_KEY"],
        needs_key: true,
        reasoning_effort: true,
    },
    Provider {
        name: "ollama",
        wire: Wire::OpenAi,
        base: "http://localhost:11434/v1",
        base_env: &["OLLAMA_API_BASE"],
        key_env: &[],
        needs_key: false,
        reasoning_effort: false,
    },
];

/// The provider and wire model name a LiteLLM model string routes to.
pub fn route(model: &str) -> Result<(&'static str, String)> {
    let model = model.strip_prefix("litellm/").unwrap_or(model);
    let (provider, name) = match model.split_once('/') {
        Some((provider, name)) => (provider, name),
        None => ("openai", model),
    };
    PROVIDERS
        .iter()
        .find(|p| p.name == provider)
        .map(|p| (p.name, name.to_owned()))
        .ok_or_else(|| {
            PageIndexError::new(
                ErrorKind::Llm,
                format!(
                    "'{model}' routes through LiteLLM, but '{provider}' is not a supported provider. For an \
                     OpenAI-compatible server serving this model id, use 'openai/{model}' and point OPENAI_BASE_URL at the server."
                ),
            )
            .with_status(Some(404))
        })
}

/// The default client: real HTTP calls to the routed provider.
#[derive(Debug, Clone, Default)]
pub struct LiteLlm {
    http: reqwest::Client,
    backend: Option<Backend>,
}

impl LiteLlm {
    pub fn new() -> Self {
        Self::default()
    }

    /// Route every call through these connection overrides.
    pub fn with_backend(backend: Backend) -> Self {
        Self { http: reqwest::Client::new(), backend: Some(backend) }
    }

    fn credentials(&self, provider: &Provider) -> Result<(String, Option<String>)> {
        let base = self
            .backend
            .as_ref()
            .and_then(|b| b.api_base.clone())
            .or_else(|| provider.base_env.iter().find_map(|k| env_var(k)))
            .unwrap_or_else(|| provider.base.to_owned());
        let key = self.backend.as_ref().and_then(|b| b.api_key.clone()).or_else(|| {
            if provider.name == "openai" { openai_api_key() } else { provider.key_env.iter().find_map(|k| env_var(k)) }
        });
        if key.is_none() && provider.needs_key && self.backend.as_ref().and_then(|b| b.api_base.as_ref()).is_none() {
            let var = if provider.name == "openai" {
                "OPENAI_API_KEY"
            } else {
                provider.key_env.first().copied().unwrap_or("API key")
            };
            return Err(PageIndexError::new(
                ErrorKind::Llm,
                format!("Missing credentials for {}: set {var}", provider.name),
            )
            .with_status(Some(401)));
        }
        Ok((base.trim_end_matches('/').to_owned(), key))
    }
}

#[async_trait]
impl LlmClient for LiteLlm {
    async fn complete(&self, model: &str, messages: &[ChatMessage], params: &CompletionParams) -> Result<String> {
        let (provider_name, wire_model) = route(model)?;
        let provider = PROVIDERS.iter().find(|p| p.name == provider_name).expect("routed provider exists");
        let (base, key) = self.credentials(provider)?;
        let msgs: Vec<Value> = messages.iter().map(|m| json!({"role": m.role, "content": m.content})).collect();
        let request = match provider.wire {
            Wire::OpenAi => {
                let mut body = json!({"model": wire_model, "messages": msgs});
                if let (Some(effort), true) = (&params.reasoning_effort, provider.reasoning_effort) {
                    body["reasoning_effort"] = json!(effort);
                }
                if let Some(max) = params.max_tokens {
                    body["max_tokens"] = json!(max);
                }
                merge_extra(&mut body, self.backend.as_ref());
                let mut req = self.http.post(format!("{base}/chat/completions")).json(&body);
                if let Some(key) = &key {
                    req = req.bearer_auth(key);
                }
                req
            }
            Wire::Anthropic => {
                let system: Vec<&str> =
                    messages.iter().filter(|m| m.role == "system").map(|m| m.content.as_str()).collect();
                let msgs: Vec<Value> = messages
                    .iter()
                    .filter(|m| m.role != "system")
                    .map(|m| json!({"role": m.role, "content": m.content}))
                    .collect();
                let mut body =
                    json!({"model": wire_model, "messages": msgs, "max_tokens": params.max_tokens.unwrap_or(8192)});
                if !system.is_empty() {
                    body["system"] = json!(system.join("\n\n"));
                }
                merge_extra(&mut body, self.backend.as_ref());
                self.http
                    .post(format!("{base}/messages"))
                    .header("x-api-key", key.unwrap_or_default())
                    .header("anthropic-version", "2023-06-01")
                    .json(&body)
            }
        };
        let response = request
            .send()
            .await
            .map_err(|e| PageIndexError::new(ErrorKind::Llm, format!("{provider_name} request failed: {e}")))?;
        let status = response.status();
        let payload: Value = response.json().await.unwrap_or(Value::Null);
        if !status.is_success() {
            let detail = payload
                .pointer("/error/message")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| payload.to_string());
            return Err(PageIndexError::new(ErrorKind::Llm, format!("{provider_name} returned {status}: {detail}"))
                .with_status(Some(status.as_u16())));
        }
        let text = match provider.wire {
            Wire::OpenAi => payload.pointer("/choices/0/message/content").and_then(Value::as_str).map(str::to_owned),
            Wire::Anthropic => payload.get("content").and_then(Value::as_array).map(|blocks| {
                blocks.iter().filter_map(|b| b.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("")
            }),
        };
        Ok(text.unwrap_or_default())
    }
}

fn merge_extra(body: &mut Value, backend: Option<&Backend>) {
    if let (Some(backend), Some(obj)) = (backend, body.as_object_mut()) {
        for (k, v) in &backend.extra {
            obj.insert(k.clone(), v.clone());
        }
    }
}

#[cfg(test)]
pub mod testing {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// Replies from a script, in order; the last reply repeats.
    pub struct ScriptedLlm {
        replies: Mutex<Vec<String>>,
        calls: AtomicUsize,
        pub models: Mutex<Vec<String>>,
        pub prompts: Mutex<Vec<String>>,
    }

    impl ScriptedLlm {
        pub fn new(replies: Vec<String>) -> Self {
            Self {
                replies: Mutex::new(replies),
                calls: AtomicUsize::new(0),
                models: Mutex::default(),
                prompts: Mutex::default(),
            }
        }

        pub fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl LlmClient for ScriptedLlm {
        async fn complete(&self, model: &str, messages: &[ChatMessage], _: &CompletionParams) -> Result<String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.models.lock().unwrap().push(model.to_owned());
            self.prompts.lock().unwrap().push(messages.last().map(|m| m.content.clone()).unwrap_or_default());
            let mut replies = self.replies.lock().unwrap();
            Ok(if replies.len() > 1 { replies.remove(0) } else { replies.first().cloned().unwrap_or_default() })
        }
    }

    /// Fails every call with `status`.
    pub struct FailingLlm(pub Option<u16>, pub AtomicUsize);

    #[async_trait]
    impl LlmClient for FailingLlm {
        async fn complete(&self, _: &str, _: &[ChatMessage], _: &CompletionParams) -> Result<String> {
            self.1.fetch_add(1, Ordering::SeqCst);
            Err(PageIndexError::new(ErrorKind::Llm, "boom").with_status(self.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::testing::FailingLlm;
    use super::*;

    #[test]
    fn routes_litellm_names() {
        assert_eq!(route("gpt-6-luna").unwrap(), ("openai", "gpt-6-luna".to_string()));
        assert_eq!(route("anthropic/claude-sonnet-5-5").unwrap().0, "anthropic");
        assert_eq!(route("litellm/openrouter/x/y").unwrap(), ("openrouter", "x/y".to_string()));
        assert_eq!(route("nope/model").unwrap_err().status_code, Some(404));
    }

    #[tokio::test]
    async fn retry_ladder() {
        let transient = FailingLlm(Some(503), AtomicUsize::new(0));
        let err = complete_prompt(&transient, "m", "p", &CompletionParams::default()).await.unwrap_err();
        assert_eq!(transient.1.load(Ordering::SeqCst), MAX_RETRIES);
        assert_eq!(err.status_code, Some(503));
        assert!(!err.is_unrecoverable());
        let fatal = FailingLlm(Some(401), AtomicUsize::new(0));
        let err = complete_prompt(&fatal, "m", "p", &CompletionParams::default()).await.unwrap_err();
        assert_eq!(fatal.1.load(Ordering::SeqCst), 1);
        assert!(err.is_unrecoverable());
    }
}
