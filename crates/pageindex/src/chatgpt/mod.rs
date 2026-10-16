//! Official Sign in with ChatGPT for open-source apps (plan usage).
//!
//! First registration uses `client_id=dynamic_agent_client`; later sign-ins
//! reuse the issued `oaiapp_…` client id. Inference is `POST
//! https://api.openai.com/v1/responses` with the OAuth bearer token.
//! This module never uses a Codex client id or `chatgpt.com/backend-api`.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::config::env_var;
use crate::error::{ErrorKind, PageIndexError, Result};
use crate::llm::{ChatMessage, Completion, CompletionParams, LlmClient, ToolSpec};

mod id_token;
mod oauth;
mod responses;
mod store;

pub use oauth::{
    Callback, InProgressLogin, PendingAuth, build_authorize_url, open_browser, parse_callback_query,
    redact_authorize_url,
};
pub use store::{AccountRecord, CredentialStore, default_config_dir};

/// First-time registration entrypoint — not an issued client id and never
/// used for token exchange.
pub const DYNAMIC_CLIENT_ID: &str = "dynamic_agent_client";
pub const ISSUER: &str = "https://auth.openai.com";
pub const RESOURCE: &str = "https://api.openai.com/v1";
pub const SCOPES: &[&str] =
    &["openid", "profile", "email", "offline_access", "resource.invoke", "chatgpt.tokens.use.direct"];

const AUTH_BASE: &str = "https://auth.openai.com";
const API_BASE: &str = "https://api.openai.com/v1";
const REFRESH_SKEW_SECS: i64 = 60;

#[derive(Clone, Debug)]
pub struct Endpoints {
    pub authorize: String,
    pub token: String,
    pub jwks: String,
    pub revoke: String,
    pub api_base: String,
}

impl Endpoints {
    pub fn production() -> Self {
        Self::from_bases(AUTH_BASE, API_BASE)
    }

    pub fn from_bases(auth_base: &str, api_base: &str) -> Self {
        let auth = auth_base.trim_end_matches('/');
        let api = api_base.trim_end_matches('/');
        Self {
            authorize: format!("{auth}/api/accounts/authorize"),
            token: format!("{auth}/api/accounts/oauth/token"),
            jwks: format!("{auth}/.well-known/jwks.json"),
            revoke: format!("{auth}/api/accounts/oauth/revoke"),
            api_base: api.to_owned(),
        }
    }

    pub fn from_env() -> Self {
        let auth = env_var("PAGEINDEX_CHATGPT_AUTH_BASE").unwrap_or_else(|| AUTH_BASE.into());
        let api = env_var("PAGEINDEX_CHATGPT_API_BASE").unwrap_or_else(|| API_BASE.into());
        Self::from_bases(&auth, &api)
    }
}

#[derive(Clone)]
pub struct ChatGptClient {
    pub http: reqwest::Client,
    pub endpoints: Endpoints,
    pub store_dir: PathBuf,
    refresh_lock: Arc<tokio::sync::Mutex<()>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatGptStatus {
    pub signed_in: bool,
    pub email: Option<String>,
    pub plan_usage: bool,
    pub client_id: Option<String>,
    pub host_id: Option<String>,
}

impl ChatGptStatus {
    pub fn render(&self) -> String {
        if !self.signed_in {
            return "Not signed in with ChatGPT. Run `pageindex chatgpt login` (API keys remain the default).".into();
        }
        let who = self.email.as_deref().unwrap_or("a ChatGPT account");
        let plan = if self.plan_usage { "enabled" } else { "not granted — re-run login and approve plan usage" };
        format!("Signed in as {who}. ChatGPT plan usage: {plan}.")
    }
}

impl ChatGptClient {
    pub fn from_env() -> Self {
        Self::new(Endpoints::from_env(), store::default_config_dir())
    }

    pub fn new(endpoints: Endpoints, store_dir: PathBuf) -> Self {
        Self { http: reqwest::Client::new(), endpoints, store_dir, refresh_lock: Arc::new(tokio::sync::Mutex::new(())) }
    }

    pub fn status(&self) -> Result<ChatGptStatus> {
        let Some(store) = store::load(&self.store_dir)? else {
            return Ok(ChatGptStatus {
                signed_in: false,
                email: None,
                plan_usage: false,
                client_id: None,
                host_id: None,
            });
        };
        match store.active() {
            Some(account) if account.signed_in() => Ok(ChatGptStatus {
                signed_in: true,
                email: account.email.clone(),
                plan_usage: account.plan_usage_enabled(),
                client_id: Some(account.client_id.clone()),
                host_id: Some(store.ext_agent_host_id),
            }),
            Some(account) => Ok(ChatGptStatus {
                signed_in: false,
                email: account.email.clone(),
                plan_usage: false,
                client_id: Some(account.client_id.clone()),
                host_id: Some(store.ext_agent_host_id),
            }),
            None => Ok(ChatGptStatus {
                signed_in: false,
                email: None,
                plan_usage: false,
                client_id: None,
                host_id: Some(store.ext_agent_host_id),
            }),
        }
    }

    pub async fn begin_login(&self) -> Result<InProgressLogin> {
        oauth::begin_login(self).await
    }

    pub async fn login(&self, open_browser: bool, timeout: Duration) -> Result<AccountRecord> {
        let flow = self.begin_login().await?;
        if open_browser {
            let _ = oauth::open_browser(&flow.authorize_url);
        }
        flow.finish(self, timeout).await
    }

    pub async fn complete_authorization(
        &self,
        pending: &oauth::PendingAuth,
        callback: oauth::Callback,
    ) -> Result<AccountRecord> {
        oauth::complete_authorization(self, pending, callback).await
    }

    pub async fn logout(&self) -> Result<LogoutOutcome> {
        let mut store = store::load_or_init(&self.store_dir)?;
        let Some(account) = store.active().cloned() else {
            return Ok(LogoutOutcome { revoked: true, message: "Not signed in with ChatGPT.".into() });
        };
        let revoked = match oauth::revoke_refresh(self, &account).await {
            Ok(()) => true,
            Err(_) => false,
        };
        if let Some(existing) = store.accounts.iter_mut().find(|a| a.client_id == account.client_id) {
            existing.clear_tokens();
        }
        store::save(&self.store_dir, &store)?;
        let message = if revoked {
            "Signed out of ChatGPT. Local tokens were cleared; the client registration was kept for a later sign-in."
                .into()
        } else {
            "Signed out locally, but remote revocation was not confirmed. Disconnect the app in ChatGPT Settings if needed.".into()
        };
        Ok(LogoutOutcome { revoked, message })
    }

    pub async fn access_token(&self) -> Result<String> {
        let _guard = self.refresh_lock.lock().await;
        let store = store::load(&self.store_dir)?.ok_or_else(not_signed_in)?;
        let account = store.active().cloned().ok_or_else(not_signed_in)?;
        if !account.plan_usage_enabled() && account.access_token.is_none() {
            return Err(not_signed_in());
        }
        if !account.scopes.iter().any(|s| s == "chatgpt.tokens.use.direct") {
            return Err(PageIndexError::new(
                ErrorKind::Llm,
                "ChatGPT plan usage was not granted. Re-run `pageindex chatgpt login` and approve plan usage, or set OPENAI_API_KEY.",
            )
            .with_status(Some(403)));
        }
        if !account.access_expired(oauth::unix_now(), REFRESH_SKEW_SECS)
            && let Some(token) = account.access_token.clone()
        {
            return Ok(token);
        }
        let refreshed = oauth::refresh_tokens(self, &account).await?;
        refreshed
            .access_token
            .ok_or_else(|| PageIndexError::new(ErrorKind::Llm, "Refresh succeeded but no access token was returned."))
    }
}

#[derive(Clone, Debug)]
pub struct LogoutOutcome {
    pub revoked: bool,
    pub message: String,
}

fn not_signed_in() -> PageIndexError {
    PageIndexError::new(
        ErrorKind::Llm,
        "Not signed in with ChatGPT. Run `pageindex chatgpt login`, then use a chatgpt/ model (API keys remain the default).",
    )
    .with_status(Some(401))
}

/// Prefix a bare / OpenAI model id for the ChatGPT plan-usage provider.
pub fn chatgpt_model(model: &str) -> Result<String> {
    if model.starts_with("chatgpt/") {
        return Ok(model.to_owned());
    }
    if let Some(rest) = model.strip_prefix("openai/") {
        return Ok(format!("chatgpt/{rest}"));
    }
    if model.contains('/') {
        return Err(PageIndexError::invalid(format!("--chatgpt cannot be combined with non-OpenAI model '{model}'")));
    }
    Ok(format!("chatgpt/{model}"))
}

#[async_trait]
impl LlmClient for ChatGptClient {
    async fn complete(&self, model: &str, messages: &[ChatMessage], params: &CompletionParams) -> Result<String> {
        self.complete_turn(model, messages, params, &[]).await.map(|c| c.text)
    }

    async fn complete_turn(
        &self,
        model: &str,
        messages: &[ChatMessage],
        params: &CompletionParams,
        tools: &[ToolSpec],
    ) -> Result<Completion> {
        let wire_model = model.strip_prefix("chatgpt/").unwrap_or(model);
        let token = self.access_token().await?;
        responses::complete_responses(&self.http, &self.endpoints.api_base, &token, wire_model, messages, params, tools)
            .await
    }
}

#[cfg(test)]
mod tests;
