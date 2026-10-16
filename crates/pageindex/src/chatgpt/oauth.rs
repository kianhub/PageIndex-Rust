//! Authorization Code + PKCE against OpenAI's documented OSS endpoints.

use std::collections::HashMap;
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use uuid::Uuid;

use crate::error::{ErrorKind, PageIndexError, Result};

use super::id_token::verify_id_token;
use super::store::{self, AGENT_NAME_HINT, AccountRecord};
use super::{DYNAMIC_CLIENT_ID, Endpoints, ISSUER, RESOURCE, SCOPES};

/// Unreserved characters stay unreserved in query values (RFC 3986).
const QS: &AsciiSet = &NON_ALPHANUMERIC.remove(b'-').remove(b'.').remove(b'_').remove(b'~');

pub struct PendingAuth {
    pub state: String,
    pub nonce: String,
    pub verifier: String,
    pub redirect_uri: String,
    /// `dynamic_agent_client` on first registration; otherwise the saved issued id.
    pub requested_client_id: String,
    pub include_name_hint: bool,
    pub id_token_hint: Option<String>,
    pub login_hint: Option<String>,
}

pub struct Callback {
    pub code: Option<String>,
    pub state: String,
    pub client_id: Option<String>,
    pub error: Option<String>,
    pub scope: Option<String>,
}

pub struct InProgressLogin {
    pub authorize_url: String,
    pub redirect_uri: String,
    pub state: String,
    pub nonce: String,
    pending: PendingAuth,
    listener: TcpListener,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: Option<String>,
    refresh_token: Option<String>,
    id_token: Option<String>,
    token_type: Option<String>,
    expires_in: Option<u64>,
    scope: Option<String>,
    error: Option<String>,
}

impl InProgressLogin {
    pub async fn finish(self, client: &super::ChatGptClient, timeout: Duration) -> Result<AccountRecord> {
        let callback = tokio::time::timeout(timeout, wait_callback(self.listener, &self.pending.state))
            .await
            .map_err(|_| PageIndexError::new(ErrorKind::Api, "timed out waiting for the ChatGPT sign-in callback"))??;
        client.complete_authorization(&self.pending, callback).await
    }
}

pub fn new_pending(redirect_uri: String, requested_client_id: String, include_name_hint: bool) -> PendingAuth {
    PendingAuth {
        state: random_value(16),
        nonce: random_value(16),
        verifier: random_value(32),
        redirect_uri,
        requested_client_id,
        include_name_hint,
        id_token_hint: None,
        login_hint: None,
    }
}

pub fn challenge(verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

pub fn build_authorize_url(endpoints: &Endpoints, host_id: &str, pending: &PendingAuth) -> String {
    let challenge = challenge(&pending.verifier);
    let mut pairs: Vec<(&str, String)> = vec![
        ("client_id", pending.requested_client_id.clone()),
        ("ext_agent_host_id", host_id.to_owned()),
        ("response_type", "code".into()),
        ("redirect_uri", pending.redirect_uri.clone()),
        ("scope", SCOPES.join(" ")),
        ("resource", RESOURCE.into()),
        ("state", pending.state.clone()),
        ("nonce", pending.nonce.clone()),
        ("code_challenge_method", "S256".into()),
        ("code_challenge", challenge),
    ];
    if pending.include_name_hint {
        pairs.insert(1, ("agent_name_hint", AGENT_NAME_HINT.into()));
    }
    if let Some(hint) = &pending.id_token_hint {
        pairs.push(("id_token_hint", hint.clone()));
    }
    if let Some(hint) = &pending.login_hint {
        pairs.push(("login_hint", hint.clone()));
    }
    let query = pairs
        .iter()
        .map(|(k, v)| format!("{}={}", utf8_percent_encode(k, QS), utf8_percent_encode(v, QS)))
        .collect::<Vec<_>>()
        .join("&");
    format!("{}?{query}", endpoints.authorize)
}

/// Strip `id_token_hint` so the URL is safe to print.
pub fn redact_authorize_url(url: &str) -> String {
    match url.split_once('?') {
        Some((base, query)) => {
            let filtered: Vec<&str> = query.split('&').filter(|part| !part.starts_with("id_token_hint=")).collect();
            if filtered.is_empty() { base.to_owned() } else { format!("{base}?{}", filtered.join("&")) }
        }
        None => url.to_owned(),
    }
}

pub async fn bind_loopback() -> Result<(TcpListener, String)> {
    let listener = match TcpListener::bind("127.0.0.1:1455").await {
        Ok(listener) => listener,
        Err(_) => TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| PageIndexError::new(ErrorKind::Io, format!("could not bind loopback callback: {e}")))?,
    };
    let port = listener.local_addr()?.port();
    Ok((listener, format!("http://127.0.0.1:{port}/auth/callback")))
}

pub async fn begin_login(client: &super::ChatGptClient) -> Result<InProgressLogin> {
    let store = store::load_or_init(&client.store_dir)?;
    let host_id = store.ext_agent_host_id.clone();
    let (requested_client_id, include_name_hint, id_token_hint, login_hint) = match store.active() {
        Some(account) if account.client_id != DYNAMIC_CLIENT_ID => {
            (account.client_id.clone(), false, account.id_token.clone(), account.email.clone())
        }
        _ => (DYNAMIC_CLIENT_ID.into(), true, None, None),
    };
    let (listener, redirect_uri) = bind_loopback().await?;
    let mut pending = new_pending(redirect_uri.clone(), requested_client_id, include_name_hint);
    pending.id_token_hint = id_token_hint;
    pending.login_hint = login_hint;
    let authorize_url = build_authorize_url(&client.endpoints, &host_id, &pending);
    Ok(InProgressLogin {
        authorize_url,
        redirect_uri,
        state: pending.state.clone(),
        nonce: pending.nonce.clone(),
        pending,
        listener,
    })
}

pub fn open_browser(url: &str) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(url)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;
    }
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        std::process::Command::new("xdg-open")
            .arg(url)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;
    }
    Ok(())
}

pub async fn complete_authorization(
    client: &super::ChatGptClient,
    pending: &PendingAuth,
    callback: Callback,
) -> Result<AccountRecord> {
    if callback.state != pending.state {
        return Err(PageIndexError::new(ErrorKind::Api, "Sign-in could not be verified (state mismatch)."));
    }
    if let Some(error) = callback.error.as_deref() {
        if error == "access_denied" {
            return Err(PageIndexError::new(
                ErrorKind::Api,
                "ChatGPT sign-in was declined. Plan usage is not enabled. Re-run `pageindex chatgpt login` to grant it, or set OPENAI_API_KEY.",
            ));
        }
        return Err(PageIndexError::new(ErrorKind::Api, format!("ChatGPT sign-in failed: {error}")));
    }
    let issued_client_id = resolve_issued_client_id(pending, &callback)?;
    let code = callback
        .code
        .ok_or_else(|| PageIndexError::new(ErrorKind::Api, "The callback did not contain an authorization code."))?;
    let tokens = exchange_code(client, pending, &issued_client_id, &code).await?;
    persist_tokens(client, pending, &issued_client_id, tokens).await
}

fn resolve_issued_client_id(pending: &PendingAuth, callback: &Callback) -> Result<String> {
    if pending.requested_client_id == DYNAMIC_CLIENT_ID {
        let issued = callback.client_id.as_deref().unwrap_or("");
        if issued.is_empty() || issued == DYNAMIC_CLIENT_ID {
            return Err(PageIndexError::new(
                ErrorKind::Api,
                "Registration was incomplete: the callback did not include an issued client_id.",
            ));
        }
        return Ok(issued.to_owned());
    }
    if let Some(returned) = callback.client_id.as_deref()
        && returned != pending.requested_client_id
    {
        return Err(PageIndexError::new(
            ErrorKind::Api,
            "The callback client_id did not match the selected account's registration.",
        ));
    }
    Ok(pending.requested_client_id.clone())
}

async fn exchange_code(
    client: &super::ChatGptClient,
    pending: &PendingAuth,
    issued_client_id: &str,
    code: &str,
) -> Result<TokenResponse> {
    let form = [
        ("grant_type", "authorization_code"),
        ("client_id", issued_client_id),
        ("code", code),
        ("code_verifier", pending.verifier.as_str()),
        ("redirect_uri", pending.redirect_uri.as_str()),
        ("resource", RESOURCE),
    ];
    token_request(client, &form).await
}

pub async fn refresh_tokens(client: &super::ChatGptClient, account: &AccountRecord) -> Result<AccountRecord> {
    let refresh = account.refresh_token.as_deref().ok_or_else(|| {
        PageIndexError::new(ErrorKind::Api, "No refresh token is stored. Run `pageindex chatgpt login`.")
    })?;
    let form = [
        ("grant_type", "refresh_token"),
        ("client_id", account.client_id.as_str()),
        ("refresh_token", refresh),
        ("resource", RESOURCE),
    ];
    let tokens = match token_request(client, &form).await {
        Ok(tokens) => tokens,
        Err(err) if is_unusable_refresh(&err) => {
            let mut store = store::load_or_init(&client.store_dir)?;
            if let Some(existing) = store.accounts.iter_mut().find(|a| a.client_id == account.client_id) {
                existing.clear_tokens();
            }
            store::save(&client.store_dir, &store)?;
            return Err(PageIndexError::new(
                ErrorKind::Api,
                "The ChatGPT refresh token is no longer valid. Run `pageindex chatgpt login`.",
            )
            .with_status(err.status_code));
        }
        Err(err) => return Err(err),
    };
    persist_tokens(client, &reauth_pending(account), &account.client_id, tokens).await
}

fn reauth_pending(account: &AccountRecord) -> PendingAuth {
    PendingAuth {
        state: String::new(),
        nonce: String::new(),
        verifier: String::new(),
        redirect_uri: String::new(),
        requested_client_id: account.client_id.clone(),
        include_name_hint: false,
        id_token_hint: None,
        login_hint: None,
    }
}

fn is_unusable_refresh(err: &PageIndexError) -> bool {
    let msg = err.message.to_ascii_lowercase();
    [
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    ]
    .iter()
    .any(|code| msg.contains(code))
}

async fn token_request(client: &super::ChatGptClient, form: &[(&str, &str)]) -> Result<TokenResponse> {
    let response = client
        .http
        .post(&client.endpoints.token)
        .header("accept", "application/json")
        .header("content-type", "application/x-www-form-urlencoded")
        .form(form)
        .send()
        .await
        .map_err(|e| PageIndexError::new(ErrorKind::Api, format!("token request failed: {e}")))?;
    let status = response.status();
    let request_id = response
        .headers()
        .get("openai-request-id")
        .or_else(|| response.headers().get("x-request-id"))
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-")
        .to_owned();
    let payload: Value = response.json().await.unwrap_or(Value::Null);
    if !status.is_success() {
        let error = payload.get("error").and_then(Value::as_str).unwrap_or("token_exchange_failed");
        return Err(PageIndexError::new(
            ErrorKind::Api,
            format!("OpenAI token exchange failed ({status}, request {request_id}): {error}"),
        )
        .with_status(Some(status.as_u16())));
    }
    serde_json::from_value(payload)
        .map_err(|e| PageIndexError::new(ErrorKind::Api, format!("invalid token response: {e}")))
}

async fn persist_tokens(
    client: &super::ChatGptClient,
    pending: &PendingAuth,
    issued_client_id: &str,
    tokens: TokenResponse,
) -> Result<AccountRecord> {
    if let Some(error) = tokens.error {
        return Err(PageIndexError::new(ErrorKind::Api, format!("token response error: {error}")));
    }
    let mut store = store::load_or_init(&client.store_dir)?;
    let now = unix_now();
    let expires_in = tokens.expires_in.unwrap_or(3600);
    let login = !pending.nonce.is_empty();
    let (subject, email, id_token) = if login {
        let id_token = tokens
            .id_token
            .as_deref()
            .ok_or_else(|| PageIndexError::new(ErrorKind::Api, "The token response did not contain an ID token."))?;
        let claims =
            verify_id_token(&client.http, &client.endpoints.jwks, id_token, issued_client_id, Some(&pending.nonce))
                .await?;
        if let Some(existing) = store.by_client_id(issued_client_id)
            && existing.subject != claims.sub
        {
            return Err(PageIndexError::new(
                ErrorKind::Api,
                "The signed-in ChatGPT account does not match the selected registration.",
            ));
        }
        (claims.sub, claims.email, Some(id_token.to_owned()))
    } else {
        let existing = store.by_client_id(issued_client_id).ok_or_else(|| {
            PageIndexError::new(ErrorKind::Api, "Refresh did not match a stored ChatGPT registration.")
        })?;
        let id_token = if let Some(tok) = tokens.id_token.as_deref() {
            let claims = verify_id_token(&client.http, &client.endpoints.jwks, tok, issued_client_id, None).await?;
            if claims.sub != existing.subject {
                return Err(PageIndexError::new(
                    ErrorKind::Api,
                    "The signed-in ChatGPT account does not match the selected registration.",
                ));
            }
            Some(tok.to_owned())
        } else {
            existing.id_token.clone()
        };
        (existing.subject.clone(), existing.email.clone(), id_token)
    };
    let scopes = if tokens.scope.is_some() {
        parse_scopes(tokens.scope.as_deref())
    } else {
        store.by_client_id(issued_client_id).map(|a| a.scopes.clone()).unwrap_or_default()
    };
    let refresh_token =
        tokens.refresh_token.or_else(|| store.by_client_id(issued_client_id).and_then(|a| a.refresh_token.clone()));
    let record = AccountRecord {
        email,
        issuer: ISSUER.into(),
        subject,
        client_id: issued_client_id.to_owned(),
        ext_agent_host_id: store.ext_agent_host_id.clone(),
        id_token,
        access_token: tokens.access_token,
        refresh_token,
        token_type: tokens.token_type.or(Some("Bearer".into())),
        expires_in: Some(expires_in),
        scopes,
        saved_at: Some(rfc3339_utc(now)),
        expires_at: Some(now + expires_in as i64),
    };
    store.upsert(record.clone());
    store::save(&client.store_dir, &store)?;
    Ok(record)
}

pub async fn revoke_refresh(client: &super::ChatGptClient, account: &AccountRecord) -> Result<()> {
    let Some(refresh) = account.refresh_token.as_deref() else {
        return Ok(());
    };
    let form = [("token", refresh), ("token_type_hint", "refresh_token"), ("client_id", account.client_id.as_str())];
    let response = client
        .http
        .post(&client.endpoints.revoke)
        .header("content-type", "application/x-www-form-urlencoded")
        .form(&form)
        .send()
        .await
        .map_err(|e| PageIndexError::new(ErrorKind::Api, format!("revocation request failed: {e}")))?;
    let status = response.status();
    if status.is_success() {
        return Ok(());
    }
    if status.is_server_error() {
        return Err(PageIndexError::new(ErrorKind::Api, format!("revocation returned {status}"))
            .with_status(Some(status.as_u16())));
    }
    Ok(())
}

async fn wait_callback(listener: TcpListener, expected_state: &str) -> Result<Callback> {
    let (mut socket, addr) = listener
        .accept()
        .await
        .map_err(|e| PageIndexError::new(ErrorKind::Io, format!("callback accept failed: {e}")))?;
    if !addr.ip().is_loopback() {
        return Err(PageIndexError::new(ErrorKind::Api, "callback connection was not from loopback"));
    }
    let mut buf = vec![0u8; 8192];
    let n = socket.read(&mut buf).await?;
    let request = String::from_utf8_lossy(&buf[..n]);
    let callback = parse_callback_request(&request)?;
    let body = callback_html(&callback);
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let _ = socket.write_all(response.as_bytes()).await;
    let _ = socket.shutdown().await;
    if callback.state != expected_state {
        return Err(PageIndexError::new(ErrorKind::Api, "Sign-in could not be verified (state mismatch)."));
    }
    Ok(callback)
}

fn parse_callback_request(request: &str) -> Result<Callback> {
    let first = request.lines().next().unwrap_or("");
    let path = first.split_whitespace().nth(1).unwrap_or("");
    let (route, query) = path.split_once('?').unwrap_or((path, ""));
    if route != "/auth/callback" {
        return Err(PageIndexError::new(ErrorKind::Api, format!("unexpected callback path {route}")));
    }
    Ok(parse_callback_query(query))
}

pub fn parse_callback_query(query: &str) -> Callback {
    let mut map: HashMap<String, String> = HashMap::new();
    for part in query.split('&') {
        if let Some((k, v)) = part.split_once('=') {
            map.insert(url_decode(k), url_decode(v));
        }
    }
    Callback {
        code: map.remove("code"),
        state: map.remove("state").unwrap_or_default(),
        client_id: map.remove("client_id"),
        error: map.remove("error"),
        scope: map.remove("scope"),
    }
}

fn callback_html(callback: &Callback) -> String {
    if callback.error.is_some() {
        "<html><body><p>Sign in with ChatGPT was not completed. You can close this window.</p></body></html>".into()
    } else {
        "<html><body><p>PageIndex is signed in with ChatGPT. You can close this window.</p></body></html>".into()
    }
}

fn url_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hex = &input[i + 1..i + 3];
                if let Ok(v) = u8::from_str_radix(hex, 16) {
                    out.push(v);
                    i += 3;
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub fn parse_scopes(scope: Option<&str>) -> Vec<String> {
    scope.unwrap_or("").split([' ', '+']).filter(|s| !s.is_empty()).map(str::to_owned).collect()
}

fn random_value(nbytes: usize) -> String {
    let mut bytes = Vec::with_capacity(nbytes);
    while bytes.len() < nbytes {
        bytes.extend_from_slice(Uuid::new_v4().as_bytes());
    }
    bytes.truncate(nbytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

pub fn unix_now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

pub fn rfc3339_utc(secs: i64) -> String {
    let secs = secs.max(0) as u64;
    let mut days = secs / 86400;
    let tod = secs % 86400;
    let hour = tod / 3600;
    let min = (tod % 3600) / 60;
    let sec = tod % 60;
    let mut year = 1970i32;
    loop {
        let diy = if is_leap(year) { 366 } else { 365 };
        if days < diy {
            break;
        }
        days -= diy;
        year += 1;
    }
    let mdays = [31u64, if is_leap(year) { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 1u32;
    for m in mdays {
        if days < m {
            break;
        }
        days -= m;
        month += 1;
    }
    format!("{year:04}-{month:02}-{:02}T{hour:02}:{min:02}:{sec:02}Z", days + 1)
}

fn is_leap(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}
