use std::sync::{Arc, Mutex};
use std::time::Duration;

use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use serde::Serialize;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::llm::{ChatMessage, CompletionParams, LiteLlm, LlmClient};

use super::oauth::{self, Callback};
use super::store::{self, AccountRecord};
use super::{ChatGptClient, DYNAMIC_CLIENT_ID, Endpoints, ISSUER, RESOURCE};

const TEST_KID: &str = "pageindex-test";
const TEST_N: &str = "whTki_7bgyw5KSGXZKrL3IaTtmdEtCcpWTAt4LhWGq7fIZnF9b1aPNF4K4RFCk2_6K9Q-GLfRcy-2mMC_Uk-17jhQbwEBKrtQQR2vbk7GgLpgcsYr3Z5ZauGTDQ4Zvrgyi_7Irxipv19FpZIdzZKU3N-MuOMXAdJR869Hfgm0tuMLVHYr_ppn4vacfshFkpRATQCZtmYVZb-QU3OLGKQzKizoX1nd7CFfUD1MhXv7jx4cax-CjtGuA8uz7CcLF_TpSMpfEGiwZb3r8fqxHk0J-gvsmUsMTKeEwu-8xj_Dq9GAjZeMc0D5cuR0tNp7fY2NByrw29hTuyn6W1zf-xBjw";
const TEST_E: &str = "AQAB";
const TEST_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDCFOSL/tuDLDkp
IZdkqsvchpO2Z0S0JylZMC3guFYart8hmcX1vVo80XgrhEUKTb/or1D4Yt9FzL7a
YwL9ST7XuOFBvAQEqu1BBHa9uTsaAumByxivdnllq4ZMNDhm+uDKL/sivGKm/X0W
lkh3NkpTc34y44xcB0lHzr0d+CbS24wtUdiv+mmfi9px+yEWSlEBNAJm2ZhVlv5B
Tc4sYpDMqLOhfWd3sIV9QPUyFe/uPHhxrH4KO0a4Dy7PsJwsX9OlIyl8QaLBlvev
x+rEeTQn6C+yZSwxMp4TC77zGP8Or0YCNl4xzQPly5HS02nt9jY0HKvDb2FO7Kfp
bXN/7EGPAgMBAAECggEAKeM5em+Mb/Cc+6AY3SyxqAxRmBYMsXccSPgJhs0yJDsm
xEk8fO+bqLGq5nV7MyCZsT4ZUyp/tsBcVqT08babvhJUZ4wNL/a2EwjEGel+9Hu5
av5q+v+ubr+I0iyZGby+xFR7tsEC57ihBT11MWmBvCYakCAlXH1kFv/rOCslMe+4
1T/zRME55wH8gEE0WdZVRFBB70XewJpwWUTJd3Vn93fyzVhoeXwJ0Yx1hJ0wWWBI
WfE9Ev9SZODqAuJkSnVaBOfPMQOadXWEuurYaJkFjEmvJyG/+CNkGFqSZ6kAF00W
0mA276qH4cjCdEfaym32sQB5H1cZW37F2FiRe+lnAQKBgQD1au1ZTdCqV/tGgFR6
VPWfeHUZP2rfFXWniufX+8r7ryED6LLngNmj0qmJ43n24tlpAe2nR+W8+kgo/Plv
U1TmYJ7jlrSTi0wQd1v3mP0CVRWmMVQy/o9DGMXgDW9EUAohHijl5KnTLPAXkmZc
Plbs9MA1+HfwUEFpHQO2asdhAQKBgQDKc0lCJrCuRXnx76776OjpoQsHv+taLWvq
iBSqin3f1HUMduFKJiijSXMaPaD1sevygFctVwuyTB6o3up7f9ShaRRf8beZ9QmS
X7aJ47lnCaq9G4PdDAbQTvRnH5ccNf69kufPO4cfYK1I6c7eIpKkUBKRr8crA3+9
CUWt+rsSjwKBgQCwpGTv6V1urmLzDL1BS9N7E9Wc9ZNzvFtx2z4irF4ep5/o012p
2YlNK/zRfcS0mfUIatXmmI5YZgD4V5UltuId65+RilCw/gsisvakER7SxfEOLOVA
yw7MqZELQkA6+C1CRxYC2oINuIObQ8mfTVN9vHB4qD2T4AAkqziX3OVjAQKBgQC/
AVY021wGj/YllQV7FpvpEM/fV+stAX23oPweituHumNUhzS3SJk+taWrNEoQhPUf
roaReHbImZizqt1/93Mddo6Klvuqfc2yFCucnQXneO9ly/f1Lcuf7V4ZjWx8aVUW
duvOUVOC/yt+h1ltqyC/deyPF1DJ/FsgdwcOxmiJcQKBgQCDFqPohdjwwJw68qjZ
QTU4yswPaH23bDQkvt+Cs9LYSWQcyyFUjB4rjaYSOJJXMbXhRiAq2A9gJLWKKhJZ
eWuwX59r9mCBx6fIHXhvTrh2YnD6g/jz0b6ADHj3MUc1/J1JAizKmuadmvNUch2f
PLDhhGrvRmkSqnovQ/1B3UYPgQ==
-----END PRIVATE KEY-----
";

const ISSUED: &str = "oaiapp_pageindex_test";
const SCOPES: &str = "chatgpt.tokens.use.direct email offline_access openid profile resource.invoke";

struct Mock {
    addr: std::net::SocketAddr,
    state: Arc<MockState>,
    task: tokio::task::JoinHandle<()>,
}

struct MockState {
    jwks: Value,
    token: Mutex<Value>,
    token_status: Mutex<u16>,
    sse: Mutex<String>,
    responses_status: Mutex<u16>,
    revoke_status: Mutex<u16>,
    token_forms: Mutex<Vec<String>>,
    response_bodies: Mutex<Vec<Value>>,
    revoke_forms: Mutex<Vec<String>>,
}

impl Drop for Mock {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl Mock {
    fn auth_base(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn api_base(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    fn client(&self, dir: &std::path::Path) -> ChatGptClient {
        ChatGptClient::new(Endpoints::from_bases(&self.auth_base(), &self.api_base()), dir.to_path_buf())
    }
}

fn test_jwks() -> Value {
    json!({"keys":[{"kty":"RSA","use":"sig","alg":"RS256","kid":TEST_KID,"n":TEST_N,"e":TEST_E}]})
}

fn hello_sse() -> String {
    "event: response.output_text.delta\n\
     data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\
     \n\
     event: response.completed\n\
     data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\"}]}]}}\n\
     \n"
        .into()
}

async fn spawn_mock() -> Mock {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = Arc::new(MockState {
        jwks: test_jwks(),
        token: Mutex::new(json!({})),
        token_status: Mutex::new(200),
        sse: Mutex::new(hello_sse()),
        responses_status: Mutex::new(200),
        revoke_status: Mutex::new(200),
        token_forms: Mutex::new(Vec::new()),
        response_bodies: Mutex::new(Vec::new()),
        revoke_forms: Mutex::new(Vec::new()),
    });
    let task_state = state.clone();
    let task = tokio::spawn(async move {
        loop {
            let Ok((sock, _)) = listener.accept().await else { break };
            let state = task_state.clone();
            tokio::spawn(async move {
                handle_conn(sock, &state).await;
            });
        }
    });
    Mock { addr, state, task }
}

async fn handle_conn(mut sock: TcpStream, state: &MockState) {
    let Some((method, path, body)) = read_http(&mut sock).await else {
        return;
    };
    let path = path.split('?').next().unwrap_or(&path);
    let (status, content_type, payload) = if method == "GET" && path == "/.well-known/jwks.json" {
        (200, "application/json", state.jwks.to_string())
    } else if method == "POST" && path == "/api/accounts/oauth/token" {
        state.token_forms.lock().unwrap().push(String::from_utf8_lossy(&body).into_owned());
        let status = *state.token_status.lock().unwrap();
        (status, "application/json", state.token.lock().unwrap().to_string())
    } else if method == "POST" && path == "/api/accounts/oauth/revoke" {
        state.revoke_forms.lock().unwrap().push(String::from_utf8_lossy(&body).into_owned());
        (*state.revoke_status.lock().unwrap(), "text/plain", String::new())
    } else if method == "POST" && path == "/v1/responses" {
        if let Ok(value) = serde_json::from_slice::<Value>(&body) {
            state.response_bodies.lock().unwrap().push(value);
        }
        let status = *state.responses_status.lock().unwrap();
        (status, "text/event-stream", state.sse.lock().unwrap().clone())
    } else {
        (404, "application/json", json!({"error":"not_found"}).to_string())
    };
    let phrase = if status == 200 { "OK" } else { "ERR" };
    let response = format!(
        "HTTP/1.1 {status} {phrase}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    );
    let _ = sock.write_all(response.as_bytes()).await;
    let _ = sock.shutdown().await;
}

async fn read_http(sock: &mut TcpStream) -> Option<(String, String, Vec<u8>)> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 2048];
    loop {
        let n = sock.read(&mut tmp).await.ok()?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        let Some(idx) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
            if buf.len() > 65_536 {
                return None;
            }
            continue;
        };
        let headers = String::from_utf8_lossy(&buf[..idx]);
        let first = headers.lines().next().unwrap_or("");
        let mut parts = first.split_whitespace();
        let method = parts.next()?.to_owned();
        let path = parts.next()?.to_owned();
        let len = headers
            .lines()
            .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
            .and_then(|l| l.split(':').nth(1))
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0);
        let mut body = buf[idx + 4..].to_vec();
        while body.len() < len {
            let n = sock.read(&mut tmp).await.ok()?;
            if n == 0 {
                break;
            }
            body.extend_from_slice(&tmp[..n]);
        }
        body.truncate(len);
        return Some((method, path, body));
    }
    None
}

#[derive(Serialize)]
struct TestClaims {
    sub: String,
    iss: String,
    aud: String,
    exp: i64,
    iat: i64,
    nonce: String,
    email: String,
}

fn mint_id_token(client_id: &str, nonce: &str, sub: &str) -> String {
    let now = oauth::unix_now();
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some(TEST_KID.into());
    let claims = TestClaims {
        sub: sub.into(),
        iss: ISSUER.into(),
        aud: client_id.into(),
        exp: now + 3600,
        iat: now,
        nonce: nonce.into(),
        email: "user@example.com".into(),
    };
    encode(&header, &claims, &EncodingKey::from_rsa_pem(TEST_PEM.as_bytes()).unwrap()).unwrap()
}

fn token_json(id_token: &str) -> Value {
    json!({
        "access_token": "access-1",
        "refresh_token": "refresh-1",
        "id_token": id_token,
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": SCOPES,
    })
}

fn signed_in_record(dir_host: &str) -> AccountRecord {
    AccountRecord {
        email: Some("user@example.com".into()),
        issuer: ISSUER.into(),
        subject: "sub-1".into(),
        client_id: ISSUED.into(),
        ext_agent_host_id: dir_host.into(),
        id_token: Some("id".into()),
        access_token: Some("access-live".into()),
        refresh_token: Some("refresh-live".into()),
        token_type: Some("Bearer".into()),
        expires_in: Some(3600),
        scopes: SCOPES.split_whitespace().map(str::to_owned).collect(),
        saved_at: Some(oauth::rfc3339_utc(oauth::unix_now())),
        expires_at: Some(oauth::unix_now() + 3600),
    }
}

#[test]
fn production_endpoints_are_the_documented_public_urls() {
    let e = Endpoints::production();
    assert_eq!(e.authorize, "https://auth.openai.com/api/accounts/authorize");
    assert_eq!(e.token, "https://auth.openai.com/api/accounts/oauth/token");
    assert_eq!(e.jwks, "https://auth.openai.com/.well-known/jwks.json");
    assert_eq!(e.revoke, "https://auth.openai.com/api/accounts/oauth/revoke");
    assert_eq!(e.api_base, "https://api.openai.com/v1");
    assert!(!e.api_base.contains("chatgpt.com"));
    assert!(!e.api_base.contains("backend-api"));
    assert_eq!(DYNAMIC_CLIENT_ID, "dynamic_agent_client");
    assert_eq!(RESOURCE, "https://api.openai.com/v1");
}

#[test]
fn first_registration_authorize_url_matches_the_docs() {
    let pending = oauth::new_pending("http://127.0.0.1:1455/auth/callback".into(), DYNAMIC_CLIENT_ID.into(), true);
    let url =
        oauth::build_authorize_url(&Endpoints::production(), "urn:uuid:11111111-1111-4111-8111-111111111111", &pending);
    assert!(url.starts_with("https://auth.openai.com/api/accounts/authorize?"));
    assert!(url.contains("client_id=dynamic_agent_client"));
    assert!(url.contains("agent_name_hint=PageIndex"));
    assert!(url.contains("ext_agent_host_id=urn%3Auuid%3A11111111-1111-4111-8111-111111111111"));
    assert!(url.contains("response_type=code"));
    assert!(url.contains("redirect_uri=http%3A%2F%2F127.0.0.1%3A1455%2Fauth%2Fcallback"));
    assert!(url.contains("code_challenge_method=S256"));
    assert!(url.contains("resource=https%3A%2F%2Fapi.openai.com%2Fv1"));
    assert!(url.contains("openid"));
    assert!(url.contains("offline_access"));
    assert!(url.contains("resource.invoke"));
    assert!(url.contains("chatgpt.tokens.use.direct"));
    assert!(!url.contains("localhost"));
    assert!(!url.contains("backend-api"));
    let reauth = oauth::new_pending("http://127.0.0.1:54321/auth/callback".into(), ISSUED.into(), false);
    let url = oauth::build_authorize_url(&Endpoints::production(), "urn:uuid:abc", &reauth);
    assert!(url.contains(&format!("client_id={ISSUED}")));
    assert!(!url.contains("agent_name_hint"));
}

#[test]
fn redacts_id_token_hint_from_printed_urls() {
    let redacted =
        oauth::redact_authorize_url("https://auth.openai.com/api/accounts/authorize?foo=1&id_token_hint=SECRET&bar=2");
    assert!(!redacted.contains("SECRET"));
    assert!(redacted.contains("foo=1") && redacted.contains("bar=2"));
}

#[test]
fn rfc3339_is_utc() {
    assert_eq!(oauth::rfc3339_utc(0), "1970-01-01T00:00:00Z");
    assert_eq!(oauth::rfc3339_utc(1_000_000_000), "2001-09-09T01:46:40Z");
}

#[test]
fn chatgpt_model_prefix_is_opt_in() {
    assert_eq!(super::chatgpt_model("gpt-6.1-sol").unwrap(), "chatgpt/gpt-6.1-sol");
    assert_eq!(super::chatgpt_model("openai/gpt-6.1-sol").unwrap(), "chatgpt/gpt-6.1-sol");
    assert_eq!(super::chatgpt_model("chatgpt/gpt-6.1-sol").unwrap(), "chatgpt/gpt-6.1-sol");
    assert!(super::chatgpt_model("anthropic/claude").is_err());
}

#[tokio::test]
async fn login_exchanges_pkce_validates_id_token_and_writes_0600_store() {
    let mock = spawn_mock().await;
    let dir = tempfile::tempdir().unwrap();
    let client = mock.client(dir.path());
    let pending = oauth::new_pending("http://127.0.0.1:1455/auth/callback".into(), DYNAMIC_CLIENT_ID.into(), true);
    let id_token = mint_id_token(ISSUED, &pending.nonce, "sub-1");
    *mock.state.token.lock().unwrap() = token_json(&id_token);
    let account = client
        .complete_authorization(
            &pending,
            Callback {
                code: Some("auth-code".into()),
                state: pending.state.clone(),
                client_id: Some(ISSUED.into()),
                error: None,
                scope: None,
            },
        )
        .await
        .unwrap();
    assert_eq!(account.email.as_deref(), Some("user@example.com"));
    assert_eq!(account.client_id, ISSUED);
    assert!(account.plan_usage_enabled());
    let form = mock.state.token_forms.lock().unwrap()[0].clone();
    assert!(form.contains("grant_type=authorization_code"));
    assert!(form.contains(&format!("client_id={ISSUED}")));
    assert!(form.contains("code=auth-code"));
    assert!(form.contains("code_verifier="));
    assert!(form.contains("redirect_uri="));
    assert!(form.contains("resource=https%3A%2F%2Fapi.openai.com%2Fv1"));
    assert!(!form.contains("client_secret"));
    assert!(!form.contains(DYNAMIC_CLIENT_ID));
    let path = store::store_path(dir.path());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }
    let on_disk = std::fs::read_to_string(&path).unwrap();
    assert!(on_disk.contains("access-1"));
    assert!(client.status().unwrap().render().contains("user@example.com"));
}

#[tokio::test]
async fn login_rejects_bad_nonce_and_missing_issued_client_id() {
    let mock = spawn_mock().await;
    let dir = tempfile::tempdir().unwrap();
    let client = mock.client(dir.path());
    let pending = oauth::new_pending("http://127.0.0.1:1455/auth/callback".into(), DYNAMIC_CLIENT_ID.into(), true);
    let id_token = mint_id_token(ISSUED, "other-nonce", "sub-1");
    *mock.state.token.lock().unwrap() = token_json(&id_token);
    let err = client
        .complete_authorization(
            &pending,
            Callback {
                code: Some("auth-code".into()),
                state: pending.state.clone(),
                client_id: Some(ISSUED.into()),
                error: None,
                scope: None,
            },
        )
        .await
        .unwrap_err();
    assert!(err.message.contains("nonce"));
    let err = client
        .complete_authorization(
            &pending,
            Callback {
                code: Some("auth-code".into()),
                state: pending.state.clone(),
                client_id: None,
                error: None,
                scope: None,
            },
        )
        .await
        .unwrap_err();
    assert!(err.message.contains("issued client_id"));
}

#[tokio::test]
async fn refresh_then_responses_sends_only_supported_fields() {
    let mock = spawn_mock().await;
    let dir = tempfile::tempdir().unwrap();
    let mut store = store::load_or_init(dir.path()).unwrap();
    let mut record = signed_in_record(&store.ext_agent_host_id);
    record.expires_at = Some(oauth::unix_now() - 10);
    record.access_token = Some("stale".into());
    store.upsert(record);
    store::save(dir.path(), &store).unwrap();
    *mock.state.token.lock().unwrap() = json!({
        "access_token": "access-2",
        "refresh_token": "refresh-2",
        "token_type": "Bearer",
        "expires_in": 3600,
    });
    let client = mock.client(dir.path());
    let llm = LiteLlm::new().with_chatgpt(client.clone());
    let text = llm
        .complete(
            "chatgpt/gpt-6.1-sol",
            &[ChatMessage::system("Be brief."), ChatMessage::user("Hi")],
            &CompletionParams { temperature: Some(0.2), max_tokens: Some(16), ..Default::default() },
        )
        .await
        .unwrap();
    assert_eq!(text, "Hello");
    let form = mock.state.token_forms.lock().unwrap()[0].clone();
    assert!(form.contains("grant_type=refresh_token"));
    assert!(form.contains(&format!("client_id={ISSUED}")));
    assert!(form.contains("refresh_token=refresh-live"));
    assert!(form.contains("resource=https%3A%2F%2Fapi.openai.com%2Fv1"));
    assert!(!form.contains("scope="));
    assert!(!form.contains(DYNAMIC_CLIENT_ID));
    let body = mock.state.response_bodies.lock().unwrap()[0].clone();
    assert_eq!(body["store"], false);
    assert_eq!(body["stream"], true);
    assert_eq!(body["model"], "gpt-6.1-sol");
    assert_eq!(body["instructions"], "Be brief.");
    assert!(body.get("temperature").is_none());
    assert!(body.get("max_tokens").is_none());
    assert!(body.get("max_output_tokens").is_none());
    assert!(body.get("previous_response_id").is_none());
    assert!(body.get("tools").is_none());
}

#[tokio::test]
async fn logout_revokes_and_clears_tokens_but_keeps_client_mapping() {
    let mock = spawn_mock().await;
    let dir = tempfile::tempdir().unwrap();
    let mut store = store::load_or_init(dir.path()).unwrap();
    let record = signed_in_record(&store.ext_agent_host_id);
    store.upsert(record);
    store::save(dir.path(), &store).unwrap();
    let client = mock.client(dir.path());
    let out = client.logout().await.unwrap();
    assert!(out.revoked);
    let form = mock.state.revoke_forms.lock().unwrap()[0].clone();
    assert!(form.contains("token=refresh-live"));
    assert!(form.contains("token_type_hint=refresh_token"));
    assert!(form.contains(&format!("client_id={ISSUED}")));
    let stored = store::load(dir.path()).unwrap().unwrap();
    let account = stored.by_client_id(ISSUED).unwrap();
    assert!(account.access_token.is_none() && account.refresh_token.is_none() && account.id_token.is_none());
    assert_eq!(account.client_id, ISSUED);
    assert_eq!(stored.ext_agent_host_id, account.ext_agent_host_id);
    assert!(!client.status().unwrap().signed_in);
}

#[tokio::test]
async fn loopback_callback_completes_login() {
    let mock = spawn_mock().await;
    let dir = tempfile::tempdir().unwrap();
    let client = mock.client(dir.path());
    let flow = client.begin_login().await.unwrap();
    assert!(flow.redirect_uri.starts_with("http://127.0.0.1:"));
    assert!(flow.redirect_uri.ends_with("/auth/callback"));
    assert!(!flow.redirect_uri.contains("localhost"));
    let id_token = mint_id_token(ISSUED, &flow.nonce, "sub-loop");
    *mock.state.token.lock().unwrap() = token_json(&id_token);
    let callback_url = format!("{}?code=loop-code&state={}&client_id={ISSUED}", flow.redirect_uri, flow.state);
    let finish = flow.finish(&client, Duration::from_secs(5));
    let ping = async {
        tokio::time::sleep(Duration::from_millis(50)).await;
        reqwest::Client::new().get(&callback_url).send().await.unwrap()
    };
    let (account, page) = tokio::join!(finish, ping);
    let account = account.unwrap();
    assert_eq!(account.client_id, ISSUED);
    assert!(page.status().is_success());
    assert!(page.text().await.unwrap().contains("signed in"));
}
