//! Local credential file: owner-only permissions, never logged or committed.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::env_var;
use crate::error::{ErrorKind, PageIndexError, Result};

/// App display name sent as `agent_name_hint` on first registration only.
pub const AGENT_NAME_HINT: &str = "PageIndex";

const STORE_FILE: &str = "chatgpt.json";

/// One issued-client registration, matching the documented credential record.
#[derive(Clone, Serialize, Deserialize)]
pub struct AccountRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    pub issuer: String,
    pub subject: String,
    pub client_id: String,
    pub ext_agent_host_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub access_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_in: Option<u64>,
    #[serde(default)]
    pub scopes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saved_at: Option<String>,
    /// Unix seconds when the current access token expires (derived, for refresh).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
}

impl AccountRecord {
    pub fn plan_usage_enabled(&self) -> bool {
        self.scopes.iter().any(|s| s == "chatgpt.tokens.use.direct") && self.access_token.is_some()
    }

    pub fn signed_in(&self) -> bool {
        self.access_token.is_some() || self.refresh_token.is_some()
    }

    pub fn access_expired(&self, now: i64, skew_secs: i64) -> bool {
        match self.expires_at {
            Some(exp) => now + skew_secs >= exp,
            None => true,
        }
    }

    pub fn clear_tokens(&mut self) {
        self.id_token = None;
        self.access_token = None;
        self.refresh_token = None;
        self.token_type = None;
        self.expires_in = None;
        self.saved_at = None;
        self.expires_at = None;
    }
}

impl std::fmt::Debug for AccountRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccountRecord")
            .field("email", &self.email)
            .field("issuer", &self.issuer)
            .field("subject", &self.subject)
            .field("client_id", &self.client_id)
            .field("ext_agent_host_id", &self.ext_agent_host_id)
            .field("id_token", &self.id_token.as_ref().map(|_| "<redacted>"))
            .field("access_token", &self.access_token.as_ref().map(|_| "<redacted>"))
            .field("refresh_token", &self.refresh_token.as_ref().map(|_| "<redacted>"))
            .field("token_type", &self.token_type)
            .field("expires_in", &self.expires_in)
            .field("scopes", &self.scopes)
            .field("saved_at", &self.saved_at)
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CredentialStore {
    pub ext_agent_host_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_client_id: Option<String>,
    #[serde(default)]
    pub accounts: Vec<AccountRecord>,
}

impl CredentialStore {
    pub fn active(&self) -> Option<&AccountRecord> {
        let id = self.active_client_id.as_deref()?;
        self.accounts.iter().find(|a| a.client_id == id)
    }

    pub fn active_mut(&mut self) -> Option<&mut AccountRecord> {
        let id = self.active_client_id.clone()?;
        self.accounts.iter_mut().find(|a| a.client_id == id)
    }

    pub fn by_client_id(&self, client_id: &str) -> Option<&AccountRecord> {
        self.accounts.iter().find(|a| a.client_id == client_id)
    }

    pub fn upsert(&mut self, record: AccountRecord) {
        if let Some(existing) = self.accounts.iter_mut().find(|a| a.client_id == record.client_id) {
            *existing = record.clone();
        } else {
            self.accounts.push(record.clone());
        }
        self.active_client_id = Some(record.client_id);
        self.ext_agent_host_id = record.ext_agent_host_id;
    }
}

pub fn default_config_dir() -> PathBuf {
    if let Some(dir) = env_var("PAGEINDEX_CONFIG_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME")
        && !xdg.is_empty()
    {
        return PathBuf::from(xdg).join("pageindex");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".config").join("pageindex")
}

pub fn store_path(dir: &Path) -> PathBuf {
    dir.join(STORE_FILE)
}

pub fn load(dir: &Path) -> Result<Option<CredentialStore>> {
    let path = store_path(dir);
    if !path.is_file() {
        return Ok(None);
    }
    let text = fs::read_to_string(&path)?;
    let store: CredentialStore = serde_json::from_str(&text)
        .map_err(|e| PageIndexError::new(ErrorKind::Io, format!("invalid ChatGPT credential file: {e}")))?;
    Ok(Some(store))
}

pub fn save(dir: &Path, store: &CredentialStore) -> Result<()> {
    fs::create_dir_all(dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(dir, fs::Permissions::from_mode(0o700));
    }
    let json = serde_json::to_vec_pretty(store)?;
    write_private_file(&store_path(dir), &json)
}

pub fn load_or_init(dir: &Path) -> Result<CredentialStore> {
    if let Some(store) = load(dir)? {
        return Ok(store);
    }
    let store = CredentialStore { ext_agent_host_id: new_host_id(), active_client_id: None, accounts: Vec::new() };
    save(dir, &store)?;
    Ok(store)
}

pub fn new_host_id() -> String {
    format!("urn:uuid:{}", uuid::Uuid::new_v4())
}

fn write_private_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("json.tmp");
    {
        let mut opts = OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut file = opts.open(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}
