//! Config shapes for the client's `index` / `chat` slots, mirroring
//! `pageindex/types.py`. Field names are the flat constructor arguments with
//! the side prefix factored out (`index.model` is `index_model`).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub const PAGEINDEX_CLOUD: &str = "pageindex-cloud";

/// How much of the tree-optimization pass to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OptimizeMode {
    /// Deterministic merge, then LLM expand.
    #[default]
    Full,
    /// Deterministic merge only (no model calls).
    Merge,
    Off,
}

impl std::str::FromStr for OptimizeMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "full" => Ok(Self::Full),
            "merge" => Ok(Self::Merge),
            "off" => Ok(Self::Off),
            other => Err(format!("optimize must be 'full', 'merge', or 'off', got {other:?}")),
        }
    }
}

/// Documents hosted on PageIndex cloud. Without `api_key` it comes from
/// `PAGEINDEX_API_KEY`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CloudIndexConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

/// Documents indexed and stored locally.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LocalIndexConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary_max_words: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary_concurrency: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary_reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_embedded_toc: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimize: Option<OptimizeMode>,
    /// Connection overrides for the indexing lane (`api_base`, `api_key`, ...).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<Backend>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum IndexConfig {
    Cloud(CloudIndexConfig),
    Local(Box<LocalIndexConfig>),
}

/// The chat side: `Local` is your own model running in-process on your
/// keys; `Cloud` is the managed chat.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum ChatConfig {
    Cloud,
    Local {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        backend: Option<Backend>,
    },
}

/// `chat(show_process=...)` display options; every field defaults on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatProcessOptions {
    pub thinking: bool,
    pub tool_call: bool,
    pub tool_result: bool,
    pub max_chars: usize,
}

impl Default for ChatProcessOptions {
    fn default() -> Self {
        Self { thinking: true, tool_call: true, tool_result: true, max_chars: 200 }
    }
}

/// Per-lane connection settings, the LiteLLM `api_base` / `api_key` pair
/// plus extra request parameters merged into every call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Backend {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_config_declares_its_side_by_mode() {
        let local: IndexConfig =
            serde_json::from_str(r#"{"mode": "local", "model": "m", "optimize": "merge"}"#).unwrap();
        assert_eq!(
            local,
            IndexConfig::Local(Box::new(LocalIndexConfig {
                model: Some("m".into()),
                optimize: Some(OptimizeMode::Merge),
                ..Default::default()
            }))
        );
        let cloud: IndexConfig = serde_json::from_str(r#"{"mode": "cloud", "api_key": "k"}"#).unwrap();
        assert_eq!(cloud, IndexConfig::Cloud(CloudIndexConfig { api_key: Some("k".into()) }));
    }

    #[test]
    fn process_options_default_on() {
        let opts: ChatProcessOptions = serde_json::from_str(r#"{"thinking": false}"#).unwrap();
        assert_eq!(opts, ChatProcessOptions { thinking: false, ..Default::default() });
    }
}
