//! `config.yaml` defaults, model-role resolution, and secret lookup.
//!
//! Secrets are only ever read from the process environment or a local
//! `.env` file (never written back, never stored in config files).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::error::{PageIndexError, Result};

pub const DEFAULT_INDEX_MODEL: &str = "gpt-6-luna";
pub const DEFAULT_CHAT_MODEL: &str = "gpt-6-sol";

const DEFAULT_CONFIG_YAML: &str = include_str!("../config.yaml");

const MODEL_KEYS: [&str; 5] = ["model", "summary_model", "retrieve_model", "index_model", "chat_model"];

/// A resolved configuration: `config.yaml` merged with caller overrides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub model: String,
    pub index_model: String,
    pub summary_model: String,
    pub chat_model: String,
    pub retrieve_model: String,
    pub toc_check_page_num: usize,
    pub max_page_num_each_node: usize,
    pub max_token_num_each_node: usize,
    pub if_add_node_id: bool,
    pub if_add_node_summary: bool,
    pub if_add_doc_description: bool,
    pub if_add_node_text: bool,
}

/// Caller-supplied values; unset fields fall back to the loaded file.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConfigOverrides {
    pub model: Option<String>,
    pub index_model: Option<String>,
    pub summary_model: Option<String>,
    pub chat_model: Option<String>,
    pub retrieve_model: Option<String>,
    pub toc_check_page_num: Option<usize>,
    pub max_page_num_each_node: Option<usize>,
    pub max_token_num_each_node: Option<usize>,
    pub if_add_node_id: Option<bool>,
    pub if_add_node_summary: Option<bool>,
    pub if_add_doc_description: Option<bool>,
    pub if_add_node_text: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct ConfigLoader {
    defaults: HashMap<String, serde_yaml::Value>,
}

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::from_yaml(DEFAULT_CONFIG_YAML).expect("bundled config.yaml is valid")
    }
}

impl ConfigLoader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_yaml(&std::fs::read_to_string(path)?)
    }

    pub fn from_yaml(text: &str) -> Result<Self> {
        let value: serde_yaml::Value =
            serde_yaml::from_str(text).map_err(|e| PageIndexError::invalid(format!("Invalid config: {e}")))?;
        let defaults = match value {
            serde_yaml::Value::Null => HashMap::new(),
            serde_yaml::Value::Mapping(map) => {
                map.into_iter().filter_map(|(k, v)| k.as_str().map(|k| (k.to_owned(), v))).collect()
            }
            _ => return Err(PageIndexError::invalid("Invalid config: expected a mapping")),
        };
        let known = Self::default_keys();
        let mut unknown: Vec<&str> =
            defaults.keys().map(String::as_str).filter(|k| !known.contains(k) && !MODEL_KEYS.contains(k)).collect();
        if !unknown.is_empty() {
            unknown.sort_unstable();
            return Err(PageIndexError::invalid(format!("Unknown config keys: {unknown:?}")));
        }
        Ok(Self { defaults })
    }

    fn default_keys() -> [&'static str; 7] {
        [
            "toc_check_page_num",
            "max_page_num_each_node",
            "max_token_num_each_node",
            "if_add_node_id",
            "if_add_node_summary",
            "if_add_doc_description",
            "if_add_node_text",
        ]
    }

    pub fn load(&self, overrides: &ConfigOverrides) -> Config {
        let text = |key: &str| self.defaults.get(key).and_then(|v| v.as_str()).map(str::to_owned);
        let int = |key: &str, fallback: usize| {
            self.defaults.get(key).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(fallback)
        };
        let flag = |key: &str, fallback: bool| self.defaults.get(key).map(yes_no).unwrap_or(fallback);

        let given = |over: &Option<String>, key: &str| over.clone().or_else(|| text(key)).filter(|s| !s.is_empty());
        let model = given(&overrides.model, "model");
        let index_model = given(&overrides.index_model, "index_model");
        let summary_model = given(&overrides.summary_model, "summary_model");
        let chat_model = given(&overrides.chat_model, "chat_model");
        let retrieve_model = given(&overrides.retrieve_model, "retrieve_model");

        let index = index_model.clone().or_else(|| model.clone()).unwrap_or_else(|| DEFAULT_INDEX_MODEL.into());
        let summary =
            summary_model.or(index_model).or_else(|| model.clone()).unwrap_or_else(|| DEFAULT_INDEX_MODEL.into());
        let chat = chat_model.or(retrieve_model).or(model).unwrap_or_else(|| DEFAULT_CHAT_MODEL.into());

        Config {
            model: index.clone(),
            index_model: index,
            summary_model: summary,
            chat_model: chat.clone(),
            retrieve_model: chat,
            toc_check_page_num: overrides.toc_check_page_num.unwrap_or_else(|| int("toc_check_page_num", 20)),
            max_page_num_each_node: overrides
                .max_page_num_each_node
                .unwrap_or_else(|| int("max_page_num_each_node", 10)),
            max_token_num_each_node: overrides
                .max_token_num_each_node
                .unwrap_or_else(|| int("max_token_num_each_node", 20000)),
            if_add_node_id: overrides.if_add_node_id.unwrap_or_else(|| flag("if_add_node_id", true)),
            if_add_node_summary: overrides.if_add_node_summary.unwrap_or_else(|| flag("if_add_node_summary", true)),
            if_add_doc_description: overrides
                .if_add_doc_description
                .unwrap_or_else(|| flag("if_add_doc_description", false)),
            if_add_node_text: overrides.if_add_node_text.unwrap_or_else(|| flag("if_add_node_text", false)),
        }
    }
}

fn yes_no(value: &serde_yaml::Value) -> bool {
    match value {
        serde_yaml::Value::Bool(b) => *b,
        serde_yaml::Value::String(s) => parse_yes_no(s).unwrap_or(false),
        _ => false,
    }
}

/// `"yes"` / `"no"` (and boolean spellings), as the CLI's `--if-add-*` flags take.
pub fn parse_yes_no(text: &str) -> Option<bool> {
    match text.trim().to_ascii_lowercase().as_str() {
        "yes" | "y" | "true" | "1" | "on" => Some(true),
        "no" | "n" | "false" | "0" | "off" => Some(false),
        _ => None,
    }
}

static DOTENV: OnceLock<HashMap<String, String>> = OnceLock::new();

fn dotenv() -> &'static HashMap<String, String> {
    DOTENV.get_or_init(|| find_dotenv().map(|path| parse_dotenv(&path)).unwrap_or_default())
}

fn find_dotenv() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let candidate = dir.join(".env");
        if candidate.is_file() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn parse_dotenv(path: &Path) -> HashMap<String, String> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return HashMap::new();
    };
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            let line = line.strip_prefix("export ").unwrap_or(line);
            if line.starts_with('#') {
                return None;
            }
            let (key, value) = line.split_once('=')?;
            let value = value.trim();
            let value = value
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
                .unwrap_or(value);
            Some((key.trim().to_owned(), value.to_owned()))
        })
        .collect()
}

/// A secret or setting from the environment, then from `.env`. Present but
/// empty in the environment wins over `.env` (and reads as unset).
pub fn env_var(key: &str) -> Option<String> {
    match std::env::var(key) {
        Ok(value) => Some(value).filter(|v| !v.is_empty()),
        Err(_) => dotenv().get(key).cloned().filter(|v| !v.is_empty()),
    }
}

/// The OpenAI key, honoring the deprecated `CHATGPT_API_KEY` spelling.
pub fn openai_api_key() -> Option<String> {
    env_var("OPENAI_API_KEY").or_else(|| env_var("CHATGPT_API_KEY"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_config_yaml() {
        let cfg = ConfigLoader::new().load(&ConfigOverrides::default());
        assert_eq!(cfg.toc_check_page_num, 20);
        assert_eq!(cfg.max_page_num_each_node, 10);
        assert_eq!(cfg.max_token_num_each_node, 20000);
        assert!(cfg.if_add_node_id && cfg.if_add_node_summary);
        assert!(!cfg.if_add_doc_description && !cfg.if_add_node_text);
        assert_eq!(cfg.model, DEFAULT_INDEX_MODEL);
        assert_eq!(cfg.summary_model, DEFAULT_INDEX_MODEL);
        assert_eq!(cfg.chat_model, DEFAULT_CHAT_MODEL);
    }

    #[test]
    fn model_roles_resolve_specific_over_general() {
        let loader = ConfigLoader::new();
        let cfg = loader.load(&ConfigOverrides { model: Some("m".into()), ..Default::default() });
        assert_eq!((cfg.index_model.as_str(), cfg.summary_model.as_str(), cfg.chat_model.as_str()), ("m", "m", "m"));
        let cfg = loader.load(&ConfigOverrides {
            model: Some("m".into()),
            index_model: Some("i".into()),
            retrieve_model: Some("r".into()),
            ..Default::default()
        });
        assert_eq!((cfg.model.as_str(), cfg.summary_model.as_str(), cfg.chat_model.as_str()), ("i", "i", "r"));
    }

    #[test]
    fn a_file_summary_model_outranks_a_cli_model() {
        let loader = ConfigLoader::from_yaml(&format!("{DEFAULT_CONFIG_YAML}\nsummary_model: yaml-summary\n")).unwrap();
        let cfg = loader.load(&ConfigOverrides { model: Some("cli-model".into()), ..Default::default() });
        assert_eq!(cfg.summary_model, "yaml-summary");
    }

    #[test]
    fn unknown_keys_are_rejected() {
        let err = ConfigLoader::from_yaml("toc_check_page_num: 3\nbogus: 1\n").unwrap_err();
        assert!(err.message.contains("bogus"));
    }
}
