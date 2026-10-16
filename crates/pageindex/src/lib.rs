//! PageIndex: vectorless, reasoning-based document indexing.
//!
//! Builds a table-of-contents tree for a PDF (Flash: from layout statistics,
//! no LLM required) or a Markdown file, optionally refined for search cost
//! and summarized with an LLM, and stores it for retrieval.

pub mod agent;
pub mod chat;
pub mod chat_stream;
pub mod classic;
pub mod client;
pub mod cloud;
pub mod config;
pub mod error;
pub mod flash;
pub mod imaging;
pub mod jsonx;
pub mod llm;
pub mod markdown;
pub mod mcp;
pub mod naming;
pub mod store;
pub mod summary;
pub mod text;
pub mod tree;
pub mod tree_optimize;
pub mod types;

pub use agent::{
    AGENT_INSTRUCTIONS, AgentBackend, CITE_PROMPT, as_anthropic_tools, as_claude_mcp, as_openai_tools, call_tool,
    get_citations, resolve_citations, tool_names,
};
pub use chat::{
    ChatRequest, run_chat_completions, run_chat_stream, run_cloud_chat_stream, run_messages, run_responses,
};
pub use chat_stream::{ChatEvent, ChatStream};
pub use classic::{ClassicOptions, ClassicResult, page_index, page_index_main};
pub use client::{OcrFormat, OcrResult, PageIndexClient, PageIndexLocalClient, SubmitOptions, SubmitResult};
pub use cloud::{CloudApi, PageIndexCloudClient};
pub use config::{Config, ConfigLoader, ConfigOverrides};
pub use error::{ErrorKind, PageIndexError, Result};
pub use flash::{FlashOptions, FlashResult, TocSource, flash_rejection_reason, page_index_flash};
pub use imaging::highlight_region;
pub use llm::{LiteLlm, LlmClient};
pub use markdown::{MdOptions, MdResult, md_to_tree};
pub use mcp::McpBridge;
pub use tree::Node;
pub use tree_optimize::{OptimizeOptions, OptimizeReport, merge_tree, optimize};
pub use types::{
    ChatConfig, ChatProcessOptions, CloudIndexConfig, IndexConfig, LocalIndexConfig, OptimizeMode, PAGEINDEX_CLOUD,
};

/// `PageIndexAPIError`, by its upstream name.
pub type PageIndexAPIError = PageIndexError;
