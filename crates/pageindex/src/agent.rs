//! Agent tools: the cloud MCP tool contract, executed against a PageIndex client.

use std::collections::{BTreeMap, HashSet};
use std::sync::OnceLock;

use async_trait::async_trait;
use regex::Regex;
use serde_json::{Value, json};

use crate::error::{PageIndexError, Result};
use crate::mcp::{McpBridge, render_prompt_text, render_text};

pub const TOOL_RESPONSE_CHAR_LIMIT: usize = 100_000;
pub const STRUCTURE_FIRST_PAGE_THRESHOLD: usize = 20;
pub const PAGEINDEX_BASE_URL: &str = "https://api.pageindex.ai";

const CHAR_BUDGET: usize = TOOL_RESPONSE_CHAR_LIMIT * 95 / 100;
const MAX_REQUESTED_PAGES: usize = 10_000;
const SIMILAR_NAMES_LIMIT: usize = 3;

const READ_TOOLS: &[&str] = &["browse_documents", "get_document", "get_document_structure", "get_page_content"];
const MANAGEMENT_TOOLS: &[&str] = &["remove_document"];

const STRUCTURE_KEY_ORDER: &[&str] =
    &["title", "node_id", "start_index", "end_index", "page_index", "prefix_summary", "summary", "nodes"];

/// Document store operations the tool layer needs, local or cloud.
#[async_trait]
pub trait AgentBackend: Send + Sync {
    fn api_key(&self) -> Option<&str>;
    fn instructions(&self) -> Option<&str> {
        None
    }
    fn base_url(&self) -> &str {
        PAGEINDEX_BASE_URL
    }

    async fn list_documents(&self, limit: usize, offset: usize) -> Result<Value>;
    async fn get_document(&self, doc_id: &str) -> Result<Value>;
    async fn get_tree(&self, doc_id: &str, node_summary: bool, include_text: bool) -> Result<Value>;
    async fn get_ocr_pages(&self, doc_id: &str) -> Result<Value>;
    async fn delete_document(&self, doc_id: &str) -> Result<()>;
    async fn get_block(&self, doc_id: &str, block_id: &str) -> Result<Value> {
        let _ = (doc_id, block_id);
        Err(PageIndexError::api("get_block is cloud-only — page images and layout blocks live on PageIndex cloud."))
    }
    async fn list_folders(&self) -> Result<Value> {
        Err(PageIndexError::api("Folders are cloud-only."))
    }
}

#[async_trait]
impl<T: AgentBackend + ?Sized> AgentBackend for std::sync::Arc<T> {
    fn api_key(&self) -> Option<&str> {
        (**self).api_key()
    }
    fn instructions(&self) -> Option<&str> {
        (**self).instructions()
    }
    fn base_url(&self) -> &str {
        (**self).base_url()
    }
    async fn list_documents(&self, limit: usize, offset: usize) -> Result<Value> {
        (**self).list_documents(limit, offset).await
    }
    async fn get_document(&self, doc_id: &str) -> Result<Value> {
        (**self).get_document(doc_id).await
    }
    async fn get_tree(&self, doc_id: &str, node_summary: bool, include_text: bool) -> Result<Value> {
        (**self).get_tree(doc_id, node_summary, include_text).await
    }
    async fn get_ocr_pages(&self, doc_id: &str) -> Result<Value> {
        (**self).get_ocr_pages(doc_id).await
    }
    async fn delete_document(&self, doc_id: &str) -> Result<()> {
        (**self).delete_document(doc_id).await
    }
    async fn get_block(&self, doc_id: &str, block_id: &str) -> Result<Value> {
        (**self).get_block(doc_id, block_id).await
    }
    async fn list_folders(&self) -> Result<Value> {
        (**self).list_folders().await
    }
}

fn page_spec_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^(\d+(-\d+)?)(,\s*\d+(-\d+)?)*$").expect("page spec regex"))
}

/// Tool names, descriptions, and parameter schemas, identical to the cloud MCP server.
pub fn tool_contract() -> Value {
    json!({
        "browse_documents": {
            "annotations": {"readOnlyHint": true, "openWorldHint": false},
            "description": "Primary document retrieval tool. After orienting with get_folder_structure() (when available), use this for all document-related questions. The bare call returns root-level sub-folders and documents; pass folder_id to drill into a sub-folder level by level. Use sort=\"relevance\" + query for semantic ranking. Do NOT jump to search_documents() first — it is an escalation path, only after browse_documents(sort=\"relevance\") has failed.",
            "schema": {
                "type": "object",
                "properties": {
                    "folder_id": {"type": "string", "default": "root", "description": "Folder scope (default \"root\")."},
                    "recursive": {"type": "boolean", "default": false, "description": "Whether to include documents from descendant folders."},
                    "sort": {"type": "string", "enum": ["time", "relevance"], "default": "time", "description": "Sort order."},
                    "query": {"type": "string", "description": "Search query for relevance ranking. Required when sort=\"relevance\"."},
                    "offset": {"type": "integer", "minimum": 0, "default": 0, "description": "Zero-based pagination offset."},
                    "limit": {"type": "number", "minimum": 1, "maximum": 50, "default": 10, "description": "Number of documents to return per page (1-50, default 10)"}
                },
                "required": []
            }
        },
        "get_document": {
            "annotations": {"readOnlyHint": true, "openWorldHint": false},
            "description": "Check a document's processing status and metadata. `status` is one of \"pending\", \"queued\", \"processing\", \"completed\", or \"failed\" — call this before `get_document_structure()` or `get_page_content()` to confirm the document is ready.",
            "schema": {
                "type": "object",
                "properties": {
                    "doc_name": {"type": "string", "minLength": 1, "description": "Copy the `name` field verbatim from a browse_documents() response (case-sensitive, include extension)."},
                    "folder_id": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Disambiguator for same-name documents."},
                    "wait_for_completion": {"type": "boolean", "default": false, "description": "If true and document is processing, automatically wait up to 3 minutes until completed."}
                },
                "required": ["doc_name"]
            }
        },
        "get_document_structure": {
            "annotations": {"readOnlyHint": true, "openWorldHint": false},
            "description": format!("Extract a document's hierarchical outline (headers, sections, page references). REQUIRED for documents over {STRUCTURE_FIRST_PAGE_THRESHOLD} pages — call this first to locate relevant sections, then pass their page numbers to `get_page_content()`. Use the `part` parameter to iterate large outlines until `pagination.has_more` is false."),
            "schema": {
                "type": "object",
                "properties": {
                    "doc_name": {"type": "string", "minLength": 1, "description": "Copy the `name` field verbatim from a browse_documents() response."},
                    "folder_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "part": {"type": "integer", "minimum": 1, "default": 1, "description": "Part number for pagination (1-based, default 1)."},
                    "wait_for_completion": {"type": "boolean", "default": false}
                },
                "required": ["doc_name"]
            }
        },
        "get_page_content": {
            "annotations": {"readOnlyHint": true, "openWorldHint": false},
            "description": format!("Extract page content from a processed document. Use tight, targeted page ranges — never the whole document at once. For documents over {STRUCTURE_FIRST_PAGE_THRESHOLD} pages, call `get_document_structure()` first to pick relevant sections."),
            "schema": {
                "type": "object",
                "properties": {
                    "doc_name": {"type": "string", "minLength": 1},
                    "folder_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "pages": {"type": "string", "minLength": 1, "pattern": r"^(\d+(-\d+)?)(,\s*\d+(-\d+)?)*$", "description": "Page specification: \"5\", \"3,7,10\", \"5-10\", or \"1-3,7,9-12\""},
                    "wait_for_completion": {"type": "boolean", "default": false}
                },
                "required": ["doc_name", "pages"]
            }
        },
        "remove_document": {
            "annotations": {"readOnlyHint": false, "destructiveHint": true, "idempotentHint": true, "openWorldHint": false},
            "description": "Permanently delete documents and all associated data. Only invoke when the user explicitly names the documents AND confirms deletion.",
            "schema": {
                "type": "object",
                "properties": {
                    "doc_names": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1, "maxItems": 10},
                    "folder_id": {"anyOf": [{"type": "string"}, {"type": "null"}]}
                },
                "required": ["doc_names"]
            }
        }
    })
}

pub fn tool_names(include_management: bool) -> Vec<&'static str> {
    let mut names = READ_TOOLS.to_vec();
    if include_management {
        names.extend_from_slice(MANAGEMENT_TOOLS);
    }
    names
}

fn local_hidden(name: &str) -> &'static [&'static str] {
    match name {
        "browse_documents" => &["folder_id", "recursive", "sort", "query"],
        "get_document" | "get_document_structure" | "get_page_content" | "remove_document" => &["folder_id"],
        _ => &[],
    }
}

fn local_description(name: &str, cloud: &str) -> String {
    match name {
        "browse_documents" => "Primary document retrieval tool — first choice for any document-related question. Lists your documents newest first with names and descriptions; match them against the user's intent and page through with `offset: next_offset` (limit up to 50) while `has_more` is true. Folder browsing and semantic ranking (sort=\"relevance\") are not supported in local mode yet — they work on PageIndex cloud.".into(),
        "get_page_content" => {
            let re = Regex::new(r"\s*[^.]*`get_document_image\(\)`[^.]*\.").expect("image tool sentence");
            re.replace(cloud, "").into_owned()
        }
        _ => cloud.to_owned(),
    }
}

/// Local JSON schema: the cloud contract minus hidden cloud-only parameters.
pub fn local_schema(name: &str) -> Value {
    let mut schema = tool_contract()[name]["schema"].clone();
    if let Some(props) = schema.get_mut("properties").and_then(Value::as_object_mut) {
        for hidden in local_hidden(name) {
            props.remove(*hidden);
        }
    }
    schema
}

pub fn local_tool_spec(name: &str) -> crate::llm::ToolSpec {
    let contract = tool_contract();
    let cloud_desc = contract[name]["description"].as_str().unwrap_or_default().to_owned();
    crate::llm::ToolSpec {
        name: name.to_owned(),
        description: local_description(name, &cloud_desc),
        parameters: local_schema(name),
    }
}

pub const AGENT_INSTRUCTIONS: &str = concat!(
    "PageIndex by Vectify AI is a document platform for uploading and managing long PDFs (research papers, financial reports, legal docs, textbooks, etc.).\n\n",
    "READING WORKFLOW:\n",
    "- For documents over 20 pages: call get_document_structure() first to locate relevant sections, then get_page_content() with targeted page ranges.\n",
    "- For small documents (20 pages or fewer): call get_page_content() directly.\n\n",
    "TOOL USAGE RULES:\n",
    "- Invoke a tool only when all required parameters are present or clearly inferable. Never invent placeholder values.\n",
    "- If a tool returns an error, present the provided next_steps/options to the user instead of retrying blindly.\n\n",
    "DOCUMENT DISCOVERY:\n",
    "- browse_documents() — DEFAULT discovery tool, first choice for any document-related question. The bare call returns your documents newest first with names and descriptions; match them against the user's intent.\n\n",
    "DECISION:\n",
    "- \"What do I have / list / recent\" → browse_documents()\n",
    "- ANY question that needs a document to answer (including \"find THE paper about Y\") → browse_documents(), then pick the documents whose name/description matches the question\n\n",
    "- Skip discovery ONLY for questions with NO possible document connection (e.g., \"capital of France\").\n",
    "- After discovery: 1 match or 1 clearly best match → proceed to read and answer without asking. Multiple equally relevant → ask user to pick.\n",
    "- Results returned ≠ correct results. If the returned documents do not clearly match the user's intent (e.g., wrong topic, wrong time period, wrong document type), treat it the same as \"not found\" and continue the PERSISTENCE protocol below.\n\n",
    "PERSISTENCE (before concluding the target document is not in the library):\n",
    "This protocol applies both when results are empty AND when results are returned but none match the user's intent. Do NOT give up after a single discovery attempt. Follow these steps in order:\n",
    "1. browse_documents() and compare every returned name/description against the user's intent\n",
    "2. Rephrase the query with synonyms or alternative terms and browse again\n",
    "3. Page through the ENTIRE library with `limit: 50` and `offset: next_offset` until has_more is false — MANDATORY, must be completed before concluding \"not found\"\n",
    "Only after ALL steps have been tried may you conclude the document is not in the library. Do NOT fall back to general knowledge — if the user's question references their own documents, exhaust every discovery path first."
);

pub const CHAT_HEADER: &str = "You are PageIndex by Vectify AI, a document-focused assistant. Be concise, never use emojis, and do not expose tool names.";

pub const CITE_PROMPT: &str = "\
GROUNDING
- Answer only from the user's PageIndex documents. Call get_page_content() and state only what was actually read there.
- Never fill a gap from general knowledge. When the documents do not answer the question, say so.

CITATIONS
- Cite only statements supported by tool outputs: <cite doc=\"{docName}\" page=\"{pageNumber}\"/> or <cite doc=\"{docName}\" page=\"{pageNumber}\" block=\"{blockId}\"/>. Place immediately after the claim.
- When page content includes block_id values, citations MUST be block-level: copy the exact block_id of the supporting block. Page-only cites are allowed ONLY when the tool output carries no block_id (legacy documents, structure outlines). NEVER invent or alter block_id values.
- For a claim drawn from multiple blocks on one page, add one tag per supporting block (at most 3); beyond that, cite the single strongest block.
- Each tag must reference a SINGLE page integer. For multi-page citations, use separate tags.";

pub const MARKDOWN_CITE_PROMPT: &str = "\
GROUNDING
- Answer only from the user's PageIndex documents. Call get_page_content() and state only what was actually read there.
- Never fill a gap from general knowledge. When the documents do not answer the question, say so.

CITATIONS
- Cite only statements supported by tool outputs, as a bracketed reference: [{docName}, p. {pageNumber}] or [{docName}, p. {pageNumber}, block {blockId}]. Place immediately after the claim.
- When page content includes block_id values, citations MUST be block-level: copy the exact block_id of the supporting block. Page-only cites are allowed ONLY when the tool output carries no block_id (legacy documents, structure outlines). NEVER invent or alter block_id values.
- For a claim drawn from multiple blocks on one page, add one reference per supporting block (at most 3); beyond that, cite the single strongest block.
- Each reference must reference a SINGLE page integer. For multi-page citations, use separate references.";

fn success(mut data: Value, next_steps: Value) -> (Value, bool) {
    if let Some(obj) = data.as_object_mut() {
        obj.insert("success".into(), json!(true));
        obj.insert("next_steps".into(), next_steps);
    }
    (data, false)
}

fn failure(error: &str, details: Option<Value>, next_steps: Value, error_code: Option<&str>) -> (Value, bool) {
    let mut payload = json!({"error": error, "next_steps": next_steps});
    if let Some(code) = error_code {
        payload["errorCode"] = json!(code);
    }
    if let Some(Value::Object(details)) = details
        && let Some(obj) = payload.as_object_mut()
    {
        for (k, v) in details {
            obj.insert(k, v);
        }
    }
    (payload, true)
}

fn dumps(payload: &Value) -> String {
    serde_json::to_string(payload).unwrap_or_else(|_| "{}".into())
}

async fn all_documents(client: &dyn AgentBackend, stop_ids: Option<&HashSet<String>>) -> Result<Vec<Value>> {
    let mut documents = Vec::new();
    let mut offset = 0usize;
    let mut remaining = stop_ids.cloned();
    loop {
        let page = client.list_documents(100, offset).await?;
        let batch = page.get("documents").and_then(Value::as_array).cloned().unwrap_or_default();
        documents.extend(batch.iter().cloned());
        if let Some(left) = remaining.as_mut() {
            for doc in &batch {
                if let Some(id) = doc.get("id").and_then(Value::as_str) {
                    left.remove(id);
                }
            }
            if left.is_empty() {
                return Ok(documents);
            }
        }
        offset += batch.len();
        let total = page.get("total").and_then(Value::as_u64).map(|n| n as usize);
        if batch.is_empty() || total.is_some_and(|t| offset >= t) {
            return Ok(documents);
        }
    }
}

fn normalize_created_at(value: &Value) -> String {
    let Some(text) = value.as_str().filter(|s| !s.is_empty()) else {
        return String::new();
    };
    if text.ends_with('Z') {
        return text.to_owned();
    }
    if text.len() >= 19 {
        return format!("{}Z", text.trim_end_matches('Z'));
    }
    text.to_owned()
}

fn flat_metadata(value: &Value) -> Option<Value> {
    let obj = value.as_object()?;
    let flat: serde_json::Map<String, Value> = obj
        .iter()
        .filter(|(_, v)| matches!(v, Value::String(_) | Value::Number(_) | Value::Bool(_)))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if flat.is_empty() { None } else { Some(Value::Object(flat)) }
}

fn scope_documents(documents: &[Value], allowed: Option<&HashSet<String>>) -> Vec<Value> {
    match allowed {
        None => documents.to_vec(),
        Some(ids) => documents
            .iter()
            .filter(|d| d.get("id").and_then(Value::as_str).is_some_and(|id| ids.contains(id)))
            .cloned()
            .collect(),
    }
}

fn similar_names(target: &str, names: &[String]) -> Vec<String> {
    let mut scored: Vec<(f64, &String)> =
        names.iter().map(|n| (ratio(target, n), n)).filter(|(s, _)| *s >= 0.5).collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(SIMILAR_NAMES_LIMIT).map(|(_, n)| n.clone()).collect()
}

fn ratio(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut prev = vec![0usize; b.len() + 1];
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        for (j, &cb) in b.iter().enumerate() {
            cur[j + 1] = if ca == cb { prev[j] + 1 } else { prev[j + 1].max(cur[j]) };
        }
        std::mem::swap(&mut prev, &mut cur);
        let _ = i;
    }
    2.0 * prev[b.len()] as f64 / (a.len() + b.len()) as f64
}

async fn resolve_document(
    client: &dyn AgentBackend,
    doc_name: &str,
    documents: Option<Vec<Value>>,
    allowed: Option<&HashSet<String>>,
) -> Result<(Option<Value>, Option<(Value, bool)>)> {
    let documents = match documents {
        Some(d) => d,
        None => all_documents(client, allowed).await?,
    };
    let documents = scope_documents(&documents, allowed);
    let mut matches: Vec<Value> =
        documents.iter().filter(|d| d.get("name").and_then(Value::as_str) == Some(doc_name)).cloned().collect();
    if !matches.is_empty() {
        matches.sort_by(|a, b| {
            b.get("createdAt")
                .and_then(Value::as_str)
                .unwrap_or("")
                .cmp(a.get("createdAt").and_then(Value::as_str).unwrap_or(""))
        });
        return Ok((matches.into_iter().next(), None));
    }
    let names: Vec<String> =
        documents.iter().filter_map(|d| d.get("name").and_then(Value::as_str).map(str::to_owned)).collect();
    let similar = similar_names(doc_name, &names);
    let message = if similar.is_empty() {
        "Document not found or you do not have access to it".to_owned()
    } else {
        format!(
            "Document not found. Did you mean: {}?",
            similar.iter().map(|n| format!("\"{n}\"")).collect::<Vec<_>>().join(", ")
        )
    };
    Ok((
        None,
        Some(failure(
            &message,
            Some(json!({"doc_name": doc_name, "similar_files": similar})),
            json!({
                "summary": "The requested document does not exist or is not accessible",
                "options": [
                    "Verify the document name is correct",
                    "Use browse_documents() to see your recent documents",
                    "Check if the document was deleted"
                ]
            }),
            Some("NOT_FOUND"),
        )),
    ))
}

fn folder_unsupported(param: &str) -> (Value, bool) {
    failure(
        &format!("Folders are not supported in local mode yet — omit {param}."),
        None,
        json!({
            "summary": "This local library does not have folders yet",
            "options": [
                "Retry the call without a folder_id",
                "Use browse_documents() to list the library root",
                "Folders are available on PageIndex cloud (PageIndexCloudClient with an API key)"
            ]
        }),
        Some("INVALID_INPUT"),
    )
}

fn not_ready_error(doc_name: &str, status: &str, operation: &str, timed_out: bool) -> (Value, bool) {
    if status == "failed" {
        return failure(
            &format!("Document processing failed. Current status: {status}"),
            Some(json!({"doc_name": doc_name})),
            json!({
                "summary": "Document processing has failed",
                "options": [
                    "Index the document again with PageIndexClient.submit_document()",
                    "Use browse_documents() to work with other documents"
                ]
            }),
            Some("INVALID_INPUT"),
        );
    }
    if timed_out {
        return failure(
            &format!("Document is still processing. Current status: {status}"),
            Some(json!({"doc_name": doc_name})),
            json!({
                "summary": "Document processing timeout",
                "options": ["Try again later when processing is complete", "Check status with get_document()"]
            }),
            Some("INVALID_INPUT"),
        );
    }
    failure(
        &format!("Document is not ready for {operation}. Current status: {status}"),
        Some(json!({"doc_name": doc_name})),
        json!({
            "summary": "Document is still processing",
            "options": [
                "Wait for document processing to complete",
                "Check status with browse_documents() or get_document()"
            ]
        }),
        Some("INVALID_INPUT"),
    )
}

#[derive(Debug)]
enum PageSpecError {
    Invalid,
    TooMany,
    Nonpositive,
}

fn expand_pages(pages: &str) -> std::result::Result<Vec<usize>, PageSpecError> {
    if !page_spec_re().is_match(pages) {
        return Err(PageSpecError::Invalid);
    }
    let mut expanded = HashSet::new();
    for part in pages.split(',') {
        let part = part.trim();
        let (start, end) = if let Some((a, b)) = part.split_once('-') {
            let start: usize = a.parse().map_err(|_| PageSpecError::Invalid)?;
            let end: usize = b.parse().map_err(|_| PageSpecError::Invalid)?;
            if start > end {
                return Err(PageSpecError::Invalid);
            }
            (start, end)
        } else {
            let n: usize = part.parse().map_err(|_| PageSpecError::Invalid)?;
            (n, n)
        };
        if end.saturating_sub(start).saturating_add(1) > MAX_REQUESTED_PAGES {
            return Err(PageSpecError::TooMany);
        }
        for page in start..=end {
            expanded.insert(page);
            if expanded.len() > MAX_REQUESTED_PAGES {
                return Err(PageSpecError::TooMany);
            }
        }
    }
    if expanded.iter().copied().min().unwrap_or(0) < 1 {
        return Err(PageSpecError::Nonpositive);
    }
    let mut out: Vec<usize> = expanded.into_iter().collect();
    out.sort_unstable();
    Ok(out)
}

fn parse_page_spec(pages: &str, doc_name: &str) -> std::result::Result<Vec<usize>, (Value, bool)> {
    match expand_pages(pages) {
        Ok(pages) => Ok(pages),
        Err(PageSpecError::TooMany) => Err(failure(
            &format!("Too many pages requested (over {MAX_REQUESTED_PAGES})"),
            Some(json!({"doc_name": doc_name})),
            json!({"summary": "The page specification spans too many pages", "options": ["Request a narrower page range"]}),
            Some("INVALID_INPUT"),
        )),
        Err(PageSpecError::Nonpositive) => Err(failure(
            "Invalid page numbers. Page numbers must be positive integers",
            Some(json!({"doc_name": doc_name})),
            json!({"summary": "Invalid page numbers provided", "options": ["Page numbers must be positive integers (>= 1)"]}),
            Some("INVALID_INPUT"),
        )),
        Err(PageSpecError::Invalid) => Err(failure(
            "Invalid page specification format",
            Some(json!({"doc_name": doc_name})),
            json!({"summary": "Failed to parse the pages parameter", "options": ["Use valid formats: \"5\", \"3,7,10\", \"5-10\", or \"1-3,7,9-12\""]}),
            Some("INVALID_INPUT"),
        )),
    }
}

pub fn format_page_spec(pages: &[usize]) -> String {
    if pages.is_empty() {
        return String::new();
    }
    let mut ordered = pages.to_vec();
    ordered.sort_unstable();
    ordered.dedup();
    let mut ranges = Vec::new();
    let (mut start, mut prev) = (ordered[0], ordered[0]);
    for &page in &ordered[1..] {
        if page == prev + 1 {
            prev = page;
            continue;
        }
        ranges.push(if start == prev { format!("{start}") } else { format!("{start}-{prev}") });
        start = page;
        prev = page;
    }
    ranges.push(if start == prev { format!("{start}") } else { format!("{start}-{prev}") });
    ranges.join(",")
}

fn format_structure(node: &Value) -> Value {
    match node {
        Value::Array(items) => Value::Array(items.iter().map(format_structure).collect()),
        Value::Object(map) => {
            let mut stripped: BTreeMap<String, Value> =
                map.iter().filter(|(k, _)| k.as_str() != "text").map(|(k, v)| (k.clone(), v.clone())).collect();
            if let Some(nodes) = stripped.get("nodes").cloned() {
                stripped.insert("nodes".into(), format_structure(&nodes));
            }
            let mut ordered = serde_json::Map::new();
            for key in STRUCTURE_KEY_ORDER {
                if let Some(value) = stripped.remove(*key) {
                    ordered.insert((*key).to_owned(), value);
                }
            }
            for (k, v) in stripped {
                ordered.insert(k, v);
            }
            Value::Object(ordered)
        }
        other => other.clone(),
    }
}

fn serialized_size(value: &Value) -> usize {
    serde_json::to_string(value).map(|s| s.len()).unwrap_or(0)
}

fn split_structure(structure: &Value, budget: usize) -> Vec<Value> {
    if serialized_size(structure) <= budget {
        return vec![structure.clone()];
    }
    let nodes = match structure {
        Value::Array(items) => items.clone(),
        other => vec![other.clone()],
    };
    let mut chunks = Vec::new();
    let mut group = Vec::new();
    let mut group_size = 0usize;
    for node in nodes {
        let size = serialized_size(&node);
        if size > budget {
            if !group.is_empty() {
                chunks.push(Value::Array(std::mem::take(&mut group)));
                group_size = 0;
            }
            chunks.extend(split_oversized_node(&node, budget).into_iter().map(|p| json!([p])));
            continue;
        }
        if !group.is_empty() && group_size + size > budget {
            chunks.push(Value::Array(std::mem::take(&mut group)));
            group_size = 0;
        }
        group.push(node);
        group_size += size;
    }
    if !group.is_empty() {
        chunks.push(Value::Array(group));
    }
    if chunks.is_empty() { vec![structure.clone()] } else { chunks }
}

fn split_oversized_node(node: &Value, budget: usize) -> Vec<Value> {
    let Some(children) = node.get("nodes") else {
        return vec![node.clone()];
    };
    let mut shell = node.clone();
    if let Some(obj) = shell.as_object_mut() {
        obj.remove("nodes");
    }
    let shell_size = serialized_size(&shell);
    let child_budget = (budget.saturating_sub(shell_size)).max(budget / 2);
    split_structure(children, child_budget)
        .into_iter()
        .map(|chunk| {
            let mut part = shell.clone();
            part["nodes"] = if chunk.is_array() { chunk } else { json!([chunk]) };
            part
        })
        .collect()
}

async fn browse_documents(
    client: &dyn AgentBackend,
    args: &Value,
    allowed: Option<&HashSet<String>>,
) -> Result<(Value, bool)> {
    let folder_id = args.get("folder_id").and_then(Value::as_str).unwrap_or("root");
    if folder_id != "root" {
        return Ok(folder_unsupported("folder_id"));
    }
    let sort = args.get("sort").and_then(Value::as_str).unwrap_or("time");
    if sort != "time" && sort != "relevance" {
        return Ok(failure(
            "Invalid sort mode — only the default \"time\" sort is available in local mode.",
            None,
            json!({"summary": "Invalid sort mode", "options": ["Use sort=\"time\" (newest first) or omit sort"]}),
            Some("INVALID_INPUT"),
        ));
    }
    if sort == "relevance" || args.get("query").is_some() {
        return Ok(failure(
            "Relevance ranking is not supported in local mode yet — use the default time sort.",
            None,
            json!({"summary": "This local library does not have semantic ranking yet", "options": ["Retry without sort/query and match the returned names and descriptions against the intent yourself"]}),
            Some("INVALID_INPUT"),
        ));
    }
    let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(10).clamp(1, 50) as usize;
    let recursive = args.get("recursive").and_then(Value::as_bool).unwrap_or(false);
    let (window, total) = if let Some(ids) = allowed {
        let scoped = scope_documents(&all_documents(client, Some(ids)).await?, Some(ids));
        let total = scoped.len();
        (scoped.into_iter().skip(offset).take(limit).collect::<Vec<_>>(), total)
    } else {
        let listing = client.list_documents(limit, offset).await?;
        let window = listing.get("documents").and_then(Value::as_array).cloned().unwrap_or_default();
        let total = listing.get("total").and_then(Value::as_u64).map(|n| n as usize).unwrap_or(window.len());
        (window, total)
    };
    let window_end = offset + window.len();
    let has_more = !window.is_empty() && window_end < total;
    let next_offset = if has_more { Some(window_end) } else { None };
    let mut page_has_processing = false;
    let mut page_has_failed = false;
    let mut items = Vec::new();
    for doc in &window {
        let status = doc.get("status").and_then(Value::as_str).unwrap_or("unknown");
        if status == "failed" {
            page_has_failed = true;
        } else if status != "completed" {
            page_has_processing = true;
        }
        let mut item = json!({
            "name": doc.get("name").and_then(Value::as_str).unwrap_or("Unknown Document"),
            "description": doc.get("description").and_then(Value::as_str).unwrap_or("No description provided"),
            "status": status,
            "created_at": normalize_created_at(doc.get("createdAt").unwrap_or(&Value::Null)),
        });
        if let Some(meta) = flat_metadata(doc.get("metadata").unwrap_or(&Value::Null)) {
            item["metadata"] = meta;
        }
        items.push(item);
    }
    let mut data = json!({
        "documents": items,
        "sort": sort,
        "next_offset": next_offset,
        "has_more": has_more,
    });
    if !recursive {
        data["folders"] = json!([]);
    }
    if items.is_empty() && offset == 0 {
        return Ok(success(
            data,
            json!({
                "summary": "Nothing to show",
                "options": ["Nothing here. Index documents with PageIndexClient.submit_document() to get started."],
                "auto_retry": "Index a document with PageIndexClient.submit_document() to get started"
            }),
        ));
    }
    let mut options = Vec::new();
    if !items.is_empty() {
        options.push(json!("Use get_document() with a document name to view details"));
        options.push(json!(format!(
            "Results returned ≠ correct results. Verify these documents match the user's actual intent (topic, time period, document type) before proceeding.{} Do NOT use general knowledge as a substitute.",
            if has_more { " If they do not match, page through the rest of the library." } else { "" }
        )));
    }
    if page_has_processing {
        options.push(json!(
            "Some documents on this page are still processing. Use get_document() to check individual status."
        ));
    }
    if page_has_failed {
        options.push(json!("Some documents on this page failed processing. Use get_document() to see error details."));
    }
    if has_more {
        options.push(json!("Use browse_documents() with `offset: next_offset` to load more documents"));
    }
    let summary = if items.is_empty() {
        "Nothing to show".into()
    } else {
        format!("Showing {} document(s){}", items.len(), if has_more { " (more available)" } else { "" })
    };
    Ok(success(data, json!({"summary": summary, "options": options})))
}

async fn get_document_tool(
    client: &dyn AgentBackend,
    args: &Value,
    allowed: Option<&HashSet<String>>,
) -> Result<(Value, bool)> {
    if !matches!(args.get("folder_id").and_then(Value::as_str), None | Some("root")) {
        return Ok(folder_unsupported("folder_id"));
    }
    let doc_name = args.get("doc_name").and_then(Value::as_str).unwrap_or("");
    let (entry, error) = resolve_document(client, doc_name, None, allowed).await?;
    if let Some(error) = error {
        return Ok(error);
    }
    let entry = entry.expect("resolved");
    let status = entry.get("status").and_then(Value::as_str).unwrap_or("unknown");
    let is_processing = !matches!(status, "completed" | "failed");
    let is_ready = status == "completed";
    let page_num = entry.get("pageNum").and_then(Value::as_u64).unwrap_or(0);
    let name = entry.get("name").and_then(Value::as_str).unwrap_or("Unknown Document");
    let mut suggestions = Vec::new();
    if is_processing {
        suggestions.push("Document is still processing. Processing status can be checked later.".to_owned());
    } else if is_ready {
        suggestions.push("Document is ready for analysis.".into());
        if page_num > 0 && page_num <= 5 {
            suggestions.extend([
                format!("This is a short document with {page_num} pages."),
                format!("First explore structure: get_document_structure(doc_name: \"{name}\")"),
                format!("Then extract all content: get_page_content(doc_name: \"{name}\", pages: \"1-{page_num}\")"),
            ]);
        } else if page_num > 5 && page_num <= STRUCTURE_FIRST_PAGE_THRESHOLD as u64 {
            suggestions.extend([
                format!("This document has {page_num} pages."),
                format!("First explore structure: get_document_structure(doc_name: \"{name}\")"),
                format!("Then extract key pages: get_page_content(doc_name: \"{name}\", pages: \"1,5,10\")"),
            ]);
        } else if page_num > STRUCTURE_FIRST_PAGE_THRESHOLD as u64 {
            suggestions.extend([
                format!("This is a large document with {page_num} pages."),
                format!("First explore structure: get_document_structure(doc_name: \"{name}\")"),
                format!("Then target specific sections: get_page_content(doc_name: \"{name}\", pages: \"1-3\")"),
            ]);
        }
    } else {
        suggestions.push(
            "Document processing failed. Index the document again with PageIndexClient.submit_document().".into(),
        );
    }
    let mut data = json!({
        "name": name,
        "description": entry.get("description").and_then(Value::as_str).unwrap_or("No description provided"),
        "status": status,
        "created_at": normalize_created_at(entry.get("createdAt").unwrap_or(&Value::Null)),
        "page_count": if page_num == 0 { Value::Null } else { json!(page_num) },
        "folder_id": entry.get("folderId").cloned().unwrap_or(Value::Null),
    });
    if let Some(meta) = flat_metadata(entry.get("metadata").unwrap_or(&Value::Null)) {
        data["metadata"] = meta;
    }
    let mut next = json!({
        "summary": if is_ready { "Document is ready for analysis and querying." } else if is_processing { "Document is still being processed." } else { "Document processing has failed." },
        "options": suggestions,
    });
    if is_processing {
        next["auto_retry"] = json!("Document processing status can be monitored periodically");
    }
    Ok(success(data, next))
}

async fn get_document_structure(
    client: &dyn AgentBackend,
    args: &Value,
    allowed: Option<&HashSet<String>>,
) -> Result<(Value, bool)> {
    if !matches!(args.get("folder_id").and_then(Value::as_str), None | Some("root")) {
        return Ok(folder_unsupported("folder_id"));
    }
    let doc_name = args.get("doc_name").and_then(Value::as_str).unwrap_or("");
    let (entry, error) = resolve_document(client, doc_name, None, allowed).await?;
    if let Some(error) = error {
        return Ok(error);
    }
    let entry = entry.expect("resolved");
    let status = entry.get("status").and_then(Value::as_str).unwrap_or("unknown");
    if status != "completed" {
        return Ok(not_ready_error(doc_name, status, "structure retrieval", false));
    }
    let id = entry.get("id").and_then(Value::as_str).unwrap_or("");
    let tree = match client.get_tree(id, true, false).await {
        Ok(v) => v.get("result").cloned(),
        Err(exc) => {
            return Ok(failure(
                &format!("Failed to retrieve document structure: {exc}"),
                Some(json!({"doc_name": doc_name})),
                json!({"summary": "Failed to retrieve document structure due to an error", "options": ["The document may not exist or is not accessible"]}),
                Some("INTERNAL_ERROR"),
            ));
        }
    };
    let Some(tree) = tree else {
        return Ok(failure(
            "Structure not available for this document",
            Some(json!({"doc_name": doc_name})),
            json!({"summary": "Structure not available for this document", "options": ["Try processing the document again if possible"]}),
            Some("INTERNAL_ERROR"),
        ));
    };
    let formatted = format_structure(&tree);
    let chunks = split_structure(&formatted, CHAR_BUDGET);
    let total_parts = chunks.len().max(1);
    let requested = args.get("part").and_then(Value::as_u64).unwrap_or(1) as usize;
    let current = requested.clamp(1, total_parts);
    if total_parts == 1 {
        return Ok(success(
            json!({"doc_name": doc_name, "structure": chunks[0]}),
            json!({"summary": "Document structure retrieved successfully.", "options": ["Use get_page_content() to extract specific content from pages"]}),
        ));
    }
    let next_steps = if current < total_parts {
        json!({"summary": format!("Showing part {current} of {total_parts}."), "options": [format!("Request next part with part: {}", current + 1), "Proceed to get_page_content() for specific sections"]})
    } else {
        json!({"summary": "All parts retrieved for current pagination.", "options": ["Use get_page_content() to extract specific content from pages"]})
    };
    Ok(success(
        json!({
            "doc_name": doc_name,
            "total_parts": total_parts,
            "structure": chunks[current - 1],
            "pagination": {"part": current, "total_parts": total_parts, "has_more": current < total_parts}
        }),
        next_steps,
    ))
}

async fn get_page_content(
    client: &dyn AgentBackend,
    args: &Value,
    allowed: Option<&HashSet<String>>,
) -> Result<(Value, bool)> {
    if !matches!(args.get("folder_id").and_then(Value::as_str), None | Some("root")) {
        return Ok(folder_unsupported("folder_id"));
    }
    let doc_name = args.get("doc_name").and_then(Value::as_str).unwrap_or("");
    let (entry, error) = resolve_document(client, doc_name, None, allowed).await?;
    if let Some(error) = error {
        return Ok(error);
    }
    let entry = entry.expect("resolved");
    let status = entry.get("status").and_then(Value::as_str).unwrap_or("unknown");
    if status != "completed" {
        return Ok(not_ready_error(doc_name, status, "page content retrieval", false));
    }
    let pages = args.get("pages").and_then(Value::as_str).unwrap_or("");
    let requested = match parse_page_spec(pages, doc_name) {
        Ok(p) => p,
        Err(err) => return Ok(err),
    };
    let id = entry.get("id").and_then(Value::as_str).unwrap_or("");
    let page_data = match client.get_ocr_pages(id).await {
        Ok(v) => v.get("result").and_then(Value::as_array).cloned().unwrap_or_default(),
        Err(exc) => {
            return Ok(failure(
                &format!("Failed to retrieve page content: {exc}"),
                Some(json!({"doc_name": doc_name})),
                json!({"summary": "Unable to retrieve page content due to a service issue.", "options": ["Verify the document name is correct using browse_documents()"]}),
                Some("INTERNAL_ERROR"),
            ));
        }
    };
    let mut by_index = BTreeMap::new();
    for item in page_data {
        if let Some(idx) = item.get("page_index").and_then(Value::as_u64) {
            by_index.insert(idx as usize, item);
        }
    }
    let max_page = by_index.keys().copied().max().unwrap_or(0);
    let out_of_range: Vec<usize> = requested.iter().copied().filter(|p| *p > max_page).collect();
    let valid: Vec<usize> = requested.iter().copied().filter(|p| *p <= max_page).collect();
    if !out_of_range.is_empty() && valid.is_empty() {
        return Ok(failure(
            &format!(
                "All requested pages are out of range. Document has {max_page} pages, but you requested pages: {}",
                format_page_spec(&out_of_range)
            ),
            Some(
                json!({"doc_name": doc_name, "max_pages": max_page, "requested_pages": format_page_spec(&out_of_range)}),
            ),
            json!({"summary": "All requested pages are out of range for this document", "options": [format!("Request pages between 1 and {max_page}")]}),
            Some("INVALID_INPUT"),
        ));
    }
    let mut content = Vec::new();
    let mut included = Vec::new();
    let mut remaining = Vec::new();
    let mut budget = CHAR_BUDGET as isize;
    for page in valid {
        let text = by_index
            .get(&page)
            .and_then(|i| i.get("markdown"))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("Page {page} content not available"));
        let entry = json!({"page": page, "text": text});
        let size = serialized_size(&entry) as isize + 2;
        if included.is_empty() || budget - size >= 0 {
            content.push(entry);
            included.push(page);
            budget -= size;
        } else {
            remaining.push(page);
        }
    }
    let mut options = vec![
        json!("Use get_document_structure() to understand document organization"),
        json!("Request additional pages as needed"),
    ];
    if !remaining.is_empty() {
        options.insert(0, json!(format!("For remaining pages, request: {}", format_page_spec(&remaining))));
    }
    if !out_of_range.is_empty() {
        options.insert(0, json!(format!("Document has {max_page} pages total - request pages 1-{max_page}")));
    }
    let summary = if remaining.is_empty() && out_of_range.is_empty() {
        format!(
            "Successfully retrieved content for {} page{}.",
            content.len(),
            if content.len() == 1 { "" } else { "s" }
        )
    } else {
        let mut parts = vec![format!("Retrieved {} of {} requested pages.", included.len(), requested.len())];
        if !remaining.is_empty() {
            parts.push(format!("Pages {} were omitted due to response size limits.", format_page_spec(&remaining)));
        }
        if !out_of_range.is_empty() {
            parts.push(format!("Pages {} were out of range.", format_page_spec(&out_of_range)));
        }
        parts.join(" ")
    };
    Ok(success(
        json!({
            "doc_name": doc_name,
            "total_pages": max_page,
            "requested_pages": format_page_spec(&requested),
            "returned_pages": format_page_spec(&included),
            "content": content
        }),
        json!({"summary": summary, "options": options}),
    ))
}

async fn remove_document(
    client: &dyn AgentBackend,
    args: &Value,
    allowed: Option<&HashSet<String>>,
) -> Result<(Value, bool)> {
    if !matches!(args.get("folder_id").and_then(Value::as_str), None | Some("root")) {
        return Ok(folder_unsupported("folder_id"));
    }
    let Some(names) = args.get("doc_names").and_then(Value::as_array) else {
        return Ok(failure(
            "At least one document name is required",
            None,
            json!({"summary": "No document names provided", "options": ["Pass doc_names as a non-empty array"]}),
            Some("INVALID_INPUT"),
        ));
    };
    if names.is_empty() || names.iter().any(|n| n.as_str().is_none_or(str::is_empty)) {
        return Ok(failure(
            "doc_names must be an array of non-empty document name strings",
            None,
            json!({"summary": "Invalid document names", "options": ["Copy each name verbatim from a browse_documents() response"]}),
            Some("INVALID_INPUT"),
        ));
    }
    let mut doc_names: Vec<String> = Vec::new();
    for name in names {
        let name = name.as_str().unwrap();
        if !doc_names.iter().any(|n| n == name) {
            doc_names.push(name.to_owned());
        }
    }
    if doc_names.len() > 10 {
        return Ok(failure(
            "Maximum 10 documents can be deleted at once",
            None,
            json!({"summary": "Too many documents in one call", "options": ["Delete at most 10 documents per call"]}),
            Some("INVALID_INPUT"),
        ));
    }
    let documents = all_documents(client, allowed).await?;
    let mut results = Vec::new();
    for doc_name in &doc_names {
        let (entry, error) = resolve_document(client, doc_name, Some(documents.clone()), allowed).await?;
        if error.is_some() || entry.is_none() {
            results.push(json!({"doc_name": doc_name, "status": "not_found"}));
            continue;
        }
        let id = entry.unwrap().get("id").and_then(Value::as_str).unwrap_or("").to_owned();
        match client.delete_document(&id).await {
            Ok(()) => results.push(json!({"doc_name": doc_name, "status": "deleted"})),
            Err(exc) => results.push(json!({"doc_name": doc_name, "status": "failed", "error": exc.to_string()})),
        }
    }
    let deleted = results.iter().filter(|r| r["status"] == "deleted").count();
    Ok(success(
        json!({"results": results}),
        json!({"summary": format!("Deleted {deleted} of {} document(s).", doc_names.len()), "options": ["Use browse_documents() to review the remaining library"]}),
    ))
}

fn coerce_bool_args(args: &mut Value) {
    let Some(obj) = args.as_object_mut() else {
        return;
    };
    for key in ["recursive", "wait_for_completion"] {
        if let Some(Value::String(s)) = obj.get(key) {
            let truthy = !matches!(s.trim().to_ascii_lowercase().as_str(), "false" | "no" | "0" | "");
            obj.insert(key.into(), json!(truthy));
        }
    }
}

/// Run one contract tool; returns `(envelope_json, is_error)`. Never raises
/// for tool-level failures.
pub async fn call_tool(
    client: &dyn AgentBackend,
    name: &str,
    arguments: Value,
    doc_ids: Option<&[String]>,
) -> (String, bool) {
    if !matches!(
        name,
        "browse_documents" | "get_document" | "get_document_structure" | "get_page_content" | "remove_document"
    ) {
        let (payload, _) = failure(
            &format!("Unknown tool: {name}"),
            Some(json!({"tool_name": name, "available_tools": READ_TOOLS})),
            json!({"summary": "Tool not found", "options": [format!("Available tools: {}", READ_TOOLS.join(", "))]}),
            Some("INVALID_INPUT"),
        );
        return (dumps(&payload), true);
    }
    let mut args = if arguments.is_null() { json!({}) } else { arguments };
    if !args.is_object() {
        let (payload, is_error) = failure(
            &format!("Invalid arguments for {name}: expected a JSON object"),
            None,
            json!({"summary": "Invalid tool arguments", "options": [format!("Pass {name}() arguments as a JSON object of its parameters")]}),
            Some("INVALID_INPUT"),
        );
        return (dumps(&payload), is_error);
    }
    if let Some(obj) = args.as_object_mut() {
        obj.retain(|k, v| !k.starts_with('_') && !v.is_null());
    }
    coerce_bool_args(&mut args);
    let allowed = doc_ids.map(|ids| ids.iter().cloned().collect::<HashSet<_>>());
    let result = match name {
        "browse_documents" => browse_documents(client, &args, allowed.as_ref()).await,
        "get_document" => get_document_tool(client, &args, allowed.as_ref()).await,
        "get_document_structure" => get_document_structure(client, &args, allowed.as_ref()).await,
        "get_page_content" => get_page_content(client, &args, allowed.as_ref()).await,
        "remove_document" => remove_document(client, &args, allowed.as_ref()).await,
        _ => unreachable!(),
    };
    match result {
        Ok((payload, is_error)) => (dumps(&payload), is_error),
        Err(exc) => {
            let (payload, is_error) = failure(
                &format!("{name} failed: {exc}"),
                None,
                json!({"summary": "Unexpected error while running the tool", "options": ["Try the request again"], "auto_retry": "This is likely a temporary issue - you can try the request again"}),
                Some("INTERNAL_ERROR"),
            );
            (dumps(&payload), is_error)
        }
    }
}

pub async fn fetch_citation_prompt(client: &dyn AgentBackend, format: &str) -> Result<String> {
    let local = match format {
        "cite" => CITE_PROMPT,
        "markdown" => MARKDOWN_CITE_PROMPT,
        other => {
            return Err(PageIndexError::api(format!("citations format {other:?} is not one of cite, markdown.")));
        }
    };
    if client.api_key().is_none() {
        return Ok(local.to_owned());
    }
    let key = client.api_key().unwrap();
    let url = format!("{}/mcp", client.base_url());
    let bridge = McpBridge::new(url, vec![("Authorization".into(), format!("Bearer {key}"))]);
    let (_, messages) = bridge.get_prompt("cited_answer", Some(&json!({"format": format}))).await?;
    let text = render_prompt_text(&messages);
    if text.trim().is_empty() {
        return Err(PageIndexError::api("The MCP server returned an empty cited_answer prompt."));
    }
    Ok(text)
}

pub async fn base_instructions(client: &dyn AgentBackend, include_management: bool) -> Result<String> {
    let base = if client.api_key().is_none() {
        AGENT_INSTRUCTIONS.to_owned()
    } else {
        let key = client.api_key().unwrap();
        let suffix = if include_management { "" } else { "?tools=read" };
        let url = format!("{}/mcp{suffix}", client.base_url());
        let bridge = McpBridge::new(url, vec![("Authorization".into(), format!("Bearer {key}"))]);
        match bridge.instructions().await? {
            Some(text) if !text.trim().is_empty() => text,
            _ => {
                return Err(PageIndexError::api(
                    "The MCP server returned no agent instructions — refusing to substitute the SDK's local-subset guidance, which does not cover the cloud tool set.",
                ));
            }
        }
    };
    Ok(match client.instructions() {
        Some(own) if !own.is_empty() => format!("{base}\n\n{own}"),
        _ => base,
    })
}

pub async fn doc_targeting_block(client: &dyn AgentBackend, doc_id: &[String]) -> Result<Option<String>> {
    if doc_id.is_empty() {
        return Ok(None);
    }
    let mut details = Vec::new();
    let mut missing = Vec::new();
    for one_id in doc_id {
        match client.get_document(one_id).await {
            Ok(doc) => details.push(doc),
            Err(exc) if matches!(exc.status_code, None | Some(403 | 404)) => missing.push(one_id.clone()),
            Err(exc) => return Err(exc),
        }
    }
    if !missing.is_empty() {
        return Err(PageIndexError::api(format!("Documents not found or access denied: {}", missing.join(", "))));
    }
    if details.len() == 1 {
        return Ok(Some(format!(
            "The user has specified document: {}\nDocument metadata: {}\nUse this document's name to retrieve its content with get_document_structure() and get_page_content().",
            details[0].get("name").and_then(Value::as_str).unwrap_or(""),
            details[0]
        )));
    }
    let names: Vec<&str> = details.iter().filter_map(|d| d.get("name").and_then(Value::as_str)).collect();
    Ok(Some(format!(
        "The user has specified documents: {}\nDocuments metadata: {}\nUse these documents' names to retrieve their content with get_document_structure() and get_page_content().",
        names.join(", "),
        Value::Array(details)
    )))
}

pub async fn targeting_block(
    client: &dyn AgentBackend,
    doc_id: Option<&[String]>,
    folder_id: Option<&str>,
) -> Result<Option<String>> {
    if let Some(folder) = folder_id
        && !folder.is_empty()
        && folder != "root"
        && client.api_key().is_none()
    {
        return Err(PageIndexError::api(
            "folder_id is cloud-only — folders are not supported in local mode. Create the client with an api_key to use folders.",
        ));
    }
    let doc = match doc_id {
        Some(ids) if !ids.is_empty() => doc_targeting_block(client, ids).await?,
        _ => None,
    };
    Ok(doc)
}

/// OpenAI Chat Completions / Agents SDK function-tool list.
pub fn as_openai_tools(include_management: bool, hosted: bool, api_key: Option<&str>, base_url: &str) -> Value {
    if hosted && let Some(key) = api_key {
        let suffix = if include_management { "" } else { "?tools=read" };
        return json!([{
            "type": "mcp",
            "server_label": "pageindex",
            "server_url": format!("{base_url}/mcp{suffix}"),
            "headers": {"Authorization": format!("Bearer {key}")},
            "require_approval": "never"
        }]);
    }
    Value::Array(
        tool_names(include_management)
            .into_iter()
            .map(|name| {
                let spec = local_tool_spec(name);
                json!({
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                        "strict": true
                    }
                })
            })
            .collect(),
    )
}

/// Anthropic Messages `tools` list.
pub fn as_anthropic_tools(include_management: bool) -> Value {
    Value::Array(
        tool_names(include_management)
            .into_iter()
            .map(|name| {
                let spec = local_tool_spec(name);
                json!({
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.parameters
                })
            })
            .collect(),
    )
}

/// MCP `tools/list` descriptors (`inputSchema` + annotations).
pub fn as_mcp_tools(include_management: bool) -> Value {
    let contract = tool_contract();
    Value::Array(
        tool_names(include_management)
            .into_iter()
            .map(|name| {
                let spec = local_tool_spec(name);
                let mut tool = json!({
                    "name": spec.name,
                    "description": spec.description,
                    "inputSchema": spec.parameters,
                });
                if let Some(annotations) = contract[name].get("annotations") {
                    tool["annotations"] = annotations.clone();
                }
                tool
            })
            .collect(),
    )
}

/// Claude Agent SDK MCP server config (HTTP for cloud, in-process schema for local).
pub fn as_claude_mcp(include_management: bool, api_key: Option<&str>, base_url: &str) -> Value {
    if let Some(key) = api_key {
        let suffix = if include_management { "" } else { "?tools=read" };
        return json!({
            "type": "http",
            "url": format!("{base_url}/mcp{suffix}"),
            "headers": {"Authorization": format!("Bearer {key}")}
        });
    }
    json!({
        "type": "sdk",
        "name": "pageindex",
        "version": env!("CARGO_PKG_VERSION"),
        "tools": as_anthropic_tools(include_management)
    })
}

pub fn openai_agent_config(
    include_management: bool,
    api_key: Option<&str>,
    base_url: &str,
    instructions: &str,
) -> Value {
    json!({
        "instructions": instructions,
        "tools": as_openai_tools(include_management, api_key.is_some(), api_key, base_url)
    })
}

pub fn anthropic_runner_config(include_management: bool, instructions: &str) -> Value {
    json!({
        "system": instructions,
        "tools": as_anthropic_tools(include_management)
    })
}

pub fn claude_agent_config(
    include_management: bool,
    api_key: Option<&str>,
    base_url: &str,
    instructions: &str,
) -> Value {
    json!({
        "system": instructions,
        "mcp_servers": {"pageindex": as_claude_mcp(include_management, api_key, base_url)}
    })
}

pub fn citation_key(document: &str, page: usize, block_id: Option<&str>) -> String {
    match block_id {
        Some(block) => format!("{document}:{page}:{block}"),
        None => format!("{document}:{page}"),
    }
}

/// Parse `<cite .../>` and legacy `<doc=...>` tags.
pub fn parse_citations(text: &str) -> Vec<Value> {
    let cite = Regex::new(r#"<cite\s+doc="([^"]+)"\s+page="(\d+)"(?:\s+block="([^"]+)")?\s*/>"#).expect("cite");
    let legacy = Regex::new(r"<doc=([^;>]+);page=(\d+)(?:;(?:block|block_id)=([^>]+))?>").expect("legacy cite");
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for cap in cite.captures_iter(text).chain(legacy.captures_iter(text)) {
        let document = cap.get(1).map(|m| m.as_str().to_owned()).unwrap_or_default();
        let page: usize = cap.get(2).and_then(|m| m.as_str().parse().ok()).unwrap_or(0);
        let block = cap.get(3).map(|m| m.as_str().to_owned());
        let key = citation_key(&document, page, block.as_deref());
        if !seen.insert(key) {
            continue;
        }
        let mut item = json!({"document": document, "page": page});
        if let Some(block) = block {
            item["block_id"] = json!(block);
        }
        out.push(item);
    }
    out
}

pub async fn get_citations(client: &dyn AgentBackend, answer: &str, doc_id: Option<&[String]>) -> Result<Vec<Value>> {
    let parsed = parse_citations(answer);
    if parsed.is_empty() {
        return Ok(Vec::new());
    }
    let library = all_documents(client, None).await?;
    let mut resolved = Vec::new();
    for cite in parsed {
        let name = cite.get("document").and_then(Value::as_str).unwrap_or("");
        let matches: Vec<&Value> =
            library.iter().filter(|d| d.get("name").and_then(Value::as_str) == Some(name)).collect();
        let id = if let Some(ids) = doc_id {
            if ids.len() == 1 {
                Some(ids[0].clone())
            } else {
                matches.iter().find_map(|d| d.get("id").and_then(Value::as_str).map(str::to_owned))
            }
        } else if matches.len() == 1 {
            matches[0].get("id").and_then(Value::as_str).map(str::to_owned)
        } else if matches.len() > 1 {
            return Err(PageIndexError::api(format!(
                "Citation document {name:?} is ambiguous — pass doc_id= to disambiguate."
            )));
        } else {
            None
        };
        let mut item = cite;
        if let Some(id) = id {
            item["doc_id"] = json!(id);
            if let Some(block) = item.get("block_id").and_then(Value::as_str).map(str::to_owned)
                && let Ok(block_json) = client.get_block(item["doc_id"].as_str().unwrap_or(""), &block).await
                && let Some(obj) = item.as_object_mut()
            {
                for (k, v) in block_json.as_object().cloned().unwrap_or_default() {
                    obj.entry(k).or_insert(v);
                }
            }
        }
        resolved.push(item);
    }
    Ok(resolved)
}

pub async fn resolve_citations(client: &dyn AgentBackend, answer: &str, doc_id: Option<&[String]>) -> Result<Value> {
    let citations = get_citations(client, answer, doc_id).await?;
    let mut rewritten = answer.to_owned();
    let mut numbered = Vec::new();
    for (i, cite) in citations.into_iter().enumerate() {
        let index = i + 1;
        let anchor = format!("pageindex-citation-{index:02}");
        let tag = if let Some(block) = cite.get("block_id").and_then(Value::as_str) {
            format!(
                r#"<cite doc="{}" page="{}" block="{block}"/>"#,
                cite["document"].as_str().unwrap_or(""),
                cite["page"]
            )
        } else {
            format!(r#"<cite doc="{}" page="{}"/>"#, cite["document"].as_str().unwrap_or(""), cite["page"])
        };
        let link = format!("[[{index}]](#{anchor})");
        rewritten = rewritten.replacen(&tag, &link, 1);
        let mut item = cite;
        item["anchor"] = json!(anchor);
        item["index"] = json!(index);
        numbered.push(item);
    }
    Ok(json!({"answer": rewritten, "citations": numbered}))
}

/// Cloud MCP tool call, returning the text rendering of the content blocks.
pub async fn call_cloud_tool(bridge: &McpBridge, name: &str, arguments: &Value) -> Result<(String, bool)> {
    let (blocks, is_error) = bridge.call_tool(name, arguments).await?;
    Ok((render_text(&blocks), is_error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_specs_round_trip() {
        assert_eq!(expand_pages("5-7").unwrap(), vec![5, 6, 7]);
        assert_eq!(format_page_spec(&[1, 2, 3, 5]), "1-3,5");
        assert!(expand_pages("0").is_err());
        assert!(expand_pages("7-5").is_err());
    }

    #[test]
    fn citations_parse_both_spellings() {
        let tags = parse_citations(r#"claim <cite doc="A.pdf" page="3"/> and <doc=B.pdf;page=2;block=p2_text_1>"#);
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0]["document"], "A.pdf");
        assert_eq!(tags[1]["block_id"], "p2_text_1");
    }
}
