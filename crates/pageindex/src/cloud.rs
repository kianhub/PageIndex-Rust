//! PageIndex cloud HTTP client (`cloud_api.py`).

use std::path::Path;
use std::time::Duration;

use percent_encoding::{NON_ALPHANUMERIC, utf8_percent_encode};
use serde_json::{Value, json};

use crate::error::{PageIndexError, Result};
use crate::naming::{sanitize_filename, validate_folder_name};

pub const PAGEINDEX_BASE_URL: &str = "https://api.pageindex.ai";

fn enc(value: &str) -> String {
    utf8_percent_encode(value, NON_ALPHANUMERIC).to_string()
}

/// HTTP wrapper for `https://api.pageindex.ai`.
#[derive(Debug, Clone)]
pub struct CloudApi {
    pub base_url: String,
    pub api_key: String,
    http: reqwest::Client,
}

impl CloudApi {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base(PAGEINDEX_BASE_URL, api_key)
    }

    pub fn with_base(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key: api_key.into(),
            http: reqwest::Client::new(),
        }
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Ok(value) = reqwest::header::HeaderValue::from_str(&self.api_key) {
            headers.insert("api_key", value);
        }
        headers
    }

    async fn fail(operation: &str, response: reqwest::Response) -> PageIndexError {
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        PageIndexError::api(format!("Failed to {operation}: {text}")).with_status(Some(status))
    }

    pub async fn submit_document(
        &self,
        file_path: impl AsRef<Path>,
        mode: Option<&str>,
        beta_headers: Option<&[String]>,
        folder_id: Option<&str>,
        metadata: Option<&Value>,
    ) -> Result<Value> {
        let path = file_path.as_ref();
        let bytes = std::fs::read(path)?;
        let name = sanitize_filename(&path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        let mut form = reqwest::multipart::Form::new()
            .text("if_retrieval", "true")
            .part("file", reqwest::multipart::Part::bytes(bytes).file_name(name));
        if let Some(mode) = mode {
            form = form.text("mode", mode.to_owned());
        }
        if let Some(headers) = beta_headers {
            form = form.text("beta_headers", serde_json::to_string(headers)?);
        }
        if let Some(folder_id) = folder_id {
            form = form.text("folder_id", folder_id.to_owned());
        }
        if let Some(metadata) = metadata {
            form = form.text("metadata", serde_json::to_string(metadata)?);
        }
        let response = self
            .http
            .post(format!("{}/doc/", self.base_url))
            .headers(self.headers())
            .multipart(form)
            .send()
            .await
            .map_err(|e| PageIndexError::api(format!("Failed to submit document: {e}")))?;
        if response.status() != reqwest::StatusCode::OK {
            return Err(Self::fail("submit document", response).await);
        }
        Ok(response.json().await.unwrap_or(Value::Null))
    }

    pub async fn get_ocr(&self, doc_id: &str, format: &str) -> Result<Value> {
        if !matches!(format, "page" | "node" | "raw") {
            return Err(PageIndexError::invalid("Format parameter must be 'page', 'node', or 'raw'"));
        }
        let response = self
            .http
            .get(format!("{}/doc/{}/?type=ocr&format={format}", self.base_url, enc(doc_id)))
            .headers(self.headers())
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| PageIndexError::api(format!("Failed to get OCR result: {e}")))?;
        if response.status() != reqwest::StatusCode::OK {
            return Err(Self::fail("get OCR result", response).await);
        }
        Ok(response.json().await.unwrap_or(Value::Null))
    }

    pub async fn get_block(&self, doc_id: &str, block_id: &str) -> Result<Value> {
        let response = self
            .http
            .get(format!("{}/doc/{}/block/{}/", self.base_url, enc(doc_id), enc(block_id)))
            .headers(self.headers())
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| PageIndexError::api(format!("Failed to get block: {e}")))?;
        if response.status() != reqwest::StatusCode::OK {
            return Err(Self::fail("get block", response).await);
        }
        Ok(response.json().await.unwrap_or(Value::Null))
    }

    pub async fn get_page_image(&self, doc_id: &str, page: usize) -> Result<String> {
        let response = self
            .http
            .get(format!("{}/doc/s3/{}/images", self.base_url, enc(doc_id)))
            .headers(self.headers())
            .query(&[("start", page.to_string()), ("end", page.to_string())])
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| PageIndexError::api(format!("Failed to get page image: {e}")))?;
        if response.status() != reqwest::StatusCode::OK {
            return Err(Self::fail("get page image", response).await);
        }
        let payload: Value = response.json().await.unwrap_or(Value::Null);
        for img in payload.get("images").and_then(Value::as_array).cloned().unwrap_or_default() {
            if img.get("page").and_then(Value::as_u64) == Some(page as u64)
                && let Some(url) = img.get("url").and_then(Value::as_str)
            {
                return Ok(url.to_owned());
            }
        }
        Err(PageIndexError::api(format!("No image URL returned for page {page}.")))
    }

    pub async fn get_document_image(&self, doc_id: &str, img_id: &str) -> Result<String> {
        let response = self
            .http
            .get(format!("{}/doc/{}/image/{}/", self.base_url, enc(doc_id), enc(img_id)))
            .headers(self.headers())
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| PageIndexError::api(format!("Failed to get document image: {e}")))?;
        if response.status() != reqwest::StatusCode::OK {
            return Err(Self::fail("get document image", response).await);
        }
        let payload: Value = response.json().await.unwrap_or(Value::Null);
        payload
            .get("url")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| PageIndexError::api(format!("No image URL returned for {img_id:?}.")))
    }

    pub async fn get_tree(&self, doc_id: &str, node_summary: bool, include_text: bool) -> Result<Value> {
        let response = self
            .http
            .get(format!(
                "{}/doc/{}/?type=tree&summary={node_summary}&include_text={}",
                self.base_url,
                enc(doc_id),
                if include_text { "true" } else { "false" }
            ))
            .headers(self.headers())
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| PageIndexError::api(format!("Failed to get tree result: {e}")))?;
        if response.status() != reqwest::StatusCode::OK {
            return Err(Self::fail("get tree result", response).await);
        }
        Ok(response.json().await.unwrap_or(Value::Null))
    }

    pub async fn submit_query(&self, doc_id: &str, query: &str, thinking: bool) -> Result<Value> {
        let response = self
            .http
            .post(format!("{}/retrieval/", self.base_url))
            .headers(self.headers())
            .json(&json!({"doc_id": doc_id, "query": query, "thinking": thinking}))
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| PageIndexError::api(format!("Failed to submit retrieval: {e}")))?;
        if response.status() != reqwest::StatusCode::OK {
            return Err(Self::fail("submit retrieval", response).await);
        }
        Ok(response.json().await.unwrap_or(Value::Null))
    }

    pub async fn get_retrieval(&self, retrieval_id: &str) -> Result<Value> {
        let response = self
            .http
            .get(format!("{}/retrieval/{}/", self.base_url, enc(retrieval_id)))
            .headers(self.headers())
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| PageIndexError::api(format!("Failed to get retrieval result: {e}")))?;
        if response.status() != reqwest::StatusCode::OK {
            return Err(Self::fail("get retrieval result", response).await);
        }
        Ok(response.json().await.unwrap_or(Value::Null))
    }

    pub async fn chat_completions(&self, payload: &Value, stream: bool) -> Result<Value> {
        let response = self
            .http
            .post(format!("{}/chat/completions/", self.base_url))
            .headers(self.headers())
            .json(payload)
            .timeout(Duration::from_secs(if stream { 120 } else { 600 }))
            .send()
            .await
            .map_err(|e| PageIndexError::api(format!("Failed to get chat completion: {e}")))?;
        if response.status() != reqwest::StatusCode::OK {
            return Err(Self::fail("get chat completion", response).await);
        }
        if stream {
            let text = response.text().await.unwrap_or_default();
            let mut chunks = Vec::new();
            for line in text.lines() {
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };
                if data == "[DONE]" {
                    break;
                }
                if let Ok(chunk) = serde_json::from_str::<Value>(data) {
                    if chunk.get("error").is_some() {
                        return Err(PageIndexError::api(format!(
                            "Chat completion failed mid-stream: {}",
                            chunk["error"]
                        )));
                    }
                    chunks.push(chunk);
                }
            }
            Ok(Value::Array(chunks))
        } else {
            Ok(response.json().await.unwrap_or(Value::Null))
        }
    }

    pub async fn get_document(&self, doc_id: &str) -> Result<Value> {
        let response = self
            .http
            .get(format!("{}/doc/{}/metadata/", self.base_url, enc(doc_id)))
            .headers(self.headers())
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| PageIndexError::api(format!("Failed to get document metadata: {e}")))?;
        if response.status() != reqwest::StatusCode::OK {
            return Err(Self::fail("get document metadata", response).await);
        }
        Ok(response.json().await.unwrap_or(Value::Null))
    }

    pub async fn delete_document(&self, doc_id: &str) -> Result<Value> {
        let response = self
            .http
            .delete(format!("{}/doc/{}/", self.base_url, enc(doc_id)))
            .headers(self.headers())
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| PageIndexError::api(format!("Failed to delete document: {e}")))?;
        if response.status() != reqwest::StatusCode::OK {
            return Err(Self::fail("delete document", response).await);
        }
        let bytes = response.bytes().await.unwrap_or_default();
        if bytes.is_empty() { Ok(json!({})) } else { Ok(serde_json::from_slice(&bytes).unwrap_or(json!({}))) }
    }

    pub async fn list_documents(
        &self,
        limit: usize,
        offset: usize,
        folder_id: Option<&str>,
        name: Option<&str>,
        recursive: bool,
    ) -> Result<Value> {
        if !(1..=10000).contains(&limit) {
            return Err(PageIndexError::invalid("limit must be between 1 and 10000"));
        }
        let mut query: Vec<(String, String)> =
            vec![("limit".into(), limit.to_string()), ("offset".into(), offset.to_string())];
        if let Some(folder_id) = folder_id {
            query.push(("folder_id".into(), folder_id.to_owned()));
        }
        if let Some(name) = name {
            query.push(("name".into(), name.to_owned()));
        }
        if recursive {
            query.push(("recursive".into(), "true".into()));
        }
        let response = self
            .http
            .get(format!("{}/docs/", self.base_url))
            .headers(self.headers())
            .query(&query)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| PageIndexError::api(format!("Failed to list documents: {e}")))?;
        if response.status() != reqwest::StatusCode::OK {
            return Err(Self::fail("list documents", response).await);
        }
        Ok(response.json().await.unwrap_or(Value::Null))
    }

    pub async fn create_folder(
        &self,
        name: &str,
        description: Option<&str>,
        parent_folder_id: Option<&str>,
    ) -> Result<Value> {
        validate_folder_name(name)?;
        let mut payload = json!({"name": name});
        if let Some(description) = description {
            payload["description"] = json!(description);
        }
        if let Some(parent) = parent_folder_id {
            payload["parent_folder_id"] = json!(parent);
        }
        let response = self
            .http
            .post(format!("{}/folder/", self.base_url))
            .headers(self.headers())
            .json(&payload)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| PageIndexError::api(format!("Failed to create folder: {e}")))?;
        if response.status() != reqwest::StatusCode::OK {
            return Err(Self::fail("create folder", response).await);
        }
        Ok(response.json().await.unwrap_or(Value::Null))
    }

    pub async fn list_folders(&self, parent_folder_id: Option<&str>) -> Result<Value> {
        let mut req = self.http.get(format!("{}/folders/", self.base_url)).headers(self.headers());
        if let Some(parent) = parent_folder_id {
            req = req.query(&[("parent_folder_id", parent)]);
        }
        let response = req
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| PageIndexError::api(format!("Failed to list folders: {e}")))?;
        if response.status() != reqwest::StatusCode::OK {
            return Err(Self::fail("list folders", response).await);
        }
        Ok(response.json().await.unwrap_or(Value::Null))
    }
}

/// Cloud-pinned client: documents live on PageIndex cloud.
pub struct PageIndexCloudClient {
    pub api: CloudApi,
    pub instructions: Option<String>,
    pub chat_model: Option<String>,
    pub chat_backend: Option<crate::types::Backend>,
}

impl PageIndexCloudClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self { api: CloudApi::new(api_key), instructions: None, chat_model: None, chat_backend: None }
    }

    pub fn from_env() -> Result<Self> {
        let key = crate::config::env_var("PAGEINDEX_API_KEY").ok_or_else(|| {
            PageIndexError::api(
                "PageIndexCloudClient requires a PageIndex API key — pass api_key=..., or export PAGEINDEX_API_KEY. Get one at https://developer.pageindex.ai/api-keys.",
            )
        })?;
        Ok(Self::new(key))
    }

    pub fn with_chat_model(mut self, model: impl Into<String>) -> Self {
        self.chat_model = Some(model.into());
        self
    }

    pub fn with_instructions(mut self, text: impl Into<String>) -> Self {
        self.instructions = Some(text.into());
        self
    }
}

#[async_trait::async_trait]
impl crate::agent::AgentBackend for PageIndexCloudClient {
    fn api_key(&self) -> Option<&str> {
        Some(&self.api.api_key)
    }

    fn instructions(&self) -> Option<&str> {
        self.instructions.as_deref()
    }

    fn base_url(&self) -> &str {
        &self.api.base_url
    }

    async fn list_documents(&self, limit: usize, offset: usize) -> Result<Value> {
        self.api.list_documents(limit, offset, None, None, false).await
    }

    async fn get_document(&self, doc_id: &str) -> Result<Value> {
        self.api.get_document(doc_id).await
    }

    async fn get_tree(&self, doc_id: &str, node_summary: bool, include_text: bool) -> Result<Value> {
        self.api.get_tree(doc_id, node_summary, include_text).await
    }

    async fn get_ocr_pages(&self, doc_id: &str) -> Result<Value> {
        self.api.get_ocr(doc_id, "page").await
    }

    async fn delete_document(&self, doc_id: &str) -> Result<()> {
        self.api.delete_document(doc_id).await.map(|_| ())
    }

    async fn get_block(&self, doc_id: &str, block_id: &str) -> Result<Value> {
        self.api.get_block(doc_id, block_id).await
    }

    async fn list_folders(&self) -> Result<Value> {
        self.api.list_folders(None).await
    }
}
