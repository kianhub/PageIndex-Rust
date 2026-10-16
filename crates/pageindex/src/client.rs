//! The local client: index documents in-process and keep them under a
//! storage directory. Responses use the cloud API's envelope and field names.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::config::{ConfigLoader, ConfigOverrides};
use crate::error::{ErrorKind, PageIndexError, Result};
use crate::flash::{FlashOptions, flash_rejection_reason, page_index_flash};
use crate::llm::{LiteLlm, LlmClient};
use crate::naming::{MAX_NAME_BYTES, sanitize_filename, truncate_filename};
use crate::store::{DocMeta, DocStore, StoredPage};
use crate::summary::generate_doc_description;
use crate::tree::{Node, add_node_text, write_node_ids};
use crate::types::{LocalIndexConfig, OptimizeMode};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SubmitResult {
    pub doc_id: String,
    pub name: String,
}

/// `{doc_id, status, retrieval_ready, result, metadata}` as both modes return it.
#[derive(Debug, Clone, Serialize)]
pub struct Envelope<T> {
    pub doc_id: String,
    pub status: String,
    pub retrieval_ready: bool,
    pub result: T,
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DocumentList {
    pub documents: Vec<DocMeta>,
    pub total: usize,
    pub limit: usize,
    pub offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrFormat {
    Page,
    Node,
    Raw,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum OcrResult {
    Pages(Vec<StoredPage>),
    Nodes(Vec<OcrNode>),
    Raw(String),
}

#[derive(Debug, Clone, Serialize)]
pub struct OcrNode {
    pub title: String,
    pub level: usize,
    pub page_index: usize,
    pub text: String,
}

#[derive(Debug, Clone, Default)]
pub struct SubmitOptions {
    /// `"flash"` (default) or `"standard"`.
    pub mode: Option<String>,
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

pub struct PageIndexLocalClient {
    store: DocStore,
    summary_model: String,
    config: LocalIndexConfig,
    llm: Arc<dyn LlmClient>,
}

impl PageIndexLocalClient {
    pub fn new(storage_path: impl Into<PathBuf>) -> Self {
        Self::with_config(LocalIndexConfig { storage_path: Some(storage_path.into()), ..Default::default() })
    }

    pub fn with_config(config: LocalIndexConfig) -> Self {
        let resolved = ConfigLoader::new().load(&ConfigOverrides {
            model: config.model.clone(),
            summary_model: config.summary_model.clone(),
            ..Default::default()
        });
        let llm: Arc<dyn LlmClient> = match &config.backend {
            Some(backend) => Arc::new(LiteLlm::with_backend(backend.clone())),
            None => Arc::new(LiteLlm::new()),
        };
        let root = config.storage_path.clone().unwrap_or_else(|| PathBuf::from("./pageindex_store"));
        Self { store: DocStore::new(root), summary_model: resolved.summary_model, config, llm }
    }

    /// Use this model client for every indexing call.
    pub fn with_llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.llm = llm;
        self
    }

    fn unique_name(&self, name: &str) -> Result<String> {
        let taken: Vec<String> = self.store.list_metas().into_iter().map(|m| m.name).collect();
        if !taken.iter().any(|t| t == name) {
            return Ok(name.to_owned());
        }
        for n in 1..100 {
            let candidate = truncate_filename(name, MAX_NAME_BYTES, &format!("_{n}"))?;
            if !taken.contains(&candidate) {
                return Ok(candidate);
            }
        }
        Err(PageIndexError::api(
            "Failed to submit document: Too many files with similar names. Please use a different file name.",
        ))
    }

    /// Index a PDF (Flash by default) and store it. Returns the stored name,
    /// which gains a `_N` suffix when the name is taken.
    pub async fn submit_document(&self, file_path: impl AsRef<Path>, options: SubmitOptions) -> Result<SubmitResult> {
        let mode = options.mode.as_deref().unwrap_or("flash");
        if mode == "standard" {
            return Err(PageIndexError::api(
                "Failed to submit document: standard mode is not available in this build yet; use mode='flash'.",
            ));
        }
        if mode != "flash" {
            return Err(PageIndexError::api(format!(
                "Failed to submit document: unknown local processing mode {mode:?}. Supported: 'flash' (default) or 'standard'."
            )));
        }
        let path = file_path.as_ref();
        if !path.is_file() {
            return Err(PageIndexError::new(ErrorKind::NotFound, format!("No such file: {}", path.display())));
        }
        let file_name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let doc_name = sanitize_filename(&file_name);
        if !doc_name.to_lowercase().ends_with(".pdf") {
            return Err(PageIndexError::api("Failed to submit document: only PDF files are supported in local mode."));
        }
        let flash = FlashOptions {
            summary: true,
            summary_model: Some(self.summary_model.clone()),
            optimize: self.config.optimize.unwrap_or(OptimizeMode::Full),
            optimize_model: Some(self.summary_model.clone()),
            summary_concurrency: self.config.summary_concurrency,
            use_embedded_toc: self.config.use_embedded_toc.unwrap_or(true),
            summary_max_words: self.config.summary_max_words,
            summary_reasoning_effort: self.config.summary_reasoning_effort.clone(),
            llm: Some(self.llm.clone()),
        };
        let result = page_index_flash(path, &flash).await.map_err(|e| match e.kind {
            ErrorKind::NotFound => e,
            _ => PageIndexError::api(format!("Failed to submit document: {}", e.message)).with_status(e.status_code),
        })?;
        if !result.page_texts.iter().any(|t| !t.trim().is_empty()) {
            return Err(PageIndexError::api("Failed to submit document: PDF has no content. All pages are blank."));
        }
        if let Some(reason) = flash_rejection_reason(&result.structure, result.toc_source, "mode='standard'") {
            return Err(PageIndexError::api(format!("Failed to submit document: {reason}")));
        }
        let mut structure = result.structure;
        write_node_ids(&mut structure);
        let description = generate_doc_description(&structure, self.llm.as_ref(), &self.summary_model).await?;
        let page_count = result.page_texts.len();
        check_page_bounds(&structure, page_count)?;
        let pages: Vec<StoredPage> = result
            .page_texts
            .into_iter()
            .enumerate()
            .map(|(i, markdown)| StoredPage { page_index: i + 1, markdown })
            .collect();
        crate::tree::remove_text(&mut structure);
        let meta = DocMeta {
            id: new_doc_id(),
            name: self.unique_name(&doc_name)?,
            description: Some(description),
            status: "completed".into(),
            created_at: now_iso(),
            page_num: page_count,
            folder_id: None,
            metadata: options.metadata,
            mode: Some(mode.to_owned()),
        };
        self.store.save_document(&meta, &structure, &pages)?;
        Ok(SubmitResult { doc_id: meta.id, name: meta.name })
    }

    fn require_doc(&self, doc_id: &str, prefix: &str) -> Result<DocMeta> {
        self.store
            .get_meta(doc_id)
            .ok_or_else(|| PageIndexError::api(format!("{prefix}: Document not found.")).with_status(Some(404)))
    }

    fn require_pages(&self, doc_id: &str, prefix: &str) -> Result<Vec<StoredPage>> {
        match self.store.get_pages(doc_id) {
            None => Err(PageIndexError::api(format!("{prefix}: stored document data is unreadable."))),
            Some(pages) if pages.is_empty() => {
                Err(PageIndexError::api(format!("{prefix}: stored document has no page content.")))
            }
            Some(pages) => Ok(pages),
        }
    }

    fn tree_with_text(&self, doc_id: &str, prefix: &str) -> Result<Vec<Node>> {
        let mut tree = self
            .store
            .get_tree(doc_id)
            .ok_or_else(|| PageIndexError::api(format!("{prefix}: stored document data is unreadable.")))?;
        let pages: Vec<String> = self.require_pages(doc_id, prefix)?.into_iter().map(|p| p.markdown).collect();
        add_node_text(&mut tree, &pages);
        Ok(tree)
    }

    pub fn get_document(&self, doc_id: &str) -> Result<DocMeta> {
        self.require_doc(doc_id, "Failed to get document metadata")
    }

    pub fn delete_document(&self, doc_id: &str) -> Result<()> {
        if self.store.delete_document(doc_id) {
            Ok(())
        } else {
            Err(PageIndexError::api("Failed to delete document: Document not found.").with_status(Some(404)))
        }
    }

    /// Newest first; `name` filters by exact stored name.
    pub fn list_documents(&self, limit: usize, offset: usize, name: Option<&str>) -> Result<DocumentList> {
        if !(1..=10000).contains(&limit) {
            return Err(PageIndexError::invalid("limit must be between 1 and 10000"));
        }
        let mut metas = self.store.list_metas();
        metas.sort_by(|a, b| b.created_at.cmp(&a.created_at).then(a.id.cmp(&b.id)));
        if let Some(name) = name {
            metas.retain(|m| m.name == name);
        }
        let total = metas.len();
        let documents = metas.into_iter().skip(offset).take(limit).collect();
        Ok(DocumentList { documents, total, limit, offset })
    }

    /// The stored tree; node text is each node's own pages, summaries on request.
    pub fn get_tree(&self, doc_id: &str, node_summary: bool, include_text: bool) -> Result<Envelope<Vec<Node>>> {
        let prefix = "Failed to get tree result";
        let meta = self.require_doc(doc_id, prefix)?;
        let mut tree = if include_text {
            self.tree_with_text(doc_id, prefix)?
        } else {
            self.store
                .get_tree(doc_id)
                .ok_or_else(|| PageIndexError::api(format!("{prefix}: stored document data is unreadable.")))?
        };
        if !node_summary {
            crate::tree::for_each_mut(&mut tree, &mut |n| n.summary = None);
        }
        Ok(envelope(doc_id, tree, meta))
    }

    /// The structure without text, summaries included.
    pub fn get_document_structure(&self, doc_id: &str) -> Result<Vec<Node>> {
        Ok(self.get_tree(doc_id, true, false)?.result)
    }

    pub fn get_ocr(&self, doc_id: &str, format: OcrFormat) -> Result<Envelope<OcrResult>> {
        let prefix = "Failed to get OCR result";
        let meta = self.require_doc(doc_id, prefix)?;
        let result = match format {
            OcrFormat::Node => {
                let tree = self.tree_with_text(doc_id, prefix)?;
                let mut out = Vec::new();
                fn walk(nodes: &[Node], level: usize, out: &mut Vec<OcrNode>) {
                    for n in nodes {
                        out.push(OcrNode {
                            title: n.title.clone(),
                            level,
                            page_index: n.start_index,
                            text: n.text.clone().unwrap_or_default(),
                        });
                        walk(&n.nodes, level + 1, out);
                    }
                }
                walk(&tree, 1, &mut out);
                OcrResult::Nodes(out)
            }
            OcrFormat::Page => OcrResult::Pages(self.require_pages(doc_id, prefix)?),
            OcrFormat::Raw => OcrResult::Raw(
                self.require_pages(doc_id, prefix)?.into_iter().map(|p| p.markdown).collect::<Vec<_>>().join("\n\n"),
            ),
        };
        Ok(envelope(doc_id, result, meta))
    }

    /// Pages by specifier: `"5-7"`, `"3,8"` or `"12"`.
    pub fn get_page_content(&self, doc_id: &str, pages: &str) -> Result<Vec<StoredPage>> {
        let wanted = parse_pages(pages)?;
        let all = self.require_pages(doc_id, "Failed to get OCR result")?;
        Ok(all.into_iter().filter(|p| wanted.contains(&p.page_index)).collect())
    }

    pub fn is_retrieval_ready(&self, doc_id: &str) -> bool {
        self.store.get_meta(doc_id).is_some_and(|m| m.status == "completed")
    }
}

fn envelope<T>(doc_id: &str, result: T, meta: DocMeta) -> Envelope<T> {
    Envelope {
        doc_id: doc_id.to_owned(),
        status: "completed".into(),
        retrieval_ready: true,
        result,
        metadata: meta.metadata,
    }
}

fn check_page_bounds(structure: &[Node], page_count: usize) -> Result<()> {
    for node in crate::tree::subtree(structure) {
        if node.start_index < 1 || node.end_index > page_count {
            return Err(PageIndexError::api(format!(
                "Failed to submit document: the extracted structure references pages {}-{} outside the PDF's {page_count} readable pages.",
                node.start_index, node.end_index
            )));
        }
    }
    Ok(())
}

/// `"5-7"`, `"3,8"`, `"12"` (and combinations) to page numbers.
pub fn parse_pages(spec: &str) -> Result<Vec<usize>> {
    let bad = || PageIndexError::invalid(format!("Invalid pages specifier: {spec:?}"));
    let mut out = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b): (usize, usize) =
                    (a.trim().parse().map_err(|_| bad())?, b.trim().parse().map_err(|_| bad())?);
                if a == 0 || b < a {
                    return Err(bad());
                }
                out.extend(a..=b);
            }
            None => {
                out.push(part.parse().map_err(|_| bad()).and_then(|p: usize| if p == 0 { Err(bad()) } else { Ok(p) })?)
            }
        }
    }
    if out.is_empty() {
        return Err(bad());
    }
    Ok(out)
}

fn new_doc_id() -> String {
    format!("pi-{}", uuid::Uuid::new_v4().simple())
}

/// Naive UTC, millisecond precision (`2026-10-16T00:35:00.123`).
fn now_iso() -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = now.as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{:03}", rem / 3600, rem % 3600 / 60, rem % 60, now.subsec_millis())
}

/// Days since 1970-01-01 to (year, month, day), proleptic Gregorian.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_specifiers() {
        assert_eq!(parse_pages("5-7").unwrap(), [5, 6, 7]);
        assert_eq!(parse_pages("3,8").unwrap(), [3, 8]);
        assert_eq!(parse_pages("12").unwrap(), [12]);
        assert!(parse_pages("0").is_err() && parse_pages("x").is_err() && parse_pages("7-5").is_err());
    }

    #[test]
    fn dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_742), (2026, 10, 16));
        assert_eq!(now_iso().len(), 23);
    }
}
