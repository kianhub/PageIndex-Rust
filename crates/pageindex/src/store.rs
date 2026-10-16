//! On-disk document store behind the local client:
//! `<root>/docs/<doc_id>/{doc,tree,pages}.json` plus a `manifest.json` cache.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::error::{PageIndexError, Result};
use crate::tree::Node;

/// Stored document metadata (the cloud API's field names).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocMeta {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub status: String,
    #[serde(rename = "createdAt")]
    pub created_at: String,
    #[serde(rename = "pageNum")]
    pub page_num: usize,
    #[serde(rename = "folderId")]
    pub folder_id: Option<String>,
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default)]
    pub mode: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredPage {
    pub page_index: usize,
    pub markdown: String,
}

#[derive(Debug, Clone)]
pub struct DocStore {
    root: PathBuf,
}

fn is_safe_id(id: &str) -> bool {
    !id.is_empty() && id != "." && id != ".." && !id.contains(['/', '\\'])
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    let data = serde_json::to_vec(value)?;
    let result = std::fs::write(&tmp, data).and_then(|_| std::fs::rename(&tmp, path));
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    Ok(result?)
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Option<T> {
    let data = std::fs::read(path).ok()?;
    serde_json::from_slice(&data).ok()
}

impl DocStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn docs(&self) -> PathBuf {
        self.root.join("docs")
    }

    fn manifest_path(&self) -> PathBuf {
        self.root.join("manifest.json")
    }

    fn doc_dir(&self, id: &str) -> Option<PathBuf> {
        is_safe_id(id).then(|| self.docs().join(id))
    }

    fn read_manifest(&self) -> BTreeMap<String, DocMeta> {
        #[derive(Deserialize)]
        struct Manifest {
            docs: BTreeMap<String, DocMeta>,
        }
        read_json::<Manifest>(&self.manifest_path()).map(|m| m.docs).unwrap_or_default()
    }

    fn write_manifest(&self, docs: &BTreeMap<String, DocMeta>) {
        let _ = write_json_atomic(&self.manifest_path(), &serde_json::json!({ "docs": docs }));
    }

    pub fn save_document(&self, meta: &DocMeta, tree: &[Node], pages: &[StoredPage]) -> Result<()> {
        let dir =
            self.doc_dir(&meta.id).ok_or_else(|| PageIndexError::invalid(format!("Invalid doc_id: {:?}", meta.id)))?;
        std::fs::create_dir_all(&dir)?;
        write_json_atomic(&dir.join("tree.json"), &tree)?;
        write_json_atomic(&dir.join("pages.json"), &pages)?;
        write_json_atomic(&dir.join("doc.json"), meta)?;
        let mut manifest = self.read_manifest();
        manifest.insert(meta.id.clone(), meta.clone());
        self.write_manifest(&manifest);
        Ok(())
    }

    fn doc_file<T: DeserializeOwned>(&self, id: &str, name: &str) -> Option<T> {
        let dir = self.doc_dir(id)?;
        if !dir.join("doc.json").is_file() {
            return None;
        }
        read_json(&dir.join(name))
    }

    pub fn get_meta(&self, id: &str) -> Option<DocMeta> {
        let dir = self.doc_dir(id)?;
        if !dir.join("doc.json").is_file() {
            return None;
        }
        read_json::<DocMeta>(&dir.join("doc.json"))
            .filter(|m| m.id == id)
            .or_else(|| self.read_manifest().remove(id).filter(|m| m.id == id))
    }

    pub fn get_tree(&self, id: &str) -> Option<Vec<Node>> {
        self.doc_file(id, "tree.json")
    }

    pub fn get_pages(&self, id: &str) -> Option<Vec<StoredPage>> {
        self.doc_file(id, "pages.json")
    }

    pub fn list_metas(&self) -> Vec<DocMeta> {
        let Ok(entries) = std::fs::read_dir(self.docs()) else {
            return Vec::new();
        };
        let cached = self.read_manifest();
        let mut fresh = BTreeMap::new();
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !is_safe_id(&name) || !entry.path().join("doc.json").is_file() {
                continue;
            }
            let meta = cached
                .get(&name)
                .cloned()
                .filter(|m| m.id == name)
                .or_else(|| read_json::<DocMeta>(&entry.path().join("doc.json")).filter(|m| m.id == name));
            if let Some(meta) = meta {
                fresh.insert(name, meta);
            }
        }
        if fresh != cached {
            self.write_manifest(&fresh);
        }
        fresh.into_values().collect()
    }

    pub fn delete_document(&self, id: &str) -> bool {
        let Some(dir) = self.doc_dir(id) else {
            return false;
        };
        let existed = std::fs::remove_file(dir.join("doc.json")).is_ok();
        if dir.is_dir() {
            let _ = std::fs::remove_dir_all(&dir);
        }
        let mut manifest = self.read_manifest();
        if manifest.remove(id).is_some() {
            self.write_manifest(&manifest);
        }
        existed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(id: &str) -> DocMeta {
        DocMeta {
            id: id.into(),
            name: "a.pdf".into(),
            description: None,
            status: "completed".into(),
            created_at: "2026-01-01T00:00:00.000".into(),
            page_num: 1,
            folder_id: None,
            metadata: None,
            mode: Some("flash".into()),
        }
    }

    #[test]
    fn round_trip_and_delete() {
        let dir = tempfile::tempdir().unwrap();
        let store = DocStore::new(dir.path());
        let tree = vec![Node::new("A", 1, 1)];
        let pages = vec![StoredPage { page_index: 1, markdown: "hello".into() }];
        store.save_document(&meta("pi-1"), &tree, &pages).unwrap();
        assert_eq!(store.get_meta("pi-1").unwrap().name, "a.pdf");
        assert_eq!(store.get_tree("pi-1").unwrap(), tree);
        assert_eq!(store.get_pages("pi-1").unwrap(), pages);
        assert_eq!(store.list_metas().len(), 1);
        assert!(store.get_meta("../x").is_none());
        assert!(store.delete_document("pi-1"));
        assert!(!store.delete_document("pi-1"));
        assert!(store.list_metas().is_empty());
    }
}
