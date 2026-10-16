//! Embedded bookmarks (the document outline), flattened depth-first.

use std::collections::{HashMap, HashSet};

use lopdf::{Dictionary, Document, Object, ObjectId};

use crate::objects::{as_dict, dict_get, resolve, text_string};

/// One outline entry. `level` and `page` are 1-based.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bookmark {
    pub title: String,
    pub level: usize,
    pub page: usize,
}

const MAX_ENTRIES: usize = 100_000;

pub fn read(doc: &Document, page_numbers: &HashMap<ObjectId, usize>) -> Vec<Bookmark> {
    let Ok(catalog) = doc.catalog() else {
        return Vec::new();
    };
    let Some(outlines) = dict_get(doc, catalog, b"Outlines").and_then(|o| as_dict(doc, o)) else {
        return Vec::new();
    };
    let names = NamedDests::new(doc, catalog);
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    walk(doc, outlines.get(b"First").ok(), 1, page_numbers, &names, &mut seen, &mut out);
    out
}

fn walk<'a>(
    doc: &'a Document,
    mut current: Option<&'a Object>,
    level: usize,
    pages: &HashMap<ObjectId, usize>,
    names: &NamedDests<'a>,
    seen: &mut HashSet<ObjectId>,
    out: &mut Vec<Bookmark>,
) {
    while let Some(Object::Reference(id)) = current {
        if !seen.insert(*id) || out.len() >= MAX_ENTRIES || level > 64 {
            return;
        }
        let Some(item) = doc.get_object(*id).ok().and_then(|o| as_dict(doc, o)) else {
            return;
        };
        let title = match dict_get(doc, item, b"Title") {
            Some(Object::String(bytes, _)) => text_string(bytes),
            _ => String::new(),
        };
        let page = destination_of(doc, item, names).and_then(|dest| page_of(doc, dest, pages));
        let title = title.trim().to_owned();
        if let Some(page) = page.filter(|_| !title.is_empty()) {
            out.push(Bookmark { title, level, page });
        }
        walk(doc, item.get(b"First").ok(), level + 1, pages, names, seen, out);
        current = item.get(b"Next").ok();
    }
}

fn destination_of<'a>(doc: &'a Document, item: &'a Dictionary, names: &'a NamedDests) -> Option<&'a Object> {
    let dest = dict_get(doc, item, b"Dest").or_else(|| {
        let action = dict_get(doc, item, b"A").and_then(|a| as_dict(doc, a))?;
        dict_get(doc, action, b"D")
    })?;
    names.resolve(doc, dest)
}

fn page_of(doc: &Document, dest: &Object, pages: &HashMap<ObjectId, usize>) -> Option<usize> {
    let array = match resolve(doc, dest) {
        Object::Array(items) => items,
        Object::Dictionary(d) => match dict_get(doc, d, b"D") {
            Some(Object::Array(items)) => items,
            _ => return None,
        },
        _ => return None,
    };
    match array.first()? {
        Object::Reference(id) => pages.get(id).copied(),
        Object::Integer(n) if *n >= 0 => Some(*n as usize + 1),
        _ => None,
    }
}

struct NamedDests<'a> {
    tree: Option<&'a Dictionary>,
    legacy: Option<&'a Dictionary>,
}

impl<'a> NamedDests<'a> {
    fn new(doc: &'a Document, catalog: &'a Dictionary) -> Self {
        let tree = dict_get(doc, catalog, b"Names")
            .and_then(|n| as_dict(doc, n))
            .and_then(|n| dict_get(doc, n, b"Dests"))
            .and_then(|d| as_dict(doc, d));
        let legacy = dict_get(doc, catalog, b"Dests").and_then(|d| as_dict(doc, d));
        Self { tree, legacy }
    }

    fn resolve(&self, doc: &'a Document, dest: &'a Object) -> Option<&'a Object> {
        match dest {
            Object::Name(name) => self.legacy.and_then(|d| dict_get(doc, d, name)),
            Object::String(key, _) => self
                .tree
                .and_then(|t| name_tree_lookup(doc, t, key, 0))
                .or_else(|| self.legacy.and_then(|d| dict_get(doc, d, key))),
            other => Some(other),
        }
    }
}

fn name_tree_lookup<'a>(doc: &'a Document, node: &'a Dictionary, key: &[u8], depth: usize) -> Option<&'a Object> {
    if depth > 32 {
        return None;
    }
    if let Some(Object::Array(pairs)) = dict_get(doc, node, b"Names") {
        for pair in pairs.chunks(2) {
            if let [Object::String(name, _), value] = pair
                && name.as_slice() == key
            {
                return Some(resolve(doc, value));
            }
        }
    }
    if let Some(Object::Array(kids)) = dict_get(doc, node, b"Kids") {
        for kid in kids {
            let Some(kid) = as_dict(doc, kid) else { continue };
            if let Some(Object::Array(limits)) = dict_get(doc, kid, b"Limits")
                && let [Object::String(lo, _), Object::String(hi, _)] = limits.as_slice()
                && (key < lo.as_slice() || key > hi.as_slice())
            {
                continue;
            }
            if let Some(found) = name_tree_lookup(doc, kid, key, depth + 1) {
                return Some(found);
            }
        }
    }
    None
}
