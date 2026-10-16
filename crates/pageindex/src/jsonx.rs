//! JSON extraction from model replies, matching `utils.extract_json`.

use serde_json::Value;

/// The first JSON object or array in a model reply, with light repairs.
/// Returns an empty object when nothing parseable is found.
pub fn extract_json(content: &str) -> Value {
    let mut text = content.trim().to_owned();
    if let Some(start) = text.find("```json") {
        text = text[start + 7..].to_owned();
        if let Some(end) = text.rfind("```") {
            text = text[..end].to_owned();
        }
    } else if text.contains("```")
        && let (Some(start), Some(end)) = (text.find("```"), text.rfind("```"))
        && end > start + 3
    {
        let inner = &text[start + 3..end];
        let inner = inner.strip_prefix("json").unwrap_or(inner);
        text = inner.to_owned();
    }
    text = text.replace("None", "null");
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let repaired = collapsed.replace(",]", "]").replace(",}", "}");
    for candidate in [collapsed.as_str(), repaired.as_str(), text.trim()] {
        if let Some(value) = parse_value(candidate) {
            return value;
        }
    }
    Value::Object(serde_json::Map::new())
}

fn parse_value(text: &str) -> Option<Value> {
    if let Ok(value) = serde_json::from_str::<Value>(text) {
        return Some(value);
    }
    let start = text.find(['{', '['])?;
    let end = text.rfind(['}', ']'])?;
    if end <= start {
        return None;
    }
    serde_json::from_str(&text[start..=end]).ok()
}

/// `"yes"` / `"no"` (any case) from a JSON field, defaulting to `no`.
pub fn json_yes(value: &Value, key: &str) -> bool {
    match value.get(key) {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => s.eq_ignore_ascii_case("yes") || s.eq_ignore_ascii_case("true"),
        _ => false,
    }
}

pub fn json_str(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(|v| match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Null => None,
        other => Some(other.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_fenced_and_repaired_json() {
        let value = extract_json("```json\n{\"a\": 1,}\n```");
        assert_eq!(value["a"], 1);
        let value = extract_json("prefix [{\"x\": true}] suffix");
        assert_eq!(value[0]["x"], true);
        assert!(extract_json("not json").as_object().is_some_and(|o| o.is_empty()));
    }
}
