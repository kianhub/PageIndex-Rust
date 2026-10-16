//! `chat(stream=True)`'s return type: one run, one view — text or events.

use std::sync::Mutex;

use crate::error::{PageIndexError, Result};

/// A typed process event from a chat run.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ChatEvent {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<serde_json::Value>,
}

impl ChatEvent {
    pub fn thinking(delta: impl Into<String>) -> Self {
        Self {
            kind: "thinking".into(),
            delta: Some(delta.into()),
            call_id: None,
            name: None,
            arguments: None,
            output: None,
        }
    }

    pub fn answer(delta: impl Into<String>) -> Self {
        Self {
            kind: "answer".into(),
            delta: Some(delta.into()),
            call_id: None,
            name: None,
            arguments: None,
            output: None,
        }
    }

    pub fn tool_call(call_id: impl Into<String>, name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            kind: "tool_call".into(),
            delta: None,
            call_id: Some(call_id.into()),
            name: Some(name.into()),
            arguments: Some(arguments),
            output: None,
        }
    }

    pub fn tool_result(call_id: impl Into<String>, name: impl Into<String>, output: serde_json::Value) -> Self {
        Self {
            kind: "tool_result".into(),
            delta: None,
            call_id: Some(call_id.into()),
            name: Some(name.into()),
            arguments: None,
            output: Some(output),
        }
    }
}

/// `chat(stream=True)`: iterate it for answer text, or read `.events` for
/// typed process dicts. One run serves one view.
pub struct ChatStream {
    text: Mutex<Option<Vec<String>>>,
    events: Mutex<Option<Result<Vec<ChatEvent>>>>,
    view: Mutex<Option<&'static str>>,
    closed: Mutex<bool>,
}

impl ChatStream {
    pub fn from_parts(text: Vec<String>, events: Vec<ChatEvent>) -> Self {
        Self {
            text: Mutex::new(Some(text)),
            events: Mutex::new(Some(Ok(events))),
            view: Mutex::new(None),
            closed: Mutex::new(false),
        }
    }

    pub fn from_text(text: impl Into<String>) -> Self {
        let text = text.into();
        let events = vec![ChatEvent::answer(text.clone())];
        Self::from_parts(if text.is_empty() { Vec::new() } else { vec![text] }, events)
    }

    /// Events view refused (managed cloud chunk stream has no event iterator).
    pub fn text_only(text: Vec<String>, refusal: impl Into<String>) -> Self {
        Self {
            text: Mutex::new(Some(text)),
            events: Mutex::new(Some(Err(PageIndexError::api(refusal.into())))),
            view: Mutex::new(None),
            closed: Mutex::new(false),
        }
    }

    fn claim(&self, view: &'static str) -> Result<()> {
        let mut held = self.view.lock().expect("chat stream lock");
        if let Some(current) = *held {
            if current != view {
                return Err(PageIndexError::api(format!(
                    "This chat stream is being consumed as {current}; one run serves one view — call chat() again for the other."
                )));
            }
        } else {
            *held = Some(view);
        }
        Ok(())
    }

    /// The answer text pieces (and, with `show_process`, the woven display).
    pub fn collect_text(&self) -> Result<String> {
        self.claim("text")?;
        if *self.closed.lock().expect("chat stream lock") {
            return Ok(String::new());
        }
        Ok(self.text.lock().expect("chat stream lock").take().unwrap_or_default().join(""))
    }

    /// Typed process events. Consuming — not merely constructing the stream
    /// — claims the view.
    pub fn collect_events(&self) -> Result<Vec<ChatEvent>> {
        if let Some(Err(err)) = self.events.lock().expect("chat stream lock").as_ref() {
            return Err(err.clone());
        }
        self.claim("events")?;
        if *self.closed.lock().expect("chat stream lock") {
            return Ok(Vec::new());
        }
        match self.events.lock().expect("chat stream lock").take() {
            Some(Ok(events)) => Ok(events),
            Some(Err(err)) => Err(err),
            None => Ok(Vec::new()),
        }
    }

    pub fn close(&self) {
        *self.closed.lock().expect("chat stream lock") = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_view_per_run() {
        let stream = ChatStream::from_text("hello");
        assert_eq!(stream.collect_text().unwrap(), "hello");
        assert!(stream.collect_events().unwrap_err().message.contains("text"));
    }
}
