use std::fmt;

/// The SDK's error type, mirroring `PageIndexAPIError`: a message plus the
/// HTTP status when the failing call had one. `None` does not imply the
/// failure was local.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageIndexError {
    pub message: String,
    pub status_code: Option<u16>,
    pub kind: ErrorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Bad arguments or unsupported option combinations.
    InvalidInput,
    /// A file that does not exist.
    NotFound,
    /// A file that could not be read or parsed.
    Document,
    /// A model call failed.
    Llm,
    /// A PageIndex API (local or cloud) call failed.
    Api,
    Io,
}

impl PageIndexError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self { message: message.into(), status_code: None, kind }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidInput, message)
    }

    pub fn api(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Api, message)
    }

    pub fn with_status(mut self, status: Option<u16>) -> Self {
        self.status_code = status;
        self
    }

    /// No retry can fix a rejected key or a model that does not exist.
    pub fn is_unrecoverable(&self) -> bool {
        matches!(self.status_code, Some(401 | 403 | 404))
    }
}

impl fmt::Display for PageIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PageIndexError {}

impl From<std::io::Error> for PageIndexError {
    fn from(err: std::io::Error) -> Self {
        Self::new(ErrorKind::Io, err.to_string())
    }
}

impl From<serde_json::Error> for PageIndexError {
    fn from(err: serde_json::Error) -> Self {
        Self::new(ErrorKind::Io, err.to_string())
    }
}

impl From<pageindex_pdf::Error> for PageIndexError {
    fn from(err: pageindex_pdf::Error) -> Self {
        let kind = match err {
            pageindex_pdf::Error::NotFound(_) => ErrorKind::NotFound,
            _ => ErrorKind::Document,
        };
        Self::new(kind, err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, PageIndexError>;
