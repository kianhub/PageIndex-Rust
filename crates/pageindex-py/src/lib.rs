//! Optional PyO3 bindings. Not built by default (`default-members` omits this crate).

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use pageindex::agent::{AGENT_INSTRUCTIONS, CITE_PROMPT, as_mcp_tools, call_tool, tool_names};
use pageindex::client::{PageIndexLocalClient, SubmitOptions};
use pageindex::flash::{FlashOptions, page_index_flash};
use pageindex::llm::testing::ScriptedLlm;
use pageindex::mcp::McpServer;
use pageindex::{LocalIndexConfig, OptimizeMode};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use serde_json::{Value, json};
use tokio::runtime::Runtime;

fn runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| Runtime::new().expect("tokio runtime"))
}

fn map_err(err: pageindex::PageIndexError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

fn value_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    let encoded = serde_json::to_string(value).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    py.import("json")?.call_method1("loads", (encoded,)).map(|obj| obj.unbind())
}

fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(json!({}));
    }
    let encoded: String = obj.py().import("json")?.call_method1("dumps", (obj,))?.extract()?;
    serde_json::from_str(&encoded).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

fn parse_optimize(value: Option<&str>) -> PyResult<OptimizeMode> {
    match value {
        None => Ok(OptimizeMode::Full),
        Some(text) => text.parse().map_err(PyRuntimeError::new_err),
    }
}

#[pyclass(name = "McpServer", module = "pageindex")]
struct PyMcpServer {
    handle: Option<pageindex::McpServerHandle>,
    url: String,
}

#[pymethods]
impl PyMcpServer {
    #[getter]
    fn url(&self) -> &str {
        &self.url
    }

    fn close(&mut self) {
        if let Some(handle) = self.handle.take() {
            runtime().block_on(handle.shutdown());
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.close();
        Ok(false)
    }
}

impl Drop for PyMcpServer {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            drop(handle);
        }
    }
}

/// Local PageIndex client: index documents and run the agent tools.
#[pyclass(name = "Client", module = "pageindex")]
struct Client {
    inner: Arc<PageIndexLocalClient>,
}

#[pymethods]
impl Client {
    #[new]
    #[pyo3(signature = (storage_path=None, optimize=None, scripted_replies=None))]
    fn new(
        storage_path: Option<String>,
        optimize: Option<&str>,
        scripted_replies: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let config = LocalIndexConfig {
            storage_path: storage_path.map(PathBuf::from),
            optimize: Some(parse_optimize(optimize)?),
            ..Default::default()
        };
        let mut client = PageIndexLocalClient::with_config(config);
        if let Some(replies) = scripted_replies {
            client = client.with_llm(Arc::new(ScriptedLlm::new(replies)));
        }
        Ok(Self { inner: Arc::new(client) })
    }

    #[pyo3(signature = (path, mode=None))]
    fn submit_document(&self, py: Python<'_>, path: &str, mode: Option<&str>) -> PyResult<Py<PyAny>> {
        let options = SubmitOptions { mode: mode.map(str::to_owned), metadata: None };
        let inner = self.inner.clone();
        let path = path.to_owned();
        let result = py.allow_threads(|| runtime().block_on(inner.submit_document(path, options))).map_err(map_err)?;
        let value = serde_json::to_value(&result).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        value_to_py(py, &value)
    }

    #[pyo3(signature = (limit=10, offset=0))]
    fn list_documents(&self, py: Python<'_>, limit: usize, offset: usize) -> PyResult<Py<PyAny>> {
        let listed = self.inner.list_documents(limit, offset, None).map_err(map_err)?;
        let value = serde_json::to_value(&listed).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        value_to_py(py, &value)
    }

    fn get_document(&self, py: Python<'_>, doc_id: &str) -> PyResult<Py<PyAny>> {
        let doc = self.inner.get_document(doc_id).map_err(map_err)?;
        let value = serde_json::to_value(&doc).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        value_to_py(py, &value)
    }

    #[pyo3(signature = (doc_id, node_summary=true, include_text=false))]
    fn get_tree(&self, py: Python<'_>, doc_id: &str, node_summary: bool, include_text: bool) -> PyResult<Py<PyAny>> {
        let tree = self.inner.get_tree(doc_id, node_summary, include_text).map_err(map_err)?;
        let value = serde_json::to_value(&tree).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        value_to_py(py, &value)
    }

    fn delete_document(&self, doc_id: &str) -> PyResult<()> {
        self.inner.delete_document(doc_id).map_err(map_err)
    }

    #[pyo3(signature = (name, arguments=None))]
    fn call_tool(&self, py: Python<'_>, name: &str, arguments: Option<&Bound<'_, PyAny>>) -> PyResult<(String, bool)> {
        let args = match arguments {
            Some(obj) => py_to_value(obj)?,
            None => json!({}),
        };
        let inner = self.inner.clone();
        let name = name.to_owned();
        Ok(py.allow_threads(|| runtime().block_on(call_tool(inner.as_ref(), &name, args, None))))
    }

    #[pyo3(signature = (include_management=false))]
    fn agent_instructions(&self, py: Python<'_>, include_management: bool) -> PyResult<String> {
        let inner = self.inner.clone();
        py.allow_threads(|| runtime().block_on(inner.agent_instructions(include_management))).map_err(map_err)
    }

    #[pyo3(signature = (include_management=false))]
    fn serve_mcp(&self, include_management: bool) -> PyResult<PyMcpServer> {
        let backend: Arc<dyn pageindex::AgentBackend> = self.inner.clone();
        let handle = runtime().block_on(McpServer::new(backend, include_management).start()).map_err(map_err)?;
        let url = handle.url();
        Ok(PyMcpServer { handle: Some(handle), url })
    }
}

#[pyfunction(name = "page_index_flash")]
#[pyo3(signature = (path, summary=false, optimize=None))]
fn page_index_flash_py(py: Python<'_>, path: &str, summary: bool, optimize: Option<&str>) -> PyResult<Py<PyAny>> {
    let optimize = match optimize {
        None => OptimizeMode::Off,
        Some(text) => text.parse().map_err(PyRuntimeError::new_err)?,
    };
    let opts = FlashOptions { summary, optimize, ..FlashOptions::structure_only() };
    let path = path.to_owned();
    let result = py.allow_threads(|| runtime().block_on(page_index_flash(path, &opts))).map_err(map_err)?;
    let value = serde_json::to_value(&result).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    value_to_py(py, &value)
}

#[pyfunction(name = "call_tool")]
#[pyo3(signature = (client, name, arguments=None))]
fn call_tool_py(
    py: Python<'_>,
    client: &Client,
    name: &str,
    arguments: Option<&Bound<'_, PyAny>>,
) -> PyResult<(String, bool)> {
    client.call_tool(py, name, arguments)
}

#[pyfunction(name = "tool_names")]
#[pyo3(signature = (include_management=false))]
fn tool_names_py(include_management: bool) -> Vec<String> {
    tool_names(include_management).into_iter().map(str::to_owned).collect()
}

#[pyfunction(name = "as_mcp_tools")]
#[pyo3(signature = (include_management=false))]
fn as_mcp_tools_py(py: Python<'_>, include_management: bool) -> PyResult<Py<PyAny>> {
    value_to_py(py, &as_mcp_tools(include_management))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Client>()?;
    m.add_class::<PyMcpServer>()?;
    m.add_function(wrap_pyfunction!(page_index_flash_py, m)?)?;
    m.add_function(wrap_pyfunction!(call_tool_py, m)?)?;
    m.add_function(wrap_pyfunction!(tool_names_py, m)?)?;
    m.add_function(wrap_pyfunction!(as_mcp_tools_py, m)?)?;
    m.add("AGENT_INSTRUCTIONS", AGENT_INSTRUCTIONS)?;
    m.add("CITE_PROMPT", CITE_PROMPT)?;
    Ok(())
}
