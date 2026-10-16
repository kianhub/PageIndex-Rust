"""Python bindings: client, indexing, tools, and SDK helpers. No API keys."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

import pageindex

REPO = Path(__file__).resolve().parents[3]
PDF = REPO / "crates/pageindex/tests/data/ja_report.pdf"


def test_page_index_flash_structure_only():
    result = pageindex.page_index_flash(str(PDF), summary=False, optimize="off")
    assert result["doc_name"].endswith(".pdf")
    assert result["structure"]


def test_client_indexes_and_call_tool(tmp_path):
    client = pageindex.Client(
        storage_path=str(tmp_path / "store"),
        optimize="off",
        scripted_replies=['{"summary": "A Japanese report."}', "A short report."],
    )
    submitted = client.submit_document(str(PDF))
    assert submitted["name"].endswith(".pdf")
    assert submitted["doc_id"].startswith("pi-")

    listed = client.list_documents()
    assert listed["total"] == 1

    text, is_error = pageindex.call_tool(client, "browse_documents", {})
    assert not is_error
    browse = json.loads(text)
    assert browse["success"] is True
    assert browse["documents"][0]["name"] == submitted["name"]

    text, is_error = client.call_tool(
        "get_document_structure", {"doc_name": submitted["name"]}
    )
    assert not is_error
    structure = json.loads(text)
    assert structure["structure"]

    text, is_error = client.call_tool(
        "get_page_content", {"doc_name": submitted["name"], "pages": "1"}
    )
    assert not is_error
    pages = json.loads(text)
    assert pages["content"][0]["text"]


def test_tool_names_and_mcp_schema():
    assert "browse_documents" in pageindex.tool_names()
    assert "remove_document" not in pageindex.tool_names()
    assert "remove_document" in pageindex.tool_names(True)
    tools = pageindex.as_mcp_tools()
    assert tools[0]["name"] == "browse_documents"
    assert "inputSchema" in tools[0]
    assert pageindex.AGENT_INSTRUCTIONS.startswith("PageIndex")


def test_as_openai_tools_with_stub_sdk(tmp_path):
    calls = {}

    class FunctionTool:
        def __init__(self, name, description, params_json_schema, on_invoke_tool, strict_json_schema=False):
            self.name = name
            self.description = description
            self.params_json_schema = params_json_schema
            self.on_invoke_tool = on_invoke_tool
            self.strict_json_schema = strict_json_schema

    class HostedMCPTool:
        def __init__(self, tool_config):
            self.tool_config = tool_config

    fake = types.ModuleType("agents")
    fake.FunctionTool = FunctionTool
    fake.HostedMCPTool = HostedMCPTool
    sys.modules["agents"] = fake

    client = pageindex.Client(
        storage_path=str(tmp_path / "store"),
        optimize="off",
        scripted_replies=['{"summary": "s"}', "desc"],
    )
    client.submit_document(str(PDF))
    tools = pageindex.as_openai_tools(client)
    assert [tool.name for tool in tools] == pageindex.tool_names()
    assert tools[0].params_json_schema["type"] == "object"

    import asyncio

    out = asyncio.run(tools[0].on_invoke_tool(None, "{}"))
    envelope = json.loads(out)
    assert envelope["success"] is True
    assert envelope["documents"]
    calls["ok"] = True
    assert calls["ok"]


def test_as_anthropic_tools_with_stub_sdk(tmp_path):
    class ToolError(Exception):
        pass

    def beta_tool(fn, name, description, input_schema):
        fn.tool_name = name
        fn.description = description
        fn.input_schema = input_schema
        return fn

    def beta_async_tool(fn, name, description, input_schema):
        fn.tool_name = name
        fn.description = description
        fn.input_schema = input_schema
        return fn

    anthropic = types.ModuleType("anthropic")
    anthropic.beta_tool = beta_tool
    anthropic.beta_async_tool = beta_async_tool
    tools_mod = types.ModuleType("anthropic.lib.tools")
    tools_mod.ToolError = ToolError
    lib_mod = types.ModuleType("anthropic.lib")
    sys.modules["anthropic"] = anthropic
    sys.modules["anthropic.lib"] = lib_mod
    sys.modules["anthropic.lib.tools"] = tools_mod

    client = pageindex.Client(
        storage_path=str(tmp_path / "store"),
        optimize="off",
        scripted_replies=['{"summary": "s"}', "desc"],
    )
    client.submit_document(str(PDF))
    tools = pageindex.as_anthropic_tools(client)
    assert tools[0].tool_name == "browse_documents"
    assert tools[0].input_schema["type"] == "object"
    envelope = json.loads(tools[0]())
    assert envelope["documents"]


def test_serve_mcp_initialize(tmp_path):
    urllib = pytest.importorskip("urllib.request")
    client = pageindex.Client(
        storage_path=str(tmp_path / "store"),
        optimize="off",
        scripted_replies=['{"summary": "s"}', "desc"],
    )
    client.submit_document(str(PDF))
    with client.serve_mcp() as server:
        assert server.url.startswith("http://127.0.0.1")
        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "init-1",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest"},
                },
            }
        ).encode()
        req = urllib.request.Request(
            server.url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
            assert resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
        assert body["id"] == "init-1"
        assert body["result"]["protocolVersion"] == "2025-06-18"
        names = [tool["name"] for tool in pageindex.as_mcp_tools()]
        assert "browse_documents" in names
