"""Optional Python bindings for PageIndex-Rust."""

from pageindex._native import (
    AGENT_INSTRUCTIONS,
    CITE_PROMPT,
    Client,
    McpServer,
    as_mcp_tools,
    call_tool,
    page_index_flash,
    tool_names,
)
from pageindex.integrations.anthropic_sdk import as_anthropic_tools
from pageindex.integrations.openai_agents import as_openai_tools

Client.as_openai_tools = as_openai_tools
Client.as_anthropic_tools = as_anthropic_tools

__all__ = [
    "AGENT_INSTRUCTIONS",
    "CITE_PROMPT",
    "Client",
    "McpServer",
    "as_anthropic_tools",
    "as_mcp_tools",
    "as_openai_tools",
    "call_tool",
    "page_index_flash",
    "tool_names",
]
