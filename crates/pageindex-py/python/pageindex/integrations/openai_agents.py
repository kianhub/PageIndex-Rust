"""OpenAI Agents SDK adapter for ``Agent(tools=...)``.

Local clients become ``FunctionTool`` objects (the framework's own
conversion of the in-process tool contract). Cloud-hosted MCP is the
``HostedMCPTool`` / ``MCPServerStreamableHttp`` path against a running
PageIndex Streamable HTTP server (``Client.serve_mcp()`` or the cloud
``/mcp`` URL).
"""

from __future__ import annotations

import asyncio
import json


def as_openai_tools(client, include_management: bool = False, hosted: bool = False) -> list:
    try:
        from agents import FunctionTool, HostedMCPTool
    except ImportError as exc:
        raise RuntimeError(
            "as_openai_tools requires the OpenAI Agents SDK — pip install openai-agents."
        ) from exc

    if hosted:
        url = hosted if isinstance(hosted, str) else None
        if url:
            return [
                HostedMCPTool(
                    tool_config={
                        "type": "mcp",
                        "server_label": "pageindex",
                        "server_url": url,
                        "require_approval": "never",
                    }
                )
            ]
        raise RuntimeError(
            "hosted=True needs a public MCP URL (cloud /mcp or a tunneled Client.serve_mcp()). "
            "For a local Runner, omit hosted and pass FunctionTools, or attach "
            "MCPServerStreamableHttp(params={'url': client.serve_mcp().url})."
        )

    from pageindex._native import as_mcp_tools, tool_names

    specs = {item["name"]: item for item in as_mcp_tools(include_management)}

    def _make(name: str):
        spec = specs[name]

        async def on_invoke(_ctx, arguments: str) -> str:
            payload = json.loads(arguments or "{}")
            text, _is_error = await asyncio.to_thread(client.call_tool, name, payload)
            return text

        on_invoke.__name__ = name
        return FunctionTool(
            name=name,
            description=spec.get("description") or "",
            params_json_schema=spec.get("inputSchema") or {"type": "object", "properties": {}},
            on_invoke_tool=on_invoke,
            strict_json_schema=False,
        )

    return [_make(name) for name in tool_names(include_management)]
