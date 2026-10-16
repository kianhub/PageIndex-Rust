"""Anthropic SDK adapter for ``beta.messages.tool_runner(tools=...)``.

Local clients become ``beta_tool`` / ``beta_async_tool`` runnables. The
Messages API MCP connector is the server-side alternative:
``mcp_servers=[{"type": "url", "name": "pageindex", "url": ...}]`` against
``Client.serve_mcp()`` (HTTPS for Anthropic's hosted connector) or cloud ``/mcp``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any


def as_anthropic_tools(
    client,
    include_management: bool = False,
    asynchronous: bool = False,
) -> list:
    try:
        from anthropic import beta_async_tool, beta_tool
    except ImportError as exc:
        raise RuntimeError(
            "as_anthropic_tools requires the Anthropic SDK tool runner — "
            "pip install -U anthropic."
        ) from exc

    try:
        from anthropic.lib.tools import ToolError
    except ImportError:
        ToolError = RuntimeError  # type: ignore[misc, assignment]

    from pageindex._native import as_mcp_tools, tool_names

    specs = {item["name"]: item for item in as_mcp_tools(include_management)}

    def _wrap(name: str):
        spec = specs[name]
        description = spec.get("description") or ""
        schema = spec.get("inputSchema") or {"type": "object", "properties": {}}

        def run(kwargs: dict[str, Any]) -> str:
            text, is_error = client.call_tool(name, kwargs or {})
            if is_error:
                raise ToolError(text)
            return text

        if asynchronous:

            async def _afn(**kwargs: Any) -> str:
                return await asyncio.to_thread(run, kwargs)

            _afn.__name__ = name
            return beta_async_tool(_afn, name=name, description=description, input_schema=schema)

        def _fn(**kwargs: Any) -> str:
            return run(kwargs)

        _fn.__name__ = name
        return beta_tool(_fn, name=name, description=description, input_schema=schema)

    return [_wrap(name) for name in tool_names(include_management)]
