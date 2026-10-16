"""Framework adapters. SDKs are imported only when a helper is called."""

from pageindex.integrations.anthropic_sdk import as_anthropic_tools
from pageindex.integrations.openai_agents import as_openai_tools

__all__ = ["as_anthropic_tools", "as_openai_tools"]
