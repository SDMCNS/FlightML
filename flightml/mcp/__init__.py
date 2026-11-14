"""FlightML Model Context Protocol (MCP) Integration."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flightml.mcp.server import create_mcp_server, main, mcp


def __getattr__(name: str) -> Any:
    if name in ("mcp", "create_mcp_server", "main"):
        import flightml.mcp.server as s
        return getattr(s, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["mcp", "create_mcp_server", "main"]
