"""
mcp/mcp_client.py
-----------------
Client interface for agents to discover and invoke tools via the Model Context Protocol (MCP).
"""

from __future__ import annotations
import json
import uuid
from typing import Dict, Any, List, Optional
from mcp.mcp_server import default_mcp_server, MCPServer


class MCPClient:
    """Client for Model Context Protocol interactions."""

    def __init__(self, server: Optional[MCPServer] = None):
        self.server = server or default_mcp_server

    def list_tools(self) -> List[Dict[str, Any]]:
        """Invokes MCP 'tools/list' request."""
        req = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4())[:8],
            "method": "tools/list",
            "params": {}
        }
        res = self.server.handle_request(req)
        return res.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invokes MCP 'tools/call' request adhering to MCP spec."""
        req = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4())[:8],
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }
        res = self.server.handle_request(req)
        if "error" in res:
            raise RuntimeError(f"MCP Tool Call Error [{name}]: {res['error']['message']}")

        content_list = res.get("result", {}).get("content", [])
        if content_list and content_list[0].get("type") == "text":
            return json.loads(content_list[0]["text"])
        return {}


# Global default client
default_mcp_client = MCPClient()
