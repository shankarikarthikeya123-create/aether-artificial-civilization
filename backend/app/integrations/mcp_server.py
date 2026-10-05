"""Standalone MCP server for AETHER."""
from mcp.server.mcpserver import MCPServer
from .mcp_tools import (
    world_state, citizen_snapshot, knowledge_search, civilization_summary,
    resource_snapshot, inspect_location, execute_citizen_action
)
from ..simulation import get_simulation

mcp = MCPServer("AETHER Civilization")

@mcp.tool()
def get_world_state() -> dict:
    return world_state(get_simulation())

@mcp.tool()
def get_citizen(citizen_id: str) -> dict:
    return citizen_snapshot(get_simulation(), citizen_id)

@mcp.tool()
def search_aether_knowledge(query: str, limit: int = 5) -> dict:
    return knowledge_search(query, limit)

@mcp.tool()
def get_civilization_summary() -> dict:
    return civilization_summary(get_simulation())

@mcp.tool()
def get_resources() -> dict:
    return resource_snapshot(get_simulation())

@mcp.tool()
def inspect_location(location_id: str) -> dict:
    return inspect_location(get_simulation(), location_id)

@mcp.tool()
def execute_action(citizen_id: str, action: str) -> dict:
    return execute_citizen_action(get_simulation(), citizen_id, action)

if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
