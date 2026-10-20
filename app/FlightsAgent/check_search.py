from mcp.client.streamable_http import streamable_http_client
from strands.tools.mcp import MCPClient

mcp = MCPClient(lambda: streamable_http_client("https://mcp.skiplagged.com/mcp"))

with mcp:
    result = mcp.call_tool_sync(
        "test-1",
        "sk_flights_search",
        {
            "origin": "BAQ",
            "destination": "MAD",
            "departureDate": "2026-11-20",
            "sort": "price",
            "limit": 3,
            "includeHiddenCity": False,
            "includeVirtualInterlining": False,
        },
    )
    print(result["status"])
    for block in result["content"]:
        print(block.get("text", block))
