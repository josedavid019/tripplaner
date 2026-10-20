from mcp.client.streamable_http import streamable_http_client
from strands.tools.mcp import MCPClient

from flight_results import format_flight_results

CASES = [
    ("BAQ", "MAD", "2026-11-20"),
    ("BAQ", "MAD", "2026-11-25"),
]

mcp = MCPClient(lambda: streamable_http_client("https://mcp.skiplagged.com/mcp"))

with mcp:
    for origin, destination, day in CASES:
        result = mcp.call_tool_sync(
            f"fmt-{day}",
            "sk_flights_search",
            {
                "origin": origin,
                "destination": destination,
                "departureDate": day,
                "sort": "price",
                "limit": 5,
                "includeHiddenCity": False,
                "includeVirtualInterlining": False,
            },
        )
        if result["status"] != "success":
            raise RuntimeError(f"Flight search failed for {day}: {result['status']}")

        text = "\n".join(
            block["text"]
            for block in result["content"]
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
        print(f"--- {origin} -> {destination} {day} ---")
        print(format_flight_results(text, origin, destination, day))
        print()
