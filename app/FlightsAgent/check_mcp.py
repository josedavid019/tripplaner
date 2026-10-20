from mcp.client.streamable_http import streamable_http_client
from strands.tools.mcp import MCPClient

URL = "https://mcp.skiplagged.com/mcp"
mcp = MCPClient(lambda: streamable_http_client(URL))

with mcp:
    for t in mcp.list_tools_sync():
        print(t.tool_name)
        print(t.tool_spec["inputSchema"])
        print()
