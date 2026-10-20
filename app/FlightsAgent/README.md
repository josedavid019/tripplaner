# FlightsAgent

An A2A (Agent-to-Agent) agent deployed on Amazon Bedrock AgentCore using Strands SDK.

## Overview

This agent implements the A2A protocol, enabling agent-to-agent communication. Other agents can discover and interact with this agent via the `/.well-known/agent-card.json` endpoint.

## Local Development

```bash
uv sync
uv run python main.py
```

The agent starts on port 9000.

The agent searches flights through the Skiplagged MCP server. Its Markdown results are parsed
locally to sort offers by departure time and mark the lowest returned price. Set `TRAVEL_MCP_URL`
to override the MCP endpoint.

Run the flights unit tests from the repository root with:

```bash
python -m unittest tests.test_flights_agent -v
```

## Deploy

```bash
agentcore deploy
```
