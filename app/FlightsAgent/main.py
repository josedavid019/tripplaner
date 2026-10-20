# app/FlightsAgent/main.py
import logging
import os
from datetime import date, timedelta
from uuid import uuid4
from urllib.parse import quote

from mcp.client.streamable_http import streamable_http_client
from strands import Agent, tool
from strands.models import BedrockModel
from strands.multiagent.a2a import A2AServer
from strands.tools.mcp import MCPClient

from flight_results import flight_search_unavailable_message, format_flight_results
from flight_response import FlightResponseAgent

logger = logging.getLogger(__name__)
TRAVEL_MCP_URL = os.environ.get("TRAVEL_MCP_URL", "https://mcp.skiplagged.com/mcp")
MCP_READ_TIMEOUT = timedelta(seconds=25)

# Solo exponemos al LLM las tools de vuelos (el servidor trae 9, incluidas hoteles y carros).
ALLOWED_TOOLS = {"sk_flights_search", "sk_resolve_iata"}

# Sesión abierta durante toda la vida del proceso (servidor A2A de larga vida):
# por eso start() y no "with".
travel_mcp = MCPClient(lambda: streamable_http_client(TRAVEL_MCP_URL))
travel_mcp.start()
available_tools = {
    tool.tool_name: tool
    for tool in travel_mcp.list_tools_sync()
    if tool.tool_name in ALLOWED_TOOLS
}
missing_tools = ALLOWED_TOOLS - available_tools.keys()
if missing_tools:
    raise RuntimeError(
        f"El servidor MCP en {TRAVEL_MCP_URL} no expone las tools esperadas: "
        f"{', '.join(sorted(missing_tools))}"
    )


def build_system_prompt() -> str:
    return (
        f"Hoy es {date.today().isoformat()}. "
        "Eres un agente de vuelos. Para buscar usa sk_flights_search con: "
        "origin y destination como códigos IATA (si el usuario da una ciudad y no sabes "
        "el código, usa sk_resolve_iata primero), departureDate en formato YYYY-MM-DD "
        "(si el usuario no da el año, usa la próxima fecha futura), sort='price', limit=5, "
        "includeHiddenCity=false e includeVirtualInterlining=false. "
        "La herramienta devuelve el resultado final ya ordenado, con la fecha y el vuelo más "
        "barato marcados. Copia su respuesta exactamente, conservando cada salto de línea, "
        "sin volver a ordenar, recalcular, agregar moneda ni cambiar vuelos, aerolíneas o precios. "
        "No incluyas razonamiento ni etiquetas como <thinking>. "
        "Si la herramienta falla o no hay resultados, dilo en una sola línea. "
        "No agregues más texto."
    )


def create_agent(context_id: str) -> Agent:
    search_state: dict[str, str | None] = {"response": None}

    @tool
    def sk_flights_search(
        origin: str,
        destination: str,
        departureDate: str,
        sort: str = "price",
        limit: int = 5,
        includeHiddenCity: bool = False,
        includeVirtualInterlining: bool = False,
    ) -> str:
        """Busca vuelos y devuelve resultados ordenados con el precio mínimo marcado."""
        try:
            result = travel_mcp.call_tool_sync(
                uuid4().hex,
                "sk_flights_search",
                {
                    "origin": origin,
                    "destination": destination,
                    "departureDate": departureDate,
                    "sort": sort,
                    "limit": limit,
                    "includeHiddenCity": includeHiddenCity,
                    "includeVirtualInterlining": includeVirtualInterlining,
                },
                read_timeout_seconds=MCP_READ_TIMEOUT,
            )
            if result["status"] != "success":
                raise RuntimeError(f"La herramienta sk_flights_search falló: {result['status']}")

            text_blocks = [
                block["text"]
                for block in result["content"]
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ]
            if not text_blocks:
                raise RuntimeError("La herramienta sk_flights_search no devolvió contenido de texto")
            response = format_flight_results(
                "\n".join(text_blocks),
                origin,
                destination,
                departureDate,
            )
        except Exception:
            logger.exception("sk_flights_search falló")
            response = flight_search_unavailable_message(origin, destination, departureDate)
        search_state["response"] = response
        return response

    return FlightResponseAgent(
        name="Flights Agent",
        description="Busca vuelos entre ciudades para una fecha dada.",
        model=BedrockModel(model_id="us.amazon.nova-lite-v1:0", region_name="us-east-2"),
        system_prompt=build_system_prompt(),
        tools=[available_tools["sk_resolve_iata"], sk_flights_search],
        callback_handler=None,
        search_state=search_state,
    )


# El servidor publica la agent card en /.well-known/agent-card.json
RUNTIME_ARN = "arn:aws:bedrock-agentcore:us-east-2:289076681196:runtime/tripplaner_FlightsAgent-NIrzy8HGru"
DEFAULT_URL = (
    "https://bedrock-agentcore.us-east-2.amazonaws.com/runtimes/"
    f"{quote(RUNTIME_ARN, safe='')}/invocations/"
)
# URL pública que va en la agent card: sin esto el cliente intenta conectarse a 0.0.0.0:9000
runtime_url = os.environ.get("AGENTCORE_RUNTIME_URL", DEFAULT_URL)

a2a_server = A2AServer(
    agent_factory=create_agent,
    http_url=runtime_url,
    serve_at_root=True,
    host="0.0.0.0",
    port=9000,
)

if __name__ == "__main__":
    a2a_server.serve()