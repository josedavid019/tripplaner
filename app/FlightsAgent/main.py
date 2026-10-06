# app/FlightsAgent/main.py
import os
from urllib.parse import quote

from strands import Agent, tool
from strands.models import BedrockModel
from strands.multiagent.a2a import A2AServer


AIRLINE_BY_CODE_PREFIX = {
    "AV": "Avianca",
    "LA": "LATAM",
}


def airline_name(flight_code: str) -> str:
    """Return the carrier for a simulated flight code, failing on unknown prefixes."""
    prefix = flight_code[:2].upper()
    try:
        return AIRLINE_BY_CODE_PREFIX[prefix]
    except KeyError as exc:
        raise ValueError(f"Unknown airline code prefix: {prefix}") from exc


@tool
def search_flights(origin: str, destination: str, date: str) -> list[dict]:
    """Busca vuelos entre dos ciudades para una fecha (datos simulados)."""
    options = [
        {"flight": "AV9821", "depart": "07:15", "price_usd": 89},
        {"flight": "LA4410", "depart": "12:40", "price_usd": 76},
        {"flight": "AV9835", "depart": "18:05", "price_usd": 102},
    ]
    return [
        {
            "origin": origin,
            "destination": destination,
            "date": date,
            **o,
            "airline": airline_name(o["flight"]),
        }
        for o in options
    ]


def create_agent(context_id: str) -> Agent:
    return Agent(
        name="Flights Agent",
        description="Busca vuelos entre ciudades para una fecha dada.",
        model=BedrockModel(model_id="us.amazon.nova-lite-v1:0", region_name="us-east-2"),
        system_prompt=(
            "Eres un agente de vuelos. Usa search_flights con origen, destino y fecha. "
            "Responde solo con una línea por vuelo, ordenadas por hora de salida, con el formato "
            "'<vuelo> | <aerolínea> | salida <hora> | <precio> USD'. Usa exactamente el campo "
            "'airline' devuelto por search_flights; no deduzcas ni inventes aerolíneas a partir "
            "del código. Agrega ' (más barato)' al final de la línea del vuelo de menor precio y "
            "solo de esa. Antes de las líneas escribe una sola línea con la ruta y la fecha "
            "consultadas en formato DD/MM/AAAA. No agregues más texto."
        ),
        tools=[search_flights],
        callback_handler=None,
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