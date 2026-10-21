# app/WeatherAgent/main.py
import hashlib
import os
from urllib.parse import quote

from strands import Agent, tool
from strands.models import BedrockModel
from strands.multiagent.a2a import A2AServer


def build_system_prompt() -> str:
    return (
        "Eres un agente de clima. Para cada solicitud, identifica la ciudad y la fecha "
        "y llama siempre a get_forecast con esos valores. No respondas según conocimiento "
        "general ni inventes datos. Después de la herramienta, responde en español indicando "
        "la ciudad, la fecha, la condición y la temperatura en °C. Si no puedes identificar "
        "la ciudad o la fecha, pide únicamente el dato que falta."
    )


@tool
def get_forecast(city: str, date: str) -> dict:
    """Devuelve un pronóstico simulado y estable para una ciudad y fecha."""
    forecasts = [
        ("Soleado", 29),
        ("Parcialmente nublado", 24),
        ("Nublado", 19),
        ("Lluvia ligera", 16),
        ("Lluvioso", 14),
        ("Tormentas aisladas", 22),
        ("Ventoso", 18),
    ]
    seed = hashlib.sha256(f"{city.casefold()}:{date}".encode("utf-8")).digest()
    condition, temperature = forecasts[int.from_bytes(seed[:2], "big") % len(forecasts)]
    return {"city": city, "date": date, "temp_c": temperature, "condition": condition}


def create_agent(context_id: str) -> Agent:
    return Agent(
        name="Weather Agent",
        description="Consulta el pronóstico del clima por ciudad y fecha.",
        model=BedrockModel(model_id="us.amazon.nova-lite-v1:0", region_name="us-east-2"),
        system_prompt=build_system_prompt(),
        tools=[get_forecast],
        callback_handler=None,
    )


RUNTIME_ARN = "arn:aws:bedrock-agentcore:us-east-2:289076681196:runtime/tripplaner_WeatherAgent-aJJvVI7ABv"
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