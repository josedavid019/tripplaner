# app/WeatherAgent/main.py
import os
from urllib.parse import quote

from strands import Agent, tool
from strands.models import BedrockModel
from strands.multiagent.a2a import A2AServer


@tool
def get_forecast(city: str, date: str) -> dict:
    """Devuelve el pronóstico del clima de una ciudad (datos simulados)."""
    return {"city": city, "date": date, "temp_c": 31, "condition": "Soleado"}


def create_agent(context_id: str) -> Agent:
    return Agent(
        name="Weather Agent",
        description="Consulta el pronóstico del clima por ciudad y fecha.",
        model=BedrockModel(model_id="us.amazon.nova-lite-v1:0", region_name="us-east-2"),
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