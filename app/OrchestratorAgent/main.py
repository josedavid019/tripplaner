import os
import re
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from strands import Agent, tool
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands.models import BedrockModel
from bedrock_agentcore.memory.integrations.strands.config import (
    AgentCoreMemoryConfig,
    RetrievalConfig,
)
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)
from strands.agent.a2a_agent import A2AAgent
from a2a.client import ClientConfig, ClientFactory

app = BedrockAgentCoreApp()
log = app.logger

REGION = os.environ.get("AWS_REGION", "us-east-2")
MODEL_ID = "us.amazon.nova-lite-v1:0"

# Vacío -> el agente sigue funcionando sin memoria gestionada.
MEMORY_ID = os.environ.get("TRIP_MEMORY_ID") or None

# ARNs de los especialistas (salen de `agentcore status`).
# Se pueden sobreescribir con variables de entorno si redespliegas.
FLIGHTS_ARN = os.environ.get(
    "FLIGHTS_AGENT_ARN",
    "arn:aws:bedrock-agentcore:us-east-2:289076681196:runtime/tripplaner_FlightsAgent-NIrzy8HGru",
)
WEATHER_ARN = os.environ.get(
    "WEATHER_AGENT_ARN",
    "arn:aws:bedrock-agentcore:us-east-2:289076681196:runtime/tripplaner_WeatherAgent-aJJvVI7ABv",
)

ACTOR_ID_HEADER = "x-amzn-bedrock-agentcore-runtime-custom-actor-id"


def runtime_url(arn: str) -> str:
    """URL A2A de un runtime de AgentCore (termina en /invocations/ con slash final)."""
    return (
        f"https://bedrock-agentcore.{REGION}.amazonaws.com/runtimes/"
        f"{quote(arn, safe='')}/invocations/"
    )


class SigV4HttpxAuth(httpx.Auth):
    """Firma cada petición httpx con SigV4 usando el rol de ejecución del runtime."""

    requires_request_body = True

    def __init__(self, region: str, service: str = "bedrock-agentcore"):
        self.region = region
        self.service = service

    def auth_flow(self, request: httpx.Request):
        creds = boto3.Session().get_credentials().get_frozen_credentials()
        aws_request = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content,
            headers=dict(request.headers),
        )
        SigV4Auth(creds, self.service, self.region).add_auth(aws_request)
        for key, value in aws_request.headers.items():
            request.headers[key] = value
        yield request


def make_remote_agent(arn: str, http_client: httpx.AsyncClient) -> A2AAgent:
    """Crea un cliente A2A autenticado usando el HTTP client de la llamada actual."""
    config = ClientConfig(httpx_client=http_client, streaming=False)
    endpoint = runtime_url(arn)
    try:
        return A2AAgent(endpoint=endpoint, client_config=config)
    except TypeError:
        # Versiones anteriores de Strands solo aceptan a2a_client_factory
        return A2AAgent(endpoint=endpoint, a2a_client_factory=ClientFactory(config))


async def _invoke_remote_agent(arn: str, request: str) -> str:
    async with httpx.AsyncClient(
        auth=SigV4HttpxAuth(REGION),
        timeout=httpx.Timeout(60.0),
    ) as http_client:
        remote = make_remote_agent(arn, http_client)
        return str(await remote.invoke_async(request))


@tool
async def ask_flights(request: str) -> str:
    """Delega en el Flights Agent la búsqueda de vuelos."""
    return await _invoke_remote_agent(FLIGHTS_ARN, request)


@tool
async def ask_weather(request: str) -> str:
    """Delega en el Weather Agent la consulta de clima."""
    return await _invoke_remote_agent(WEATHER_ARN, request)


DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]


def _fecha_legible(d) -> str:
    return f"{DIAS[d.weekday()]} {d.strftime('%d/%m/%Y')} (para las herramientas: {d.isoformat()})"


def build_system_prompt() -> str:
    """Prompt con la fecha de hoy (hora de Colombia) para resolver 'este fin de semana'."""
    today = datetime.now(timezone(timedelta(hours=-5))).date()
    saturday = today + timedelta(days=(5 - today.weekday()) % 7)
    sunday = saturday + timedelta(days=1)
    return (
        "Eres un planificador de viajes. Tienes dos herramientas: ask_flights (vuelos) y "
        "ask_weather (clima).\n\n"
        f"FECHAS: hoy es {_fecha_legible(today)}. El sábado es {_fecha_legible(saturday)} y el "
        f"domingo es {_fecha_legible(sunday)}. Usa solo estas fechas, sin calcular otras.\n\n"
        "MEMORIA DEL USUARIO: si en el contexto aparecen preferencias o datos guardados del "
        "usuario (aerolínea preferida, horario preferido, etc.), tenlos en cuenta aunque el "
        "usuario no los repita en este mensaje. Si no hay ninguno, ignora esta sección. "
        "La respuesta de ask_flights incluye una aerolínea explícita por vuelo; usa ese dato "
        "como fuente de verdad y no contradigas el nombre de aerolínea con el código.\n\n"
        "PLAN DE TRABAJO:\n"
        "- Si el usuario pide 'este fin de semana', la ida es el sábado y el regreso el domingo. "
        "Si da otras fechas o un trayecto distinto, úsalos. Nunca le pidas la fecha al usuario.\n"
        "- Llama a ask_flights una vez para la ida y otra para el regreso (con el trayecto invertido). "
        "En cada llamada incluye ciudad de origen, ciudad de destino y fecha en formato YYYY-MM-DD.\n"
        "- Llama a ask_weather para la ciudad de destino, para el sábado y el domingo, con fecha "
        "YYYY-MM-DD.\n"
        "- Haz todas las llamadas antes de responder y no escribas texto antes de llamarlas.\n"
        "- Si una herramienta falla, dilo en esa sección y continúa con el resto.\n\n"
        "RESPUESTA FINAL: en español, texto plano (sin asteriscos, sin tablas, sin etiquetas "
        "<thinking>), con exactamente esta estructura:\n\n"
        "PLAN DE VIAJE: <origen> -> <destino>\n"
        "Fin de semana del <día> <DD/MM/AAAA> al <día> <DD/MM/AAAA> "
        "(ejemplo: Fin de semana del sábado 10/10/2026 al domingo 11/10/2026)\n\n"
        "IDA - <día y fecha> (<origen> -> <destino>)\n"
        "  <vuelo> | <aerolínea> | salida <hora> | <precio> USD\n"
        "  (una línea por vuelo, ordenadas por hora de salida, sin guiones ni la palabra Flight; "
        "agrega ' (más barato)' al final de la MISMA línea del vuelo de menor precio, solo en esa línea)\n\n"
        "REGRESO - <día y fecha> (<destino> -> <origen>)\n"
        "  (mismo formato que la ida, incluida la marca del más barato)\n\n"
        "CLIMA EN <DESTINO>\n"
        "  <día y fecha>: <condición>, <temperatura> °C\n\n"
        "RECOMENDACION\n"
        "  Dos o tres frases. Primera: el vuelo de ida y el de regreso más baratos con el total de "
        "ambos (suma exacta de los dos precios) y cómo estará el clima. Segunda: si hay preferencias "
        "guardadas del usuario (aerolínea u horario), recomienda un vuelo que cumpla todas las "
        "preferencias posibles y menciona por qué. Compara la preferencia de aerolínea con el campo "
        "explícito de aerolínea devuelto por la herramienta y la de horario con la hora de salida. "
        "Usa los datos de los vuelos de esta búsqueda; no asumas códigos ni aerolíneas de ejemplos "
        "o búsquedas anteriores. Nunca digas que un vuelo no cumple una preferencia si sus datos "
        "explícitos sí la cumplen. "
        "Si ningún vuelo cumple todas, recomienda el que cumpla más y explica con precisión el "
        "conflicto. Si no hay preferencias guardadas, omite esta segunda frase.\n\n"
        "REGLAS: usa únicamente datos devueltos por las herramientas; no inventes vuelos, "
        "horarios, precios ni clima. La marca '(más barato)' se basa solo en el precio; las "
        "preferencias del usuario solo influyen en la RECOMENDACION. "
        "En la respuesta final escribe TODAS las fechas en formato DD/MM/AAAA, por ejemplo "
        "'sábado 10/10/2026'; nunca uses YYYY-MM-DD ni nombres de mes. Si una herramienta devuelve "
        "una fecha como 2026-10-10, conviértela a 10/10/2026. El formato YYYY-MM-DD solo se usa "
        "al llamar a las herramientas. No agregues notas finales."
    )


class TagFilter:
    """Quita bloques <thinking>...</thinking> y las etiquetas <response> de un texto en streaming."""

    TAG_RE = re.compile(r"<thinking>|</?response>")
    CLOSE = "</thinking>"
    KNOWN = ("<thinking>", "</thinking>", "<response>", "</response>")

    def __init__(self):
        self.buf = ""
        self.inside = False
        self.started = False

    def feed(self, text: str, final: bool = False) -> str:
        out = self._feed(text, final)
        if not self.started:
            out = out.lstrip()
            self.started = bool(out)
        return out

    def _feed(self, text: str, final: bool = False) -> str:
        self.buf += text
        out = []
        while True:
            if self.inside:
                i = self.buf.find(self.CLOSE)
                if i == -1:
                    self.buf = "" if final else self.buf[-(len(self.CLOSE) - 1):]
                    break
                self.buf = self.buf[i + len(self.CLOSE):]
                self.inside = False
                continue
            m = self.TAG_RE.search(self.buf)
            if m:
                out.append(self.buf[:m.start()])
                self.buf = self.buf[m.end():]
                self.inside = m.group(0) == "<thinking>"
                continue
            last = self.buf.rfind("<")
            if not final and last != -1 and any(t.startswith(self.buf[last:]) for t in self.KNOWN):
                out.append(self.buf[:last])
                self.buf = self.buf[last:]
            else:
                out.append(self.buf)
                self.buf = ""
            break
        return "".join(out)


tools = [ask_flights, ask_weather]


def _build_session_manager(session_id: str, actor_id: str):
    """Memoria gestionada de AgentCore. None si no hay MEMORY_ID."""
    if not MEMORY_ID:
        return None
    config = AgentCoreMemoryConfig(
        memory_id=MEMORY_ID,
        session_id=session_id,
        actor_id=actor_id,
        retrieval_config={
            f"/preferences/{actor_id}/": RetrievalConfig(top_k=5, relevance_score=0.3),
            f"/facts/{actor_id}/": RetrievalConfig(top_k=5, relevance_score=0.3),
        },
    )
    return AgentCoreMemorySessionManager(config, region_name=REGION)


# Reutiliza un Agent por (session_id, actor_id) (LRU de 128 sesiones).
def agent_factory():
    cache = OrderedDict()

    def get_or_create_agent(session_id, actor_id):
        key = (session_id, actor_id)
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        if len(cache) >= 128:
            cache.popitem(last=False)

        agent_kwargs = {
            "name": "Trip Planner Orchestrator",
            "model": BedrockModel(model_id=MODEL_ID, region_name=REGION),
            "system_prompt": build_system_prompt(),
            "tools": tools,
            "callback_handler": None,
        }
        session_manager = _build_session_manager(session_id, actor_id)
        if session_manager is not None:
            agent_kwargs["session_manager"] = session_manager
        else:
            agent_kwargs["conversation_manager"] = NullConversationManager()

        cache[key] = Agent(**agent_kwargs)
        return cache[key]

    return get_or_create_agent


get_or_create_agent = agent_factory()


def strip_trailing_tool_use(messages: Any) -> list[dict]:
    """Quita bloques toolUse del final hasta que el último mensaje no tenga ninguno."""
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    messages = list(messages)
    while messages:
        last = messages[-1]
        if not isinstance(last, dict):
            raise ValueError("each message must be an object")
        original_content = last.get("content", [])
        if not isinstance(original_content, list) or not all(isinstance(b, dict) for b in original_content):
            raise ValueError("each message content value must be a list of content blocks")

        content = [block for block in original_content if "toolUse" not in block]
        if len(content) == len(original_content):
            break
        if content:
            messages[-1] = {**last, "content": content}
            break
        messages.pop()

    return messages


def _extract_prompt(payload: dict):
    """Acepta mensajes del harness, tool results o un prompt simple."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    if "messages" in payload:
        return strip_trailing_tool_use(payload["messages"])
    if "tool_results" in payload:
        tool_results = payload["tool_results"]
        if not isinstance(tool_results, list) or not all(
            isinstance(tr, dict) and isinstance(tr.get("toolUseId"), str) for tr in tool_results
        ):
            raise ValueError("tool_results must contain objects with a toolUseId string")
        return [{"role": "user", "content": [{"toolResult": {
            "toolUseId": tr["toolUseId"],
            "status": tr.get("status", "success"),
            "content": tr.get("content", []),
        }} for tr in tool_results]}]
    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    return prompt


def _resolve_actor_id(payload, context) -> str:
    """Identidad del usuario: payload['actor_id'] o header custom (sin distinguir mayúsculas)."""
    if isinstance(payload, dict) and isinstance(payload.get("actor_id"), str) and payload["actor_id"]:
        return payload["actor_id"]
    headers = getattr(context, "request_headers", None) or {}
    lowered = {str(k).lower(): v for k, v in headers.items()}
    log.info(f"headers recibidos: {list(lowered.keys())}")
    value = lowered.get(ACTOR_ID_HEADER) or lowered.get("x-actor-id")
    if isinstance(value, str) and value:
        return value
    return "default-user"


@app.entrypoint
async def invoke(payload, context):
    log.info("Invoking Trip Planner Orchestrator...")

    session_id = getattr(context, "session_id", "default-session")
    actor_id = _resolve_actor_id(payload, context)
    log.info(f"session_id={session_id} actor_id={actor_id}")
    agent = get_or_create_agent(session_id, actor_id)

    prompt = _extract_prompt(payload)

    tag_filter = TagFilter()
    async for event in agent.stream_async(prompt):
        if not isinstance(event, dict) or "event" not in event:
            continue
        ev = event["event"]
        cbs = ev.get("contentBlockStart")
        if cbs is not None and not cbs.get("start"):
            continue

        delta = ev.get("contentBlockDelta")
        if delta is not None and "text" in delta.get("delta", {}):
            clean = tag_filter.feed(delta["delta"]["text"])
            if not clean:
                continue
            delta["delta"]["text"] = clean
        elif "contentBlockStop" in ev:
            rest = tag_filter.feed("", final=True)
            if rest:
                idx = ev["contentBlockStop"].get("contentBlockIndex", 0)
                yield {"event": {"contentBlockDelta": {"delta": {"text": rest}, "contentBlockIndex": idx}}}
            tag_filter = TagFilter()
        yield event


if __name__ == "__main__":
    app.run()