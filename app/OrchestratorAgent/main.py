import os
import re
import unicodedata
from contextvars import ContextVar
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
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
flight_search_results: ContextVar[list[str] | None] = ContextVar(
    "flight_search_results",
    default=None,
)

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
    try:
        response = await _invoke_remote_agent(FLIGHTS_ARN, request)
    except Exception as exc:
        log.error(f"ask_flights falló: {exc!r}")
        response = "No se pudo consultar vuelos en este momento. Intenta de nuevo en unos minutos."
    results = flight_search_results.get()
    if results is not None:
        results.append(response)
    return response


@tool
async def ask_weather(request: str) -> str:
    """Delega en el Weather Agent la consulta de clima."""
    try:
        return await _invoke_remote_agent(WEATHER_ARN, request)
    except Exception:
        log.exception("ask_weather falló para la solicitud %r", request)
        raise


DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]
WEEKDAY_DATE_PATTERN = re.compile(
    r"\b(lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo)"
    r"(\s+)(\d{2}/\d{2}/\d{4})",
    re.IGNORECASE,
)
BARE_DATE_PATTERN = re.compile(r"^\s*(\d{2}/\d{2}/\d{4})(\s*:)")
DIRECT_FLIGHT_QUERY_PATTERN = re.compile(
    r"^\s*vuelos?\s+de\s+(?P<origin>.+?)\s+a\s+(?P<destination>.+?)"
    r"\s+(?:el|para el)\s+(?P<date>.+?)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
MONTH_NUMBERS = {
    name: number
    for number, name in enumerate(
        (
            "enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
        ),
        start=1,
    )
}


def _remove_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def parse_direct_flight_query(prompt: Any) -> tuple[str, str, date] | None:
    if not isinstance(prompt, str):
        return None
    match = DIRECT_FLIGHT_QUERY_PATTERN.fullmatch(prompt)
    if match is None:
        return None

    date_text = match.group("date").strip()
    try:
        travel_date = datetime.strptime(date_text, "%d/%m/%Y").date()
    except ValueError:
        normalized_date = _remove_accents(date_text)
        date_match = re.fullmatch(
            r"(?P<day>\d{1,2})\s+de\s+(?P<month>[a-z]+)"
            r"(?:\s+de\s+(?P<year>\d{4}))?",
            normalized_date,
        )
        if date_match is None:
            return None
        month = MONTH_NUMBERS.get(date_match.group("month"))
        if month is None:
            return None
        year = int(date_match.group("year") or datetime.now(
            timezone(timedelta(hours=-5))
        ).year)
        try:
            travel_date = date(year, month, int(date_match.group("day")))
        except ValueError:
            return None
    return (
        match.group("origin").strip(),
        match.group("destination").strip(),
        travel_date,
    )


def correct_weekdays(text: str) -> str:
    """Corrects Spanish weekday labels using their adjacent DD/MM/YYYY date."""

    def replace(match: re.Match[str]) -> str:
        try:
            actual_weekday = datetime.strptime(match.group(3), "%d/%m/%Y").date().weekday()
        except ValueError:
            return match.group(0)
        weekday = DIAS[actual_weekday]
        if match.group(1)[0].isupper():
            weekday = weekday.capitalize()
        return f"{weekday}{match.group(2)}{match.group(3)}"

    return WEEKDAY_DATE_PATTERN.sub(replace, text)


FLIGHT_PRICE_PATTERN = re.compile(
    r"^\s*(?P<currency>[^\d\s]+)?\s*(?P<amount>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s+(?P<suffix>[A-Z]{3}))?\s*$"
)


def _flight_rows(search_result: str) -> list[str]:
    lines = [line.strip() for line in search_result.splitlines() if line.strip()]
    if lines and re.match(r"^\d{2}/\d{2}/\d{4}\s*\|", lines[0]):
        lines = lines[1:]
    return lines


def _flight_price(row: str) -> tuple[Decimal, str, str] | None:
    price = row.rsplit("|", 1)[-1].strip()
    price = re.sub(r"\s*\(más barato\)\s*$", "", price)
    match = FLIGHT_PRICE_PATTERN.fullmatch(price)
    if match is None:
        return None
    amount = Decimal(match.group("amount").replace(",", ""))
    currency = match.group("currency") or match.group("suffix") or ""
    return amount, currency, price


def _best_flight_row(rows: list[str]) -> str | None:
    priced_rows = [(parsed[0], row) for row in rows if (parsed := _flight_price(row))]
    if not priced_rows:
        return None
    return min(priced_rows, key=lambda item: item[0])[1]


def _replace_section(lines: list[str], name: str, content: list[str]) -> bool:
    section_pattern = re.compile(rf"^\s*{name}\b", re.IGNORECASE)
    section_names = ("IDA", "REGRESO", "CLIMA EN", "RECOMENDACI[ÓO]N")
    next_section_pattern = re.compile(
        r"^\s*(?:" + "|".join(section_names) + r")\b",
        re.IGNORECASE,
    )
    start = next((i for i, line in enumerate(lines) if section_pattern.match(line)), None)
    if start is None:
        return False
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if next_section_pattern.match(lines[i])
        ),
        len(lines),
    )
    lines[start + 1 : end] = content + [""]
    return True


def _format_total(first: str, second: str) -> str | None:
    first_price = _flight_price(first)
    second_price = _flight_price(second)
    if first_price is None or second_price is None or first_price[1] != second_price[1]:
        return None
    total = first_price[0] + second_price[0]
    if total == total.to_integral_value():
        amount = f"{total:,.0f}"
    else:
        amount = f"{total:,.2f}"
    return f"{first_price[1]}{amount}"


def _normalize_climate_section(lines: list[str]) -> str | None:
    climate_index = next(
        (
            i for i, line in enumerate(lines)
            if re.match(r"^\s*CLIMA EN\b", line, re.IGNORECASE)
        ),
        None,
    )
    if climate_index is None:
        return None

    recommendation_index = next(
        (
            i for i, line in enumerate(lines[climate_index + 1 :], climate_index + 1)
            if re.match(r"^\s*RECOMENDACI[ÓO]N\b", line, re.IGNORECASE)
        ),
        len(lines),
    )
    climate_lines = lines[climate_index + 1 : recommendation_index]
    normalized_lines = []
    for line in climate_lines:
        match = BARE_DATE_PATTERN.match(line)
        if match is None:
            normalized_lines.append(line)
            continue
        try:
            day = datetime.strptime(match.group(1), "%d/%m/%Y").date()
        except ValueError:
            normalized_lines.append(line)
            continue
        normalized_lines.append(
            f"{DIAS[day.weekday()]} {match.group(1)}{line[match.end(1):]}"
        )

    lines[climate_index + 1 : recommendation_index] = normalized_lines
    climate_facts = [line.strip() for line in normalized_lines if line.strip()]
    if not climate_facts:
        return None
    city = re.sub(r"^\s*CLIMA EN\s+", "", lines[climate_index], flags=re.IGNORECASE).strip()
    summary = "; ".join(climate_facts)
    return f"Clima en {city}: {summary}" if city else f"Clima: {summary}"


WEATHER_CONDITIONS = (
    (("parcialmente nublado", "partly cloudy"), "Parcialmente nublado"),
    (("nublado", "nublada", "cloudy"), "Nublado"),
    (("soleado", "soleada", "sunny"), "Soleado"),
    (("lluvia ligera", "light rain"), "Lluvia ligera"),
    (("lluvioso", "lluviosa", "rainy"), "Lluvioso"),
    (("tormentas aisladas", "isolated thunderstorms"), "Tormentas aisladas"),
    (("ventoso", "ventosa", "windy"), "Ventoso"),
)
TEMPERATURE_PATTERN = re.compile(r"(?P<temperature>-?\d{1,2}(?:[.,]\d+)?)\s*°\s*C", re.IGNORECASE)


def normalize_weather_result(response: str) -> str:
    """Remove model reasoning and retain only a recognized simulated condition and temperature."""
    tag_filter = TagFilter()
    clean_response = tag_filter.feed(response, final=True)
    normalized = _remove_accents(clean_response)

    condition = next(
        (
            display
            for phrases, display in WEATHER_CONDITIONS
            if any(re.search(rf"\b{re.escape(phrase)}\b", normalized) for phrase in phrases)
        ),
        None,
    )
    if condition is None:
        log.warning("No se reconoció la condición del pronóstico: %r", clean_response)
        return "Pronóstico no disponible."

    temperature = TEMPERATURE_PATTERN.search(clean_response)
    if temperature is None:
        log.warning("No se reconoció la temperatura del pronóstico: %r", clean_response)
        return "Pronóstico no disponible."
    return f"{condition}, {temperature.group('temperature').replace(',', '.')} °C"


def normalize_trip_response(response: str, search_results: list[str]) -> str:
    """Restore tool-authoritative flight rows and calculate the recommendation from them."""
    if not search_results:
        return correct_weekdays(response).strip()

    lines = correct_weekdays(response).splitlines()
    climate_summary = _normalize_climate_section(lines)
    flight_rows = [_flight_rows(result) for result in search_results]
    ida_present = _replace_section(lines, "IDA", flight_rows[0])

    if len(flight_rows) > 1:
        if not _replace_section(lines, "REGRESO", flight_rows[1]):
            climate_index = next(
                (
                    i for i, line in enumerate(lines)
                    if re.match(r"^\s*CLIMA EN\b", line, re.IGNORECASE)
                ),
                len(lines),
            )
            lines[climate_index:climate_index] = ["REGRESO", *flight_rows[1], ""]
    else:
        lines = [
            line for line in lines
            if not re.match(r"^\s*REGRESO\b", line, re.IGNORECASE)
            and not re.match(r"^\s*No hay regreso programado\b", line, re.IGNORECASE)
        ]

    recommendations = ["RECOMENDACIÓN"]
    ida_best = _best_flight_row(flight_rows[0])
    if ida_best is None:
        recommendations.append("No se pudo determinar la opción más económica de ida.")
    else:
        recommendations.append(
            f"Opción más económica de ida: {ida_best.removesuffix(' (más barato)')}."
        )
    if len(flight_rows) > 1:
        regreso_best = _best_flight_row(flight_rows[1])
        if regreso_best is None:
            recommendations.append("No se pudo determinar la opción más económica de regreso.")
        else:
            recommendations.append(
                f"Opción más económica de regreso: "
                f"{regreso_best.removesuffix(' (más barato)')}."
            )
            if ida_best is not None:
                total = _format_total(ida_best, regreso_best)
                if total is not None:
                    recommendations.append(f"Total de ida y vuelta: {total}.")
    if climate_summary:
        recommendations.append(climate_summary)

    recommendation_index = next(
        (
            i for i, line in enumerate(lines)
            if re.match(r"^\s*RECOMENDACI[ÓO]N\b", line, re.IGNORECASE)
        ),
        None,
    )
    if recommendation_index is None:
        lines.extend(["", *recommendations])
    else:
        lines[recommendation_index:] = recommendations
    return "\n".join(lines).strip()


async def _direct_flight_response(
    origin: str,
    destination: str,
    travel_date: date,
) -> str:
    flight_request = (
        f"Busca vuelos de {origin} a {destination} para el {travel_date.isoformat()}."
    )
    flight_result = await ask_flights(flight_request)

    try:
        weather_result = await _invoke_remote_agent(
            WEATHER_ARN,
            f"Consulta el pronóstico del clima para {destination} "
            f"el {travel_date.isoformat()}.",
        )
    except Exception:
        log.exception(
            "No se pudo obtener el clima en %s para %s",
            destination,
            travel_date.isoformat(),
        )
        weather_result = "No se pudo consultar el clima en este momento."

    display_date = travel_date.strftime("%d/%m/%Y")
    weekday = DIAS[travel_date.weekday()]
    weather_text = normalize_weather_result(weather_result)
    response = "\n".join(
        [
            f"PLAN DE VIAJE: {origin} -> {destination}",
            f"Fecha: {weekday} {display_date}",
            "",
            f"IDA - {weekday} {display_date} ({origin} -> {destination})",
            "",
            f"CLIMA EN {destination}",
            f"  {weekday} {display_date}: {weather_text}",
            "",
            "RECOMENDACIÓN",
        ]
    )
    return normalize_trip_response(response, [flight_result])


async def _stream_text_response(text: str):
    yield {
        "event": {
            "contentBlockDelta": {
                "delta": {"text": text},
                "contentBlockIndex": 0,
            }
        }
    }
    yield {"event": {"contentBlockStop": {"contentBlockIndex": 0}}}


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
        f"domingo es {_fecha_legible(sunday)}. Para solicitudes de fin de semana usa estas fechas; "
        "si el usuario indica una fecha explícita, usa esa fecha.\n\n"
        "MEMORIA DEL USUARIO: usa datos guardados solo para preferencias personales, como "
        "aerolínea u horario. Nunca uses memoria para reemplazar el origen, destino o fechas "
        "explícitos del mensaje actual. Los datos explícitos del mensaje actual siempre tienen "
        "prioridad sobre cualquier viaje anterior o memoria. "
        "La memoria solo aporta preferencias (aerolínea, horario): NUNCA define el origen, "
        "el destino ni las fechas, que salen siempre del mensaje actual del usuario, aunque "
        "la memoria mencione otros viajes. "
        "La respuesta de ask_flights incluye una aerolínea explícita por vuelo; usa ese dato "
        "como fuente de verdad y no contradigas el nombre de aerolínea con el código.\n\n"
        "PLAN DE TRABAJO:\n"
        "- El origen y el destino son los que dice el usuario en su mensaje actual (ciudad o código). "
        "No conviertas nombres de ciudades a códigos IATA ni inventes códigos: cuando el usuario "
        "dé una ciudad, pásala por su nombre a ask_flights para que Flights Agent la resuelva con "
        "sk_resolve_iata. Si el usuario da un código, conserva ese código exactamente.\n"
        "Si pide 'este fin de semana', la ida es el sábado y el regreso el domingo. "
        "Si da una fecha (por ejemplo '20 de noviembre'), esa es la fecha de la ida (año 2026 si no "
        "dice otro) y no hay regreso salvo que pida ida y vuelta o dé una segunda fecha. "
        "Nunca le pidas la fecha al usuario.\n"
        "- Si no hay regreso, omite la sección REGRESO, cambia la línea 'Fin de semana del ...' "
        "por 'Fecha: <día> <DD/MM/AAAA>' y consulta el clima solo de esa fecha.\n"
        "- El día de la semana debe corresponder al calendario de cada fecha. No lo deduzcas "
        "del contexto del viaje; verifica siempre el día a partir de DD/MM/AAAA.\n"
        "- Llama a ask_flights una vez para cada trayecto solicitado. Para el regreso, invierte el "
        "trayecto y usa su fecha. En cada llamada incluye ciudad de origen, ciudad de destino y fecha "
        "en formato YYYY-MM-DD.\n"
        "- Llama a ask_weather para la ciudad de destino, una vez por cada fecha del viaje "
        "(sábado y domingo en un fin de semana; solo la fecha de ida si no hay regreso), con fecha "
        "YYYY-MM-DD. Escribe las condiciones del clima en español.\n"
        "- Esta regla también aplica a solicitudes directas de vuelos aunque el usuario no mencione "
        "un plan de viaje ni el clima: consulta siempre ask_weather para cada fecha solicitada.\n"
        "- Haz todas las llamadas antes de responder y no escribas texto antes de llamarlas.\n"
        "- Si una herramienta falla, dilo en esa sección y continúa con el resto.\n\n"
        "RESPUESTA FINAL: en español, texto plano (sin asteriscos, sin tablas, sin etiquetas "
        "<thinking>), con exactamente esta estructura:\n\n"
        "PLAN DE VIAJE: <origen> -> <destino>\n"
        "Fin de semana del <día> <DD/MM/AAAA> al <día> <DD/MM/AAAA> "
        "(ejemplo: Fin de semana del sábado 10/10/2026 al domingo 11/10/2026; "
        "si no hay regreso, usa 'Fecha: <día> <DD/MM/AAAA>')\n\n"
        "IDA - <día y fecha> (<origen> -> <destino>)\n"
        "  <vuelo> | <aerolínea> | salida <hora> | <precio>\n"
        "  (copia las líneas de vuelo tal cual las devuelve ask_flights, sin su primera línea de ruta "
        "y fecha: mismo orden, mismo texto, mismo precio y misma marca ' (más barato)'. No las "
        "reordenes ni agregues texto. Si ask_flights devuelve que no hay vuelos o un error, "
        "escribe ese mensaje en su lugar)\n\n"
        "REGRESO - <día y fecha> (<destino> -> <origen>)\n"
        "  (mismo formato que la ida, incluida la marca del más barato; omite esta sección "
        "si el usuario no solicitó regreso)\n\n"
        "CLIMA EN <DESTINO>\n"
        "  <día y fecha>: <condición>, <temperatura> °C "
        "(incluye solo las fechas solicitadas)\n\n"
        "RECOMENDACION\n"
        "  Una o dos frases. Resume los vuelos más baratos (si hay ida y regreso, incluye el total "
        "exacto de ambos) y cómo estará el clima. Si hay preferencias "
        "guardadas del usuario (aerolínea u horario), recomienda un vuelo que cumpla todas las "
        "preferencias posibles y menciona por qué. Compara la preferencia de aerolínea con el campo "
        "explícito de aerolínea devuelto por la herramienta y la de horario con la hora de salida. "
        "Usa los datos de los vuelos de esta búsqueda; no asumas códigos ni aerolíneas de ejemplos "
        "o búsquedas anteriores. Nunca digas que un vuelo no cumple una preferencia si sus datos "
        "explícitos sí la cumplen. "
        "Si ningún vuelo cumple todas, recomienda el que cumpla más y explica con precisión el "
        "conflicto. Si no hay preferencias guardadas, omite la recomendación de preferencias.\n\n"
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

    prompt = _extract_prompt(payload)
    direct_query = parse_direct_flight_query(prompt)
    if direct_query is not None:
        origin, destination, travel_date = direct_query
        response = await _direct_flight_response(
            origin,
            destination,
            travel_date,
        )
        async for event in _stream_text_response(response):
            yield event
        return

    session_id = getattr(context, "session_id", "default-session")
    actor_id = _resolve_actor_id(payload, context)
    log.info(f"session_id={session_id} actor_id={actor_id}")
    agent = get_or_create_agent(session_id, actor_id)

    search_results: list[str] = []
    token = flight_search_results.set(search_results)
    try:
        tag_filter = TagFilter()
        response_parts = []
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
                if clean:
                    response_parts.append(clean)
            elif "contentBlockStop" in ev:
                rest = tag_filter.feed("", final=True)
                if rest:
                    response_parts.append(rest)
                tag_filter = TagFilter()
        response_parts.append(tag_filter.feed("", final=True))
        response = normalize_trip_response("".join(response_parts), search_results)
    finally:
        flight_search_results.reset(token)
    async for event in _stream_text_response(response):
        yield event


if __name__ == "__main__":
    app.run()