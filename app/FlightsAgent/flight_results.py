import html
import re
from dataclasses import dataclass
from datetime import datetime, time
from decimal import Decimal
from typing import Sequence


@dataclass(frozen=True)
class FlightOffer:
    departure_time: time
    flight: str
    airlines: str
    price: str
    amount: Decimal


PRICE_PATTERN = re.compile(
    r"^\s*(?P<prefix>[^\d\s,.-]+)?\s*(?P<amount>\d[\d,]*(?:\.\d+)?)"
    r"\s*(?P<suffix>[A-Z]{3})?\s*$"
)
TRIP_PATTERN = re.compile(r"#trip=([A-Z0-9]+(?:-[A-Z0-9]+)*)")
FLIGHT_CODE_PATTERN = re.compile(r"\b[A-Z]{2,3}\d{2,4}\b")
DEPARTURE_PATTERN = re.compile(r"\b(\d{1,2}:\d{2})(?::\d{2})?")


def _clean_cell(value: str) -> str:
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    return html.unescape(value).strip()


def _split_markdown_row(row: str) -> list[str]:
    cells: list[str] = []
    cell: list[str] = []
    escaped = False
    for character in row.strip().strip("|"):
        if escaped:
            cell.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(cell).strip())
            cell.clear()
        else:
            cell.append(character)
    if escaped:
        cell.append("\\")
    cells.append("".join(cell).strip())
    return cells


def _column_index(headers: Sequence[str], names: tuple[str, ...]) -> int | None:
    normalized = [re.sub(r"[^a-z]", "", header.lower()) for header in headers]
    return next(
        (
            index
            for index, header in enumerate(normalized)
            if any(header == name or header.startswith(name) for name in names)
        ),
        None,
    )


def _parse_price(value: str) -> tuple[str, Decimal]:
    value = _clean_cell(value)
    match = PRICE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Formato de precio MCP no reconocido: {value!r}")

    prefix = (match.group("prefix") or "").strip()
    suffix = match.group("suffix") or ""
    amount_text = match.group("amount")
    if "," in amount_text and not re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", amount_text):
        raise ValueError(f"Formato numérico de precio MCP no reconocido: {amount_text!r}")
    amount = Decimal(amount_text.replace(",", ""))
    display = f"{prefix}{' ' if prefix and prefix.isalpha() else ''}{amount_text}"
    if suffix:
        display = f"{display} {suffix}".strip()
    return display, amount


def _parse_offer(row: str, columns: dict[str, int | None]) -> FlightOffer | None:
    cells = _split_markdown_row(row)
    if cells and re.fullmatch(r"[-:\s]+", cells[0]):
        return None
    price_index = columns.get("price")
    airlines_index = columns.get("airlines")
    segments_index = columns.get("segments")
    if (
        price_index is None
        or airlines_index is None
        or segments_index is None
        or max(price_index, airlines_index, segments_index) >= len(cells)
    ):
        return None
    price_header = re.sub(r"[^a-z]", "", cells[price_index].lower())
    if price_header.startswith("price"):
        return None

    price, amount = _parse_price(cells[price_index])
    airlines = _clean_cell(cells[airlines_index])
    segments = _clean_cell(cells[segments_index])
    booking_index = columns.get("booking")
    booking = cells[booking_index] if booking_index is not None and booking_index < len(cells) else ""

    departure_match = DEPARTURE_PATTERN.search(segments)
    if departure_match is None:
        raise ValueError("La fila de vuelo MCP no contiene una hora de salida reconocible")
    departure_time = time.fromisoformat(departure_match.group(1))

    trip_match = TRIP_PATTERN.search(booking)
    if trip_match:
        flight = " / ".join(trip_match.group(1).split("-"))
    else:
        flight_codes = FLIGHT_CODE_PATTERN.findall(segments)
        if not flight_codes:
            raise ValueError("La fila de vuelo MCP no contiene códigos de vuelo")
        flight = " / ".join(flight_codes)

    if not airlines:
        raise ValueError("La fila de vuelo MCP no contiene aerolíneas")

    return FlightOffer(departure_time, flight, airlines, price, amount)


def _find_table_columns(markdown: str) -> dict[str, int | None] | None:
    for line in markdown.splitlines():
        if "|" not in line:
            continue
        headers = _split_markdown_row(line)
        price = _column_index(headers, ("price",))
        airlines = _column_index(headers, ("airline",))
        segments = _column_index(headers, ("segment", "itinerary"))
        if price is None or airlines is None or segments is None:
            continue
        return {
            "price": price,
            "airlines": airlines,
            "segments": segments,
            "booking": _column_index(headers, ("booking", "link")),
        }
    return None


def format_flight_results(
    markdown: str,
    origin: str,
    destination: str,
    departure_date: str,
) -> str:
    """Parse the Skiplagged Markdown table and render stable, sorted flight lines."""
    try:
        display_date = datetime.strptime(departure_date, "%Y-%m-%d").strftime("%d/%m/%Y")
    except ValueError as exc:
        raise ValueError("La fecha de salida debe tener el formato YYYY-MM-DD") from exc

    lowered = markdown.lower()
    if "no flights" in lowered or "no results" in lowered or "0 of 0 results" in lowered:
        return f"No se encontraron vuelos para {origin} a {destination} el {display_date}."

    columns = _find_table_columns(markdown)
    if columns is None:
        raise ValueError("La respuesta MCP no contiene una tabla de vuelos reconocible")

    offers = [
        offer
        for line in markdown.splitlines()
        if (offer := _parse_offer(line, columns)) is not None
    ]
    if not offers:
        raise ValueError("La respuesta MCP contiene una tabla pero ninguna fila de vuelos reconocible")

    offers.sort(key=lambda offer: offer.departure_time)
    cheapest_index = min(range(len(offers)), key=lambda index: offers[index].amount)
    lines = [f"{display_date} | {origin} → {destination}"]
    for index, offer in enumerate(offers):
        line = (
            f"{offer.flight} | {offer.airlines} | "
            f"salida {offer.departure_time.strftime('%H:%M')} | {offer.price}"
        )
        if index == cheapest_index:
            line += " (más barato)"
        lines.append(line)
    return "\n".join(lines)


def flight_search_unavailable_message(origin: str, destination: str, departure_date: str) -> str:
    """Return a fixed message when flight search is unavailable."""
    try:
        display_date = datetime.strptime(departure_date, "%Y-%m-%d").strftime("%d/%m/%Y")
    except ValueError:
        display_date = departure_date
    return (
        f"No se pudo consultar vuelos para {origin} a {destination} el {display_date} "
        "en este momento. Intenta de nuevo en unos minutos."
    )
