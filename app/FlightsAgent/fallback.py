# Mock de vuelos (datos simulados). Ya NO lo usa el agente en producción:
# queda para tests unitarios y como base de un fallback futuro.
from strands import tool

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
