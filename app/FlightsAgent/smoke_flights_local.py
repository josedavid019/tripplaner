import re

import httpx

URL = "http://localhost:9000/"
CASES = [
    ("ruta + fecha 1", "Vuelos de BAQ a MAD el 20 de noviembre"),
    ("ruta + fecha 2", "Vuelos de BAQ a MAD el 25 de noviembre"),
    ("ciudades por nombre", "Vuelos de Barranquilla a Madrid el 20 de noviembre"),
    ("fuera de tema", "¿Qué clima hace en Madrid?"),
]
HEADER = re.compile(r"^\d{2}/\d{2}/\d{4} \| [A-Z]{3} → [A-Z]{3}$")
TIME = re.compile(r"salida (\d{2}:\d{2})")


def texts(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "text" and isinstance(value, str):
                yield value
            else:
                yield from texts(value)
    elif isinstance(node, list):
        for item in node:
            yield from texts(item)


def ask(prompt: str, n: int) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id": str(n),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "kind": "message",
                "messageId": f"smoke-{n}",
                "parts": [{"kind": "text", "text": prompt}],
            }
        },
    }
    response = httpx.post(URL, json=payload, timeout=120)
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise RuntimeError(body["error"])
    result = body["result"]
    source = result.get("artifacts") or result.get("status", {}).get("message") or result
    return "\n".join(texts(source)).strip()


def check_flights(text: str) -> list[str]:
    lines = text.splitlines()
    problems = []
    if not lines or not HEADER.match(lines[0]):
        problems.append("la primera línea no es 'DD/MM/AAAA | ORIGEN → DESTINO'")
    flights = lines[1:]
    if not flights:
        problems.append("no hay líneas de vuelo")
    if sum("(más barato)" in line for line in flights) != 1:
        problems.append("debe haber exactamente un '(más barato)'")
    times = [m.group(1) for line in flights if (m := TIME.search(line))]
    if len(times) != len(flights):
        problems.append("hay líneas sin 'salida HH:MM'")
    if times != sorted(times):
        problems.append("los vuelos no están ordenados por hora")
    if "<thinking>" in text:
        problems.append("aparece <thinking> en la respuesta")
    return problems


answers = {}
for n, (label, prompt) in enumerate(CASES, start=1):
    print(f"=== {label}: {prompt}")
    text = ask(prompt, n)
    answers[label] = text
    print(text)
    if label != "fuera de tema":
        problems = check_flights(text)
        print("RESULTADO:", "OK" if not problems else "FALLA -> " + "; ".join(problems))
    print()

same = answers["ruta + fecha 1"] == answers["ruta + fecha 2"]
print("Dos fechas distintas dan respuestas distintas:", "NO (revisar)" if same else "SI")
