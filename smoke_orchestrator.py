import json
import sys
import uuid

import boto3

REGION = "us-east-2"
ARN = "arn:aws:bedrock-agentcore:us-east-2:289076681196:runtime/tripplaner_OrchestratorAgent-arYrJOAxNh"
ACTOR = sys.argv[1] if len(sys.argv) > 1 else f"test-{uuid.uuid4().hex[:8]}"

# (prompt, expected cities, expected date)
CASES = [
    ("Plan de viaje de Barranquilla a Madrid este fin de semana", [("Barranquilla", "BAQ"), ("Madrid", "MAD")], "24/10/2026"),
    ("Vuelos de BAQ a MAD el 20 de noviembre", [("Barranquilla", "BAQ"), ("Madrid", "MAD")], "20/11/2026"),
    ("Vuelos de BAQ a MAD el 25 de noviembre", [("Barranquilla", "BAQ"), ("Madrid", "MAD")], "25/11/2026"),
]
FORBIDDEN = ["Cartagena", "Bogotá"]

client = boto3.client("bedrock-agentcore", region_name=REGION)


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


def read_text(response) -> str:
    raw = response["response"].read().decode("utf-8")
    if "event-stream" in response.get("contentType", ""):
        parts: list[str] = []
        for line in raw.splitlines():
            if line.startswith("data:"):
                chunk = line[5:].strip()
                try:
                    parts.extend(texts(json.loads(chunk)))
                except ValueError:
                    parts.append(chunk)
        return "".join(parts).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, str) else json.dumps(parsed, ensure_ascii=False)
    except ValueError:
        return raw


print("Actor:", ACTOR)
for prompt, cities, expected_date in CASES:
    print(f"\n=== {prompt}")
    response = client.invoke_agent_runtime(
        agentRuntimeArn=ARN,
        runtimeSessionId=f"{uuid.uuid4()}-smoke",
        payload=json.dumps({"prompt": prompt, "actor_id": ACTOR}).encode("utf-8"),
    )
    text = read_text(response)
    print(text)
    route_ok = all(any(alternative in text for alternative in pair) for pair in cities) and not any(
        forbidden in text for forbidden in FORBIDDEN
    )
    date_ok = expected_date in text
    print(f"\nRUTA:  {'OK' if route_ok else 'FALLA'}")
    print(f"FECHA: {'OK' if date_ok else 'FALLA'}")
