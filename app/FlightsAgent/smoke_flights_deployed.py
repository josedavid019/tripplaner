import json
from urllib.parse import quote

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

REGION = "us-east-2"
ARN = "arn:aws:bedrock-agentcore:us-east-2:289076681196:runtime/tripplaner_FlightsAgent-NIrzy8HGru"
URL = (
    f"https://bedrock-agentcore.{REGION}.amazonaws.com/runtimes/"
    f"{quote(ARN, safe='')}/invocations/"
)
PROMPTS = [
    "Vuelos de BAQ a MAD el 20 de noviembre",
    "Vuelos de BAQ a MAD el 25 de noviembre",
]


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
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": str(n),
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "kind": "message",
                    "messageId": f"deployed-{n}",
                    "parts": [{"kind": "text", "text": prompt}],
                }
            },
        }
    )
    creds = boto3.Session().get_credentials().get_frozen_credentials()
    request = AWSRequest(
        method="POST", url=URL, data=body, headers={"Content-Type": "application/json"}
    )
    SigV4Auth(creds, "bedrock-agentcore", REGION).add_auth(request)
    response = httpx.post(URL, content=body, headers=dict(request.headers), timeout=180)
    print("HTTP", response.status_code)
    response.raise_for_status()
    result = response.json()["result"]
    source = result.get("artifacts") or result.get("status", {}).get("message") or result
    return "\n".join(texts(source)).strip()


answers = []
for n, prompt in enumerate(PROMPTS, start=1):
    print(f"=== {prompt}")
    text = ask(prompt, n)
    answers.append(text)
    print(text)
    print()

print(
    "Dos fechas distintas dan respuestas distintas:",
    "NO (revisar)" if answers[0] == answers[1] else "SI",
)
