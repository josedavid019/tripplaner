"""Pide la agent card de un especialista desplegado en AgentCore (firmada con SigV4).

Uso (desde la carpeta del proyecto, con tus credenciales de AWS configuradas):
    uv run python check_agent_card.py flights
    uv run python check_agent_card.py weather
"""
import json
import sys
from urllib.parse import quote

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

REGION = "us-east-2"
ARNS = {
    "flights": "arn:aws:bedrock-agentcore:us-east-2:289076681196:runtime/tripplaner_FlightsAgent-NIrzy8HGru",
    "weather": "arn:aws:bedrock-agentcore:us-east-2:289076681196:runtime/tripplaner_WeatherAgent-aJJvVI7ABv",
}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "flights"
    arn = ARNS[name]
    url = (
        f"https://bedrock-agentcore.{REGION}.amazonaws.com/runtimes/"
        f"{quote(arn, safe='')}/invocations/.well-known/agent-card.json"
    )
    creds = boto3.Session().get_credentials().get_frozen_credentials()
    request = AWSRequest(method="GET", url=url)
    SigV4Auth(creds, "bedrock-agentcore", REGION).add_auth(request)

    response = httpx.get(url, headers=dict(request.headers), timeout=30)
    print("HTTP", response.status_code)
    try:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except ValueError:
        print(response.text)


if __name__ == "__main__":
    main()