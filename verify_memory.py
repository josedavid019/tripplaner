import boto3

REGION = "us-east-2"
MEMORY_ID = "tripplaner_TripPlannerMemory-Z0ubr16QhZ"
ACTOR = "default-user"

client = boto3.client("bedrock-agentcore", region_name=REGION)


def list_all(operation, result_key, **kwargs):
    results = []
    next_token = None
    while True:
        request = dict(kwargs)
        if next_token:
            request["nextToken"] = next_token
        response = getattr(client, operation)(**request)
        results.extend(response.get(result_key, []))
        next_token = response.get("nextToken")
        if not next_token:
            return results


sessions = list_all(
    "list_sessions",
    "sessionSummaries",
    memoryId=MEMORY_ID,
    actorId=ACTOR,
    maxResults=100,
)
print("Sesiones:", [session["sessionId"] for session in sessions])

for session in sessions:
    session_id = session["sessionId"]
    events = list_all(
        "list_events",
        "events",
        memoryId=MEMORY_ID,
        actorId=ACTOR,
        sessionId=session_id,
        maxResults=100,
    )
    print(session_id, "->", len(events), "eventos")

for namespace in (f"/preferences/{ACTOR}/", f"/facts/{ACTOR}/"):
    records = list_all(
        "retrieve_memory_records",
        "memoryRecordSummaries",
        memoryId=MEMORY_ID,
        namespace=namespace,
        searchCriteria={"searchQuery": "preferencias de vuelo", "topK": 5},
        maxResults=100,
    )
    print(namespace)
    for record in records:
        text = record.get("content", {}).get("text", "")
        print("  -", text)