from strands import Agent, tool
from strands.multiagent.a2a.executor import StrandsA2AExecutor
from bedrock_agentcore.runtime import serve_a2a
from model.load import load_model


@tool
def add_numbers(a: int, b: int) -> int:
    """Return the sum of two numbers."""
    return a + b


tools = [add_numbers]


SYSTEM_PROMPT = """
You are a helpful assistant. Use tools when appropriate.

"""

agent = Agent(
    model=load_model(),
    system_prompt=SYSTEM_PROMPT,
    tools=tools,
)

if __name__ == "__main__":
    serve_a2a(StrandsA2AExecutor(agent))
