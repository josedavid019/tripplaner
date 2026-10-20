from collections.abc import AsyncIterator
from typing import Any

from strands import Agent


async def final_response_stream(
    events: AsyncIterator[Any],
    search_state: dict[str, str | None],
) -> AsyncIterator[Any]:
    search_state["response"] = None
    async for event in events:
        if "result" not in event:
            continue

        response = search_state["response"]
        if response is not None:
            event["result"].message = {
                "role": "assistant",
                "content": [{"text": response}],
            }
        yield event


class FlightResponseAgent(Agent):
    def __init__(
        self,
        *args: Any,
        search_state: dict[str, str | None],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._search_state = search_state

    async def stream_async(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async for event in final_response_stream(
            super().stream_async(*args, **kwargs),
            self._search_state,
        ):
            yield event
