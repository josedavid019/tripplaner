import asyncio
import unittest

from app.FlightsAgent.flight_response import final_response_stream


class FlightResponseStreamTests(unittest.TestCase):
    def test_suppresses_intermediate_text_and_returns_exact_search_output(self):
        response = "20/11/2026 | BAQ → MAD\nFlight ABC | Airline | salida 15:40 | $100"
        state = {"response": response}
        result = type("Result", (), {"message": {"role": "assistant", "content": []}})()

        async def events():
            yield {"data": "<thinking>reasoning</thinking>"}
            yield {"data": "Here are the flights"}
            state["response"] = response
            yield {"result": result}

        output = asyncio.run(self._collect(events(), state))

        self.assertEqual(len(output), 1)
        self.assertEqual(
            output[0]["result"].message,
            {"role": "assistant", "content": [{"text": response}]},
        )

    def test_preserves_final_answer_when_no_search_tool_was_used(self):
        state = {"response": "stale previous result"}
        result = type("Result", (), {"message": {"role": "assistant", "content": [{"text": "Hola"}]}})()

        async def events():
            yield {"data": "Hola"}
            yield {"result": result}

        output = asyncio.run(self._collect(events(), state))

        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["result"], result)
        self.assertIsNone(state["response"])

    @staticmethod
    async def _collect(events, state):
        return [event async for event in final_response_stream(events, state)]


if __name__ == "__main__":
    unittest.main()
