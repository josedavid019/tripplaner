import unittest
from unittest.mock import AsyncMock, patch

from app.OrchestratorAgent import main as orchestrator


class RemoteAgentLifecycleTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.OrchestratorAgent.main.make_remote_agent")
    @patch("app.OrchestratorAgent.main.httpx.AsyncClient")
    async def test_remote_http_client_is_closed_after_success(self, async_client_class, make_remote_agent):
        http_client = AsyncMock()
        async_client_class.return_value.__aenter__.return_value = http_client
        remote_agent = AsyncMock()
        remote_agent.invoke_async.return_value = "response"
        make_remote_agent.return_value = remote_agent

        response = await orchestrator._invoke_remote_agent("runtime-arn", "request")

        self.assertEqual(response, "response")
        make_remote_agent.assert_called_once_with("runtime-arn", http_client)
        remote_agent.invoke_async.assert_awaited_once_with("request")
        async_client_class.return_value.__aexit__.assert_awaited_once()

    @patch("app.OrchestratorAgent.main.make_remote_agent")
    @patch("app.OrchestratorAgent.main.httpx.AsyncClient")
    async def test_remote_http_client_is_closed_after_failure(self, async_client_class, make_remote_agent):
        http_client = AsyncMock()
        async_client_class.return_value.__aenter__.return_value = http_client
        remote_agent = AsyncMock()
        remote_agent.invoke_async.side_effect = RuntimeError("remote failure")
        make_remote_agent.return_value = remote_agent

        with self.assertRaisesRegex(RuntimeError, "remote failure"):
            await orchestrator._invoke_remote_agent("runtime-arn", "request")

        async_client_class.return_value.__aexit__.assert_awaited_once()


class StandaloneFlightRoutingTests(unittest.IsolatedAsyncioTestCase):
    def test_only_direct_flight_queries_bypass_trip_planning(self):
        self.assertTrue(
            orchestrator._is_standalone_flight_search(
                "Vuelos de BAQ a MAD el 20 de noviembre"
            )
        )
        self.assertTrue(
            orchestrator._is_standalone_flight_search(
                "Vuelos de Barranquilla a Madrid el 20 de noviembre"
            )
        )
        self.assertFalse(
            orchestrator._is_standalone_flight_search(
                "Plan de viaje de BAQ a MAD este fin de semana"
            )
        )
        self.assertFalse(
            orchestrator._is_standalone_flight_search(
                "¿Qué clima hace en Madrid?"
            )
        )
        self.assertFalse(orchestrator._is_standalone_flight_search(["message history"]))

    async def test_direct_flight_response_is_streamed_without_rewriting(self):
        expected = "20/11/2026 | BAQ → MAD\nFlight ABC | Airline | salida 15:40 | $100"

        events = [
            event async for event in orchestrator._stream_text_response(expected)
        ]

        self.assertEqual(
            events[0]["event"]["contentBlockDelta"]["delta"]["text"],
            expected,
        )
        self.assertEqual(
            events[1],
            {"event": {"contentBlockStop": {"contentBlockIndex": 0}}},
        )

    @patch("app.OrchestratorAgent.main.get_or_create_agent")
    @patch("app.OrchestratorAgent.main._invoke_remote_agent", new_callable=AsyncMock)
    async def test_standalone_query_bypasses_orchestrator_model_and_memory(
        self,
        invoke_remote_agent,
        get_or_create_agent,
    ):
        invoke_remote_agent.return_value = "20/11/2026 | BAQ → MAD\nFlight ABC"

        events = [
            event
            async for event in orchestrator.invoke(
                {"prompt": "Vuelos de BAQ a MAD el 20 de noviembre"},
                context=None,
            )
        ]

        invoke_remote_agent.assert_awaited_once_with(
            orchestrator.FLIGHTS_ARN,
            "Vuelos de BAQ a MAD el 20 de noviembre",
        )
        get_or_create_agent.assert_not_called()
        self.assertEqual(
            events[0]["event"]["contentBlockDelta"]["delta"]["text"],
            "20/11/2026 | BAQ → MAD\nFlight ABC",
        )


class OrchestratorPromptTests(unittest.TestCase):
    def test_prompt_prioritizes_current_trip_and_supports_one_way_search(self):
        prompt = orchestrator.build_system_prompt()

        self.assertIn("NUNCA define el origen", prompt)
        self.assertIn("la fecha de la ida", prompt)
        self.assertIn("no hay regreso", prompt)
        self.assertIn("consulta el clima solo de esa fecha", prompt)
        self.assertIn("Escribe las condiciones del clima en español", prompt)
        self.assertIn("mismo orden, mismo texto, mismo precio", prompt)
        self.assertNotIn("<precio> USD", prompt)


class AskFlightsFallbackTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.OrchestratorAgent.main._invoke_remote_agent", new_callable=AsyncMock)
    async def test_ask_flights_returns_fixed_message_when_flights_fails(self, invoke_remote):
        invoke_remote.side_effect = RuntimeError("timeout")

        response = await orchestrator.ask_flights("Vuelos de BAQ a MAD")

        self.assertEqual(
            response,
            "No se pudo consultar vuelos en este momento. Intenta de nuevo en unos minutos.",
        )


if __name__ == "__main__":
    unittest.main()
