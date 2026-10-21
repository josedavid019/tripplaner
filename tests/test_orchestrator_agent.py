from datetime import date
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


class FlightWeatherRoutingTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.OrchestratorAgent.main._invoke_remote_agent", new_callable=AsyncMock)
    @patch("app.OrchestratorAgent.main.get_or_create_agent")
    async def test_direct_flight_query_calls_flights_and_weather_deterministically(
        self, get_or_create_agent, invoke_remote_agent
    ):
        invoke_remote_agent.side_effect = [
            "20/11/2026 | BAQ -> MAD\n"
            "LA4157 / LA4907 | LATAM Airlines | salida 20:50 | $720 (más barato)",
            "Soleado, 29 °C",
        ]

        events = [
            event
            async for event in orchestrator.invoke(
                {"prompt": "Vuelos de BAQ a MAD el 20 de noviembre"},
                context=None,
            )
        ]

        get_or_create_agent.assert_not_called()
        self.assertEqual(invoke_remote_agent.await_count, 2)
        self.assertEqual(
            invoke_remote_agent.await_args_list[0].args,
            (
                orchestrator.FLIGHTS_ARN,
                "Busca vuelos de BAQ a MAD para el 2026-11-20.",
            ),
        )
        self.assertEqual(
            invoke_remote_agent.await_args_list[1].args,
            (
                orchestrator.WEATHER_ARN,
                "Consulta el pronóstico del clima para MAD el 2026-11-20.",
            ),
        )
        self.assertEqual(
            events[0]["event"]["contentBlockDelta"]["delta"]["text"],
            "PLAN DE VIAJE: BAQ -> MAD\n"
            "Fecha: viernes 20/11/2026\n\n"
            "IDA - viernes 20/11/2026 (BAQ -> MAD)\n"
            "LA4157 / LA4907 | LATAM Airlines | salida 20:50 | $720 (más barato)\n\n"
            "CLIMA EN MAD\n"
            "  viernes 20/11/2026: Soleado, 29 °C\n\n"
            "RECOMENDACIÓN\n"
            "Opción más económica de ida: LA4157 / LA4907 | LATAM Airlines | "
            "salida 20:50 | $720.\n"
            "Clima en MAD: viernes 20/11/2026: Soleado, 29 °C",
        )

    def test_direct_flight_parser_resolves_supported_date_formats(self):
        self.assertEqual(
            orchestrator.parse_direct_flight_query(
                "Vuelos de BAQ a MAD el 25 de noviembre"
            ),
            ("BAQ", "MAD", date(2026, 11, 25)),
        )
        self.assertEqual(
            orchestrator.parse_direct_flight_query(
                "Vuelos de Barranquilla a Madrid el 20/11/2026"
            ),
            ("Barranquilla", "Madrid", date(2026, 11, 20)),
        )
        self.assertIsNone(
            orchestrator.parse_direct_flight_query("¿Qué clima hace en Madrid?")
        )


class OrchestratorPromptTests(unittest.TestCase):
    def test_prompt_prioritizes_current_trip_and_supports_one_way_search(self):
        prompt = orchestrator.build_system_prompt()

        self.assertIn("NUNCA define el origen", prompt)
        self.assertIn("la fecha de la ida", prompt)
        self.assertIn("no hay regreso", prompt)
        self.assertIn("consulta el clima solo de esa fecha", prompt)
        self.assertIn("Escribe las condiciones del clima en español", prompt)
        self.assertIn(
            "también aplica a solicitudes directas de vuelos",
            prompt,
        )
        self.assertIn("No conviertas nombres de ciudades a códigos IATA", prompt)
        self.assertIn("pásala por su nombre a ask_flights", prompt)
        self.assertIn("mismo orden, mismo texto, mismo precio", prompt)
        self.assertNotIn("<precio> USD", prompt)

    def test_weekday_labels_are_corrected_from_the_calendar_date(self):
        self.assertEqual(
            orchestrator.correct_weekdays(
                "sábado 20/11/2026\ndomingo 25/11/2026\nsábado 24/10/2026"
            ),
            "viernes 20/11/2026\nmiércoles 25/11/2026\nsábado 24/10/2026",
        )

    def test_invalid_calendar_dates_are_left_unchanged(self):
        self.assertEqual(
            orchestrator.correct_weekdays("sábado 31/02/2026"),
            "sábado 31/02/2026",
        )

    def test_weather_result_removes_thinking_tags_and_keeps_forecast_facts(self):
        self.assertEqual(
            orchestrator.normalize_weather_result(
                "<thinking>I should explain the tool call.</thinking> "
                "El pronóstico para Madrid es parcialmente nublado con una "
                "temperatura de 24°C."
            ),
            "Parcialmente nublado, 24 °C",
        )
        self.assertEqual(
            orchestrator.normalize_weather_result(
                "<thinking>I have the forecast.</thinking> "
                "Madrid will have isolated thunderstorms, 22 °C."
            ),
            "Tormentas aisladas, 22 °C",
        )

    def test_weather_result_does_not_echo_unrecognized_model_prose(self):
        self.assertEqual(
            orchestrator.normalize_weather_result(
                "<thinking>Internal reasoning.</thinking> I cannot provide that."
            ),
            "Pronóstico no disponible.",
        )

    def test_normalizer_restores_flight_order_and_calculates_round_trip_total(self):
        response = (
            "PLAN DE VIAJE: Barranquilla -> Madrid\n"
            "IDA - sábado 24/10/2026 (Barranquilla -> Madrid)\n"
            "AV9525 / IB154 | Avianca, Iberia | salida 18:20 | $816 (más barato)\n"
            "REGRESO - domingo 25/10/2026 (Madrid -> Barranquilla)\n"
            "AV183 / AV9532 | Avianca | salida 10:10 | $1,035 (más barato)\n"
            "CLIMA EN MADRID\n"
            "24/10/2026: Lluvioso, 14 °C\n"
            "RECOMENDACION\n"
            "El vuelo más barato es AV9525. El totalde tu viaje es $1,831."
        )
        ida = (
            "24/10/2026 | BAQ -> MAD\n"
            "AV8419 / IB156 | Avianca, Iberia | salida 06:25 | $823\n"
            "AV9525 / IB154 | Avianca, Iberia | salida 18:20 | $816 (más barato)"
        )
        regreso = (
            "25/10/2026 | MAD -> BAQ\n"
            "AV183 / AV9532 | Avianca | salida 10:10 | $1,035 (más barato)"
        )

        normalized = orchestrator.normalize_trip_response(response, [ida, regreso])

        self.assertIn(
            "AV8419 / IB156 | Avianca, Iberia | salida 06:25 | $823\n"
            "AV9525 / IB154 | Avianca, Iberia | salida 18:20 | $816 (más barato)",
            normalized,
        )
        self.assertIn("Total de ida y vuelta: $1,851.", normalized)
        self.assertIn("sábado 24/10/2026: Lluvioso, 14 °C", normalized)
        self.assertIn(
            "Clima en MADRID: sábado 24/10/2026: Lluvioso, 14 °C",
            normalized,
        )
        self.assertNotIn("$1,831", normalized)
        self.assertNotIn("totalde", normalized)

    def test_normalizer_removes_false_return_message_for_one_way_trip(self):
        response = (
            "IDA - viernes 20/11/2026 (BAQ -> MAD)\n"
            "Otra opción: $751\n"
            "RECOMENDACION\n"
            "No hay regreso programado para este viaje."
        )
        flights = (
            "20/11/2026 | BAQ -> MAD\n"
            "LA4157 / LA4907 | LATAM Airlines | salida 20:50 | $720 (más barato)"
        )

        normalized = orchestrator.normalize_trip_response(response, [flights])

        self.assertIn("$720 (más barato)", normalized)
        self.assertIn(
            "Opción más económica de ida: LA4157 / LA4907 | LATAM Airlines | "
            "salida 20:50 | $720.",
            normalized,
        )
        self.assertNotIn("$751", normalized)
        self.assertNotIn("No hay regreso programado", normalized)


class AskFlightsFallbackTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.OrchestratorAgent.main._invoke_remote_agent", new_callable=AsyncMock)
    async def test_successful_flight_results_are_captured_for_final_response(self, invoke_remote):
        invoke_remote.return_value = (
            "20/11/2026 | BAQ -> MAD\n"
            "LA4157 | LATAM Airlines | salida 20:50 | $720 (más barato)"
        )
        captured_results = []
        token = orchestrator.flight_search_results.set(captured_results)
        try:
            response = await orchestrator.ask_flights("BAQ a MAD el 20 de noviembre")
        finally:
            orchestrator.flight_search_results.reset(token)

        self.assertEqual(captured_results, [response])
        self.assertIn("$720 (más barato)", response)

    @patch("app.OrchestratorAgent.main._invoke_remote_agent", new_callable=AsyncMock)
    async def test_ask_flights_returns_fixed_message_when_flights_fails(self, invoke_remote):
        invoke_remote.side_effect = RuntimeError("timeout")

        response = await orchestrator.ask_flights("Vuelos de BAQ a MAD")

        self.assertEqual(
            response,
            "No se pudo consultar vuelos en este momento. Intenta de nuevo en unos minutos.",
        )


class AskWeatherLoggingTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.OrchestratorAgent.main._invoke_remote_agent", new_callable=AsyncMock)
    async def test_ask_weather_logs_and_reraises_remote_failures(self, invoke_remote):
        invoke_remote.side_effect = RuntimeError("weather runtime unavailable")

        with self.assertRaisesRegex(RuntimeError, "weather runtime unavailable"):
            await orchestrator.ask_weather("Madrid 2026-10-24")


if __name__ == "__main__":
    unittest.main()
