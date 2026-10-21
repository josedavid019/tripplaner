import unittest
from unittest.mock import patch

from app.WeatherAgent import main as weather


class WeatherAgentTests(unittest.TestCase):
    @patch("app.WeatherAgent.main.Agent")
    @patch("app.WeatherAgent.main.BedrockModel")
    def test_agent_is_instructed_to_call_forecast_tool(
        self, bedrock_model, agent_class
    ):
        weather.create_agent("test-context")

        agent_class.assert_called_once()
        self.assertIn(
            "llama siempre a get_forecast",
            agent_class.call_args.kwargs["system_prompt"],
        )
        self.assertEqual(
            agent_class.call_args.kwargs["tools"],
            [weather.get_forecast],
        )
        bedrock_model.assert_called_once_with(
            model_id="us.amazon.nova-lite-v1:0",
            region_name="us-east-2",
        )

    def test_forecast_is_deterministic_and_varies_by_city_or_date(self):
        dates = [
            "2026-10-24",
            "2026-10-25",
            "2026-11-20",
            "2026-11-25",
            "2026-12-01",
        ]
        forecasts = [weather.get_forecast("Madrid", date) for date in dates]

        self.assertEqual(forecasts[0], weather.get_forecast("Madrid", dates[0]))
        self.assertGreater(len({forecast["condition"] for forecast in forecasts}), 1)
        self.assertTrue(
            all(
                forecast["city"] == "Madrid"
                and forecast["date"] == date
                and isinstance(forecast["temp_c"], int)
                for forecast, date in zip(forecasts, dates)
            )
        )

    def test_forecast_includes_varied_simulated_conditions(self):
        conditions = {
            weather.get_forecast("Madrid", f"2026-10-{day:02d}")["condition"]
            for day in range(1, 29)
        }

        self.assertGreater(len(conditions), 1)
        self.assertLessEqual(
            conditions,
            {
                "Soleado",
                "Parcialmente nublado",
                "Nublado",
                "Lluvia ligera",
                "Lluvioso",
                "Tormentas aisladas",
                "Ventoso",
            },
        )


if __name__ == "__main__":
    unittest.main()
