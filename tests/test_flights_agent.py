import unittest

from app.FlightsAgent.fallback import airline_name, search_flights
from app.FlightsAgent.flight_results import format_flight_results


class FlightAirlineTests(unittest.TestCase):
    def test_search_results_include_explicit_airlines(self):
        flights = search_flights("BOG", "CTG", "2026-10-10")

        self.assertEqual(
            [(flight["flight"], flight["airline"]) for flight in flights],
            [
                ("AV9821", "Avianca"),
                ("LA4410", "LATAM"),
                ("AV9835", "Avianca"),
            ],
        )
        self.assertEqual(
            [(flight["flight"], flight["depart"]) for flight in flights],
            [("AV9821", "07:15"), ("LA4410", "12:40"), ("AV9835", "18:05")],
        )

    def test_unknown_airline_prefix_fails_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError, "Unknown airline code prefix: ZZ"):
            airline_name("ZZ1234")


class FlightResultFormattingTests(unittest.TestCase):
    def test_results_are_sorted_and_only_actual_lowest_price_is_marked(self):
        markdown = """
| Price | Duration | Stops | Type | Airlines | Segments | Booking |
| --- | --- | --- | --- | --- | --- | --- |
| $660 | 32h | 2 stops | — | LATAM Airlines | Outbound:<br/>BAQ → MAD (2026-11-20 21:20:00-05:00 → 2026-11-22 11:30:00+01:00) | [Book](https://example.test/#trip=LA4157-LA4907-LA8076) |
| $632 | 24h | 2 stops | — | LATAM Airlines | Outbound:<br/>BAQ → MAD (2026-11-20 20:50:00-05:00 → 2026-11-21 13:40:00+01:00) | [Book](https://example.test/#trip=LA4145-LA8211-LA8066) |
| $660 | 26h | 2 stops | — | LATAM Airlines | Outbound:<br/>BAQ → MAD (2026-11-20 15:40:00-05:00 → 2026-11-21 13:40:00+01:00) | [Book](https://example.test/#trip=LA4143-LA8211-LA8066) |
"""

        self.assertEqual(
            format_flight_results(markdown, "BAQ", "MAD", "2026-11-20"),
            "\n".join(
                [
                    "20/11/2026 | BAQ → MAD",
                    "LA4143 / LA8211 / LA8066 | LATAM Airlines | salida 15:40 | $660",
                    "LA4145 / LA8211 / LA8066 | LATAM Airlines | salida 20:50 | $632 (más barato)",
                    "LA4157 / LA4907 / LA8076 | LATAM Airlines | salida 21:20 | $660",
                ]
            ),
        )

    def test_equal_lowest_prices_mark_only_first_sorted_offer(self):
        markdown = """
| Price | Duration | Stops | Type | Airlines | Segments | Booking |
| --- | --- | --- | --- | --- | --- | --- |
| $573 | 24h | 2 stops | — | Copa Airlines, KLM | Outbound:<br/>BAQ → MAD (2026-11-25 13:41:00-05:00 → 2026-11-26 13:41:00+01:00) | [Book](https://example.test/#trip=CM1234-KL5678) |
| $573 | 26h | 2 stops | — | Copa Airlines, KLM | Outbound:<br/>BAQ → MAD (2026-11-25 09:04:00-05:00 → 2026-11-26 13:41:00+01:00) | [Book](https://example.test/#trip=CM4321-KL8765) |
"""

        formatted = format_flight_results(markdown, "BAQ", "MAD", "2026-11-25")
        lines = formatted.splitlines()
        self.assertEqual(lines[1], "CM4321 / KL8765 | Copa Airlines, KLM | salida 09:04 | $573 (más barato)")
        self.assertEqual(lines[2], "CM1234 / KL5678 | Copa Airlines, KLM | salida 13:41 | $573")


    def test_no_results_are_reported_without_inventing_flights(self):
        self.assertEqual(
            format_flight_results("No flights found", "BAQ", "MAD", "2026-11-20"),
            "No se encontraron vuelos para BAQ a MAD el 20/11/2026.",
        )

    def test_unrecognized_result_format_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "no contiene una tabla de vuelos reconocible"):
            format_flight_results("Search response changed", "BAQ", "MAD", "2026-11-20")

    def test_parser_uses_markdown_headers_instead_of_fixed_column_positions(self):
        markdown = """
| Booking | Segments | Airlines | Stops | Price |
| --- | --- | --- | --- | --- |
| [Book](https://example.test/#trip=AV1234-IB5678) | BAQ → MAD (2026-11-20 08:45:00-05:00 → 2026-11-21 07:00:00+01:00) | Avianca, Iberia | 1 stop | USD 1,234.50 |
"""

        self.assertEqual(
            format_flight_results(markdown, "BAQ", "MAD", "2026-11-20"),
            "20/11/2026 | BAQ → MAD\n"
            "AV1234 / IB5678 | Avianca, Iberia | salida 08:45 | USD 1,234.50 (más barato)",
        )

    def test_parser_accepts_escaped_pipes_in_markdown_cells(self):
        markdown = r"""
| Price | Airlines | Segments | Booking |
| --- | --- | --- | --- |
| $720 | Airline A \| Airline B | BAQ → MAD (2026-11-20 08:45:00-05:00 → 2026-11-21 07:00:00+01:00) | [Book](https://example.test/#trip=AV1234-IB5678) |
"""

        self.assertIn(
            "AV1234 / IB5678 | Airline A | Airline B | salida 08:45 | $720 (más barato)",
            format_flight_results(markdown, "BAQ", "MAD", "2026-11-20"),
        )


if __name__ == "__main__":
    unittest.main()
