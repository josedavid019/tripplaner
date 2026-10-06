import unittest

from app.FlightsAgent.main import airline_name, search_flights


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


if __name__ == "__main__":
    unittest.main()
