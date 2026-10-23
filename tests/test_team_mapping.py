"""
Unit tests for canonical team name mapping and franchise boundaries.
"""

import unittest
from src.data_processing.team_mapping import (
    map_team_name,
    CANONICAL_TEAM_MAP,
    VALID_CANONICAL_TEAMS,
)


class TestTeamMapping(unittest.TestCase):
    def test_rebranded_teams(self):
        self.assertEqual(map_team_name("Delhi Daredevils"), "Delhi Capitals")
        self.assertEqual(map_team_name("Delhi Capitals"), "Delhi Capitals")
        self.assertEqual(map_team_name("Kings XI Punjab"), "Punjab Kings")
        self.assertEqual(map_team_name("Punjab Kings"), "Punjab Kings")
        self.assertEqual(
            map_team_name("Royal Challengers Bangalore"),
            "Royal Challengers Bengaluru"
        )
        self.assertEqual(
            map_team_name("Royal Challengers Bengaluru"),
            "Royal Challengers Bengaluru"
        )
        self.assertEqual(
            map_team_name("Rising Pune Supergiants"),
            "Rising Pune Supergiant"
        )
        self.assertEqual(
            map_team_name("Rising Pune Supergiant"),
            "Rising Pune Supergiant"
        )

    def test_franchise_independence_deccan_chargers_not_srh(self):
        """Verify Deccan Chargers is strictly NOT mapped to Sunrisers Hyderabad."""
        self.assertEqual(map_team_name("Deccan Chargers"), "Deccan Chargers")
        self.assertNotEqual(map_team_name("Deccan Chargers"), "Sunrisers Hyderabad")
        self.assertEqual(map_team_name("Sunrisers Hyderabad"), "Sunrisers Hyderabad")

    def test_defunct_franchises_preserved(self):
        self.assertEqual(map_team_name("Pune Warriors"), "Pune Warriors")
        self.assertEqual(map_team_name("Gujarat Lions"), "Gujarat Lions")
        self.assertEqual(map_team_name("Kochi Tuskers Kerala"), "Kochi Tuskers Kerala")

    def test_active_teams_unchanged(self):
        active_teams = [
            "Chennai Super Kings",
            "Mumbai Indians",
            "Kolkata Knight Riders",
            "Rajasthan Royals",
            "Sunrisers Hyderabad",
            "Gujarat Titans",
            "Lucknow Super Giants",
        ]
        for t in active_teams:
            self.assertEqual(map_team_name(t), t)

    def test_empty_and_whitespace(self):
        self.assertEqual(map_team_name(""), "")
        self.assertIsNone(map_team_name(None))
        self.assertEqual(map_team_name("  Mumbai Indians  "), "Mumbai Indians")

    def test_canonical_team_count(self):
        self.assertEqual(len(VALID_CANONICAL_TEAMS), 15)


if __name__ == "__main__":
    unittest.main()
