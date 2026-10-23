"""
Unit and regression tests verifying the cleaned datasets against expected schema and constraints.
"""

import unittest
import os
import csv
from src.data_processing.team_mapping import VALID_CANONICAL_TEAMS


class TestCleanDatasets(unittest.TestCase):
    MATCHES_PATH = "data/processed/matches_clean.csv"
    DELIVERIES_PATH = "data/processed/deliveries_clean.csv"

    def test_matches_schema_and_counts(self):
        if not os.path.exists(self.MATCHES_PATH):
            self.skipTest(f"{self.MATCHES_PATH} does not exist yet.")

        expected_cols = [
            "match_id",
            "season",
            "date",
            "venue",
            "city",
            "team_1",
            "team_2",
            "toss_winner",
            "toss_decision",
            "winner",
            "match_result_type",
            "method",
        ]

        match_ids = set()
        row_count = 0
        result_types = set()

        with open(self.MATCHES_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, expected_cols)
            for row in reader:
                row_count += 1
                self.assertNotIn(row["match_id"], match_ids, f"Duplicate match_id: {row['match_id']}")
                match_ids.add(row["match_id"])
                result_types.add(row["match_result_type"])

                # Check canonical teams
                self.assertIn(row["team_1"], VALID_CANONICAL_TEAMS)
                self.assertIn(row["team_2"], VALID_CANONICAL_TEAMS)
                if row["winner"]:
                    self.assertIn(row["winner"], VALID_CANONICAL_TEAMS)

                # Winner constraint: no_result must have empty winner
                if row["match_result_type"] == "no_result":
                    self.assertEqual(row["winner"], "")
                else:
                    self.assertTrue(len(row["winner"]) > 0)

        self.assertEqual(row_count, 1243)
        self.assertEqual(result_types, {"normal", "super_over", "no_result"})

    def test_deliveries_schema_and_counts(self):
        if not os.path.exists(self.DELIVERIES_PATH):
            self.skipTest(f"{self.DELIVERIES_PATH} does not exist yet.")

        expected_cols = [
            "match_id",
            "season",
            "start_date",
            "venue",
            "innings",
            "is_super_over",
            "ball",
            "actual_delivery",
            "batting_team",
            "bowling_team",
            "striker",
            "non_striker",
            "bowler",
            "runs_off_bat",
            "extras",
            "wides",
            "noballs",
            "byes",
            "legbyes",
            "penalty",
            "non_boundary",
            "wicket_type",
            "player_dismissed",
        ]

        row_count = 0
        super_over_count = 0

        with open(self.DELIVERIES_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, expected_cols)
            for row in reader:
                row_count += 1
                innings = int(row["innings"])
                is_so = int(row["is_super_over"])

                if innings >= 3:
                    self.assertEqual(is_so, 1)
                    super_over_count += 1
                else:
                    self.assertEqual(is_so, 0)

                # Extras must be non-empty integers
                for ext in ["wides", "noballs", "byes", "legbyes", "penalty"]:
                    self.assertTrue(row[ext].isdigit())

        self.assertEqual(row_count, 295732)
        self.assertEqual(super_over_count, 175)


if __name__ == "__main__":
    unittest.main()
