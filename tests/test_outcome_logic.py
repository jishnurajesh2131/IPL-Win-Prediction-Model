"""
Unit tests for match outcome resolution and Super Over / No Result handling.
"""

import unittest
from src.data_processing.clean_matches import resolve_match_outcome, clean_single_match


class TestOutcomeLogic(unittest.TestCase):
    def test_normal_match(self):
        raw = {
            "winner": "Delhi Daredevils",
            "outcome": "",
            "eliminator": ""
        }
        res = resolve_match_outcome(raw)
        self.assertEqual(res["match_result_type"], "normal")
        self.assertEqual(res["winner"], "Delhi Capitals")

    def test_super_over_tie(self):
        raw = {
            "winner": "",
            "outcome": "tie",
            "eliminator": "Kings XI Punjab"
        }
        res = resolve_match_outcome(raw)
        self.assertEqual(res["match_result_type"], "super_over")
        self.assertEqual(res["winner"], "Punjab Kings")

    def test_no_result_abandoned(self):
        raw = {
            "winner": "",
            "outcome": "no result",
            "eliminator": ""
        }
        res = resolve_match_outcome(raw)
        self.assertEqual(res["match_result_type"], "no_result")
        self.assertEqual(res["winner"], "")

    def test_clean_single_match_end_to_end(self):
        raw_match = {
            "match_id": "335982",
            "season": "2007/08",
            "date": "2008/04/18",
            "venue": "M Chinnaswamy Stadium",
            "city": "Bangalore",
            "team_1": "Royal Challengers Bangalore",
            "team_2": "Kolkata Knight Riders",
            "toss_winner": "Royal Challengers Bangalore",
            "toss_decision": "field",
            "winner": "Kolkata Knight Riders",
            "outcome": "",
            "eliminator": "",
            "method": "",
        }
        cleaned = clean_single_match(raw_match)
        self.assertEqual(cleaned["match_id"], "335982")
        self.assertEqual(cleaned["team_1"], "Royal Challengers Bengaluru")
        self.assertEqual(cleaned["team_2"], "Kolkata Knight Riders")
        self.assertEqual(cleaned["toss_winner"], "Royal Challengers Bengaluru")
        self.assertEqual(cleaned["winner"], "Kolkata Knight Riders")
        self.assertEqual(cleaned["match_result_type"], "normal")
        self.assertEqual(cleaned["venue"], "M Chinnaswamy Stadium")


if __name__ == "__main__":
    unittest.main()
