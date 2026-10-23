"""
Unit tests for pre-match feature engineering, state tracking, and leakage prevention.
"""

import unittest
from src.features.historical_stats import HistoricalStateTracker


class TestPreMatchFeatures(unittest.TestCase):
    def setUp(self):
        self.tracker = HistoricalStateTracker()

    def test_cold_start_defaults(self):
        """Verify cold start defaults on a clean state."""
        features = self.tracker.compute_pre_match_features(
            team_a="Royal Challengers Bengaluru",
            team_b="Kolkata Knight Riders",
            venue="M Chinnaswamy Stadium",
            city="Bengaluru",
            toss_winner="Royal Challengers Bengaluru",
            toss_decision="field",
        )

        # 0 matches played
        self.assertEqual(features["team_a_matches_played"], 0)
        self.assertEqual(features["team_b_matches_played"], 0)
        self.assertEqual(features["h2h_matches_played"], 0)

        # Bayesian smoothed neutral priors
        self.assertEqual(features["team_a_form_last_5"], 0.50)
        self.assertEqual(features["team_b_form_last_5"], 0.50)
        self.assertEqual(features["team_a_h2h_win_rate"], 0.50)
        self.assertEqual(features["team_a_venue_win_rate"], 0.50)
        self.assertEqual(features["team_b_venue_win_rate"], 0.50)
        self.assertEqual(features["form_diff_last_5"], 0.0)

    def test_toss_and_batting_first_logic(self):
        """Verify exact role determination based on toss winner and decision."""
        # Case 1: Team A wins toss and bats
        f1 = self.tracker.compute_pre_match_features(
            team_a="Mumbai Indians", team_b="Chennai Super Kings",
            venue="Wankhede Stadium", city="Mumbai",
            toss_winner="Mumbai Indians", toss_decision="bat",
        )
        self.assertEqual(f1["toss_winner_is_team_a"], 1)
        self.assertEqual(f1["toss_decision_is_bat"], 1)
        self.assertEqual(f1["team_a_bats_first"], 1)

        # Case 2: Team A wins toss and fields
        f2 = self.tracker.compute_pre_match_features(
            team_a="Mumbai Indians", team_b="Chennai Super Kings",
            venue="Wankhede Stadium", city="Mumbai",
            toss_winner="Mumbai Indians", toss_decision="field",
        )
        self.assertEqual(f2["toss_winner_is_team_a"], 1)
        self.assertEqual(f2["toss_decision_is_bat"], 0)
        self.assertEqual(f2["team_a_bats_first"], 0)

        # Case 3: Team B wins toss and bats
        f3 = self.tracker.compute_pre_match_features(
            team_a="Mumbai Indians", team_b="Chennai Super Kings",
            venue="Wankhede Stadium", city="Mumbai",
            toss_winner="Chennai Super Kings", toss_decision="bat",
        )
        self.assertEqual(f3["toss_winner_is_team_a"], 0)
        self.assertEqual(f3["toss_decision_is_bat"], 1)
        self.assertEqual(f3["team_a_bats_first"], 0)

        # Case 4: Team B wins toss and fields (Team A is put in to bat)
        f4 = self.tracker.compute_pre_match_features(
            team_a="Mumbai Indians", team_b="Chennai Super Kings",
            venue="Wankhede Stadium", city="Mumbai",
            toss_winner="Chennai Super Kings", toss_decision="field",
        )
        self.assertEqual(f4["toss_winner_is_team_a"], 0)
        self.assertEqual(f4["toss_decision_is_bat"], 0)
        self.assertEqual(f4["team_a_bats_first"], 1)

    def test_chronological_state_transition(self):
        """Verify state updates occur strictly after feature calculation."""
        team_a = "Delhi Capitals"
        team_b = "Rajasthan Royals"
        venue = "Arun Jaitley Stadium"

        # Match 1: Before any match is recorded
        f1 = self.tracker.compute_pre_match_features(
            team_a=team_a, team_b=team_b, venue=venue, city="Delhi",
            toss_winner=team_a, toss_decision="field"
        )
        self.assertEqual(f1["team_a_matches_played"], 0)
        self.assertEqual(f1["h2h_matches_played"], 0)

        # Update: Team A wins Match 1
        self.tracker.update_with_match_result(
            team_a=team_a, team_b=team_b, winner=team_a, venue=venue,
            innings_stats=[
                {"batting_team": team_b, "bowling_team": team_a, "runs": 150, "legal_balls": 120, "wickets": 7},
                {"batting_team": team_a, "bowling_team": team_b, "runs": 152, "legal_balls": 110, "wickets": 4},
            ],
            chasing_team_won=True
        )

        # Match 2: Next match between same teams
        f2 = self.tracker.compute_pre_match_features(
            team_a=team_a, team_b=team_b, venue=venue, city="Delhi",
            toss_winner=team_b, toss_decision="bat"
        )
        self.assertEqual(f2["team_a_matches_played"], 1)
        self.assertEqual(f2["team_b_matches_played"], 1)
        self.assertEqual(f2["h2h_matches_played"], 1)

        # Team A won 1 of 1: (1 + 1)/(1 + 2) = 2/3 ≈ 0.6667
        self.assertAlmostEqual(f2["team_a_h2h_win_rate"], 0.6667, places=3)
        self.assertAlmostEqual(f2["team_a_form_last_5"], 0.6667, places=3)

        # Team B lost 1 of 1: (0 + 1)/(1 + 2) = 1/3 ≈ 0.3333
        self.assertAlmostEqual(f2["team_b_form_last_5"], 0.3333, places=3)
        self.assertGreater(f2["form_diff_last_5"], 0.0)


if __name__ == "__main__":
    unittest.main()
