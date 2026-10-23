"""
Traceable Unit Tests and Anti-Leakage Audit for Live Prediction Pipeline.

Tests:
1. Known Match State Progression (Score, wickets, legal deliveries, batter/bowler figures)
2. Second Innings Chase Dynamics & Target Setting
3. Innings 1 Zero-Lookahead Invariants
4. Future Delivery Perturbation Leakage Audit
5. Future Match Perturbation Historical Player Leakage Audit
6. Matchup Fallback Hierarchy on Zero-Prior Encounters
"""

import unittest
import copy
import os
import pandas as pd
from src.live_prediction.live_state_builder import LiveMatchStateMachine
from src.live_prediction.match_state_features import extract_match_state_features
from src.live_prediction.player_features import (
    HistoricalPlayerTracker,
    extract_current_player_features,
)
from src.live_prediction.matchup_features import (
    HistoricalMatchupTracker,
    extract_matchup_features,
)
from src.live_prediction.walk_forward_pre_match import (
    predict_single_match_walk_forward,
    get_training_boundary_for_match,
    load_walk_forward_prob_map,
)


class TestLivePipeline(unittest.TestCase):

    def setUp(self):
        self.inn1_deliveries = [
            {
                "match_id": "999001",
                "innings": 1,
                "ball": "0.1",
                "batting_team": "Sunrisers Hyderabad",
                "bowling_team": "Royal Challengers Bangalore",
                "striker": "DA Warner",
                "non_striker": "S Dhawan",
                "bowler": "TS Mills",
                "runs_off_bat": 0,
                "extras": 0,
                "wides": 0,
                "noballs": 0,
                "byes": 0,
                "legbyes": 0,
                "penalty": 0,
                "wicket_type": "",
                "player_dismissed": "",
            },
            {
                "match_id": "999001",
                "innings": 1,
                "ball": "0.2",
                "batting_team": "Sunrisers Hyderabad",
                "bowling_team": "Royal Challengers Bangalore",
                "striker": "DA Warner",
                "non_striker": "S Dhawan",
                "bowler": "TS Mills",
                "runs_off_bat": 4,
                "extras": 0,
                "wides": 0,
                "noballs": 0,
                "byes": 0,
                "legbyes": 0,
                "penalty": 0,
                "wicket_type": "",
                "player_dismissed": "",
            },
            {
                "match_id": "999001",
                "innings": 1,
                "ball": "0.3",
                "batting_team": "Sunrisers Hyderabad",
                "bowling_team": "Royal Challengers Bangalore",
                "striker": "DA Warner",
                "non_striker": "S Dhawan",
                "bowler": "TS Mills",
                "runs_off_bat": 0,
                "extras": 1,
                "wides": 1,
                "noballs": 0,
                "byes": 0,
                "legbyes": 0,
                "penalty": 0,
                "wicket_type": "",
                "player_dismissed": "",
            },
            {
                "match_id": "999001",
                "innings": 1,
                "ball": "0.4",
                "batting_team": "Sunrisers Hyderabad",
                "bowling_team": "Royal Challengers Bangalore",
                "striker": "DA Warner",
                "non_striker": "S Dhawan",
                "bowler": "TS Mills",
                "runs_off_bat": 0,
                "extras": 0,
                "wides": 0,
                "noballs": 0,
                "byes": 0,
                "legbyes": 0,
                "penalty": 0,
                "wicket_type": "caught",
                "player_dismissed": "DA Warner",
            },
        ]

        self.inn2_deliveries = [
            {
                "match_id": "999001",
                "innings": 2,
                "ball": "0.1",
                "batting_team": "Royal Challengers Bangalore",
                "bowling_team": "Sunrisers Hyderabad",
                "striker": "CH Gayle",
                "non_striker": "Mandeep Singh",
                "bowler": "B Kumar",
                "runs_off_bat": 6,
                "extras": 0,
                "wides": 0,
                "noballs": 0,
                "byes": 0,
                "legbyes": 0,
                "penalty": 0,
                "wicket_type": "",
                "player_dismissed": "",
            },
            {
                "match_id": "999001",
                "innings": 2,
                "ball": "0.2",
                "batting_team": "Royal Challengers Bangalore",
                "bowling_team": "Sunrisers Hyderabad",
                "striker": "CH Gayle",
                "non_striker": "Mandeep Singh",
                "bowler": "B Kumar",
                "runs_off_bat": 0,
                "extras": 0,
                "wides": 0,
                "noballs": 0,
                "byes": 0,
                "legbyes": 0,
                "penalty": 0,
                "wicket_type": "",
                "player_dismissed": "",
            },
        ]

    def test_known_match_state_progression(self):
        """Test state machine on known sequence with wide and wicket."""
        sm = LiveMatchStateMachine("999001", "Sunrisers Hyderabad", "Royal Challengers Bangalore")

        # Ball 1: 0 runs, legal ball
        s1 = sm.update_with_delivery(self.inn1_deliveries[0])
        self.assertEqual(s1["current_innings_runs"], 0)
        self.assertEqual(s1["current_innings_wickets"], 0)
        self.assertEqual(s1["legal_balls_bowled"], 1)
        self.assertEqual(s1["batter_runs_current_inn"], 0)
        self.assertEqual(s1["batter_balls_current_inn"], 1)
        self.assertEqual(s1["bowler_runs_conceded_current"], 0)
        self.assertEqual(s1["bowler_legal_balls_current"], 1)

        # Ball 2: 4 runs off bat
        s2 = sm.update_with_delivery(self.inn1_deliveries[1])
        self.assertEqual(s2["current_innings_runs"], 4)
        self.assertEqual(s2["current_innings_wickets"], 0)
        self.assertEqual(s2["legal_balls_bowled"], 2)
        self.assertEqual(s2["batter_runs_current_inn"], 4)
        self.assertEqual(s2["batter_boundaries_current"], 1)
        self.assertEqual(s2["bowler_runs_conceded_current"], 4)

        # Ball 3: Wide (+1 extra run, 0 legal balls)
        s3 = sm.update_with_delivery(self.inn1_deliveries[2])
        self.assertEqual(s3["current_innings_runs"], 5)
        self.assertEqual(s3["current_innings_wickets"], 0)
        self.assertEqual(s3["legal_balls_bowled"], 2)  # Unchanged!
        self.assertEqual(s3["batter_balls_current_inn"], 2)  # Unchanged on wide!
        self.assertEqual(s3["bowler_runs_conceded_current"], 5)  # Wide charged to bowler!

        # Ball 4: Wicket (caught)
        s4 = sm.update_with_delivery(self.inn1_deliveries[3])
        self.assertEqual(s4["current_innings_runs"], 5)
        self.assertEqual(s4["current_innings_wickets"], 1)
        self.assertEqual(s4["legal_balls_bowled"], 3)
        self.assertEqual(s4["bowler_wickets_current"], 1)
        self.assertEqual(s4["partnership_runs"], 5)

    def test_innings_2_chase_dynamics(self):
        """Test target establishment and chase calculations in Innings 2."""
        sm = LiveMatchStateMachine("999001", "Sunrisers Hyderabad", "Royal Challengers Bangalore")

        for d in self.inn1_deliveries:
            sm.update_with_delivery(d)

        # Innings 1 finished with 5 runs.
        # Innings 2 starts: target should be 5 + 1 = 6 runs.
        s_inn2_b1 = sm.update_with_delivery(self.inn2_deliveries[0])
        feats = extract_match_state_features(s_inn2_b1, "Sunrisers Hyderabad", 0.55)

        self.assertEqual(feats["target_runs"], 6)
        self.assertEqual(feats["current_innings_runs"], 6)  # Gayle hit a 6
        self.assertEqual(feats["runs_needed"], 0)
        self.assertEqual(feats["required_run_rate"], 0.0)
        self.assertEqual(feats["wickets_remaining"], 10)
        self.assertEqual(feats["legal_balls_bowled"], 1)
        self.assertEqual(feats["legal_balls_remaining"], 119)

    def test_innings_1_zero_lookahead(self):
        """Verify that Innings 1 state features contain ZERO lookahead into Innings 2."""
        sm = LiveMatchStateMachine("999001", "Sunrisers Hyderabad", "Royal Challengers Bangalore")

        for d in self.inn1_deliveries:
            state = sm.update_with_delivery(d)
            feats = extract_match_state_features(state, "Sunrisers Hyderabad", 0.50)
            self.assertEqual(feats["target_runs"], 0)
            self.assertEqual(feats["runs_needed"], 0)
            self.assertEqual(feats["required_run_rate"], 0.0)
            self.assertEqual(feats["rrr_crr_differential"], 0.0)

    def test_future_delivery_perturbation_leakage_audit(self):
        """
        CRITICAL LEAKAGE AUDIT:
        Verify that altering or removing future deliveries (k+1) has ZERO effect
        on the feature vector generated at delivery k.
        """
        # Run original sequence up to delivery index 1 (Ball 2)
        sm_orig = LiveMatchStateMachine("999001", "Sunrisers Hyderabad", "Royal Challengers Bangalore")
        sm_orig.update_with_delivery(self.inn1_deliveries[0])
        state_k_orig = sm_orig.update_with_delivery(self.inn1_deliveries[1])
        feats_k_orig = extract_match_state_features(state_k_orig, "Sunrisers Hyderabad", 0.50)

        # Create perturbed future: modify delivery 3 and 4 to 6s and wickets
        perturbed_inn1 = copy.deepcopy(self.inn1_deliveries)
        perturbed_inn1[2]["runs_off_bat"] = 6
        perturbed_inn1[3]["runs_off_bat"] = 6

        sm_pert = LiveMatchStateMachine("999001", "Sunrisers Hyderabad", "Royal Challengers Bangalore")
        sm_pert.update_with_delivery(perturbed_inn1[0])
        state_k_pert = sm_pert.update_with_delivery(perturbed_inn1[1])
        feats_k_pert = extract_match_state_features(state_k_pert, "Sunrisers Hyderabad", 0.50)

        # Invariant: features at delivery k MUST be bitwise identical!
        self.assertEqual(feats_k_orig, feats_k_pert)

    def test_historical_player_future_match_leakage_audit(self):
        """
        CRITICAL LEAKAGE AUDIT:
        Altering player performance in match M+1 must NOT change historical features
        at match M.
        """
        tracker = HistoricalPlayerTracker()

        # Query prior before any match:
        priors_m1 = tracker.get_historical_batter_stats("DA Warner")

        # Now simulate Match M concluding
        tracker.update_with_match(self.inn1_deliveries)

        # Query prior at match M+1
        priors_m2 = tracker.get_historical_batter_stats("DA Warner")
        self.assertEqual(priors_m2["historical_batter_balls_faced"], 3)
        self.assertNotEqual(priors_m2["historical_batter_strike_rate"], priors_m1["historical_batter_strike_rate"])

        # Now verify that prior for M1 was NOT retrospectively corrupted
        self.assertEqual(priors_m1["historical_batter_balls_faced"], 0)

    def test_matchup_fallback_hierarchy(self):
        """Verify that 0-prior matchups fall back cleanly to the batter's historical strike rate."""
        tracker = HistoricalMatchupTracker()
        striker = "V Kohli"
        bowler = "JJ Bumrah"
        historical_sr = 135.50

        feats = tracker.get_matchup_features(striker, bowler, historical_sr)
        self.assertEqual(feats["matchup_balls_prior"], 0)
        self.assertEqual(feats["matchup_runs_prior"], 0)
        self.assertEqual(feats["matchup_wickets_prior"], 0)
        # Fallback to batter's historical SR:
        self.assertEqual(feats["matchup_strike_rate_shrunk"], historical_sr)

    def test_walk_forward_has_no_self_inclusion(self):
        """
        Verify that for every match in the walk-forward dataset:
        1. The current match is NEVER present in its training set.
        2. The historical training count exactly equals the match's chronological position (excluding self).
        """
        audit_csv = "data/processed/walk_forward_pre_match_probs.csv"
        self.assertTrue(os.path.exists(audit_csv), f"Missing {audit_csv}")
        df_audit = pd.read_csv(audit_csv)

        # 1. Check all rows in audit table
        for idx, row in df_audit.iterrows():
            curr_mid = str(row["match_id"])
            latest_train = str(row["latest_training_match"])
            # Current match can never be the latest training match or in its training count
            if row["cold_start_flag"] == 0:
                self.assertNotEqual(curr_mid, latest_train)
                self.assertEqual(row["historical_training_match_count"], idx)

    def test_walk_forward_future_match_perturbation(self):
        """
        CRITICAL LEAKAGE TEST:
        Altering data belonging strictly AFTER match M must have ZERO effect
        on the pre-match probability generated for match M.
        """
        pre_csv = "data/processed/pre_match_dataset.csv"
        self.assertTrue(os.path.exists(pre_csv), f"Missing {pre_csv}")
        df = pd.read_csv(pre_csv)
        df["date_parsed"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["date_parsed", "match_id"]).reset_index(drop=True)

        target_idx = 45  # Past cold start (min_training_matches=40)
        prob_orig = predict_single_match_walk_forward(df, target_idx)

        # Perturb future matches (e.g. index 50, 60)
        df_pert = df.copy(deep=True)
        df_pert.loc[50:, "target_team_a_win"] = 1 - df_pert.loc[50:, "target_team_a_win"]
        df_pert.loc[50:, "team_a_form_last_5"] = 99.0
        df_pert.loc[50:, "team_b_form_last_5"] = -99.0

        prob_pert = predict_single_match_walk_forward(df_pert, target_idx)
        self.assertEqual(prob_orig, prob_pert, "Future match perturbation leaked into current match probability!")

    def test_walk_forward_current_match_outcome_perturbation(self):
        """
        CRITICAL LEAKAGE TEST:
        Altering match M's own outcome (target) must have ZERO effect on its pre-match probability.
        """
        pre_csv = "data/processed/pre_match_dataset.csv"
        self.assertTrue(os.path.exists(pre_csv), f"Missing {pre_csv}")
        df = pd.read_csv(pre_csv)
        df["date_parsed"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["date_parsed", "match_id"]).reset_index(drop=True)

        target_idx = 55
        prob_orig = predict_single_match_walk_forward(df, target_idx)

        # Invert current match's target
        df_pert = df.copy(deep=True)
        df_pert.loc[target_idx, "target_team_a_win"] = 1 - df_pert.loc[target_idx, "target_team_a_win"]

        prob_pert = predict_single_match_walk_forward(df_pert, target_idx)
        self.assertEqual(prob_orig, prob_pert, "Current match outcome influenced its own pre-match probability!")

    def test_walk_forward_training_data_boundary(self):
        """
        TRAINING DATA BOUNDARY AUDIT:
        Verify strictly expanding chronological boundaries across checkpoints:
        (date, match_id) of all training matches < (date, match_id) of current match.
        """
        pre_csv = "data/processed/pre_match_dataset.csv"
        df = pd.read_csv(pre_csv)
        df["date_parsed"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["date_parsed", "match_id"]).reset_index(drop=True)

        checkpoints = [42, 100, 350, 750, 1100]
        for c_idx in checkpoints:
            boundary = get_training_boundary_for_match(df, c_idx)
            self.assertEqual(boundary["historical_training_match_count"], c_idx)

            curr_date = pd.to_datetime(boundary["date"])
            curr_mid = int(boundary["match_id"])
            max_train_date = pd.to_datetime(boundary["training_max_date"])
            max_train_mid = int(boundary["training_max_match_id"])

            # Verify strictly chronological ordering
            self.assertTrue(
                (max_train_date < curr_date) or (max_train_date == curr_date and max_train_mid < curr_mid),
                f"Boundary violation at index {c_idx}: current ({curr_date}, {curr_mid}) vs max train ({max_train_date}, {max_train_mid})"
            )
            self.assertNotIn(str(curr_mid), boundary["training_match_ids"])


if __name__ == "__main__":
    unittest.main()
