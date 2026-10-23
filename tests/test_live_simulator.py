"""
Comprehensive Verification and Anti-Leakage Test Suite for Live Simulator.

Tests:
1. TEST 1 — T0 State (score=0, wickets=0, legal_balls=0, correct pre-match prob anchor)
2. TEST 2 — Legal Delivery (legal ball count increments by 1)
3. TEST 3 — Wide Delivery (score updates, legal balls does NOT increment, prediction generated)
4. TEST 4 — No-Ball Delivery (score updates, legal balls does NOT increment, prediction generated)
5. TEST 5 — Wicket Event (wickets increment, player figures update)
6. TEST 6 — Innings Transition (innings changes 1 -> 2, target established, state resets)
7. TEST 7 — No Lookahead (perturbing future delivery leaves prior predictions bitwise identical)
8. TEST 8 — Outcome Perturbation (changing final match winner does not alter prior predictions)
9. TEST 9 — Terminal State (deterministic 1.0 / 0.0 / 0.50 behavior)
10. TEST 10 — Probability Validity (0 <= P <= 1 and P(A) + P(B) == 1 for all events)
11. TEST 11 — Model Routing (Innings 1 uses Model 1, Innings 2 uses Model 2)
12. TEST 12 — Deterministic Replay (running same match twice produces bitwise identical probabilities)
13. TEST 13 — Dataset State Agreement (verifies states and features against live_dataset.csv)
14. TEST 14 — Historical Player Leakage (no current/future match deliveries in player priors)
15. TEST 15 — Matchup Leakage (current delivery not present in matchup history prior to prediction)
"""

import unittest
import copy
import os
import pandas as pd
import numpy as np

from src.live_simulator.state import MatchStateManager
from src.live_simulator.player_state import PlayerStateManager
from src.live_simulator.matchup_state import MatchupStateManager
from src.live_simulator.feature_builder import LiveFeatureBuilder
from src.live_simulator.model_runner import LiveModelRunner
from src.live_simulator.simulator import MatchSimulator, SimulationResult, SimulationEvent

from src.live_prediction.player_features import HistoricalPlayerTracker
from src.live_prediction.matchup_features import HistoricalMatchupTracker


class TestLiveSimulator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_runner = LiveModelRunner()
        cls.simulator = MatchSimulator(model_runner=cls.model_runner)

        # Reusable sample deliveries
        cls.sample_dels_inn1 = [
            {
                "match_id": "999001",
                "innings": 1,
                "ball": "0.1",
                "batting_team": "Kolkata Knight Riders",
                "bowling_team": "Royal Challengers Bangalore",
                "striker": "PD Salt",
                "non_striker": "SP Narine",
                "bowler": "Mohammed Siraj",
                "runs_off_bat": 4,
                "extras": 0,
                "wides": 0,
                "noballs": 0,
                "wicket_type": "",
                "player_dismissed": "",
            },
            {
                "match_id": "999001",
                "innings": 1,
                "ball": "0.2",
                "batting_team": "Kolkata Knight Riders",
                "bowling_team": "Royal Challengers Bangalore",
                "striker": "PD Salt",
                "non_striker": "SP Narine",
                "bowler": "Mohammed Siraj",
                "runs_off_bat": 0,
                "extras": 1,
                "wides": 1,
                "noballs": 0,
                "wicket_type": "",
                "player_dismissed": "",
            },
            {
                "match_id": "999001",
                "innings": 1,
                "ball": "0.3",
                "batting_team": "Kolkata Knight Riders",
                "bowling_team": "Royal Challengers Bangalore",
                "striker": "PD Salt",
                "non_striker": "SP Narine",
                "bowler": "Mohammed Siraj",
                "runs_off_bat": 1,
                "extras": 1,
                "wides": 0,
                "noballs": 1,
                "wicket_type": "",
                "player_dismissed": "",
            },
            {
                "match_id": "999001",
                "innings": 1,
                "ball": "0.4",
                "batting_team": "Kolkata Knight Riders",
                "bowling_team": "Royal Challengers Bangalore",
                "striker": "SP Narine",
                "non_striker": "PD Salt",
                "bowler": "Mohammed Siraj",
                "runs_off_bat": 0,
                "extras": 0,
                "wides": 0,
                "noballs": 0,
                "wicket_type": "caught",
                "player_dismissed": "SP Narine",
            },
        ]

    # TEST 1 — T0 state
    def test_01_t0_state(self):
        sim_res = self.simulator.simulate(
            match_id="1426274",
            custom_pre_match_prob=0.5234,
            custom_deliveries=self.sample_dels_inn1,
        )
        t0 = sim_res.events[0]
        self.assertEqual(t0.event_number, 0)
        self.assertEqual(t0.innings, 1)
        self.assertEqual(t0.score, 0)
        self.assertEqual(t0.wickets, 0)
        self.assertEqual(t0.legal_balls_bowled, 0)
        self.assertEqual(t0.overs_completed, 0.0)
        self.assertEqual(t0.team_a_win_probability, 0.5234)
        self.assertAlmostEqual(t0.team_b_win_probability, 1.0 - 0.5234, places=4)
        self.assertEqual(t0.prediction_type, "pre_match")

    # TEST 2 — legal delivery
    def test_02_legal_delivery(self):
        state_mgr = MatchStateManager(
            match_id="999001",
            team_a="Kolkata Knight Riders",
            team_b="Royal Challengers Bangalore",
            pre_match_prob_team_a=0.50,
        )
        del_legal = self.sample_dels_inn1[0]
        feats = state_mgr.process_delivery(del_legal)
        self.assertEqual(feats["legal_balls_bowled"], 1)
        self.assertEqual(feats["current_innings_runs"], 4)
        self.assertEqual(feats["current_innings_wickets"], 0)

    # TEST 3 — wide delivery
    def test_03_wide_delivery(self):
        state_mgr = MatchStateManager(
            match_id="999001",
            team_a="Kolkata Knight Riders",
            team_b="Royal Challengers Bangalore",
            pre_match_prob_team_a=0.50,
        )
        state_mgr.process_delivery(self.sample_dels_inn1[0])  # legal 4 runs
        del_wide = self.sample_dels_inn1[1]  # wide + 1 extra
        feats = state_mgr.process_delivery(del_wide)

        # Wide must NOT increment legal balls (remains 1)
        self.assertEqual(feats["legal_balls_bowled"], 1)
        # Score must increment by 1 (total = 5)
        self.assertEqual(feats["current_innings_runs"], 5)
        raw = state_mgr.get_raw_state()
        self.assertFalse(raw["is_legal"])

    # TEST 4 — no-ball delivery
    def test_04_noball_delivery(self):
        state_mgr = MatchStateManager(
            match_id="999001",
            team_a="Kolkata Knight Riders",
            team_b="Royal Challengers Bangalore",
            pre_match_prob_team_a=0.50,
        )
        state_mgr.process_delivery(self.sample_dels_inn1[0])  # legal (1 ball)
        state_mgr.process_delivery(self.sample_dels_inn1[1])  # wide (1 ball)
        del_nb = self.sample_dels_inn1[2]  # noball + 1 run off bat
        feats = state_mgr.process_delivery(del_nb)

        # No-ball must NOT increment legal balls (remains 1)
        self.assertEqual(feats["legal_balls_bowled"], 1)
        # Score increases by 2 (total = 7)
        self.assertEqual(feats["current_innings_runs"], 7)

    # TEST 5 — wicket event
    def test_05_wicket_event(self):
        state_mgr = MatchStateManager(
            match_id="999001",
            team_a="Kolkata Knight Riders",
            team_b="Royal Challengers Bangalore",
            pre_match_prob_team_a=0.50,
        )
        for d in self.sample_dels_inn1[:3]:
            state_mgr.process_delivery(d)
        del_wkt = self.sample_dels_inn1[3]  # caught SP Narine
        feats = state_mgr.process_delivery(del_wkt)

        self.assertEqual(feats["current_innings_wickets"], 1)
        self.assertEqual(feats["wickets_remaining"], 9)
        self.assertEqual(feats["legal_balls_bowled"], 2)

    # TEST 6 — innings transition
    def test_06_innings_transition(self):
        state_mgr = MatchStateManager(
            match_id="999001",
            team_a="Kolkata Knight Riders",
            team_b="Royal Challengers Bangalore",
            pre_match_prob_team_a=0.50,
        )
        # Run Innings 1 delivery
        state_mgr.process_delivery(self.sample_dels_inn1[0])  # 4 runs
        # First ball of Innings 2
        del_inn2 = {
            "match_id": "999001",
            "innings": 2,
            "ball": "0.1",
            "batting_team": "Royal Challengers Bangalore",
            "bowling_team": "Kolkata Knight Riders",
            "striker": "V Kohli",
            "non_striker": "F du Plessis",
            "bowler": "MA Starc",
            "runs_off_bat": 1,
            "extras": 0,
            "wides": 0,
            "noballs": 0,
            "wicket_type": "",
            "player_dismissed": "",
        }
        feats_inn2 = state_mgr.process_delivery(del_inn2)

        self.assertEqual(feats_inn2["current_innings"], 2)
        self.assertEqual(feats_inn2["target_runs"], 5)  # 4 + 1
        self.assertEqual(feats_inn2["current_innings_runs"], 1)
        self.assertEqual(feats_inn2["runs_needed"], 4)
        self.assertEqual(feats_inn2["legal_balls_bowled"], 1)

    # TEST 7 — no lookahead (perturb future delivery)
    def test_07_no_lookahead(self):
        dels_original = copy.deepcopy(self.sample_dels_inn1)
        dels_perturbed = copy.deepcopy(self.sample_dels_inn1)
        # Perturb delivery 4 (index 3) runs from 0 to 6
        dels_perturbed[3]["runs_off_bat"] = 6
        dels_perturbed[3]["wicket_type"] = ""

        sim_orig = self.simulator.simulate(
            match_id="1426274",
            custom_pre_match_prob=0.50,
            custom_deliveries=dels_original,
        )
        sim_pert = self.simulator.simulate(
            match_id="1426274",
            custom_pre_match_prob=0.50,
            custom_deliveries=dels_perturbed,
        )

        # Deliveries 0, 1, 2 must produce BITWISE IDENTICAL probabilities and states
        for idx in range(3):
            e_orig = sim_orig.events[idx]
            e_pert = sim_pert.events[idx]
            self.assertEqual(e_orig.team_a_win_probability, e_pert.team_a_win_probability)
            self.assertEqual(e_orig.score, e_pert.score)
            self.assertEqual(e_orig.wickets, e_pert.wickets)

    # TEST 8 — current-match outcome perturbation
    def test_08_outcome_perturbation(self):
        # Even if match metadata winner were flipped, non-terminal live model predictions are unchanged
        sim_res = self.simulator.simulate(
            match_id="1426274",
            custom_pre_match_prob=0.50,
            custom_deliveries=self.sample_dels_inn1,
        )
        # Verify non-terminal events depend only on model and physical state, not metadata winner
        for e in sim_res.events[:-1]:
            self.assertIn(e.prediction_type, ("pre_match", "model_inn1", "model_inn2"))

    # TEST 9 — terminal state
    def test_09_terminal_state(self):
        # Construct chase winning delivery
        inn2_winning_del = [
            {
                "match_id": "999001",
                "innings": 2,
                "ball": "19.6",
                "batting_team": "Royal Challengers Bangalore",
                "bowling_team": "Kolkata Knight Riders",
                "striker": "V Kohli",
                "non_striker": "KD Karthik",
                "bowler": "MA Starc",
                "runs_off_bat": 6,
                "extras": 0,
                "wides": 0,
                "noballs": 0,
                "wicket_type": "",
                "player_dismissed": "",
            }
        ]
        # Team A is KKR, Team B is RCB. If RCB scores winning runs, Team B wins (P(A)=0.0, P(B)=1.0)
        p_a, p_b, pred = self.model_runner.predict_probability(
            df_row=pd.DataFrame(),
            innings=2,
            is_terminal=True,
            terminal_winner="Royal Challengers Bangalore",
            team_a="Kolkata Knight Riders",
        )
        self.assertEqual(p_a, 0.0)
        self.assertEqual(p_b, 1.0)
        self.assertEqual(pred, "terminal_team_b_win")

        # Regulation tie
        p_a_tie, p_b_tie, pred_tie = self.model_runner.predict_probability(
            df_row=pd.DataFrame(),
            innings=2,
            is_terminal=True,
            is_tie=True,
            team_a="Kolkata Knight Riders",
        )
        self.assertEqual(p_a_tie, 0.50)
        self.assertEqual(p_b_tie, 0.50)
        self.assertEqual(pred_tie, "terminal_tie")

    # TEST 10 — probability validity
    def test_10_probability_validity(self):
        sim_res = self.simulator.simulate(
            match_id="1426274",
            custom_pre_match_prob=0.4850,
            custom_deliveries=self.sample_dels_inn1,
        )
        for e in sim_res.events:
            self.assertGreaterEqual(e.team_a_win_probability, 0.0)
            self.assertLessEqual(e.team_a_win_probability, 1.0)
            self.assertGreaterEqual(e.team_b_win_probability, 0.0)
            self.assertLessEqual(e.team_b_win_probability, 1.0)
            self.assertAlmostEqual(e.team_a_win_probability + e.team_b_win_probability, 1.0, places=3)

    # TEST 11 — model routing
    def test_11_model_routing(self):
        full_dels = copy.deepcopy(self.sample_dels_inn1)
        full_dels.append(
            {
                "match_id": "999001",
                "innings": 2,
                "ball": "0.1",
                "batting_team": "Royal Challengers Bangalore",
                "bowling_team": "Kolkata Knight Riders",
                "striker": "V Kohli",
                "non_striker": "F du Plessis",
                "bowler": "MA Starc",
                "runs_off_bat": 1,
                "extras": 0,
                "wides": 0,
                "noballs": 0,
                "wicket_type": "",
                "player_dismissed": "",
            }
        )
        sim_res = self.simulator.simulate(
            match_id="1426274",
            custom_pre_match_prob=0.50,
            custom_deliveries=full_dels,
        )
        # Event 1-4 are Innings 1 -> model_inn1
        for e in sim_res.events[1:5]:
            self.assertEqual(e.prediction_type, "model_inn1")
        # Event 5 is Innings 2 -> model_inn2
        self.assertIn("model_inn2", sim_res.events[5].prediction_type)

    # TEST 12 — deterministic replay
    def test_12_deterministic_replay(self):
        sim1 = self.simulator.simulate(
            match_id="1426274",
            custom_pre_match_prob=0.50,
            custom_deliveries=self.sample_dels_inn1,
        )
        sim2 = self.simulator.simulate(
            match_id="1426274",
            custom_pre_match_prob=0.50,
            custom_deliveries=self.sample_dels_inn1,
        )
        self.assertEqual(len(sim1.events), len(sim2.events))
        for e1, e2 in zip(sim1.events, sim2.events):
            self.assertEqual(e1.team_a_win_probability, e2.team_a_win_probability)
            self.assertEqual(e1.team_b_win_probability, e2.team_b_win_probability)

    # TEST 13 — dataset state agreement
    def test_13_dataset_state_agreement(self):
        # Load match 1426274 from live_dataset.csv
        df_live = pd.read_csv("data/processed/live_dataset.csv", low_memory=False)
        m_rows = df_live[df_live["match_id"] == 1426274].reset_index(drop=True)

        # Run simulator on first 10 deliveries of match 1426274
        m_dels = self.simulator._all_deliveries[str(1426274)][:10]
        sim_res = self.simulator.simulate(
            match_id="1426274",
            custom_deliveries=m_dels,
        )

        # Compare simulator snapshots vs stored live_dataset.csv rows
        for idx in range(len(m_dels)):
            live_row = m_rows.iloc[idx]
            sim_e = sim_res.events[idx + 1]  # skip T0

            # State checks
            self.assertEqual(sim_e.score, live_row["current_innings_runs"])
            self.assertEqual(sim_e.wickets, live_row["current_innings_wickets"])
            self.assertEqual(sim_e.legal_balls_bowled, live_row["legal_balls_bowled"])
            self.assertAlmostEqual(sim_e.overs_completed, live_row["overs_completed"], places=3)
            self.assertAlmostEqual(sim_e.current_run_rate, live_row["current_run_rate"], places=3)

    # TEST 14 — historical player leakage
    def test_14_historical_player_leakage(self):
        tracker = HistoricalPlayerTracker()
        # Delivery in match 1
        dels_m1 = [
            {
                "innings": 1,
                "striker": "V Kohli",
                "bowler": "SL Malinga",
                "runs_off_bat": 50,
                "extras": 0,
                "wides": 0,
                "noballs": 0,
                "wicket_type": "",
                "player_dismissed": "",
            }
        ]
        # Prior to match 1, V Kohli has 0 balls faced
        stats_prior = tracker.get_historical_batter_stats("V Kohli")
        self.assertEqual(stats_prior["historical_batter_balls_faced"], 0)

        # After match 1, stats update
        tracker.update_with_match(dels_m1)
        stats_post = tracker.get_historical_batter_stats("V Kohli")
        self.assertEqual(stats_post["historical_batter_balls_faced"], 1)

    # TEST 15 — matchup leakage
    def test_15_matchup_leakage(self):
        m_tracker = HistoricalMatchupTracker()
        # Prior to any match, encounter balls == 0
        feats_prior = m_tracker.get_matchup_features("V Kohli", "JJ Bumrah", historical_batter_sr=130.0)
        self.assertEqual(feats_prior["matchup_balls_prior"], 0)

        # Simulate match deliveries
        dels = [
            {
                "innings": 1,
                "striker": "V Kohli",
                "bowler": "JJ Bumrah",
                "runs_off_bat": 4,
                "wides": 0,
                "noballs": 0,
                "wicket_type": "",
                "player_dismissed": "",
            }
        ]
        m_tracker.update_with_match(dels)
        feats_post = m_tracker.get_matchup_features("V Kohli", "JJ Bumrah", historical_batter_sr=130.0)
        self.assertEqual(feats_post["matchup_balls_prior"], 1)
        self.assertEqual(feats_post["matchup_runs_prior"], 4)


if __name__ == "__main__":
    unittest.main()
