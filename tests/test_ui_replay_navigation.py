"""
Unit tests for Reversible Replay Navigation, Go To Ball, Dual-Innings Scorecards,
and Team Colors mapping.
"""

import unittest
from fastapi.testclient import TestClient
from src.backend.main import app
from src.backend.state.session_manager import session_manager


class TestUiReplayNavigation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.match_id = "1426274"  # KKR vs RCB 2024 thriller

    def setUp(self):
        session_manager.clear()
        res = self.client.post("/api/v1/simulations", json={"match_id": self.match_id})
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.session_id = data["session_id"]
        self.total_events = data["total_events"]

    def test_cannot_step_previous_at_t0(self):
        """At T0, calling previous must return 409 conflict / cannot step backward."""
        res = self.client.post(f"/api/v1/simulations/{self.session_id}/previous")
        self.assertEqual(res.status_code, 409)

    def test_next_and_previous_reversibility_t50(self):
        """
        Navigate to event 50.
        Record state 50.
        Step forward to 51.
        Step backward to 50.
        Verify state returns exactly to state 50.
        """
        # Seek to 50
        res = self.client.post(f"/api/v1/simulations/{self.session_id}/seek", json={"event_index": 50})
        self.assertEqual(res.status_code, 200)
        state_50 = res.json()
        self.assertEqual(state_50["event_index"], 50)

        # Step forward to 51
        res_next = self.client.post(f"/api/v1/simulations/{self.session_id}/next")
        self.assertEqual(res_next.status_code, 200)
        state_51 = res_next.json()
        self.assertEqual(state_51["event_index"], 51)

        # Step backward to 50
        res_prev = self.client.post(f"/api/v1/simulations/{self.session_id}/previous")
        self.assertEqual(res_prev.status_code, 200)
        state_50_rev = res_prev.json()
        self.assertEqual(state_50_rev["event_index"], 50)

        # Check exact equality
        self.assertEqual(state_50["prediction"]["team_1_probability"], state_50_rev["prediction"]["team_1_probability"])
        self.assertEqual(state_50["prediction"]["team_2_probability"], state_50_rev["prediction"]["team_2_probability"])
        self.assertEqual(state_50["state"]["score"], state_50_rev["state"]["score"])
        self.assertEqual(state_50["state"]["wickets"], state_50_rev["state"]["wickets"])
        self.assertEqual(state_50["state"]["striker"], state_50_rev["state"]["striker"])
        self.assertEqual(state_50["state"]["bowler"], state_50_rev["state"]["bowler"])

    def test_go_to_ball_seek_bounds(self):
        """Out-of-bounds seeks must be rejected."""
        res_neg = self.client.post(f"/api/v1/simulations/{self.session_id}/seek", json={"event_index": -1})
        self.assertEqual(res_neg.status_code, 422)  # Pydantic validation ge=0

        res_over = self.client.post(f"/api/v1/simulations/{self.session_id}/seek", json={"event_index": 9999})
        self.assertEqual(res_over.status_code, 400)

    def test_scorecard_causal_progression_no_leakage(self):
        """
        Verify Innings 1 and Innings 2 scorecards respect replay position:
        - At event 5 (Innings 1): Innings 1 has partial figures, Innings 2 is empty.
        - At event 200 (Innings 2): Innings 1 is complete, Innings 2 has partial figures.
        """
        # Event 5
        res_5 = self.client.post(f"/api/v1/simulations/{self.session_id}/seek", json={"event_index": 5})
        self.assertEqual(res_5.status_code, 200)
        s5 = res_5.json()["state"]
        self.assertEqual(s5["innings"], 1)
        self.assertIsNotNone(s5["innings_1"])
        self.assertIsNotNone(s5["innings_2"])
        self.assertGreater(len(s5["innings_1"]["batting"]), 0)
        # Innings 2 must be empty at ball 5 of Innings 1
        self.assertEqual(len(s5["innings_2"]["batting"]), 0)
        self.assertEqual(s5["innings_2"]["score"], 0)

        # Seek to Innings 2 (e.g. event 180)
        res_180 = self.client.post(f"/api/v1/simulations/{self.session_id}/seek", json={"event_index": 180})
        self.assertEqual(res_180.status_code, 200)
        s180 = res_180.json()["state"]
        self.assertEqual(s180["innings"], 2)
        # Innings 1 must be non-empty (completed)
        self.assertGreater(s180["innings_1"]["score"], 0)
        self.assertGreater(len(s180["innings_1"]["batting"]), 0)
        # Innings 2 must have active batters
        self.assertGreater(len(s180["innings_2"]["batting"]), 0)

    def test_probabilities_all_trajectory(self):
        """GET /probabilities with ?all=true must return all events for the match."""
        res = self.client.get(f"/api/v1/simulations/{self.session_id}/probabilities?all=true")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_points"], self.total_events)


if __name__ == "__main__":
    unittest.main()
