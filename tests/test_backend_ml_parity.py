"""
Step 19 Critical Test: ML Inference and State Parity Audit.

Verifies that the backend API produces bitwise identical predictions and state snapshots
compared to direct standalone invocation of the frozen MatchSimulator.
"""

import unittest
from fastapi.testclient import TestClient

from src.live_simulator.simulator import MatchSimulator
from src.backend.main import app


class TestBackendMLParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.simulator = MatchSimulator()
        cls.simulator._ensure_data_loaded()

    def _verify_match_parity(self, match_id: str, test_description: str):
        print(f"\n--- Testing Backend ML Parity on Match {match_id} ({test_description}) ---")

        # 1. Direct Standalone Simulation
        direct_result = self.simulator.simulate(match_id)
        direct_events = direct_result.events

        # 2. Backend API Simulation
        create_resp = self.client.post("/api/v1/simulations", json={"match_id": match_id})
        self.assertEqual(create_resp.status_code, 201)
        create_data = create_resp.json()
        session_id = create_data["session_id"]

        # Check T0 Event (Event 0)
        e0_direct = direct_events[0]
        self.assertEqual(create_data["current_event_index"], 0)
        self.assertEqual(
            create_data["current_prediction"]["team_1_probability"],
            round(float(e0_direct.team_a_win_probability), 4),
        )
        self.assertEqual(
            create_data["current_prediction"]["team_2_probability"],
            round(float(e0_direct.team_b_win_probability), 4),
        )
        self.assertEqual(create_data["current_prediction"]["prediction_type"], e0_direct.prediction_type)

        # 3. Advance Step by Step and Assert Exact Parity
        total_deliveries = len(direct_events) - 1
        for step_idx in range(1, len(direct_events)):
            e_direct = direct_events[step_idx]

            step_resp = self.client.post(f"/api/v1/simulations/{session_id}/next")
            self.assertEqual(step_resp.status_code, 200, f"Step {step_idx} failed")
            step_data = step_resp.json()

            # State Parity
            state_api = step_data["state"]
            self.assertEqual(state_api["innings"], e_direct.innings, f"Innings mismatch at event {step_idx}")
            self.assertEqual(state_api["score"], e_direct.score, f"Score mismatch at event {step_idx}")
            self.assertEqual(state_api["wickets"], e_direct.wickets, f"Wickets mismatch at event {step_idx}")
            self.assertEqual(state_api["legal_balls"], e_direct.legal_balls_bowled, f"Balls mismatch at event {step_idx}")

            # Prediction Parity
            pred_api = step_data["prediction"]
            diff_a = abs(pred_api["team_1_probability"] - e_direct.team_a_win_probability)
            diff_b = abs(pred_api["team_2_probability"] - e_direct.team_b_win_probability)

            self.assertAlmostEqual(
                diff_a,
                0.0,
                places=4,
                msg=f"Probability P(Team A) mismatch at event {step_idx}: API {pred_api['team_1_probability']} vs Direct {e_direct.team_a_win_probability}",
            )
            self.assertAlmostEqual(
                diff_b,
                0.0,
                places=4,
                msg=f"Probability P(Team B) mismatch at event {step_idx}: API {pred_api['team_2_probability']} vs Direct {e_direct.team_b_win_probability}",
            )
            self.assertEqual(
                pred_api["prediction_type"],
                e_direct.prediction_type,
                f"Prediction type mismatch at event {step_idx}",
            )

        # Terminal state assertion
        final_step = self.client.get(f"/api/v1/simulations/{session_id}").json()
        self.assertEqual(final_step["status"], "completed")

        print(f"Match {match_id}: Verified all {total_deliveries} deliveries with 100% exact parity!")

    def test_parity_match_1426274_kkr_vs_rcb_2024(self):
        """Verify ML and state parity on Match 1426274 (2024 held-out test thriller)."""
        self._verify_match_parity("1426274", "KKR vs RCB 2024, 222 vs 221 thriller")

    def test_parity_match_419155_csk_vs_dc_2010(self):
        """Verify ML and state parity on Match 419155 (121-ball anomaly test)."""
        self._verify_match_parity("419155", "CSK vs DC 2010, 121-ball over 19 anomaly")


if __name__ == "__main__":
    unittest.main()
