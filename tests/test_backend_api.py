"""
Unit & Integration Tests for Backend REST API Endpoints.
"""

import unittest
from fastapi.testclient import TestClient

from src.backend.main import app


class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_root_endpoint(self):
        """Verify root metadata endpoint."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("service", data)
        self.assertIn("docs_url", data)
        self.assertEqual(data["status"], "online")

    def test_02_health_endpoint(self):
        """Verify health check detects all model artifacts and datasets."""
        resp = self.client.get("/api/v1/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["ml_status"], "loaded")
        self.assertTrue(data["model_artifacts"]["live_model_inn1"])
        self.assertTrue(data["model_artifacts"]["live_model_inn2"])
        self.assertTrue(data["model_artifacts"]["live_preprocessor_inn1"])
        self.assertTrue(data["model_artifacts"]["live_preprocessor_inn2"])
        self.assertGreater(data["matches_available"], 1000)

    def test_03_list_matches_default(self):
        """Verify match catalog listing with default pagination."""
        resp = self.client.get("/api/v1/matches")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("matches", data)
        self.assertIn("total", data)
        self.assertEqual(data["limit"], 50)
        self.assertEqual(data["offset"], 0)
        self.assertEqual(len(data["matches"]), 50)
        self.assertGreater(data["total"], 1200)

    def test_04_list_matches_filters(self):
        """Verify filtering by season and team."""
        # Season filter
        resp_season = self.client.get("/api/v1/matches?season=2024&limit=100")
        self.assertEqual(resp_season.status_code, 200)
        data_season = resp_season.json()
        for m in data_season["matches"]:
            self.assertEqual(m["season"], "2024")

        # Team filter
        resp_team = self.client.get("/api/v1/matches?team=Kolkata&limit=10")
        self.assertEqual(resp_team.status_code, 200)
        data_team = resp_team.json()
        for m in data_team["matches"]:
            teams_str = f"{m['team_1']} {m['team_2']}".lower()
            self.assertIn("kolkata", teams_str)

    def test_05_get_match_details(self):
        """Verify retrieving single match details."""
        resp = self.client.get("/api/v1/matches/1426274")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["match_id"], "1426274")
        self.assertEqual(data["season"], "2024")
        self.assertIn("Kolkata", data["team_1"])
        self.assertIn("Royal Challengers", data["team_2"])
        self.assertIn("toss_winner", data)
        self.assertIn("toss_decision", data)

    def test_06_get_nonexistent_match(self):
        """Verify 404 on nonexistent match ID."""
        resp = self.client.get("/api/v1/matches/999999999")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("detail", resp.json())

    def test_07_simulation_lifecycle(self):
        """Verify complete simulation lifecycle: create, step, state, probabilities, and run."""
        # 1. Create simulation
        create_resp = self.client.post(
            "/api/v1/simulations",
            json={"match_id": "1426274"},
        )
        self.assertEqual(create_resp.status_code, 201)
        create_data = create_resp.json()
        session_id = create_data["session_id"]
        self.assertEqual(create_data["match_id"], "1426274")
        self.assertEqual(create_data["status"], "ready")
        self.assertEqual(create_data["current_event_index"], 0)
        self.assertGreater(create_data["total_events"], 200)

        # T0 Prediction Anchor Check
        pred_t0 = create_data["current_prediction"]
        self.assertAlmostEqual(
            pred_t0["team_1_probability"] + pred_t0["team_2_probability"],
            1.0,
            places=4,
        )
        self.assertEqual(pred_t0["prediction_type"], "pre_match")

        # 2. Get current state (without advancing)
        state_resp = self.client.get(f"/api/v1/simulations/{session_id}")
        self.assertEqual(state_resp.status_code, 200)
        state_data = state_resp.json()
        self.assertEqual(state_data["event_index"], 0)
        self.assertEqual(state_data["status"], "ready")
        self.assertIsNone(state_data["delivery"])  # T0 has no delivery

        # 3. Advance one step
        step_resp = self.client.post(f"/api/v1/simulations/{session_id}/next")
        self.assertEqual(step_resp.status_code, 200)
        step_data = step_resp.json()
        self.assertEqual(step_data["event_index"], 1)
        self.assertEqual(step_data["status"], "active")
        self.assertIsNotNone(step_data["delivery"])
        self.assertEqual(step_data["delivery"]["innings"], 1)

        # 4. Check probability trajectory history
        prob_resp = self.client.get(f"/api/v1/simulations/{session_id}/probabilities")
        self.assertEqual(prob_resp.status_code, 200)
        prob_data = prob_resp.json()
        self.assertEqual(prob_data["total_points"], 2)  # Event 0 and Event 1
        self.assertEqual(prob_data["points"][0]["event_index"], 0)
        self.assertEqual(prob_data["points"][1]["event_index"], 1)

        # 5. Run to completion
        run_resp = self.client.post(f"/api/v1/simulations/{session_id}/run?include_history=true")
        self.assertEqual(run_resp.status_code, 200)
        run_data = run_resp.json()
        self.assertEqual(run_data["status"], "completed")
        self.assertGreater(run_data["events_processed"], 200)
        self.assertEqual(run_data["winner"], "Kolkata Knight Riders")
        self.assertEqual(run_data["final_prediction"]["team_1_probability"], 1.0)
        self.assertEqual(run_data["final_prediction"]["team_2_probability"], 0.0)

        # 6. Attempting to advance completed session returns 409 Conflict
        conflict_resp = self.client.post(f"/api/v1/simulations/{session_id}/next")
        self.assertEqual(conflict_resp.status_code, 409)

    def test_08_invalid_session_errors(self):
        """Verify 404 on nonexistent session ID."""
        resp_state = self.client.get("/api/v1/simulations/invalid-uuid-12345")
        self.assertEqual(resp_state.status_code, 404)

        resp_next = self.client.post("/api/v1/simulations/invalid-uuid-12345/next")
        self.assertEqual(resp_next.status_code, 404)

        resp_probs = self.client.get("/api/v1/simulations/invalid-uuid-12345/probabilities")
        self.assertEqual(resp_probs.status_code, 404)

        resp_run = self.client.post("/api/v1/simulations/invalid-uuid-12345/run")
        self.assertEqual(resp_run.status_code, 404)


if __name__ == "__main__":
    unittest.main()
