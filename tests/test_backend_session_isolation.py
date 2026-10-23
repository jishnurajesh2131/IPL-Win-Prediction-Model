"""
Step 20 Concurrency and Session Isolation Test.

Verifies that concurrent simulation sessions maintain completely independent
event pointers, match states, and probability histories.
"""

import unittest
from fastapi.testclient import TestClient

from src.backend.main import app
from src.backend.state.session_manager import session_manager


class TestBackendSessionIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_concurrent_sessions_isolation(self):
        """Verify two simultaneous sessions maintain strictly independent states."""
        # 1. Create Session A (Match 1426274: KKR vs RCB)
        resp_a = self.client.post("/api/v1/simulations", json={"match_id": "1426274"})
        self.assertEqual(resp_a.status_code, 201)
        sess_a_id = resp_a.json()["session_id"]

        # 2. Create Session B (Match 501228: PWI vs CSK)
        resp_b = self.client.post("/api/v1/simulations", json={"match_id": "501228"})
        self.assertEqual(resp_b.status_code, 201)
        sess_b_id = resp_b.json()["session_id"]

        self.assertNotEqual(sess_a_id, sess_b_id)

        # 3. Step Session A 10 times
        for _ in range(10):
            self.client.post(f"/api/v1/simulations/{sess_a_id}/next")

        # 4. Step Session B 5 times
        for _ in range(5):
            self.client.post(f"/api/v1/simulations/{sess_b_id}/next")

        # 5. Check Session A state
        state_a = self.client.get(f"/api/v1/simulations/{sess_a_id}").json()
        self.assertEqual(state_a["event_index"], 10)
        self.assertEqual(state_a["match_id"], "1426274")
        self.assertIn("Kolkata", state_a["prediction"]["team_1"])

        # 6. Check Session B state
        state_b = self.client.get(f"/api/v1/simulations/{sess_b_id}").json()
        self.assertEqual(state_b["event_index"], 5)
        self.assertEqual(state_b["match_id"], "501228")
        self.assertIn("Pune", state_b["prediction"]["team_1"])

        # 7. Advance Session B 10 more times -> Session A must remain at event_index 10
        for _ in range(10):
            self.client.post(f"/api/v1/simulations/{sess_b_id}/next")

        state_a_after = self.client.get(f"/api/v1/simulations/{sess_a_id}").json()
        self.assertEqual(state_a_after["event_index"], 10, "Session A pointer was corrupted by Session B")

        # 8. Fast-forward Session A to completion
        run_a = self.client.post(f"/api/v1/simulations/{sess_a_id}/run").json()
        self.assertEqual(run_a["status"], "completed")

        # Verify Session B is still active at event_index 15
        state_b_after = self.client.get(f"/api/v1/simulations/{sess_b_id}").json()
        self.assertEqual(state_b_after["status"], "active")
        self.assertEqual(state_b_after["event_index"], 15)

        # 9. Clean up Session A and verify Session B remains intact
        session_manager.delete_session(sess_a_id)
        resp_a_deleted = self.client.get(f"/api/v1/simulations/{sess_a_id}")
        self.assertEqual(resp_a_deleted.status_code, 404)

        resp_b_alive = self.client.get(f"/api/v1/simulations/{sess_b_id}")
        self.assertEqual(resp_b_alive.status_code, 200)


if __name__ == "__main__":
    unittest.main()
