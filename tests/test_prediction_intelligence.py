"""
Unit & Integration Tests for Prediction Intelligence Module:
- Prediction Reliability Auditor
- Model Failure Observatory
- IPL Era-Shift Monitor
- Immutability of Core Models & Temporal Isolation
"""

import unittest
from fastapi.testclient import TestClient

from src.backend.main import app
from src.backend.services.prediction_intelligence_service import (
    prediction_intelligence_service,
)


class TestPredictionIntelligence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        prediction_intelligence_service.ensure_loaded()

    def test_01_summary_endpoint(self):
        """Verify the executive summary endpoint returns valid metrics."""
        resp = self.client.get("/api/v1/intelligence/summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("live_model_test_brier", data)
        self.assertIn("live_model_test_log_loss", data)
        self.assertIn("live_model_test_ece", data)
        self.assertIn("live_model_test_accuracy", data)
        self.assertIn("live_model_test_matches", data)
        self.assertIn("pre_match_walkforward_matches", data)
        self.assertTrue(data["era_shift_detected"])
        self.assertIn("temporal_isolation_guarantee", data)

        # Check values match verified evaluation
        self.assertEqual(data["live_model_test_matches"], 213)
        self.assertAlmostEqual(data["live_model_test_brier"], 0.1967, places=3)
        self.assertAlmostEqual(data["live_model_test_accuracy"], 0.6829, places=3)
        self.assertGreater(data["pre_match_walkforward_matches"], 1000)

    def test_02_reliability_live_model_test(self):
        """Verify Reliability Auditor on Live Model (Held-Out Test Split)."""
        resp = self.client.get("/api/v1/intelligence/reliability?model_type=live&split=test")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["model_type"], "live")
        self.assertEqual(data["split"], "test")
        self.assertEqual(data["evaluation_matches"], 213)
        self.assertEqual(data["evaluation_predictions"], 50867)
        self.assertTrue(data["temporal_isolation_verified"])
        self.assertFalse(data["insufficient_data"])

        # Check calibration bins: non-empty must have valid floats, empty must have None
        bins = data["calibration_bins"]
        self.assertEqual(len(bins), 10)
        for b in bins:
            self.assertIn("bin_index", b)
            self.assertIn("count", b)
            self.assertIn("mean_predicted_prob", b)
            if b["count"] > 0:
                self.assertIsNotNone(b["observed_frequency"])
                self.assertIsNotNone(b["calibration_error"])
                self.assertGreaterEqual(b["ci_lower"], 0.0)
                self.assertLessEqual(b["ci_upper"], 1.0)
                self.assertLessEqual(b["ci_lower"], b["ci_upper"])
            else:
                self.assertIsNone(b["observed_frequency"])
                self.assertIsNone(b["calibration_error"])
                self.assertIsNone(b["ci_lower"])
                self.assertIsNone(b["ci_upper"])

        # Clustering caveat
        self.assertIn("clustering_caveat", data)
        self.assertIsNotNone(data["clustering_caveat"])
        self.assertIn("clustering", data["clustering_caveat"].lower())

        # Phase metrics
        phases = data["phase_metrics"]
        self.assertIsNotNone(phases)
        phase_names = [p["phase"] for p in phases]
        self.assertIn("Powerplay (0-6 overs)", phase_names)
        self.assertIn("Middle (6-15 overs)", phase_names)
        self.assertIn("Death (15-20 overs)", phase_names)

        # Innings metrics
        innings = data["innings_metrics"]
        self.assertIsNotNone(innings)
        self.assertEqual(len(innings), 2)
        inn_names = [i["innings"] for i in innings]
        self.assertIn("Innings 1", inn_names)
        self.assertIn("Innings 2", inn_names)

    def test_03_reliability_live_model_validation(self):
        """Verify Reliability Auditor on Live Model (Validation Split)."""
        resp = self.client.get("/api/v1/intelligence/reliability?model_type=live&split=val")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["split"], "val")
        self.assertEqual(data["evaluation_matches"], 145)
        self.assertEqual(data["evaluation_predictions"], 35074)
        self.assertAlmostEqual(data["brier_score"], 0.2180, places=3)
        self.assertAlmostEqual(data["accuracy"], 0.6283, places=3)

    def test_04_reliability_pre_match_walkforward(self):
        """Verify Reliability Auditor on T0 Pre-Match Walk-Forward Model."""
        resp = self.client.get("/api/v1/intelligence/reliability?model_type=pre_match&split=walk_forward")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["model_type"], "pre_match")
        self.assertGreater(data["evaluation_matches"], 1150)
        self.assertFalse(data["insufficient_data"])

        # Check Brier score and calibration bounds
        self.assertGreater(data["brier_score"], 0.20)
        self.assertLess(data["brier_score"], 0.30)
        self.assertGreater(data["accuracy"], 0.50)
        self.assertEqual(len(data["calibration_bins"]), 10)

        # In pre-match model, probability extreme deciles [0, 0.1] and [0.9, 1.0] have 0 predictions
        # Verify that empty bins return null (None) for observed frequency, NOT 0.0
        empty_bins = [b for b in data["calibration_bins"] if b["count"] == 0]
        non_empty_bins = [b for b in data["calibration_bins"] if b["count"] > 0]
        self.assertGreater(len(empty_bins), 0)
        self.assertGreater(len(non_empty_bins), 0)

        for b in empty_bins:
            self.assertIsNone(b["observed_frequency"], f"Bin {b['bin_range']} has count 0 but observed_frequency is not None")
            self.assertIsNone(b["calibration_error"])
            self.assertIsNone(b["ci_lower"])
            self.assertIsNone(b["ci_upper"])

        for b in non_empty_bins:
            self.assertIsNotNone(b["observed_frequency"])
            self.assertGreater(b["observed_frequency"], 0.0)
            self.assertIsNotNone(b["calibration_error"])
            self.assertIsNotNone(b["ci_lower"])
            self.assertIsNotNone(b["ci_upper"])

    def test_05_reliability_insufficient_data_handling(self):
        """Verify insufficient data handling does not fabricate findings."""
        resp = self.client.get("/api/v1/intelligence/reliability?model_type=pre_match&season=1999")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertTrue(data["insufficient_data"])
        self.assertEqual(data["evaluation_matches"], 0)
        self.assertEqual(len(data["calibration_bins"]), 0)
        self.assertIn("Insufficient historical sample size", data["status_message"])

    def test_06_failure_observatory(self):
        """Verify Model Failure Observatory subgroups and statistical indicators."""
        resp = self.client.get("/api/v1/intelligence/failures")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["total_matches_audited"], 1234)
        self.assertEqual(data["total_errors_audited"], 560)
        self.assertAlmostEqual(data["overall_error_rate"], 0.4538, places=3)
        self.assertIn("non_causal_statement", data)
        self.assertIn("causal", data["non_causal_statement"].lower())

        subgroups = data["subgroups"]
        self.assertGreater(len(subgroups), 10)

        # Verify each subgroup pattern has required attributes and non-overlapping baseline
        for sg in subgroups:
            self.assertIn("dimension", sg)
            self.assertIn("subgroup", sg)
            self.assertIn("total_matches", sg)
            self.assertIn("error_rate", sg)
            self.assertIn("status", sg)
            self.assertIn(sg["status"], ["validated", "exploratory", "insufficient_evidence"])
            self.assertIn("caveat", sg)
            self.assertGreaterEqual(sg["ci_lower"], 0.0)
            self.assertLessEqual(sg["ci_upper"], 1.0)
            self.assertLessEqual(sg["ci_lower"], sg["ci_upper"])

            # Non-overlapping rest-of-matches check: baseline + difference == error_rate
            self.assertAlmostEqual(
                sg["baseline_error_rate"] + sg["difference_from_baseline"],
                sg["error_rate"],
                places=3,
            )

    def test_07_failure_observatory_boundary_rules(self):
        """Verify sample-size boundary rules: N < 15, 15 <= N < 50, and N >= 50."""
        resp = self.client.get("/api/v1/intelligence/failures")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        subgroups = data["subgroups"]

        # Check N < 15: Saurashtra (N=10), SuperSport Park (N=12)
        n_under_15 = [s for s in subgroups if s["total_matches"] < 15]
        self.assertGreater(len(n_under_15), 0)
        for s in n_under_15:
            self.assertEqual(s["status"], "insufficient_evidence")
            self.assertIsNone(s["p_value"])
            self.assertIsNone(s["p_value_adjusted"])
            self.assertIn("Small sample size (N < 15 matches)", s["caveat"])

        # Check 15 <= N < 50: Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium (N=17)
        aca_vdca = next(s for s in subgroups if "ACA-VDCA" in s["subgroup"])
        self.assertEqual(aca_vdca["total_matches"], 17)
        self.assertEqual(aca_vdca["status"], "exploratory")
        self.assertNotIn("below the minimum threshold", aca_vdca["caveat"].lower())
        self.assertIn("15 ≤ N < 50", aca_vdca["caveat"])
        self.assertIsNotNone(aca_vdca["p_value"])

        # Check all 15 <= N < 50 are designated exploratory
        n_15_to_49 = [s for s in subgroups if 15 <= s["total_matches"] < 50]
        for s in n_15_to_49:
            self.assertEqual(s["status"], "exploratory")
            self.assertIn("15 ≤ N < 50", s["caveat"])

        # Check N >= 50: must only be validated if p_value_adjusted < 0.05
        n_50_plus = [s for s in subgroups if s["total_matches"] >= 50]
        for s in n_50_plus:
            if s["status"] == "validated":
                self.assertIsNotNone(s["p_value_adjusted"])
                self.assertLess(s["p_value_adjusted"], 0.05)
            else:
                self.assertEqual(s["status"], "exploratory")

    def test_08_benjamini_hochberg_algorithm(self):
        """Verify Benjamini-Hochberg FDR implementation against known mathematical test cases."""
        # Test case 1: Single test
        p_single = [0.03]
        adj_single = prediction_intelligence_service._benjamini_hochberg(p_single)
        self.assertEqual(adj_single, [0.03])

        # Test case 2: Standard 4 hypotheses
        # Raw: [0.01, 0.04, 0.03, 0.20]
        # Sorted: 0.01 (k=1, q=4*0.01/1=0.04), 0.03 (k=2, q=4*0.03/2=0.06), 0.04 (k=3, q=4*0.04/3=0.0533), 0.20 (k=4, q=4*0.20/4=0.20)
        # Working backward for monotonicity:
        # k=4: min(1.0, 0.20) = 0.20
        # k=3: min(0.20, 0.0533) = 0.0533
        # k=2: min(0.0533, 0.06) = 0.0533
        # k=1: min(0.0533, 0.04) = 0.04
        # Original order: [0.04, 0.0533, 0.0533, 0.20]
        p_multi = [0.01, 0.04, 0.03, 0.20]
        adj_multi = prediction_intelligence_service._benjamini_hochberg(p_multi)
        self.assertAlmostEqual(adj_multi[0], 0.0400, places=4)
        self.assertAlmostEqual(adj_multi[1], 0.0533, places=4)
        self.assertAlmostEqual(adj_multi[2], 0.0533, places=4)
        self.assertAlmostEqual(adj_multi[3], 0.2000, places=4)

        # Monotonicity test: adjusted p-values must never exceed 1.0
        p_large = [0.8, 0.9, 0.95]
        adj_large = prediction_intelligence_service._benjamini_hochberg(p_large)
        for q in adj_large:
            self.assertLessEqual(q, 1.0)

    def test_09_chepauk_non_causal_assertions(self):
        """Verify MA Chidambaram Stadium has no unsupported causal explanations."""
        resp = self.client.get("/api/v1/intelligence/failures")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        chepauk = next(s for s in data["subgroups"] if "Chidambaram" in s["subgroup"])
        self.assertIsNotNone(chepauk)
        self.assertIn("Observed prediction error was lower in this subgroup. This analysis does not establish the cause.", chepauk["caveat"])
        # Must not contain causal buzzwords like 'home dominance', 'predictable pitch', etc.
        self.assertNotIn("home dominance", chepauk["caveat"].lower())
        self.assertNotIn("pitch condition", chepauk["caveat"].lower())

    def test_10_era_shift_monitor(self):
        """Verify Era-Shift Monitor chronological eras and statistical tests."""
        resp = self.client.get("/api/v1/intelligence/era-shifts")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        eras = data["eras"]
        self.assertEqual(len(eras), 4)

        # Methodology text must NOT claim Kolmogorov-Smirnov test was executed
        self.assertNotIn("kolmogorov-smirnov", data["methodology"].lower())
        self.assertNotIn("ks test", data["methodology"].lower())
        self.assertIn("welch's two-sample t-tests", data["methodology"].lower())

        # Check season metrics coverage
        seasons = data["seasons"]
        self.assertEqual(len(seasons), 19)

        # Check statistical shift tests
        shift_tests = data["shift_tests"]
        self.assertGreaterEqual(len(shift_tests), 3)

        # Test 1: Scoring surge
        scoring_test = next(t for t in shift_tests if t["metric"] == "Average 1st Innings Runs")
        self.assertTrue(scoring_test["is_statistically_significant"])
        self.assertEqual(scoring_test["status"], "statistically_significant")
        self.assertEqual(scoring_test["analysis_type"], "scoring_distribution_shift")
        self.assertLess(scoring_test["p_value"], 0.05)
        # Note must contain the dynamic p-value
        self.assertIn(f"p={scoring_test['p_value']}", scoring_test["notes"])

        # Test 2: 200+ scores frequency shift
        # Dynamic p-value and note MUST agree (no hardcoded p=0.0008)
        pct200_test = next(t for t in shift_tests if t["metric"] == "Matches with 200+ Runs (1st Innings %)")
        self.assertIsNotNone(pct200_test["p_value"])
        self.assertIn(f"p={pct200_test['p_value']}", pct200_test["notes"])
        self.assertNotIn("p=0.0008", pct200_test["notes"])

        # Test 3: Model calibration stability
        model_test = next(t for t in shift_tests if t["analysis_type"] == "model_performance_drift")
        self.assertFalse(model_test["is_statistically_significant"])
        self.assertEqual(model_test["status"], "provisional")
        self.assertGreater(model_test["p_value"], 0.05)
        self.assertIn(f"p={model_test['p_value']}", model_test["notes"])

        # Limitations list must mention small sample size and non-causal interpretation
        self.assertGreater(len(data["limitations"]), 2)
        combined_limitations = " ".join(data["limitations"]).lower()
        self.assertIn("small sample size", combined_limitations)
        self.assertIn("non-causal", combined_limitations)

    def test_11_core_model_invariance(self):
        """
        Verify that existing live simulator produces 100% identical predictions
        for fixed match state (KKR vs RCB 2024 first ball).
        """
        from src.backend.services.simulation_service import simulation_service

        session_resp = simulation_service.create_simulation("1426274")
        self.assertIsNotNone(session_resp)
        pred = session_resp.current_prediction
        self.assertIsNotNone(pred)

        # Ground truth check for match 1426274 T0 pre-match anchor
        # KKR probability must stay exactly at the verified 0.4956 baseline
        self.assertAlmostEqual(pred.team_1_probability, 0.4956, places=4)
        self.assertAlmostEqual(pred.team_2_probability, 0.5044, places=4)
        self.assertAlmostEqual(pred.team_1_probability + pred.team_2_probability, 1.0, places=5)
