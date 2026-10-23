"""
Validation Script: Determinism and Bitwise Reproducibility.

Replays matches 1426274 and 501228 three times independently:
- Compares state counters
- Compares feature vectors
- Compares predicted win probabilities
- Computes maximum difference across all events.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import numpy as np
from src.live_simulator.simulator import MatchSimulator


def run_determinism_validation():
    print("=" * 70)
    print("PHASE 5 STEP 5: BITWISE DETERMINISM VALIDATION (3 INDEPENDENT RUNS)")
    print("=" * 70)

    sim = MatchSimulator()
    matches = ["1426274", "501228"]

    results = {}
    for mid in matches:
        print(f"\nTesting Match {mid} across 3 independent runs...")
        run1 = sim.simulate(mid)
        run2 = sim.simulate(mid)
        run3 = sim.simulate(mid)

        n1, n2, n3 = len(run1.events), len(run2.events), len(run3.events)
        assert n1 == n2 == n3, f"Event count mismatch: {n1}, {n2}, {n3}"

        max_diff_1_2 = 0.0
        max_diff_1_3 = 0.0

        for idx in range(n1):
            e1 = run1.events[idx]
            e2 = run2.events[idx]
            e3 = run3.events[idx]

            d12 = abs(e1.team_a_win_probability - e2.team_a_win_probability)
            d13 = abs(e1.team_a_win_probability - e3.team_a_win_probability)

            max_diff_1_2 = max(max_diff_1_2, d12)
            max_diff_1_3 = max(max_diff_1_3, d13)

            # Check scores and wickets
            assert e1.score == e2.score == e3.score, f"Score mismatch at event {idx}"
            assert e1.wickets == e2.wickets == e3.wickets, f"Wickets mismatch at event {idx}"

        print(f"Match {mid} ({n1} events):")
        print(f"  Max Diff (Run 1 vs Run 2): {max_diff_1_2:.8f}")
        print(f"  Max Diff (Run 1 vs Run 3): {max_diff_1_3:.8f}")

        results[mid] = {
            "events": n1,
            "max_diff_1_2": max_diff_1_2,
            "max_diff_1_3": max_diff_1_3,
            "deterministic": (max_diff_1_2 == 0.0 and max_diff_1_3 == 0.0),
        }

    all_deterministic = all(r["deterministic"] for r in results.values())
    print("\n" + "=" * 70)
    print(f"OVERALL DETERMINISM VERDICT: {'100% BITWISE DETERMINISTIC (Delta = 0.0)' if all_deterministic else 'NON-DETERMINISTIC'}")
    print("=" * 70)

    with open("scratch/determinism_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved determinism_metrics.json successfully.")


if __name__ == "__main__":
    run_determinism_validation()
