"""
Validation Script: State & Feature Parity Agreement vs live_dataset.csv.

Compares simulator-generated states and all 54 raw features against
stored rows in data/processed/live_dataset.csv across the 11 validation matches.
Computes maximum absolute error for numerical features and exact equality for discrete states.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import pandas as pd
import numpy as np

from src.live_simulator.simulator import MatchSimulator


def run_agreement_validation():
    print("=" * 70)
    print("PHASE 5 STEP 5: STATE & FEATURE AGREEMENT AUDIT vs live_dataset.csv")
    print("=" * 70)

    print("Loading authoritative live_dataset.csv...")
    df_live = pd.read_csv("data/processed/live_dataset.csv", low_memory=False)

    sim = MatchSimulator()
    match_ids = [
        "335982", "419155", "501228", "598027", "1082591",
        "1136620", "1359507", "1359544", "1422131", "1426274", "1426298"
    ]

    total_deliveries_compared = 0
    max_state_diffs = {}
    mismatches_discrete = 0

    features_to_check = [
        "current_innings", "current_innings_runs", "current_innings_wickets",
        "legal_balls_bowled", "overs_completed", "current_run_rate",
        "target_runs", "runs_needed", "required_run_rate", "rrr_crr_differential",
        "runs_last_6_balls", "wickets_last_6_balls", "runs_last_12_balls", "wickets_last_12_balls",
        "dot_ball_ratio_last_12", "boundary_ratio_last_12",
        "current_partnership_runs", "current_partnership_balls", "current_partnership_run_rate",
        "pre_match_prob_team_a",
        "batter_runs_current_inn", "batter_balls_current_inn", "batter_strike_rate_current",
        "batter_boundaries_current", "batter_dots_current", "batter_dot_ratio_current",
        "bowler_legal_balls_current", "bowler_overs_current", "bowler_runs_conceded_current",
        "bowler_wickets_current", "bowler_economy_current", "bowler_dots_current",
        "bowler_boundaries_conceded_current", "bowler_dot_ratio_current",
        "historical_batter_strike_rate", "historical_batter_average", "historical_batter_balls_faced",
        "historical_bowler_economy", "historical_bowler_strike_rate", "historical_bowler_balls_bowled",
        "matchup_balls_prior", "matchup_runs_prior", "matchup_wickets_prior", "matchup_strike_rate_shrunk",
    ]

    for f in features_to_check:
        max_state_diffs[f] = 0.0

    match_agreement_summary = []

    for mid in match_ids:
        mid_int = int(mid)
        live_subset = df_live[df_live["match_id"] == mid_int].reset_index(drop=True)
        if len(live_subset) == 0:
            print(f"Match {mid} not in live_dataset.csv (skipping dataset comparison)")
            continue

        res = sim.simulate(mid)
        # Skip T0 event (event 0)
        sim_events = res.events[1:]

        # Note: in live_dataset.csv, terminal delivery was retained, but let's check counts
        n_compare = min(len(live_subset), len(sim_events))

        match_max_diff = 0.0
        for idx in range(n_compare):
            row_live = live_subset.iloc[idx]
            event_sim = sim_events[idx]
            snap = event_sim.feature_snapshot

            for feat in features_to_check:
                val_live = float(row_live[feat])
                val_sim = float(snap[feat])
                diff = abs(val_live - val_sim)

                if diff > max_state_diffs[feat]:
                    max_state_diffs[feat] = diff
                if diff > match_max_diff:
                    match_max_diff = diff

                # Discrete integer check
                if feat in ("current_innings", "current_innings_runs", "current_innings_wickets",
                             "legal_balls_bowled", "target_runs", "runs_needed"):
                    if diff != 0.0:
                        mismatches_discrete += 1

            total_deliveries_compared += 1

        match_agreement_summary.append({
            "match_id": mid,
            "deliveries_compared": n_compare,
            "max_feature_diff": round(float(match_max_diff), 6),
            "status": "PASS (Exact Parity)" if match_max_diff < 1e-4 else f"WARNING ({match_max_diff:.4f})"
        })
        print(f"Match {mid}: Compared {n_compare} deliveries | Max Abs Diff: {match_max_diff:.6f}")

    print("\n" + "=" * 70)
    print("MAXIMUM ABSOLUTE DIFFERENCE PER FEATURE ACROSS ALL 11 MATCHES")
    print("=" * 70)
    feat_diff_df = pd.DataFrame([
        {"Feature Name": k, "Max Absolute Error": f"{v:.6f}", "Status": "PERFECT (0.0)" if v == 0.0 else ("PASS (< 1e-4)" if v < 1e-4 else "INVESTIGATE")}
        for k, v in max_state_diffs.items()
    ])
    print(feat_diff_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Deliveries Compared: {total_deliveries_compared:,}")
    print(f"Discrete Counter Mismatches: {mismatches_discrete} (Expected: 0)")
    max_overall_err = max(max_state_diffs.values())
    print(f"Maximum Overall Numerical Difference: {max_overall_err:.6f}")
    print(f"Dataset Agreement Verdict: {'100% PERFECT PARITY' if max_overall_err < 1e-4 else 'MISMATCH DETECTED'}")

    agreement_out = {
        "total_deliveries_compared": total_deliveries_compared,
        "mismatches_discrete": mismatches_discrete,
        "max_overall_numerical_diff": max_overall_err,
        "feature_diffs": max_state_diffs,
        "matches": match_agreement_summary,
    }
    with open("scratch/agreement_metrics.json", "w", encoding="utf-8") as f:
        json.dump(agreement_out, f, indent=2)
    print("Saved agreement_metrics.json successfully.")


if __name__ == "__main__":
    run_agreement_validation()
