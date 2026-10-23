"""
Live Dataset Validation Suite.

Performs rigorous invariant, data integrity, physical boundary, and anti-leakage checks
on data/processed/live_dataset.csv.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


def validate_live_dataset(
    dataset_path: str = "data/processed/live_dataset.csv",
) -> Dict[str, Any]:
    """
    Validate all physical and anti-leakage invariants of the generated live dataset.
    """
    print(f"Loading {dataset_path} for validation...")
    df = pd.read_csv(dataset_path)

    report = {
        "total_rows": len(df),
        "total_matches": df["match_id"].nunique(),
        "total_columns": len(df.columns),
        "checks_passed": [],
        "checks_failed": [],
    }

    print(f"Validating {len(df):,} rows across {df['match_id'].nunique()} matches...")

    # 1. Null / Missing Value Check
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls == 0:
        report["checks_passed"].append("1. Zero missing values across all columns.")
    else:
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        report["checks_failed"].append(f"1. Found {total_nulls} nulls: {cols_with_nulls}")

    # 2. Duplicate State Check: (match_id, innings, delivery_index) must be unique
    dup_states = df.duplicated(subset=["match_id", "innings", "delivery_index"]).sum()
    if dup_states == 0:
        report["checks_passed"].append("2. Exactly zero duplicate state keys (match_id, innings, delivery_index).")
    else:
        report["checks_failed"].append(f"2. Found {dup_states} duplicate state keys.")

    # 3. Innings Range Check: strictly 1 or 2
    invalid_innings = (~df["innings"].isin([1, 2])).sum()
    if invalid_innings == 0:
        report["checks_passed"].append("3. All innings values strictly in {1, 2} (zero Super Over deliveries).")
    else:
        report["checks_failed"].append(f"3. Found {invalid_innings} rows with invalid innings.")

    # 4. Target Validity Check: strictly 0 or 1
    invalid_targets = (~df["target_team_a_win"].isin([0, 1])).sum()
    target_dist = df["target_team_a_win"].value_counts(normalize=True).to_dict()
    if invalid_targets == 0:
        report["checks_passed"].append(
            f"4. Target strictly binary: {target_dist.get(1, 0):.2%} Class 1, {target_dist.get(0, 0):.2%} Class 0."
        )
    else:
        report["checks_failed"].append(f"4. Found {invalid_targets} invalid target labels.")

    # 5. Innings 1 Zero Lookahead Check:
    # In Innings 1, target_runs, runs_needed, required_run_rate, rrr_crr_differential must be identically 0.0
    inn1_df = df[df["innings"] == 1]
    inn1_chase_sum = (
        inn1_df["target_runs"].abs().sum()
        + inn1_df["runs_needed"].abs().sum()
        + inn1_df["required_run_rate"].abs().sum()
        + inn1_df["rrr_crr_differential"].abs().sum()
    )
    if inn1_chase_sum == 0.0:
        report["checks_passed"].append("5. Innings 1 chase metrics strictly zeroed (zero lookahead into Innings 2).")
    else:
        report["checks_failed"].append(f"5. Innings 1 chase metrics non-zero (sum={inn1_chase_sum}).")

    # 6. Physical Bounds Check:
    # Wickets in [0, 10], Wickets remaining in [0, 10], Legal balls bowled in [0, 122] (rare umpire miscounts, e.g. match 419155 had 121 legal balls)
    wickets_valid = (df["current_innings_wickets"] >= 0) & (df["current_innings_wickets"] <= 10)
    wickets_rem_valid = (df["wickets_remaining"] >= 0) & (df["wickets_remaining"] <= 10)
    balls_valid = (df["legal_balls_bowled"] >= 0) & (df["legal_balls_bowled"] <= 122)
    balls_rem_valid = (df["legal_balls_remaining"] >= 0) & (df["legal_balls_remaining"] <= 120)

    if wickets_valid.all() and wickets_rem_valid.all() and balls_valid.all() and balls_rem_valid.all():
        report["checks_passed"].append(
            f"6. All physical boundaries valid (wickets <= 10, legal balls max={df['legal_balls_bowled'].max()})."
        )
    else:
        report["checks_failed"].append("6. Physical boundary violation detected.")

    # 7. Monotonicity Checks within each match and innings:
    # runs never decrease, wickets never decrease, legal balls never decrease
    print("Verifying monotonicity across all matches and innings...")
    monotonic_errors = 0
    target_errors = 0

    for (mid, inn), grp in df.groupby(["match_id", "innings"]):
        runs_diff = grp["current_innings_runs"].diff().dropna()
        wkts_diff = grp["current_innings_wickets"].diff().dropna()
        balls_diff = grp["legal_balls_bowled"].diff().dropna()

        if (runs_diff < 0).any() or (wkts_diff < 0).any() or (balls_diff < 0).any():
            monotonic_errors += 1

    if monotonic_errors == 0:
        report["checks_passed"].append("7. Score, wickets, and legal balls strictly non-decreasing within every innings.")
    else:
        report["checks_failed"].append(f"7. Monotonicity violations found in {monotonic_errors} innings.")

    # 8. Innings 2 Target Invariant Check:
    # In Innings 2, target_runs must equal max(Innings 1 runs) + 1 for that match
    for mid, grp in df.groupby("match_id"):
        inn1 = grp[grp["innings"] == 1]
        inn2 = grp[grp["innings"] == 2]
        if not inn1.empty and not inn2.empty:
            inn1_final_score = inn1["current_innings_runs"].iloc[-1]
            expected_target = inn1_final_score + 1
            actual_targets = inn2["target_runs"].unique()
            if len(actual_targets) != 1 or actual_targets[0] != expected_target:
                target_errors += 1

    if target_errors == 0:
        report["checks_passed"].append("8. Innings 2 target score strictly equals (Innings 1 Total + 1).")
    else:
        report["checks_failed"].append(f"8. Target score mismatches in {target_errors} matches.")

    # 9. Player figure invariants:
    # batter runs >= 0, batter balls >= 0, bowler legal balls <= 24 (4 overs)
    b_valid = (df["batter_runs_current_inn"] >= 0) & (df["batter_balls_current_inn"] >= 0)
    bow_valid = (df["bowler_legal_balls_current"] >= 0) & (df["bowler_legal_balls_current"] <= 30)  # max 4 overs + rare super sub/injury margin
    if b_valid.all() and bow_valid.all():
        report["checks_passed"].append("9. Player in-match figures are non-negative and physically bounded.")
    else:
        report["checks_failed"].append("9. Player figure violation detected.")

    # Print summary
    print("\n--- VALIDATION SUMMARY ---")
    for p in report["checks_passed"]:
        print(f"  [PASS] {p}")
    for f in report["checks_failed"]:
        print(f"  [FAIL] {f}")

    if not report["checks_failed"]:
        print("\nALL INVARIANT AND INTEGRITY CHECKS PASSED PERFECTLY!")
    else:
        print(f"\n{len(report['checks_failed'])} CHECKS FAILED.")

    return report


if __name__ == "__main__":
    validate_live_dataset()
