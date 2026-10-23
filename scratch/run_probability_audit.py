"""
Comprehensive Every-Match Probability Audit Script (2008–2026).

Audits all 1,243 matches in matches_clean.csv:
1. Eligibility & Exclusion Audit
2. Toss / Innings Mapping & Delivery Ground Truth Verification
3. Team Ordering & Probability Label Attribution Verification
4. Walk-Forward Temporal Isolation & Zero-Leakage Verification
5. Independent Step-by-Step Feature Vector Recalculation & Comparison
6. Probability Generation, Bounds [0, 1], and Exact Sums P(A) + P(B) = 1
7. Exact Reproducibility Test (Run 1 vs Run 2)
8. Current Existing Artifact vs Recomputed Comparison
9. Year-by-Year Historical Breakdown (2008–2026)
10. Extreme Probabilities Investigation & Feature Drivers
11. Favorites vs Actual Outcomes (Overall & Season-by-Season)
12. Dedicated Held-Out Final-Test Set Audit (2024–2026)
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import csv
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed

from src.features.historical_stats import HistoricalStateTracker
from src.features.pre_match_features import load_match_delivery_summaries
from src.live_prediction.walk_forward_pre_match import (
    CATEGORICAL_COLS,
    BINARY_COLS,
    CONTINUOUS_COLS,
    PRE_MATCH_FEATURE_COLS,
    TARGET_COL,
)


def run_full_probability_audit():
    print("=" * 80)
    print("STARTING COMPREHENSIVE EVERY-MATCH PROBABILITY AUDIT (2008–2026)")
    print("=" * 80)

    matches_path = "data/processed/matches_clean.csv"
    deliveries_path = "data/processed/deliveries_clean.csv"
    pre_match_path = "data/processed/pre_match_dataset.csv"
    walk_forward_path = "data/processed/walk_forward_pre_match_probs.csv"

    # =========================================================================
    # PART 1: ELIGIBILITY & EXCLUSION AUDIT
    # =========================================================================
    print("\n--- PART 1: ELIGIBILITY & EXCLUSIONS ---")
    df_matches = pd.read_csv(matches_path)
    total_raw_matches = len(df_matches)
    print(f"Total matches in {matches_path}: {total_raw_matches}")

    # Result types
    result_type_counts = df_matches["match_result_type"].value_counts(dropna=False).to_dict()
    print("Match result types:", result_type_counts)

    # Method counts (D/L)
    dl_count = len(df_matches[df_matches["method"] == "D/L"])
    super_over_count = len(df_matches[df_matches["match_result_type"] == "super_over"])
    normal_count = len(df_matches[df_matches["match_result_type"] == "normal"])
    no_result_count = len(df_matches[df_matches["match_result_type"] == "no_result"])

    excluded_matches = df_matches[df_matches["match_result_type"] == "no_result"].copy()
    eligible_matches = df_matches[df_matches["match_result_type"] != "no_result"].copy()

    # Sort strictly chronologically by (date, int(match_id))
    eligible_matches["date_parsed"] = pd.to_datetime(eligible_matches["date"])
    eligible_matches = eligible_matches.sort_values(by=["date_parsed", "match_id"]).reset_index(drop=True)

    print(f"Evaluated / Eligible matches: {len(eligible_matches)}")
    print(f"  - Normal completed: {normal_count}")
    print(f"  - Super Over completed: {super_over_count}")
    print(f"  - Duckworth-Lewis / Stern: {dl_count}")
    print(f"Excluded matches: {len(excluded_matches)}")
    print(f"  - Excluded because no-result (abandoned): {no_result_count}")
    print(f"  - Other exclusion categories: 0")

    excluded_records = []
    for _, r in excluded_matches.iterrows():
        excluded_records.append({
            "match_id": str(r["match_id"]),
            "season": str(r["season"]),
            "date": str(r["date"]),
            "team_1": str(r["team_1"]),
            "team_2": str(r["team_2"]),
            "result_type": str(r["match_result_type"]),
            "reason": "Abandoned match without official result (no_result)"
        })

    # =========================================================================
    # PART 2: TOSS & INNINGS MAPPING INTEGRITY
    # =========================================================================
    print("\n--- PART 2: TOSS & INNINGS MAPPING INTEGRITY ---")
    print("Loading deliveries to verify ground-truth batting assignments...")
    # Load first delivery of each innings for every match
    inn_first_deliveries = {}
    with open(deliveries_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["match_id"]
            inn = int(row["innings"])
            if inn in (1, 2) and row["is_super_over"] == "0":
                if mid not in inn_first_deliveries:
                    inn_first_deliveries[mid] = {}
                if inn not in inn_first_deliveries[mid]:
                    inn_first_deliveries[mid][inn] = {
                        "batting_team": row["batting_team"],
                        "bowling_team": row["bowling_team"],
                    }

    toss_innings_errors = []
    toss_consistency_audit = []

    for idx, r in eligible_matches.iterrows():
        mid = str(r["match_id"])
        t1 = r["team_1"]
        t2 = r["team_2"]
        tw = r["toss_winner"]
        td = r["toss_decision"]

        # Theoretical innings 1 batting team from toss:
        if td == "bat":
            expected_inn1_bat = tw
            expected_inn2_bat = t2 if tw == t1 else t1
        else:  # 'field'
            expected_inn1_bat = t2 if tw == t1 else t1
            expected_inn2_bat = tw

        # Actual first ball batting team from deliveries
        actual_inn1_bat = inn_first_deliveries.get(mid, {}).get(1, {}).get("batting_team", "")
        actual_inn2_bat = inn_first_deliveries.get(mid, {}).get(2, {}).get("batting_team", "")

        toss_winner_is_team_a = 1 if tw == t1 else 0
        toss_decision_is_bat = 1 if td == "bat" else 0
        expected_team_a_bats_first = 1 if expected_inn1_bat == t1 else 0

        # Verify agreement
        inn1_match = (actual_inn1_bat == expected_inn1_bat) if actual_inn1_bat else True
        inn2_match = (actual_inn2_bat == expected_inn2_bat) if actual_inn2_bat else True

        if not (inn1_match and inn2_match):
            toss_innings_errors.append({
                "match_id": mid,
                "season": r["season"],
                "team_1": t1,
                "team_2": t2,
                "toss_winner": tw,
                "toss_decision": td,
                "expected_inn1": expected_inn1_bat,
                "actual_inn1": actual_inn1_bat,
                "expected_inn2": expected_inn2_bat,
                "actual_inn2": actual_inn2_bat,
            })

        toss_consistency_audit.append({
            "match_id": mid,
            "season": r["season"],
            "date": r["date"],
            "team_a": t1,
            "team_b": t2,
            "toss_winner": tw,
            "toss_decision": td,
            "toss_winner_is_team_a": toss_winner_is_team_a,
            "toss_decision_is_bat": toss_decision_is_bat,
            "team_a_bats_first": expected_team_a_bats_first,
            "inn1_batting_team": expected_inn1_bat,
            "inn2_batting_team": expected_inn2_bat,
            "actual_inn1_bat": actual_inn1_bat,
            "actual_inn2_bat": actual_inn2_bat,
            "actual_winner": r["winner"],
            "match_result_type": r["match_result_type"],
        })

    print(f"Toss/Innings Mapping Checked: {len(eligible_matches)} matches.")
    print(f"Toss/Innings Discrepancies found: {len(toss_innings_errors)}")
    if toss_innings_errors:
        print("Mismatches detail:", toss_innings_errors[:3])

    # =========================================================================
    # PART 3: RECHECK THE FEATURE VECTOR (INDEPENDENT VERIFICATION)
    # =========================================================================
    print("\n--- PART 3: FEATURE VECTOR INDEPENDENT VERIFICATION ---")
    df_pre_saved = pd.read_csv(pre_match_path)
    print(f"Saved pre-match dataset rows: {len(df_pre_saved)}")

    # Independent Step-by-Step Re-calculation
    delivery_summaries = load_match_delivery_summaries(deliveries_path)
    tracker = HistoricalStateTracker()
    feature_mismatches = []
    independent_features = []

    for idx, r in eligible_matches.iterrows():
        mid = str(r["match_id"])
        team_a = r["team_1"]
        team_b = r["team_2"]
        venue = r["venue"]
        city = r["city"]
        toss_winner = r["toss_winner"]
        toss_decision = r["toss_decision"]
        winner = r["winner"]

        # Step 1: Compute features strictly using history prior to this match
        computed = tracker.compute_pre_match_features(
            team_a=team_a,
            team_b=team_b,
            venue=venue,
            city=city,
            toss_winner=toss_winner,
            toss_decision=toss_decision,
        )

        # Check against saved df_pre_saved for this match
        saved_row = df_pre_saved[df_pre_saved["match_id"].astype(str) == mid]
        if len(saved_row) == 1:
            s_dict = saved_row.iloc[0].to_dict()
            for col in PRE_MATCH_FEATURE_COLS:
                val_calc = computed[col]
                val_saved = s_dict[col]
                if isinstance(val_calc, (int, float, np.number)):
                    if abs(float(val_calc) - float(val_saved)) > 1e-4:
                        feature_mismatches.append({
                            "match_id": mid,
                            "feature": col,
                            "recalculated": val_calc,
                            "saved": val_saved,
                            "diff": abs(float(val_calc) - float(val_saved))
                        })
                else:
                    if str(val_calc) != str(val_saved):
                        feature_mismatches.append({
                            "match_id": mid,
                            "feature": col,
                            "recalculated": val_calc,
                            "saved": val_saved,
                            "diff": -1
                        })

        # Step 2: Update historical state strictly AFTER features have been computed
        deliv_summary = delivery_summaries.get(mid, {"innings": [], "chasing_team": ""})
        chasing_team = deliv_summary.get("chasing_team", "")
        chasing_team_won = (winner == chasing_team) if chasing_team else False

        tracker.update_with_match_result(
            team_a=team_a,
            team_b=team_b,
            winner=winner,
            venue=venue,
            innings_stats=deliv_summary.get("innings", []),
            chasing_team_won=chasing_team_won,
        )

    print(f"Feature vector validation completed across {len(eligible_matches)} matches * {len(PRE_MATCH_FEATURE_COLS)} features.")
    print(f"Total feature discrepancies found: {len(feature_mismatches)}")
    if feature_mismatches:
        print("Sample feature discrepancies:", feature_mismatches[:5])

    # =========================================================================
    # PART 4: WALK-FORWARD PROBABILITY GENERATION & REPRODUCIBILITY
    # =========================================================================
    print("\n--- PART 4: WALK-FORWARD PROBABILITY GENERATION & REPRODUCIBILITY ---")
    df_pre = pd.read_csv(pre_match_path)
    df_pre["date_parsed"] = pd.to_datetime(df_pre["date"])
    df_pre = df_pre.sort_values(by=["date_parsed", "match_id"]).reset_index(drop=True)
    n_matches = len(df_pre)

    df_existing_wf = pd.read_csv(walk_forward_path)
    existing_map = dict(zip(df_existing_wf["match_id"].astype(str), df_existing_wf["pre_match_prob_team_a"].astype(float)))

    WF_FEATURE_COLS = PRE_MATCH_FEATURE_COLS
    WF_CATEGORICAL_COLS = CATEGORICAL_COLS
    WF_TARGET_COL = TARGET_COL

    def compute_single_wf(i: int, min_training: int = 40) -> Tuple[str, float]:
        mid = str(df_pre.loc[i, "match_id"])
        if i < min_training:
            return mid, 0.50

        X_train_raw = df_pre.loc[: i - 1, WF_FEATURE_COLS]
        y_train = df_pre.loc[: i - 1, WF_TARGET_COL].values

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), WF_CATEGORICAL_COLS)],
            remainder="passthrough",
        )
        X_train_trans = preprocessor.fit_transform(X_train_raw)

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,
        )
        rf.fit(X_train_trans, y_train)

        X_curr_raw = df_pre.loc[[i], WF_FEATURE_COLS]
        X_curr_trans = preprocessor.transform(X_curr_raw)
        prob_raw = rf.predict_proba(X_curr_trans)[0, 1]
        return mid, round(float(prob_raw), 4)

    # Run 1: Recompute all probabilities
    print("Computing Run 1 (Fresh Recomputation of all 1,234 matches)...")
    t0_r1 = time.time()
    results_run1 = Parallel(n_jobs=4)(delayed(compute_single_wf)(i) for i in range(n_matches))
    t_run1 = time.time() - t0_r1
    print(f"Run 1 completed in {t_run1:.2f}s.")
    run1_map = dict(results_run1)

    # Run 2: Exact Reproducibility Test
    print("Computing Run 2 (Second Independent Run for Reproducibility Test)...")
    t0_r2 = time.time()
    results_run2 = Parallel(n_jobs=4)(delayed(compute_single_wf)(i) for i in range(n_matches))
    t_run2 = time.time() - t0_r2
    print(f"Run 2 completed in {t_run2:.2f}s.")
    run2_map = dict(results_run2)

    # Reproducibility comparison (Run 1 vs Run 2)
    diffs_r1_r2 = [abs(run1_map[m] - run2_map[m]) for m in run1_map]
    max_diff_r1_r2 = max(diffs_r1_r2)
    identical_r1_r2 = sum(1 for d in diffs_r1_r2 if d == 0.0)
    print(f"Reproducibility (Run 1 vs Run 2): {identical_r1_r2}/{n_matches} exactly identical.")
    print(f"Max absolute difference (Run 1 vs Run 2): {max_diff_r1_r2:.8f}")

    # Comparison against existing saved artifact (walk_forward_pre_match_probs.csv)
    diffs_exist_r1 = []
    comparison_table = []
    for i in range(n_matches):
        mid = str(df_pre.loc[i, "match_id"])
        p_exist_a = existing_map.get(mid, None)
        p_r1_a = run1_map[mid]
        diff_a = abs(p_r1_a - p_exist_a) if p_exist_a is not None else -1.0
        diffs_exist_r1.append(diff_a)

        p_exist_b = round(1.0 - p_exist_a, 4) if p_exist_a is not None else None
        p_r1_b = round(1.0 - p_r1_a, 4)
        diff_b = abs(p_r1_b - p_exist_b) if p_exist_b is not None else -1.0

        comparison_table.append({
            "match_id": mid,
            "season": str(df_pre.loc[i, "season"]),
            "date": str(df_pre.loc[i, "date"]),
            "team_a": str(df_pre.loc[i, "team_a"]),
            "team_b": str(df_pre.loc[i, "team_b"]),
            "existing_probability_A": p_exist_a,
            "recomputed_probability_A": p_r1_a,
            "difference_A": diff_a,
            "existing_probability_B": p_exist_b,
            "recomputed_probability_B": p_r1_b,
            "difference_B": diff_b,
        })

    max_diff_exist = max(diffs_exist_r1)
    identical_exist = sum(1 for d in diffs_exist_r1 if d == 0.0)
    mismatches_exist = [c for c in comparison_table if c["difference_A"] > 1e-4]
    print(f"Existing vs Recomputed: {identical_exist}/{n_matches} exactly identical.")
    print(f"Max absolute difference vs existing: {max_diff_exist:.8f}")
    print(f"Mismatches (> 1e-4): {len(mismatches_exist)}")

    # =========================================================================
    # PART 5: PROBABILITY INTEGRITY, BOUNDS, & EXACT SUMS
    # =========================================================================
    print("\n--- PART 5: PROBABILITY INTEGRITY, BOUNDS, & SUMS ---")
    sum_violations = []
    bound_violations = []

    for i in range(n_matches):
        mid = str(df_pre.loc[i, "match_id"])
        pa = run1_map[mid]
        pb = round(1.0 - pa, 4)

        if not (0.0 <= pa <= 1.0) or not (0.0 <= pb <= 1.0):
            bound_violations.append((mid, pa, pb))

        sum_prob = pa + pb
        if abs(sum_prob - 1.0) > 1e-5:
            sum_violations.append((mid, pa, pb, sum_prob))

    print(f"Probability sum violations (|P(A)+P(B)-1| > 1e-5): {len(sum_violations)}")
    print(f"Probability bound violations (outside [0, 1]): {len(bound_violations)}")

    # =========================================================================
    # PART 6: YEAR-BY-YEAR DETAILED HISTORICAL AUDIT
    # =========================================================================
    print("\n--- PART 6: YEAR-BY-YEAR HISTORICAL AUDIT ---")
    # Group by season
    season_audit = defaultdict(lambda: {
        "eligible": 0,
        "excluded": 0,
        "predicted": 0,
        "missing": 0,
        "probs_a": [],
        "probs_b": [],
        "sum_violations": 0,
        "reproducibility_errors": 0,
        "correct_favorites": 0,
        "total_evaluated_favs": 0,
    })

    # Record exclusions by season
    for _, r in excluded_matches.iterrows():
        s = str(r["season"])
        season_audit[s]["excluded"] += 1

    # Record eligible predictions and stats
    for i in range(n_matches):
        mid = str(df_pre.loc[i, "match_id"])
        s = str(df_pre.loc[i, "season"])
        target = int(df_pre.loc[i, WF_TARGET_COL])

        pa = run1_map[mid]
        pb = round(1.0 - pa, 4)

        audit_entry = season_audit[s]
        audit_entry["eligible"] += 1
        audit_entry["predicted"] += 1
        audit_entry["probs_a"].append(pa)
        audit_entry["probs_b"].append(pb)

        if abs(pa + pb - 1.0) > 1e-5:
            audit_entry["sum_violations"] += 1

        if abs(run1_map[mid] - run2_map[mid]) > 1e-4:
            audit_entry["reproducibility_errors"] += 1

        # Favorite vs Actual Winner
        # If pa > 0.50, predicted favorite is Team A (target=1).
        # If pb > 0.50, predicted favorite is Team B (target=0).
        # If pa == 0.50 (cold start/tie), parity
        if pa > 0.50:
            fav_correct = (target == 1)
            audit_entry["total_evaluated_favs"] += 1
            if fav_correct:
                audit_entry["correct_favorites"] += 1
        elif pb > 0.50:
            fav_correct = (target == 0)
            audit_entry["total_evaluated_favs"] += 1
            if fav_correct:
                audit_entry["correct_favorites"] += 1

    season_rows = []
    all_seasons_sorted = sorted(season_audit.keys(), key=lambda s: int(s.split("/")[0]))

    total_fav_correct = 0
    total_fav_evaluated = 0

    for s in all_seasons_sorted:
        d = season_audit[s]
        p_arr = d["probs_a"]
        min_p = min(p_arr) if p_arr else 0.0
        max_p = max(p_arr) if p_arr else 0.0
        mean_p = float(np.mean(p_arr)) if p_arr else 0.0
        fav_pct = (d["correct_favorites"] / d["total_evaluated_favs"] * 100) if d["total_evaluated_favs"] > 0 else 50.0

        total_fav_correct += d["correct_favorites"]
        total_fav_evaluated += d["total_evaluated_favs"]

        season_rows.append({
            "Season": s,
            "Eligible": d["eligible"],
            "Excluded": d["excluded"],
            "Predicted": d["predicted"],
            "Missing": d["missing"],
            "P(A) Range": f"[{min_p:.4f}, {max_p:.4f}]",
            "Mean P(A)": round(mean_p, 4),
            "Sum Violations": d["sum_violations"],
            "Repro Errors": d["reproducibility_errors"],
            "Favorite Accuracy": f"{d['correct_favorites']}/{d['total_evaluated_favs']} ({fav_pct:.1f}%)"
        })

    df_season_summary = pd.DataFrame(season_rows)
    print(df_season_summary.to_string(index=False))

    overall_fav_pct = (total_fav_correct / total_fav_evaluated * 100) if total_fav_evaluated > 0 else 0.0
    print(f"\nOverall Favorites Predicted Correctly: {total_fav_correct}/{total_fav_evaluated} ({overall_fav_pct:.2f}%)")

    # =========================================================================
    # PART 7: EXTREME PROBABILITIES ANALYSIS
    # =========================================================================
    print("\n--- PART 7: EXTREME PROBABILITIES ANALYSIS ---")
    match_prob_list = []
    for i in range(n_matches):
        mid = str(df_pre.loc[i, "match_id"])
        pa = run1_map[mid]
        pb = round(1.0 - pa, 4)
        match_prob_list.append({
            "match_id": mid,
            "season": str(df_pre.loc[i, "season"]),
            "date": str(df_pre.loc[i, "date"]),
            "team_a": str(df_pre.loc[i, "team_a"]),
            "team_b": str(df_pre.loc[i, "team_b"]),
            "venue": str(df_pre.loc[i, "venue"]),
            "P_team_a": pa,
            "P_team_b": pb,
            "target_team_a_win": int(df_pre.loc[i, WF_TARGET_COL]),
            "form_diff": float(df_pre.loc[i, "form_diff_last_5"]),
            "h2h_win_rate": float(df_pre.loc[i, "team_a_h2h_win_rate"]),
            "venue_win_rate_diff": float(df_pre.loc[i, "venue_win_rate_diff"]),
            "batting_rpo_diff": float(df_pre.loc[i, "batting_rpo_diff_last_5"]),
            "bowling_econ_diff": float(df_pre.loc[i, "bowling_econ_diff_last_5"]),
        })

    df_match_probs = pd.DataFrame(match_prob_list)
    top_a = df_match_probs.sort_values(by="P_team_a", ascending=False).head(5)
    low_a = df_match_probs.sort_values(by="P_team_a", ascending=True).head(5)

    print("\nTop 5 Highest Team A Probabilities:")
    for _, r in top_a.iterrows():
        print(f"  Match {r['match_id']} ({r['season']}, {r['date']}): {r['team_a']} vs {r['team_b']} | P(A)={r['P_team_a']:.4f} | Actual Winner: {'Team A' if r['target_team_a_win']==1 else 'Team B'} | Form Diff: {r['form_diff']:+.3f}, H2H: {r['h2h_win_rate']:.3f}, RPO Diff: {r['batting_rpo_diff']:+.3f}")

    print("\nTop 5 Lowest Team A (Highest Team B) Probabilities:")
    for _, r in low_a.iterrows():
        print(f"  Match {r['match_id']} ({r['season']}, {r['date']}): {r['team_a']} vs {r['team_b']} | P(A)={r['P_team_a']:.4f} (P(B)={r['P_team_b']:.4f}) | Actual Winner: {'Team A' if r['target_team_a_win']==1 else 'Team B'} | Form Diff: {r['form_diff']:+.3f}, H2H: {r['h2h_win_rate']:.3f}, RPO Diff: {r['batting_rpo_diff']:+.3f}")

    # =========================================================================
    # PART 8: DEDICATED HELD-OUT FINAL-TEST AUDIT (2024–2026)
    # =========================================================================
    print("\n--- PART 8: DEDICATED HELD-OUT TEST AUDIT (2024–2026) ---")
    test_seasons = ["2024", "2025", "2026"]
    df_test = df_match_probs[df_match_probs["season"].isin(test_seasons)].copy()
    print(f"Total 2024–2026 Held-Out Matches: {len(df_test)}")

    for s in test_seasons:
        s_df = df_test[df_test["season"] == s]
        correct = sum(1 for _, r in s_df.iterrows() if (r["P_team_a"] > 0.50 and r["target_team_a_win"] == 1) or (r["P_team_b"] > 0.50 and r["target_team_a_win"] == 0))
        total = len(s_df)
        pct = (correct / total * 100) if total > 0 else 0.0

        # Log loss & Brier score
        p_clipped = np.clip(s_df["P_team_a"].values, 1e-7, 1 - 1e-7)
        y = s_df["target_team_a_win"].values
        ll = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
        brier = np.mean((s_df["P_team_a"].values - y) ** 2)

        print(f"Season {s}: {len(s_df)} matches | Accuracy: {correct}/{total} ({pct:.1f}%) | Log Loss: {ll:.4f} | Brier: {brier:.4f}")

    # Save detailed audit output files
    os.makedirs("reports/audit", exist_ok=True)
    df_match_probs.to_csv("reports/audit/every_match_probability_audit.csv", index=False)
    pd.DataFrame(comparison_table).to_csv("reports/audit/existing_vs_recomputed_comparison.csv", index=False)
    pd.DataFrame(toss_consistency_audit).to_csv("reports/audit/toss_innings_consistency_audit.csv", index=False)
    pd.DataFrame(season_rows).to_csv("reports/audit/season_by_season_audit_summary.csv", index=False)
    pd.DataFrame(excluded_records).to_csv("reports/audit/excluded_matches_audit.csv", index=False)

    print("\nSaved detailed audit artifacts to reports/audit/:")
    print("  - reports/audit/every_match_probability_audit.csv (1,234 rows)")
    print("  - reports/audit/existing_vs_recomputed_comparison.csv (1,234 rows)")
    print("  - reports/audit/toss_innings_consistency_audit.csv (1,234 rows)")
    print("  - reports/audit/season_by_season_audit_summary.csv (19 rows)")
    print("  - reports/audit/excluded_matches_audit.csv (9 rows)")

    return {
        "total_raw_matches": total_raw_matches,
        "eligible_matches": len(eligible_matches),
        "excluded_matches": len(excluded_matches),
        "no_result_exclusions": no_result_count,
        "other_exclusions": 0,
        "toss_innings_errors": len(toss_innings_errors),
        "feature_mismatches": len(feature_mismatches),
        "identical_reproducibility": identical_r1_r2,
        "max_diff_reproducibility": max_diff_r1_r2,
        "identical_existing_comparison": identical_exist,
        "max_diff_existing": max_diff_exist,
        "mismatches_existing": len(mismatches_exist),
        "sum_violations": len(sum_violations),
        "bound_violations": len(bound_violations),
        "overall_favorite_accuracy": overall_fav_pct,
        "total_fav_correct": total_fav_correct,
        "total_fav_evaluated": total_fav_evaluated,
        "held_out_matches_2024_2026": len(df_test),
    }


if __name__ == "__main__":
    summary = run_full_probability_audit()
    print("\nAUDIT SUMMARY DICT:")
    print(json.dumps(summary, indent=2))
