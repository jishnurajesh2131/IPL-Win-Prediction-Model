"""
Validation Script: Probability Quality & Metric Replication on Test Split.

Evaluates the frozen Two-Model XGBoost (Feature Set D) pipeline on the active
held-out Test split (2024-2026, N = 50,867) to confirm exact alignment with Step 3 report:
- Master metrics: Log Loss, Brier Score, ECE, Accuracy, ROC-AUC
- Innings 1 vs Innings 2 metrics
- Standard 3-Phase metrics: Powerplay (0-6), Middle (6-15), Death (15-20)
- Finer 6-Phase progression: 0-2, 2-6, 6-10, 10-15, 15-18, 18-20
- Match-level aggregated metrics
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, roc_auc_score

from src.live_prediction.live_model_evaluator import (
    compute_expected_calibration_error,
    compute_standard_metrics,
    compute_phase_evaluation,
    compute_innings_breakdown,
    compute_per_match_metrics,
)


def run_probability_quality_validation():
    print("=" * 70)
    print("PHASE 5 STEP 5: PROBABILITY QUALITY & PHASE METRIC VALIDATION")
    print("=" * 70)

    print("Loading test split from live_dataset.csv...")
    df = pd.read_csv("data/processed/live_dataset.csv", low_memory=False)

    # Chronological match partitioning matching Step 3
    train_seasons = [str(y) for y in range(2008, 2022)] + ["2007/08", "2009/10", "2020/21"]
    val_seasons = ["2022", "2023"]
    test_seasons = ["2024", "2025", "2026"]

    # Exclude terminal states (1,199 rows) as established in Step 3
    is_terminal = (
        (df["current_innings"] == 2) &
        (
            (df["runs_needed"] == 0) |
            (df["wickets_remaining"] == 0) |
            (df["legal_balls_remaining"] == 0)
        )
    )
    df_active = df[~is_terminal].copy()

    df_test = df_active[df_active["season"].astype(str).isin(test_seasons)].copy()
    print(f"Active Test Deliveries: {len(df_test):,} across {df_test['match_id'].nunique()} matches")

    # Load frozen Two-Model pipeline
    model_inn1 = joblib.load("models/live/primary_live_model_inn1.joblib")
    model_inn2 = joblib.load("models/live/primary_live_model_inn2.joblib")
    prep_inn1 = joblib.load("models/live/primary_live_preprocessor_inn1.joblib")
    prep_inn2 = joblib.load("models/live/primary_live_preprocessor_inn2.joblib")

    # Split Innings 1 and 2
    test_inn1 = df_test[df_test["current_innings"] == 1]
    test_inn2 = df_test[df_test["current_innings"] == 2]

    # Predict Innings 1
    X1_trans = prep_inn1.transform(test_inn1)
    probs_inn1 = model_inn1.predict_proba(X1_trans)[:, 1]

    # Predict Innings 2
    X2_trans = prep_inn2.transform(test_inn2)
    probs_inn2 = model_inn2.predict_proba(X2_trans)[:, 1]

    # Combine predictions
    df_test.loc[test_inn1.index, "pred_prob"] = probs_inn1
    df_test.loc[test_inn2.index, "pred_prob"] = probs_inn2

    y_true_all = df_test["target_team_a_win"].values
    y_pred_all = df_test["pred_prob"].values

    # Overall Test Metrics
    overall_metrics = compute_standard_metrics(y_true_all, y_pred_all)
    print("\nOVERALL TEST METRICS (N = 50,867):")
    for k, v in overall_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Innings-Specific Metrics
    inn1_metrics = compute_standard_metrics(test_inn1["target_team_a_win"].values, probs_inn1)
    inn2_metrics = compute_standard_metrics(test_inn2["target_team_a_win"].values, probs_inn2)
    print("\nINNINGS 1 METRICS (N = 26,558):")
    print(f"  Log Loss: {inn1_metrics['log_loss']:.4f} | Brier: {inn1_metrics['brier_score']:.4f} | ECE: {inn1_metrics['ece']:.4f} | Acc: {inn1_metrics['accuracy'] * 100:.2f}% | AUC: {inn1_metrics['roc_auc']:.4f}")
    print("\nINNINGS 2 METRICS (N = 24,309):")
    print(f"  Log Loss: {inn2_metrics['log_loss']:.4f} | Brier: {inn2_metrics['brier_score']:.4f} | ECE: {inn2_metrics['ece']:.4f} | Acc: {inn2_metrics['accuracy'] * 100:.2f}% | AUC: {inn2_metrics['roc_auc']:.4f}")

    # Standard 3-Phase Breakdown
    pp_mask = df_test["overs_completed"] <= 6.0
    mid_mask = (df_test["overs_completed"] > 6.0) & (df_test["overs_completed"] <= 15.0)
    death_mask = df_test["overs_completed"] > 15.0

    pp_metrics = compute_standard_metrics(df_test.loc[pp_mask, "target_team_a_win"].values, df_test.loc[pp_mask, "pred_prob"].values)
    mid_metrics = compute_standard_metrics(df_test.loc[mid_mask, "target_team_a_win"].values, df_test.loc[mid_mask, "pred_prob"].values)
    death_metrics = compute_standard_metrics(df_test.loc[death_mask, "target_team_a_win"].values, df_test.loc[death_mask, "pred_prob"].values)

    print("\n3-PHASE BREAKDOWN:")
    print(f"  Powerplay (0-6 overs, N={pp_mask.sum():,}): Log Loss: {pp_metrics['log_loss']:.4f} | Brier: {pp_metrics['brier_score']:.4f} | ECE: {pp_metrics['ece']:.4f} | Acc: {pp_metrics['accuracy']*100:.2f}% | AUC: {pp_metrics['roc_auc']:.4f}")
    print(f"  Middle (6-15 overs, N={mid_mask.sum():,}):   Log Loss: {mid_metrics['log_loss']:.4f} | Brier: {mid_metrics['brier_score']:.4f} | ECE: {mid_metrics['ece']:.4f} | Acc: {mid_metrics['accuracy']*100:.2f}% | AUC: {mid_metrics['roc_auc']:.4f}")
    print(f"  Death (15-20 overs, N={death_mask.sum():,}):   Log Loss: {death_metrics['log_loss']:.4f} | Brier: {death_metrics['brier_score']:.4f} | ECE: {death_metrics['ece']:.4f} | Acc: {death_metrics['accuracy']*100:.2f}% | AUC: {death_metrics['roc_auc']:.4f}")

    # Finer 6-Phase Progression
    p_ranges = [
        ("0–2 overs", (df_test["overs_completed"] >= 0.0) & (df_test["overs_completed"] <= 2.0)),
        ("2–6 overs", (df_test["overs_completed"] > 2.0) & (df_test["overs_completed"] <= 6.0)),
        ("6–10 overs", (df_test["overs_completed"] > 6.0) & (df_test["overs_completed"] <= 10.0)),
        ("10–15 overs", (df_test["overs_completed"] > 10.0) & (df_test["overs_completed"] <= 15.0)),
        ("15–18 overs", (df_test["overs_completed"] > 15.0) & (df_test["overs_completed"] <= 18.0)),
        ("18–20 overs", (df_test["overs_completed"] > 18.0) & (df_test["overs_completed"] <= 20.0)),
    ]
    six_phase_rows = []
    print("\n6-PHASE PROGRESSION:")
    for label, mask in p_ranges:
        sub_y = df_test.loc[mask, "target_team_a_win"].values
        sub_p = df_test.loc[mask, "pred_prob"].values
        sub_m = compute_standard_metrics(sub_y, sub_p)
        six_phase_rows.append({
            "Window": label,
            "Deliveries": int(mask.sum()),
            "Matches": int(df_test.loc[mask, "match_id"].nunique()),
            "Log Loss": round(float(sub_m["log_loss"]), 4),
            "Brier Score": round(float(sub_m["brier_score"]), 4),
            "ECE": round(float(sub_m["ece"]), 4),
            "Accuracy": f"{sub_m['accuracy'] * 100:.2f}%",
            "ROC-AUC": round(float(sub_m["roc_auc"]), 4),
        })
        print(f"  {label:<12} (N={mask.sum():>5}): Log Loss: {sub_m['log_loss']:.4f} | Brier: {sub_m['brier_score']:.4f} | ECE: {sub_m['ece']:.4f} | Acc: {sub_m['accuracy']*100:.2f}% | AUC: {sub_m['roc_auc']:.4f}")

    # Match-Level Aggregation
    match_grouped = df_test.groupby("match_id").agg(
        mean_p=("pred_prob", "mean"),
        target=("target_team_a_win", "first")
    )
    match_ll = log_loss(match_grouped["target"], match_grouped["mean_p"])
    match_brier = brier_score_loss(match_grouped["target"], match_grouped["mean_p"])
    print("\nMATCH-LEVEL AGGREGATED METRICS (N = 213 test matches):")
    print(f"  Match-Level Mean Log Loss: {match_ll:.4f}")
    print(f"  Match-Level Mean Brier: {match_brier:.4f}")

    results_out = {
        "overall_test": overall_metrics,
        "innings_1": inn1_metrics,
        "innings_2": inn2_metrics,
        "three_phase": {"powerplay": pp_metrics, "middle": mid_metrics, "death": death_metrics},
        "six_phase": six_phase_rows,
        "match_level": {"mean_log_loss": round(float(match_ll), 4), "mean_brier": round(float(match_brier), 4)},
    }
    with open("scratch/prob_quality_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_out, f, indent=2)
    print("\nSaved prob_quality_metrics.json successfully.")


if __name__ == "__main__":
    run_probability_quality_validation()
