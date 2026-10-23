"""
Comprehensive Model Evaluation Suite for Live Win Probability Prediction.

Calculates multi-dimensional probability quality metrics:
- Log Loss, Brier Score, Expected Calibration Error (ECE)
- Accuracy, ROC-AUC
- Reliability diagrams & 10-bin calibration tables
- Match phase progression metrics (Powerplay, Middle, Death, and 6 finer bins)
- Innings 1 vs Innings 2 breakdown
- Per-match aggregated metrics (to normalize for match length)
- Permutation feature importance on validation set
- Qualitative historical match sanity case extraction
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Computes Expected Calibration Error (ECE) and bin-level reliability data.
    Uses n_bins uniform probability bins: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    N = len(y_true)
    if N == 0:
        return 0.0, []

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_records = []

    for b in range(n_bins):
        b_low = bin_boundaries[b]
        b_high = bin_boundaries[b + 1]

        if b == n_bins - 1:
            in_bin = (y_prob >= b_low) & (y_prob <= b_high)
        else:
            in_bin = (y_prob >= b_low) & (y_prob < b_high)

        count = int(np.sum(in_bin))
        if count > 0:
            mean_prob = float(np.mean(y_prob[in_bin]))
            observed_acc = float(np.mean(y_true[in_bin]))
            abs_diff = abs(observed_acc - mean_prob)
            ece += (count / N) * abs_diff
        else:
            mean_prob = float((b_low + b_high) / 2.0)
            observed_acc = float("nan")
            abs_diff = 0.0

        bin_records.append(
            {
                "bin_index": b + 1,
                "bin_range": f"[{b_low:.1f}, {b_high:.1f}]",
                "count": count,
                "fraction_of_total": round(count / N, 4) if N > 0 else 0.0,
                "mean_predicted_prob": round(mean_prob, 4),
                "observed_frequency": round(observed_acc, 4) if not np.isnan(observed_acc) else None,
                "calibration_error": round(abs_diff, 4),
            }
        )

    return round(float(ece), 4), bin_records


def compute_standard_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Computes Log Loss, Brier Score, ECE, Accuracy, and ROC-AUC.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1.0 - 1e-15)

    ll = float(log_loss(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    ece, _ = compute_expected_calibration_error(y_true, y_prob)

    preds_binary = (y_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, preds_binary))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.50

    return {
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
    }


def compute_phase_evaluation(
    df_eval: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluates probability quality across match phases:
    1. Standard 3 Phases:
       - Powerplay (overs 0-6)
       - Middle (overs 6-15)
       - Death (overs 15-20)
    2. Finer 6 Progression Bins:
       - 0-2 overs, 2-6 overs, 6-10 overs, 10-15 overs, 15-18 overs, 18-20 overs
    """
    df_temp = df_eval[["match_id", "overs_completed"]].copy()
    df_temp["y_true"] = y_true
    df_temp["y_prob"] = y_prob

    standard_phases = [
        ("Powerplay (0-6 overs)", 0.0, 6.0),
        ("Middle (6-15 overs)", 6.0, 15.0),
        ("Death (15-20 overs)", 15.0, 20.5),
    ]

    finer_bins = [
        ("0-2 overs", 0.0, 2.0),
        ("2-6 overs", 2.0, 6.0),
        ("6-10 overs", 6.0, 10.0),
        ("10-15 overs", 10.0, 15.0),
        ("15-18 overs", 15.0, 18.0),
        ("18-20 overs", 18.0, 20.5),
    ]

    results_standard = []
    for phase_name, low_o, high_o in standard_phases:
        sub = df_temp[(df_temp["overs_completed"] >= low_o) & (df_temp["overs_completed"] < high_o)]
        if len(sub) > 0:
            m = compute_standard_metrics(sub["y_true"].values, sub["y_prob"].values)
            results_standard.append(
                {
                    "phase": phase_name,
                    "rows": len(sub),
                    "matches": sub["match_id"].nunique(),
                    **m,
                }
            )

    results_finer = []
    for bin_name, low_o, high_o in finer_bins:
        sub = df_temp[(df_temp["overs_completed"] >= low_o) & (df_temp["overs_completed"] < high_o)]
        if len(sub) > 0:
            m = compute_standard_metrics(sub["y_true"].values, sub["y_prob"].values)
            results_finer.append(
                {
                    "phase": bin_name,
                    "rows": len(sub),
                    "matches": sub["match_id"].nunique(),
                    **m,
                }
            )

    return {
        "standard_phases": results_standard,
        "finer_progression_bins": results_finer,
    }


def compute_innings_breakdown(
    df_eval: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluates probability metrics separately for Innings 1 and Innings 2.
    """
    df_temp = df_eval[["match_id", "current_innings"]].copy()
    df_temp["y_true"] = y_true
    df_temp["y_prob"] = y_prob

    results = {}
    for inn in [1, 2]:
        sub = df_temp[df_temp["current_innings"] == inn]
        if len(sub) > 0:
            m = compute_standard_metrics(sub["y_true"].values, sub["y_prob"].values)
            results[f"innings_{inn}"] = {
                "rows": len(sub),
                "matches": sub["match_id"].nunique(),
                **m,
            }
    return results


def compute_per_match_metrics(
    df_eval: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculates delivery metrics grouped by match_id to assess per-match distributions.
    Prevents long matches from mathematically dominating the score.
    """
    df_temp = df_eval[["match_id"]].copy()
    df_temp["y_true"] = y_true
    df_temp["y_prob"] = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    def match_stats(g):
        y_t = g["y_true"].values
        y_p = g["y_prob"].values
        ll = log_loss(y_t, y_p, labels=[0, 1])
        brier = brier_score_loss(y_t, y_p)
        return pd.Series({"match_log_loss": ll, "match_brier": brier, "match_rows": len(g)})

    match_summary = df_temp.groupby("match_id").apply(match_stats).reset_index()

    return {
        "match_count": len(match_summary),
        "mean_match_log_loss": round(float(match_summary["match_log_loss"].mean()), 4),
        "median_match_log_loss": round(float(match_summary["match_log_loss"].median()), 4),
        "std_match_log_loss": round(float(match_summary["match_log_loss"].std()), 4),
        "mean_match_brier": round(float(match_summary["match_brier"].mean()), 4),
        "median_match_brier": round(float(match_summary["match_brier"].median()), 4),
        "std_match_brier": round(float(match_summary["match_brier"].std()), 4),
    }


def extract_qualitative_match_examples(
    df_eval: pd.DataFrame,
    y_prob: np.ndarray,
    target_match_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts representative qualitative trajectory samples for key historical matches.
    """
    df_temp = df_eval.copy()
    df_temp["predicted_team_a_win_probability"] = np.round(y_prob, 4)

    # If specific match IDs not requested, pick 5 diverse matches
    if not target_match_ids:
        target_match_ids = list(df_temp["match_id"].unique()[:5])

    samples = []
    for mid in target_match_ids:
        sub = df_temp[df_temp["match_id"].astype(str) == str(mid)]
        if sub.empty:
            continue

        team_a = sub.iloc[0]["team_a"]
        team_b = sub.iloc[0]["team_b"]
        season = sub.iloc[0]["season"]
        target = sub.iloc[0]["target_runs"]
        actual_winner_team_a = int(sub.iloc[0]["target_team_a_win"])

        # Pick 6 evenly spaced deliveries across match progression
        step = max(1, len(sub) // 6)
        checkpoints = sub.iloc[::step].head(6)

        deliveries_info = []
        for _, row in checkpoints.iterrows():
            deliveries_info.append(
                {
                    "delivery_index": int(row["delivery_index"]),
                    "innings": int(row["current_innings"]),
                    "ball": str(row["ball"]),
                    "score": f"{row['current_innings_runs']}/{row['current_innings_wickets']}",
                    "balls_remaining": int(row["legal_balls_remaining"]),
                    "runs_needed": int(row["runs_needed"]),
                    "required_run_rate": round(float(row["required_run_rate"]), 2),
                    "pre_match_prob_team_a": round(float(row["pre_match_prob_team_a"]), 4),
                    "predicted_prob_team_a": round(float(row["predicted_team_a_win_probability"]), 4),
                }
            )

        samples.append(
            {
                "match_id": str(mid),
                "season": str(season),
                "team_a": team_a,
                "team_b": team_b,
                "actual_team_a_win": actual_winner_team_a,
                "target_runs": int(target),
                "deliveries": deliveries_info,
            }
        )

    return samples
