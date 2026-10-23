"""
Probability Calibration Analysis and Reliability Diagrams.
Evaluates the alignment between predicted probabilities and empirical match outcomes.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd


def compute_calibration_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute calibration data across n_bins equal-width bins in [0, 1].

    Returns:
    - bin_df: DataFrame with columns [bin_range, count, mean_predicted_prob, empirical_win_rate, calibration_error]
    - summary_metrics: dict with ECE (Expected Calibration Error) and MCE (Maximum Calibration Error)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_records = []

    total_samples = len(y_true)
    weighted_error_sum = 0.0
    max_error = 0.0

    for i in range(n_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]

        # For the last bin, include the upper edge
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
            bin_label = f"[{low:.1f}, {high:.1f}]"
        else:
            mask = (y_prob >= low) & (y_prob < high)
            bin_label = f"[{low:.1f}, {high:.1f})"

        bin_count = int(np.sum(mask))

        if bin_count > 0:
            mean_pred = float(np.mean(y_prob[mask]))
            emp_rate = float(np.mean(y_true[mask]))
            cal_err = abs(emp_rate - mean_pred)
            weighted_error_sum += bin_count * cal_err
            if cal_err > max_error:
                max_error = cal_err
        else:
            mean_pred = np.nan
            emp_rate = np.nan
            cal_err = np.nan

        bin_records.append({
            "bin_idx": i + 1,
            "bin_range": bin_label,
            "count": bin_count,
            "pct_of_matches": round((bin_count / total_samples) * 100, 2) if total_samples > 0 else 0.0,
            "mean_pred_prob": round(mean_pred, 4) if not np.isnan(mean_pred) else None,
            "empirical_win_rate": round(emp_rate, 4) if not np.isnan(emp_rate) else None,
            "calibration_error": round(cal_err, 4) if not np.isnan(cal_err) else None,
        })

    bin_df = pd.DataFrame(bin_records)
    ece = (weighted_error_sum / total_samples) if total_samples > 0 else 0.0

    summary = {
        "expected_calibration_error_ece": round(ece, 4),
        "max_calibration_error_mce": round(max_error, 4),
    }

    return bin_df, summary


def assess_calibration_posture(bin_df: pd.DataFrame, ece: float) -> str:
    """
    Assess whether probability posture is well-calibrated, under-confident, or over-confident.
    """
    valid_bins = bin_df.dropna(subset=["mean_pred_prob", "empirical_win_rate"])
    if len(valid_bins) == 0:
        return "Insufficient data for posture assessment"

    # Count cases where extreme predictions deviate toward or away from 0.5
    high_bins = valid_bins[valid_bins["mean_pred_prob"] > 0.5]
    low_bins = valid_bins[valid_bins["mean_pred_prob"] < 0.5]

    overconfidence_votes = 0
    underconfidence_votes = 0

    for _, row in high_bins.iterrows():
        # High prediction: if empirical < predicted -> over-confident
        if row["empirical_win_rate"] < row["mean_pred_prob"]:
            overconfidence_votes += 1
        elif row["empirical_win_rate"] > row["mean_pred_prob"]:
            underconfidence_votes += 1

    for _, row in low_bins.iterrows():
        # Low prediction: if empirical > predicted -> over-confident
        if row["empirical_win_rate"] > row["mean_pred_prob"]:
            overconfidence_votes += 1
        elif row["empirical_win_rate"] < row["mean_pred_prob"]:
            underconfidence_votes += 1

    posture_desc = []
    if ece < 0.05:
        posture_desc.append("Well-calibrated (ECE < 5%)")
    elif ece < 0.10:
        posture_desc.append("Moderately calibrated (ECE 5–10%)")
    else:
        posture_desc.append("Poorly calibrated (ECE > 10%)")

    if overconfidence_votes > underconfidence_votes + 1:
        posture_desc.append("Leaning slightly over-confident (predictions slightly more extreme than actual frequency)")
    elif underconfidence_votes > overconfidence_votes + 1:
        posture_desc.append("Leaning slightly under-confident (predictions clustered closer to 0.5 than actual frequency)")
    else:
        posture_desc.append("Balanced probability dispersion around empirical base rates")

    return "; ".join(posture_desc)
