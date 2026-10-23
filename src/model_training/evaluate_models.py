"""
Model Evaluation Metrics: Classification Performance & Probability Quality.
Implements Accuracy, Precision, Recall, F1, Log Loss, Brier Score, and Calibration Metrics.
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    brier_score_loss,
)
from src.model_training.calibration import compute_calibration_data, assess_calibration_posture


def compute_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute full suite of classification and probability-quality metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Clip probabilities to prevent numerical instability in log_loss
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)

    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    ll = log_loss(y_true, y_prob_clipped)
    brier = brier_score_loss(y_true, y_prob)

    bin_df, cal_summary = compute_calibration_data(y_true, y_prob, n_bins=10)
    posture = assess_calibration_posture(bin_df, cal_summary["expected_calibration_error_ece"])

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "log_loss": round(float(ll), 4),
        "brier_score": round(float(brier), 4),
        "ece": cal_summary["expected_calibration_error_ece"],
        "mce": cal_summary["max_calibration_error_mce"],
        "calibration_posture": posture,
        "calibration_bins": bin_df,
    }
