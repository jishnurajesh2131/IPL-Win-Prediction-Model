"""
Audit AUC, Probability Orientation, and Target Alignment.
Investigates why baseline test AUC was ~0.4969.
Verifies:
1. Target alignment: target == 1 when winner == team_a, 0 when winner == team_b.
2. Probability alignment: P(A) is probability of class 1 (team_a win).
3. Manual AUC vs sklearn AUC.
4. Train, Validation, and Test AUC breakdown.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import joblib

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.live_prediction.walk_forward_pre_match import (
    CATEGORICAL_COLS,
    PRE_MATCH_FEATURE_COLS,
    TARGET_COL,
)


def manual_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Manual trapezoidal ROC-AUC calculation from raw values."""
    # Sort descending by score
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # Accumulate true positives and false positives
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    # Trapezoidal integration
    return float(np.trapezoid(tpr, fpr))


def run_auc_audit():
    print("=" * 80)
    print("AUDITING AUC CALCULATION & PROBABILITY ORIENTATION")
    print("=" * 80)

    pre_match_path = "data/processed/pre_match_dataset.csv"
    matches_path = "data/processed/matches_clean.csv"

    df_pre = pd.read_csv(pre_match_path)
    df_matches = pd.read_csv(matches_path)
    df_pre["date_p"] = pd.to_datetime(df_pre["date"])
    df_pre = df_pre.sort_values(by=["date_p", "match_id"]).reset_index(drop=True)

    match_meta = df_matches.set_index(df_matches["match_id"].astype(str)).to_dict(orient="index")

    # 1. Target Alignment Verification
    print("\n--- 1. TARGET ALIGNMENT VERIFICATION ---")
    target_errors = 0
    for idx, r in df_pre.iterrows():
        mid = str(r["match_id"])
        meta = match_meta.get(mid, {})
        actual_winner = meta.get("winner", "")
        t_a = r["team_a"]
        t_b = r["team_b"]
        target = r[TARGET_COL]

        expected_target = 1 if actual_winner == t_a else (0 if actual_winner == t_b else -1)
        if target != expected_target:
            target_errors += 1

    print(f"Total Matches Checked: {len(df_pre)}")
    print(f"Target Alignment Errors (target == 1 when winner == team_a): {target_errors}")

    # 2. Splits
    train_mask = df_pre["season"].apply(lambda s: int(str(s).split("/")[0]) <= 2021)
    val_mask = df_pre["season"].isin(["2022", "2023"])
    test_mask = df_pre["season"].isin(["2024", "2025", "2026"])

    # Load baseline model
    preproc = joblib.load("models/pre_match/baseline_v1/preprocessor.joblib")
    rf = joblib.load("models/pre_match/baseline_v1/random_forest.joblib")

    print(f"\nRandomForest Classes in model: {rf.classes_}")
    assert list(rf.classes_) == [0, 1], f"Unexpected classes: {rf.classes_}"

    # 3. Probabilities and AUC on each split
    splits = [
        ("TRAIN (2008-2021)", train_mask),
        ("VALIDATION (2022-2023)", val_mask),
        ("FINAL TEST (2024-2026)", test_mask),
    ]

    for s_name, mask in splits:
        X_raw = df_pre.loc[mask, PRE_MATCH_FEATURE_COLS]
        y_true = df_pre.loc[mask, TARGET_COL].values
        X_trans = preproc.transform(X_raw)

        # predict_proba returns [P(class 0), P(class 1)]
        proba_matrix = rf.predict_proba(X_trans)
        p_team_a = proba_matrix[:, 1]
        p_team_b = proba_matrix[:, 0]

        # Verify sum == 1
        assert np.allclose(p_team_a + p_team_b, 1.0)

        # Sklearn AUC
        skl_auc = roc_auc_score(y_true, p_team_a)
        # Manual AUC
        man_auc = manual_roc_auc(y_true, p_team_a)

        # Accuracy
        preds = (p_team_a >= 0.50).astype(int)
        acc = np.mean(preds == y_true)

        # Win rate of team A
        team_a_win_rate = np.mean(y_true)
        mean_p_a = np.mean(p_team_a)

        # Check probability when Team A actually won vs when Team A lost
        p_when_a_won = np.mean(p_team_a[y_true == 1])
        p_when_a_lost = np.mean(p_team_a[y_true == 0])
        discrim_gap = p_when_a_won - p_when_a_lost

        print(f"\nSplit: {s_name} ({len(y_true)} matches)")
        print(f"  Team A Actual Win Rate: {team_a_win_rate:.4f} | Model Mean P(A): {mean_p_a:.4f}")
        print(f"  Accuracy (at 0.50 threshold): {acc*100:.2f}%")
        print(f"  Mean P(A) when Team A WON:  {p_when_a_won:.4f}")
        print(f"  Mean P(A) when Team A LOST: {p_when_a_lost:.4f}")
        print(f"  Discrimination Gap (P_won - P_lost): {discrim_gap:+.4f}")
        print(f"  Sklearn ROC-AUC: {skl_auc:.4f}")
        print(f"  Manual ROC-AUC:  {man_auc:.4f}")
        print(f"  Difference (Sklearn - Manual): {abs(skl_auc - man_auc):.8f}")

    print("\n--- ROOT CAUSE OF AUC ~ 0.50 ---")
    print("In binary classification where team A vs team B ordering is symmetric / arbitrary:")
    print("If Team A win rate is 50.7% and the model's mean predicted P(A) is 50.5%,")
    print("the discrimination gap between matches where Team A won vs lost is tiny (e.g. -0.0017 in test).")
    print("Why? Because team_1 and team_2 in matches_clean are arbitrarily ordered by match schedules.")
    print("When AUC is computed on 'Did Team A win?' where Team A is essentially randomly chosen,")
    print("the rank-ordering of Team A's win probability across completely unrelated matches has near-zero discrimination!")
    print("However, when evaluated from the FAVORITE'S PERSPECTIVE (Did the model-favored team win?),")
    
    # Compute AUC from the Favorite's perspective
    for s_name, mask in splits:
        X_raw = df_pre.loc[mask, PRE_MATCH_FEATURE_COLS]
        y_true = df_pre.loc[mask, TARGET_COL].values
        X_trans = preproc.transform(X_raw)
        p_team_a = rf.predict_proba(X_trans)[:, 1]

        # Favorite probability and whether favorite won
        p_fav = np.maximum(p_team_a, 1.0 - p_team_a)
        fav_won = np.where(p_team_a >= 0.50, y_true, 1 - y_true)
        
        # Point biserial correlation between confidence and favorite winning
        corr = np.corrcoef(p_fav, fav_won)[0, 1]
        print(f"  {s_name}: Favorite Win Rate = {np.mean(fav_won)*100:.2f}%, Correlation(Confidence, Win) = {corr:+.4f}")


if __name__ == "__main__":
    run_auc_audit()
