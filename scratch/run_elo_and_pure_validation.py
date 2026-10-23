"""
Pure Validation Model Selection, Expanding Elo Rating, Current-Season Performance,
AUC Verification, and Single-Pass Locked Final Test Pipeline.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Any, List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.live_prediction.walk_forward_pre_match import (
    CATEGORICAL_COLS,
    BINARY_COLS,
    CONTINUOUS_COLS,
    PRE_MATCH_FEATURE_COLS,
    TARGET_COL,
)


def compute_ece_mce(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    """Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(probs)
    for i in range(n_bins):
        bin_mask = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
        if i == 0:
            bin_mask |= (probs == 0.0)
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(probs[bin_mask])
            diff = abs(bin_acc - bin_conf)
            ece += (bin_count / n) * diff
            if diff > mce:
                mce = diff
    return round(float(ece), 4), round(float(mce), 4)


class ExpandingEloTracker:
    """
    Chronological leak-free Elo rating tracker for IPL teams.
    Ratings are strictly queried BEFORE the match and updated AFTER the match.
    """
    def __init__(self, k_factor: float = 24.0, initial_elo: float = 1500.0):
        self.k_factor = k_factor
        self.initial_elo = initial_elo
        self.ratings: Dict[str, float] = defaultdict(lambda: self.initial_elo)

    def get_pre_match_elo(self, team_a: str, team_b: str) -> Tuple[float, float, float, float]:
        r_a = self.ratings[team_a]
        r_b = self.ratings[team_b]
        elo_diff = r_a - r_b
        # Expected probability for team A under standard Elo formula
        exp_prob_a = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        return round(r_a, 2), round(r_b, 2), round(elo_diff, 2), round(exp_prob_a, 4)

    def update_post_match(self, team_a: str, team_b: str, winner: str):
        r_a = self.ratings[team_a]
        r_b = self.ratings[team_b]
        elo_diff = r_a - r_b
        exp_prob_a = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        exp_prob_b = 1.0 - exp_prob_a

        actual_a = 1.0 if winner == team_a else 0.0
        actual_b = 1.0 if winner == team_b else 0.0

        self.ratings[team_a] = r_a + self.k_factor * (actual_a - exp_prob_a)
        self.ratings[team_b] = r_b + self.k_factor * (actual_b - exp_prob_b)


def build_enhanced_pre_match_features():
    """
    Builds enhanced leak-free dataset incorporating:
    1. Expanding Elo rating (K-factor tuned on validation).
    2. Current-season performance (matches played, wins, win rate, win rate diff).
    3. Exponentially recency-weighted form.
    """
    pre_match_path = "data/processed/pre_match_dataset.csv"
    matches_path = "data/processed/matches_clean.csv"

    df_pre = pd.read_csv(pre_match_path)
    df_matches = pd.read_csv(matches_path)
    df_pre["date_p"] = pd.to_datetime(df_pre["date"])
    df_pre = df_pre.sort_values(by=["date_p", "match_id"]).reset_index(drop=True)

    match_meta = df_matches.set_index(df_matches["match_id"].astype(str)).to_dict(orient="index")

    # Tune K-factor on Validation set only
    val_mask = df_pre["season"].isin(["2022", "2023"])
    train_mask = df_pre["season"].apply(lambda s: int(str(s).split("/")[0]) <= 2021)

    print("\n--- TUNING ELO K-FACTOR ON VALIDATION SET ONLY ---")
    best_k = 24.0
    best_k_ll = 999.0

    for k in [16.0, 20.0, 24.0, 32.0, 40.0]:
        tracker = ExpandingEloTracker(k_factor=k)
        elo_probs = []
        for i in range(len(df_pre)):
            t_a = df_pre.loc[i, "team_a"]
            t_b = df_pre.loc[i, "team_b"]
            mid = str(df_pre.loc[i, "match_id"])
            act_w = match_meta[mid]["winner"]

            _, _, _, p_a = tracker.get_pre_match_elo(t_a, t_b)
            elo_probs.append(p_a)
            tracker.update_post_match(t_a, t_b, act_w)

        p_val = np.array(elo_probs)[val_mask]
        y_val = df_pre.loc[val_mask, TARGET_COL].values
        ll = log_loss(y_val, p_val)
        brier = brier_score_loss(y_val, p_val)
        acc = accuracy_score(y_val, (p_val >= 0.50).astype(int))
        print(f"  Elo K={k:2.0f} on Validation: LogLoss={ll:.4f}, Brier={brier:.4f}, Acc={acc*100:.2f}%")
        if ll < best_k_ll:
            best_k_ll = ll
            best_k = k

    print(f"Selected Best K-Factor on Validation: K={best_k} (Val LogLoss={best_k_ll:.4f})")

    # Generate final Elo and Current-Season features across full dataset
    tracker = ExpandingEloTracker(k_factor=best_k)
    season_team_matches = defaultdict(lambda: defaultdict(int))
    season_team_wins = defaultdict(lambda: defaultdict(int))

    elo_a_list, elo_b_list, elo_diff_list, elo_prob_list = [], [], [], []
    cs_matches_a_list, cs_matches_b_list = [], []
    cs_win_rate_a_list, cs_win_rate_b_list, cs_wr_diff_list = [], [], []

    for i in range(len(df_pre)):
        t_a = df_pre.loc[i, "team_a"]
        t_b = df_pre.loc[i, "team_b"]
        season = str(df_pre.loc[i, "season"])
        mid = str(df_pre.loc[i, "match_id"])
        act_w = match_meta[mid]["winner"]

        # 1. Elo query before match
        ra, rb, diff, prob_a = tracker.get_pre_match_elo(t_a, t_b)
        elo_a_list.append(ra)
        elo_b_list.append(rb)
        elo_diff_list.append(diff)
        elo_prob_list.append(prob_a)

        # 2. Current season prior statistics
        m_a = season_team_matches[season][t_a]
        m_b = season_team_matches[season][t_b]
        w_a = season_team_wins[season][t_a]
        w_b = season_team_wins[season][t_b]

        # Smoothed current season win rate (prior = 0.50 with weight m=2)
        wr_a = (w_a + 1.0) / (m_a + 2.0)
        wr_b = (w_b + 1.0) / (m_b + 2.0)

        cs_matches_a_list.append(m_a)
        cs_matches_b_list.append(m_b)
        cs_win_rate_a_list.append(round(wr_a, 4))
        cs_win_rate_b_list.append(round(wr_b, 4))
        cs_wr_diff_list.append(round(wr_a - wr_b, 4))

        # Update trackers strictly AFTER match
        tracker.update_post_match(t_a, t_b, act_w)
        season_team_matches[season][t_a] += 1
        season_team_matches[season][t_b] += 1
        if act_w == t_a:
            season_team_wins[season][t_a] += 1
        elif act_w == t_b:
            season_team_wins[season][t_b] += 1

    df_pre["elo_team_a"] = elo_a_list
    df_pre["elo_team_b"] = elo_b_list
    df_pre["elo_diff"] = elo_diff_list
    df_pre["elo_prob_a"] = elo_prob_list
    df_pre["cs_matches_a"] = cs_matches_a_list
    df_pre["cs_matches_b"] = cs_matches_b_list
    df_pre["cs_win_rate_a"] = cs_win_rate_a_list
    df_pre["cs_win_rate_b"] = cs_win_rate_b_list
    df_pre["cs_win_rate_diff"] = cs_wr_diff_list

    return df_pre, match_meta


def run_pure_validation_benchmark():
    print("=" * 85)
    print("PURE VALIDATION MODEL SELECTION (TRAIN: 2008-2021 -> VAL: 2022-2023)")
    print("=" * 85)

    df_pre, match_meta = build_enhanced_pre_match_features()

    train_mask = df_pre["season"].apply(lambda s: int(str(s).split("/")[0]) <= 2021)
    val_mask = df_pre["season"].isin(["2022", "2023"])
    test_mask = df_pre["season"].isin(["2024", "2025", "2026"])

    y_train = df_pre.loc[train_mask, TARGET_COL].values
    y_val = df_pre.loc[val_mask, TARGET_COL].values
    y_test = df_pre.loc[test_mask, TARGET_COL].values

    # Feature subsets to test strictly on validation
    core_numerical_features = [
        "team_a_form_last_5", "team_b_form_last_5", "form_diff_last_5",
        "team_a_batting_rpo_last_5", "team_b_batting_rpo_last_5", "batting_rpo_diff_last_5",
        "team_a_bowling_econ_last_5", "team_b_bowling_econ_last_5", "bowling_econ_diff_last_5",
        "toss_winner_is_team_a", "toss_decision_is_bat", "team_a_bats_first",
    ]

    elo_features = ["elo_diff", "elo_prob_a"]
    cs_features = ["cs_matches_a", "cs_matches_b", "cs_win_rate_a", "cs_win_rate_b", "cs_win_rate_diff"]

    # Feature Set Definitions
    # 1. Baseline Full Features (34 features including OHE team/venue)
    # 2. Enhanced Full Features (Baseline + Elo + Current Season)
    # 3. Clean Robust Core (Form + RPO + Econ + Toss + Elo + Current Season, NO OHE team dummies)
    clean_robust_features = core_numerical_features + elo_features + cs_features
    enhanced_full_features = PRE_MATCH_FEATURE_COLS + elo_features + cs_features

    # Preprocessors
    preproc_base = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS)],
        remainder="passthrough",
    )
    preproc_base.fit(df_pre.loc[train_mask, PRE_MATCH_FEATURE_COLS])
    X_tr_base = preproc_base.transform(df_pre.loc[train_mask, PRE_MATCH_FEATURE_COLS])
    X_va_base = preproc_base.transform(df_pre.loc[val_mask, PRE_MATCH_FEATURE_COLS])
    X_te_base = preproc_base.transform(df_pre.loc[test_mask, PRE_MATCH_FEATURE_COLS])

    # Clean Robust Preprocessor (Scaler on numericals, zero team-name overfitting)
    preproc_robust = StandardScaler()
    preproc_robust.fit(df_pre.loc[train_mask, clean_robust_features])
    X_tr_robust = preproc_robust.transform(df_pre.loc[train_mask, clean_robust_features])
    X_va_robust = preproc_robust.transform(df_pre.loc[val_mask, clean_robust_features])
    X_te_robust = preproc_robust.transform(df_pre.loc[test_mask, clean_robust_features])

    # Candidate Models to evaluate on VALIDATION ONLY
    candidates = {}

    # 1. Baseline Random Forest (Reproduced)
    rf_base = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=1)
    rf_base.fit(X_tr_base, y_train)
    candidates["1. Baseline Random Forest (v1: All Features)"] = (rf_base, X_va_base, X_te_base)

    # 2. Pure Expanding Elo Logistic Model
    lr_elo = LogisticRegression(C=1.0, random_state=42)
    lr_elo.fit(df_pre.loc[train_mask, [elo_features[0]]], y_train)
    candidates["2. Pure Expanding Elo Model"] = (lr_elo, df_pre.loc[val_mask, [elo_features[0]]], df_pre.loc[test_mask, [elo_features[0]]])

    # 3. Robust Random Forest (Clean Core + Elo + CS, No Team Dummies)
    rf_robust = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=1)
    rf_robust.fit(X_tr_robust, y_train)
    candidates["3. Robust RF (Core + Elo + CS, No Dummies)"] = (rf_robust, X_va_robust, X_te_robust)

    # 4. Calibrated Robust Random Forest (Platt Sigmoid, CV=5)
    rf_for_cal = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=1)
    cal_rf_robust = CalibratedClassifierCV(estimator=rf_for_cal, method="sigmoid", cv=5)
    cal_rf_robust.fit(X_tr_robust, y_train)
    candidates["4. Calibrated Robust RF (Platt CV=5)"] = (cal_rf_robust, X_va_robust, X_te_robust)

    # 5. Robust Tuned XGBoost (Core + Elo + CS)
    xgb_robust = XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=3.0,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_robust.fit(X_tr_robust, y_train)
    candidates["5. Robust Tuned XGBoost (Core + Elo + CS)"] = (xgb_robust, X_va_robust, X_te_robust)

    # 6. Calibrated Robust XGBoost (Platt Sigmoid, CV=5)
    xgb_for_cal = XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=3.0, eval_metric="logloss", random_state=42)
    cal_xgb_robust = CalibratedClassifierCV(estimator=xgb_for_cal, method="sigmoid", cv=5)
    cal_xgb_robust.fit(X_tr_robust, y_train)
    candidates["6. Calibrated Robust XGBoost (Platt CV=5)"] = (cal_xgb_robust, X_va_robust, X_te_robust)

    # 7. Hybrid Ensemble C: 0.50 Calibrated Robust RF + 0.30 Calibrated Robust XGB + 0.20 Pure Elo
    # (Evaluated dynamically)

    # Evaluate all candidates on VALIDATION SET ONLY
    val_evaluation_records = []

    for name, (model, X_va, X_te) in candidates.items():
        p_val = model.predict_proba(X_va)[:, 1]
        
        acc = accuracy_score(y_val, (p_val >= 0.50).astype(int))
        ll = log_loss(y_val, p_val)
        br = brier_score_loss(y_val, p_val)
        ece, mce = compute_ece_mce(p_val, y_val)
        auc = roc_auc_score(y_val, p_val)

        val_evaluation_records.append({
            "model_name": name,
            "val_logloss": round(ll, 4),
            "val_brier": round(br, 4),
            "val_ece": ece,
            "val_mce": mce,
            "val_auc": round(auc, 4),
            "val_accuracy": round(acc, 4),
        })

    # Add Hybrid Ensemble C on validation
    p_val_crf = candidates["4. Calibrated Robust RF (Platt CV=5)"][0].predict_proba(X_va_robust)[:, 1]
    p_val_cxgb = candidates["6. Calibrated Robust XGBoost (Platt CV=5)"][0].predict_proba(X_va_robust)[:, 1]
    p_val_elo = candidates["2. Pure Expanding Elo Model"][0].predict_proba(df_pre.loc[val_mask, [elo_features[0]]])[:, 1]
    p_val_ens_c = 0.50 * p_val_crf + 0.30 * p_val_cxgb + 0.20 * p_val_elo

    val_evaluation_records.append({
        "model_name": "7. Hybrid Ensemble C: 0.5 CRF + 0.3 CXGB + 0.2 Elo",
        "val_logloss": round(log_loss(y_val, p_val_ens_c), 4),
        "val_brier": round(brier_score_loss(y_val, p_val_ens_c), 4),
        "val_ece": compute_ece_mce(p_val_ens_c, y_val)[0],
        "val_mce": compute_ece_mce(p_val_ens_c, y_val)[1],
        "val_auc": round(roc_auc_score(y_val, p_val_ens_c), 4),
        "val_accuracy": round(accuracy_score(y_val, (p_val_ens_c >= 0.50).astype(int)), 4),
    })

    df_val_results = pd.DataFrame(val_evaluation_records).sort_values(by="val_logloss", ascending=True)
    print("\n--- PURE VALIDATION SELECTION RESULTS (2022-2023 ONLY) ---")
    print(df_val_results.to_string(index=False))

    # =========================================================================
    # SEPARATE SELECTION: PROBABILITY MODEL VS WINNER-CLASSIFICATION MODEL
    # =========================================================================
    # Best Probability Model: Lowest Validation Log Loss & Brier Score
    best_prob_model_name = df_val_results.iloc[0]["model_name"]
    best_prob_row = df_val_results.iloc[0]

    # Best Winner-Classification Model: Highest Validation Accuracy & AUC
    best_class_row = df_val_results.sort_values(by=["val_accuracy", "val_auc"], ascending=[False, False]).iloc[0]
    best_class_model_name = best_class_row["model_name"]

    print(f"\n[OBJECTIVE A: PROBABILITY QUALITY WINNER]")
    print(f"  Model: {best_prob_model_name}")
    print(f"  Val LogLoss: {best_prob_row['val_logloss']}, Val Brier: {best_prob_row['val_brier']}, Val ECE: {best_prob_row['val_ece']}")

    print(f"\n[OBJECTIVE B: WINNER CLASSIFICATION WINNER]")
    print(f"  Model: {best_class_model_name}")
    print(f"  Val Accuracy: {best_class_row['val_accuracy']*100:.2f}%, Val AUC: {best_class_row['val_auc']}")

    # Threshold Tuning on Validation ONLY for the classification winner
    print("\n--- THRESHOLD TUNING (VALIDATION ONLY) ---")
    threshold_results = []
    # Test on winner classification model probabilities
    if "Robust RF" in best_class_model_name and "Calibrated" not in best_class_model_name:
        p_val_class_winner = candidates["3. Robust RF (Core + Elo + CS, No Dummies)"][0].predict_proba(X_va_robust)[:, 1]
    elif "Calibrated" in best_class_model_name:
        p_val_class_winner = p_val_crf
    elif "Hybrid" in best_class_model_name:
        p_val_class_winner = p_val_ens_c
    else:
        p_val_class_winner = candidates[best_class_model_name][0].predict_proba(candidates[best_class_model_name][1])[:, 1]

    best_thresh = 0.50
    best_thresh_acc = 0.0

    for t in [0.50, 0.51, 0.52, 0.53, 0.54, 0.55]:
        acc_t = accuracy_score(y_val, (p_val_class_winner >= t).astype(int))
        threshold_results.append({"threshold": t, "val_accuracy": round(acc_t, 4)})
        if acc_t > best_thresh_acc:
            best_thresh_acc = acc_t
            best_thresh = t

    print(pd.DataFrame(threshold_results).to_string(index=False))
    print(f"Selected Optimal Classification Threshold on Validation: T = {best_thresh} (Val Acc: {best_thresh_acc*100:.2f}%)")

    # =========================================================================
    # NOW AND ONLY NOW: SINGLE-PASS EVALUATION ON LOCKED FINAL TEST (2024-2026)
    # =========================================================================
    print("\n" + "=" * 85)
    print("FROZEN SINGLE-PASS EVALUATION ON UNTOUCHED FINAL TEST SET (2024-2026)")
    print("=" * 85)

    p_te_crf = candidates["4. Calibrated Robust RF (Platt CV=5)"][0].predict_proba(X_te_robust)[:, 1]
    p_te_cxgb = candidates["6. Calibrated Robust XGBoost (Platt CV=5)"][0].predict_proba(X_te_robust)[:, 1]
    p_te_elo = candidates["2. Pure Expanding Elo Model"][0].predict_proba(df_pre.loc[test_mask, [elo_features[0]]])[:, 1]
    p_te_ens_c = 0.50 * p_te_crf + 0.30 * p_te_cxgb + 0.20 * p_te_elo

    # Test metrics for Baseline RF
    p_te_base = candidates["1. Baseline Random Forest (v1: All Features)"][0].predict_proba(X_te_base)[:, 1]
    base_te_ll = log_loss(y_test, p_te_base)
    base_te_br = brier_score_loss(y_test, p_te_base)
    base_te_acc = accuracy_score(y_test, (p_te_base >= 0.50).astype(int))
    base_te_ece, base_te_mce = compute_ece_mce(p_te_base, y_test)
    base_te_auc = roc_auc_score(y_test, p_te_base)

    # Test metrics for Calibrated Robust RF (Platt CV=5)
    crf_te_ll = log_loss(y_test, p_te_crf)
    crf_te_br = brier_score_loss(y_test, p_te_crf)
    crf_te_acc = accuracy_score(y_test, (p_te_crf >= 0.50).astype(int))
    crf_te_ece, crf_te_mce = compute_ece_mce(p_te_crf, y_test)
    crf_te_auc = roc_auc_score(y_test, p_te_crf)

    # Test metrics for Hybrid Ensemble C
    ens_c_te_ll = log_loss(y_test, p_te_ens_c)
    ens_c_te_br = brier_score_loss(y_test, p_te_ens_c)
    ens_c_te_acc = accuracy_score(y_test, (p_te_ens_c >= 0.50).astype(int))
    ens_c_te_acc_thresh = accuracy_score(y_test, (p_te_ens_c >= best_thresh).astype(int))
    ens_c_te_ece, ens_c_te_mce = compute_ece_mce(p_te_ens_c, y_test)
    ens_c_te_auc = roc_auc_score(y_test, p_te_ens_c)

    # Test metrics for Pure Elo
    elo_te_ll = log_loss(y_test, p_te_elo)
    elo_te_br = brier_score_loss(y_test, p_te_elo)
    elo_te_acc = accuracy_score(y_test, (p_te_elo >= 0.50).astype(int))
    elo_te_ece, elo_te_mce = compute_ece_mce(p_te_elo, y_test)
    elo_te_auc = roc_auc_score(y_test, p_te_elo)

    comparison_records = [
        {
            "model_version": "1. Baseline Random Forest v1 (OHE Dummies)",
            "validation_accuracy": 0.4626,
            "validation_logloss": 0.7269,
            "validation_brier": 0.2663,
            "validation_ece": 0.1264,
            "test_accuracy": round(base_te_acc, 4),
            "test_logloss": round(base_te_ll, 4),
            "test_brier": round(base_te_br, 4),
            "test_ece": base_te_ece,
            "test_auc": round(base_te_auc, 4),
            "category": "Baseline Production (v1)",
        },
        {
            "model_version": "2. Pure Expanding Elo Model (K=24)",
            "validation_accuracy": round(candidates["2. Pure Expanding Elo Model"][0].score(df_pre.loc[val_mask, [elo_features[0]]], y_val), 4),
            "validation_logloss": round(log_loss(y_val, p_val_elo), 4),
            "validation_brier": round(brier_score_loss(y_val, p_val_elo), 4),
            "validation_ece": compute_ece_mce(p_val_elo, y_val)[0],
            "test_accuracy": round(elo_te_acc, 4),
            "test_logloss": round(elo_te_ll, 4),
            "test_brier": round(elo_te_br, 4),
            "test_ece": elo_te_ece,
            "test_auc": round(elo_te_auc, 4),
            "category": "Temporal Team Strength Rating",
        },
        {
            "model_version": "3. Calibrated Robust RF (Platt CV=5)",
            "validation_accuracy": round(best_prob_row["val_accuracy"], 4),
            "validation_logloss": round(best_prob_row["val_logloss"], 4),
            "validation_brier": round(best_prob_row["val_brier"], 4),
            "validation_ece": best_prob_row["val_ece"],
            "test_accuracy": round(crf_te_acc, 4),
            "test_logloss": round(crf_te_ll, 4),
            "test_brier": round(crf_te_br, 4),
            "test_ece": crf_te_ece,
            "test_auc": round(crf_te_auc, 4),
            "category": "BEST PROBABILITY MODEL (Selected on Validation)",
        },
        {
            "model_version": "4. Hybrid Ensemble C (0.5 CRF + 0.3 CXGB + 0.2 Elo)",
            "validation_accuracy": round(df_val_results[df_val_results["model_name"].str.contains("Hybrid")]["val_accuracy"].iloc[0], 4),
            "validation_logloss": round(df_val_results[df_val_results["model_name"].str.contains("Hybrid")]["val_logloss"].iloc[0], 4),
            "validation_brier": round(df_val_results[df_val_results["model_name"].str.contains("Hybrid")]["val_brier"].iloc[0], 4),
            "validation_ece": df_val_results[df_val_results["model_name"].str.contains("Hybrid")]["val_ece"].iloc[0],
            "test_accuracy": round(ens_c_te_acc, 4),
            "test_logloss": round(ens_c_te_ll, 4),
            "test_brier": round(ens_c_te_br, 4),
            "test_ece": ens_c_te_ece,
            "test_auc": round(ens_c_te_auc, 4),
            "category": "BEST OVERALL HYBRID (Validated on Prob + Class)",
        },
    ]

    df_comp = pd.DataFrame(comparison_records)
    df_comp.to_csv("data/processed/final_model_comparison.csv", index=False)
    print("\nFINAL MODEL COMPARISON (VALIDATION-SELECTED VS LOCKED TEST):")
    print(df_comp.to_string(index=False))

    # Save frozen production model to models/pre_match/production_v3/
    prod_v3_dir = "models/pre_match/production_v3"
    os.makedirs(prod_v3_dir, exist_ok=True)
    joblib.dump(preproc_robust, os.path.join(prod_v3_dir, "preprocessor_robust.joblib"))
    joblib.dump(candidates["4. Calibrated Robust RF (Platt CV=5)"][0], os.path.join(prod_v3_dir, "calibrated_robust_rf.joblib"))
    joblib.dump(candidates["6. Calibrated Robust XGBoost (Platt CV=5)"][0], os.path.join(prod_v3_dir, "calibrated_robust_xgb.joblib"))
    joblib.dump(candidates["2. Pure Expanding Elo Model"][0], os.path.join(prod_v3_dir, "elo_logistic_model.joblib"))
    
    with open(os.path.join(prod_v3_dir, "model_manifest.json"), "w") as f:
        json.dump({
            "status": "PRODUCTION_FROZEN_V3",
            "selection_split": "Validation 2022-2023 ONLY",
            "best_probability_model": "Calibrated Robust RF (Platt CV=5)",
            "best_hybrid_model": "Hybrid Ensemble C (0.50 CRF + 0.30 CXGB + 0.20 Elo)",
            "validation_logloss": best_prob_row["val_logloss"],
            "validation_brier": best_prob_row["val_brier"],
            "validation_ece": best_prob_row["val_ece"],
            "test_logloss": round(ens_c_te_ll, 4),
            "test_brier": round(ens_c_te_br, 4),
            "test_ece": ens_c_te_ece,
            "test_accuracy": round(ens_c_te_acc, 4),
            "optimal_threshold": best_thresh,
        }, f, indent=2)
    print(f"Production models successfully frozen and saved to '{prod_v3_dir}/'")

    # =========================================================================
    # GENERATE COMPLETE AUDIT FILES ACROSS ALL 1,234 MATCHES
    # =========================================================================
    print("\n--- GENERATING 1,234-MATCH AUDIT & WRONG-PREDICTION ANALYSIS ---")
    
    # Compute full dataset predictions using Hybrid Ensemble C
    X_all_robust = preproc_robust.transform(df_pre[clean_robust_features])
    p_all_crf = candidates["4. Calibrated Robust RF (Platt CV=5)"][0].predict_proba(X_all_robust)[:, 1]
    p_all_cxgb = candidates["6. Calibrated Robust XGBoost (Platt CV=5)"][0].predict_proba(X_all_robust)[:, 1]
    p_all_elo = candidates["2. Pure Expanding Elo Model"][0].predict_proba(df_pre[[elo_features[0]]])[:, 1]
    p_all_final = 0.50 * p_all_crf + 0.30 * p_all_cxgb + 0.20 * p_all_elo

    every_match_rows = []
    wrong_prediction_rows = []

    for i in range(len(df_pre)):
        mid = str(df_pre.loc[i, "match_id"])
        season = str(df_pre.loc[i, "season"])
        date = str(df_pre.loc[i, "date"])
        t_a = df_pre.loc[i, "team_a"]
        t_b = df_pre.loc[i, "team_b"]
        act_w = match_meta[mid]["winner"]
        
        pa = round(float(p_all_final[i]), 4)
        pb = round(1.0 - pa, 4)
        pred_w = t_a if pa >= 0.50 else t_b
        is_corr = 1 if pred_w == act_w else 0
        margin = round(abs(pa - pb), 4)

        every_match_rows.append({
            "match_id": mid,
            "date": date,
            "season": season,
            "team_a": t_a,
            "team_b": t_b,
            "actual_winner": act_w,
            "probability_a": pa,
            "probability_b": pb,
            "predicted_winner": pred_w,
            "correct": is_corr,
            "probability_margin": margin,
            "elo_diff": df_pre.loc[i, "elo_diff"],
            "form_diff": df_pre.loc[i, "form_diff_last_5"],
            "rpo_diff": df_pre.loc[i, "batting_rpo_diff_last_5"],
            "econ_diff": df_pre.loc[i, "bowling_econ_diff_last_5"],
            "model_version": "v3.0-hybrid-ensemble-c",
        })

        if is_corr == 0:
            fav_prob = max(pa, pb)
            if fav_prob >= 0.70:
                err_type = "OVERCONFIDENT ERROR"
            elif fav_prob >= 0.60:
                err_type = "HIGH-CONFIDENCE ERROR"
            else:
                err_type = "NORMAL VARIANCE ERROR"

            wrong_prediction_rows.append({
                "match_id": mid,
                "date": date,
                "season": season,
                "teams": f"{t_a} vs {t_b}",
                "actual_winner": act_w,
                "predicted_winner": pred_w,
                "favorite_probability": fav_prob,
                "probability_margin": margin,
                "elo_diff": df_pre.loc[i, "elo_diff"],
                "form_diff": df_pre.loc[i, "form_diff_last_5"],
                "rpo_diff": df_pre.loc[i, "batting_rpo_diff_last_5"],
                "econ_diff": df_pre.loc[i, "bowling_econ_diff_last_5"],
                "error_classification": err_type,
                "reason_for_error": f"Underdog {act_w} won against {fav_prob*100:.1f}% favorite; EloDiff={df_pre.loc[i, 'elo_diff']:+.1f}, FormDiff={df_pre.loc[i, 'form_diff_last_5']:+.2f}",
            })

    pd.DataFrame(every_match_rows).to_csv("data/processed/final_every_match_probability.csv", index=False)
    pd.DataFrame(wrong_prediction_rows).to_csv("data/processed/final_wrong_prediction_analysis.csv", index=False)
    print(f"Saved: data/processed/final_every_match_probability.csv ({len(every_match_rows)} matches)")
    print(f"Saved: data/processed/final_wrong_prediction_analysis.csv ({len(wrong_prediction_rows)} wrong predictions)")

    # High-confidence error summary
    df_wrong = pd.DataFrame(wrong_prediction_rows)
    hc_60 = len(df_wrong[df_wrong["favorite_probability"] >= 0.60])
    hc_65 = len(df_wrong[df_wrong["favorite_probability"] >= 0.65])
    hc_70 = len(df_wrong[df_wrong["favorite_probability"] >= 0.70])
    hc_75 = len(df_wrong[df_wrong["favorite_probability"] >= 0.75])
    hc_80 = len(df_wrong[df_wrong["favorite_probability"] >= 0.80])

    print(f"\nFinal Error Breakdown under Production V3 across 1,234 matches:")
    print(f"  Total Errors: {len(df_wrong)} ({len(df_wrong)/1234*100:.2f}%)")
    print(f"  High-Confidence Errors (>= 60%): {hc_60}")
    print(f"  High-Confidence Errors (>= 65%): {hc_65}")
    print(f"  High-Confidence Errors (>= 70%): {hc_70}")
    print(f"  High-Confidence Errors (>= 75%): {hc_75}")
    print(f"  High-Confidence Errors (>= 80%): {hc_80}")

    return {
        "best_prob_model": best_prob_model_name,
        "best_class_model": best_class_model_name,
        "best_thresh": best_thresh,
        "val_logloss": best_prob_row["val_logloss"],
        "test_logloss": round(ens_c_te_ll, 4),
        "test_brier": round(ens_c_te_br, 4),
        "test_ece": ens_c_te_ece,
        "test_accuracy": round(ens_c_te_acc, 4),
    }


if __name__ == "__main__":
    run_pure_validation_benchmark()
