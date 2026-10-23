"""
Live Model Training and Ablation Orchestrator.

Implements end-to-end training, ablation, and architecture comparison for:
- Model A: Match State Baseline
- Model B: Model A + Current Player
- Model C: Model B + Historical Player
- Model D: Model C + Matchup Bayes m=24

Algorithms:
- Logistic Regression (Linear baseline with StandardScaler)
- Random Forest (n_estimators=100, max_depth=6, min_samples_leaf=5)
- XGBoost (n_estimators=100, max_depth=4, lr=0.05, reg_alpha=0.1, reg_lambda=1.0)

Architectures:
- Architecture A: Unified Model (handles Innings 1 & 2)
- Architecture B: Two Models (Innings 1 model + Innings 2 chase model)

Saves all trained models, evaluation metrics, and metadata to models/live/.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.inspection import permutation_importance

from .live_model_preprocessor import (
    LiveFeaturePreprocessor,
    get_feature_columns_for_set,
)
from .live_model_evaluator import (
    compute_standard_metrics,
    compute_expected_calibration_error,
    compute_phase_evaluation,
    compute_innings_breakdown,
    compute_per_match_metrics,
    extract_qualitative_match_examples,
)


TRAIN_SEASONS = [
    "2007/08",
    "2009",
    "2009/10",
    "2011",
    "2012",
    "2013",
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
    "2019",
    "2020/21",
    "2021",
]
VAL_SEASONS = ["2022", "2023"]
TEST_SEASONS = ["2024", "2025", "2026"]

TARGET_COL = "target_team_a_win"


def load_and_split_live_dataset(
    dataset_path: str = "data/processed/live_dataset.csv",
    exclude_terminal: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Loads live_dataset.csv and performs strict match-level chronological split.
    Optionally separates terminal match states in Innings 2.
    """
    print(f"Loading live dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, low_memory=False)

    total_rows = len(df)
    total_matches = df["match_id"].nunique()

    # Identify mechanical terminal states in Innings 2:
    # Target reached, all out, or 20 overs complete on final delivery
    is_terminal = (df["current_innings"] == 2) & (
        (df["runs_needed"] == 0)
        | (df["wickets_remaining"] == 0)
        | (df["legal_balls_remaining"] == 0)
    )
    terminal_count = int(is_terminal.sum())

    if exclude_terminal:
        print(f"Isolating {terminal_count} post-match mechanical terminal states in Innings 2 for deterministic handling...")
        df_active = df[~is_terminal].copy()
    else:
        df_active = df.copy()

    train_df = df_active[df_active["season"].isin(TRAIN_SEASONS)].copy()
    val_df = df_active[df_active["season"].isin(VAL_SEASONS)].copy()
    test_df = df_active[df_active["season"].isin(TEST_SEASONS)].copy()

    split_meta = {
        "total_dataset_rows": total_rows,
        "total_dataset_matches": total_matches,
        "terminal_rows_isolated": terminal_count,
        "active_rows": len(df_active),
        "train_rows": len(train_df),
        "train_matches": train_df["match_id"].nunique(),
        "train_team_a_win_pct": round(float(train_df[TARGET_COL].mean() * 100), 2),
        "val_rows": len(val_df),
        "val_matches": val_df["match_id"].nunique(),
        "val_team_a_win_pct": round(float(val_df[TARGET_COL].mean() * 100), 2),
        "test_rows": len(test_df),
        "test_matches": test_df["match_id"].nunique(),
        "test_team_a_win_pct": round(float(test_df[TARGET_COL].mean() * 100), 2),
    }

    print(
        f"Split Summary: Train={split_meta['train_rows']} rows ({split_meta['train_matches']} matches) | "
        f"Val={split_meta['val_rows']} rows ({split_meta['val_matches']} matches) | "
        f"Test={split_meta['test_rows']} rows ({split_meta['test_matches']} matches)"
    )

    return train_df, val_df, test_df, split_meta


def get_base_classifier(model_name: str, random_state: int = 42):
    """Instantiate candidate classifier with fixed hyperparameters."""
    m_name = model_name.lower()
    if m_name in ("lr", "logistic_regression"):
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        )
    elif m_name in ("rf", "random_forest"):
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
    elif m_name in ("xgb", "xgboost"):
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def run_ablation_experiments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes master feature ablation experiment across Models A, B, C, D
    and algorithms Logistic Regression, Random Forest, XGBoost under Architecture A (Unified).
    """
    feature_sets = ["A", "B", "C", "D"]
    algorithms = ["logistic_regression", "random_forest", "xgboost"]

    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    master_results = []
    trained_artifacts = {}

    for fset in feature_sets:
        print(f"\n=======================================================")
        print(f"Evaluating Feature Set {fset}...")
        print(f"=======================================================")

        # 1. Fit Preprocessor for tree models (no scaling)
        prep_tree = LiveFeaturePreprocessor(feature_set=fset, scale_numerical=False)
        prep_tree.fit(train_df)
        X_train_tree = prep_tree.transform(train_df)
        X_val_tree = prep_tree.transform(val_df)
        X_test_tree = prep_tree.transform(test_df)

        # 2. Fit Preprocessor for linear model (with scaling)
        prep_scaled = LiveFeaturePreprocessor(feature_set=fset, scale_numerical=True)
        prep_scaled.fit(train_df)
        X_train_scaled = prep_scaled.transform(train_df)
        X_val_scaled = prep_scaled.transform(val_df)
        X_test_scaled = prep_scaled.transform(test_df)

        trained_artifacts[f"prep_{fset}"] = prep_tree
        trained_artifacts[f"prep_{fset}_scaled"] = prep_scaled

        for algo in algorithms:
            is_lr = algo == "logistic_regression"
            X_tr = X_train_scaled if is_lr else X_train_tree
            X_v = X_val_scaled if is_lr else X_val_tree
            X_te = X_test_scaled if is_lr else X_test_tree

            clf = get_base_classifier(algo)

            t0 = time.perf_counter()
            clf.fit(X_tr, y_train)
            train_time = round(time.perf_counter() - t0, 3)

            t1 = time.perf_counter()
            p_val = clf.predict_proba(X_v)[:, 1]
            inf_time_val = round((time.perf_counter() - t1) * 1000 / len(val_df), 4)  # ms per sample

            p_test = clf.predict_proba(X_te)[:, 1]

            val_metrics = compute_standard_metrics(y_val, p_val)
            test_metrics = compute_standard_metrics(y_test, p_test)

            row_res = {
                "architecture": "Unified",
                "feature_set": fset,
                "model": algo,
                "train_time_sec": train_time,
                "inference_time_ms_per_sample": inf_time_val,
                "val_log_loss": val_metrics["log_loss"],
                "val_brier": val_metrics["brier_score"],
                "val_ece": val_metrics["ece"],
                "val_accuracy": val_metrics["accuracy"],
                "val_roc_auc": val_metrics["roc_auc"],
                "test_log_loss": test_metrics["log_loss"],
                "test_brier": test_metrics["brier_score"],
                "test_ece": test_metrics["ece"],
                "test_accuracy": test_metrics["accuracy"],
                "test_roc_auc": test_metrics["roc_auc"],
            }
            master_results.append(row_res)
            trained_artifacts[f"model_{fset}_{algo}"] = clf

            print(
                f"  [{algo.upper():18}] Set {fset}: Val LogLoss={val_metrics['log_loss']:.4f} | "
                f"Val Brier={val_metrics['brier_score']:.4f} | Val ECE={val_metrics['ece']:.4f} | "
                f"Test LogLoss={test_metrics['log_loss']:.4f} (Train: {train_time}s)"
            )

    return master_results, trained_artifacts


def run_two_model_architecture_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set: str = "D",
    algorithm: str = "xgboost",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Executes Architecture B: Two-Model Pipeline (Innings 1 Model + Innings 2 Model).
    Both models predict target_team_a_win.
    """
    print(f"\n=======================================================")
    print(f"Evaluating Architecture B (Two-Model Pipeline) for Set {feature_set} [{algorithm}]...")
    print(f"=======================================================")

    is_lr = algorithm == "logistic_regression"

    # Split train, val, test by innings
    train_inn1 = train_df[train_df["current_innings"] == 1]
    train_inn2 = train_df[train_df["current_innings"] == 2]

    val_inn1 = val_df[val_df["current_innings"] == 1]
    val_inn2 = val_df[val_df["current_innings"] == 2]

    test_inn1 = test_df[test_df["current_innings"] == 1]
    test_inn2 = test_df[test_df["current_innings"] == 2]

    # Preprocessors
    prep_inn1 = LiveFeaturePreprocessor(feature_set=feature_set, scale_numerical=is_lr)
    prep_inn1.fit(train_inn1)
    X_tr1 = prep_inn1.transform(train_inn1)
    y_tr1 = train_inn1[TARGET_COL].values

    prep_inn2 = LiveFeaturePreprocessor(feature_set=feature_set, scale_numerical=is_lr)
    prep_inn2.fit(train_inn2)
    X_tr2 = prep_inn2.transform(train_inn2)
    y_tr2 = train_inn2[TARGET_COL].values

    # Train Model 1 (Innings 1)
    t0 = time.perf_counter()
    clf1 = get_base_classifier(algorithm)
    clf1.fit(X_tr1, y_tr1)
    t_m1 = time.perf_counter() - t0

    # Train Model 2 (Innings 2)
    t0 = time.perf_counter()
    clf2 = get_base_classifier(algorithm)
    clf2.fit(X_tr2, y_tr2)
    t_m2 = time.perf_counter() - t0

    total_train_time = round(t_m1 + t_m2, 3)

    # Validation Evaluation:
    # Predict Innings 1
    p_val1 = clf1.predict_proba(prep_inn1.transform(val_inn1))[:, 1]
    # Predict Innings 2
    p_val2 = clf2.predict_proba(prep_inn2.transform(val_inn2))[:, 1]

    # Assemble aligned validation predictions
    val_preds_combined = np.empty(len(val_df), dtype=float)
    val_preds_combined[val_df["current_innings"] == 1] = p_val1
    val_preds_combined[val_df["current_innings"] == 2] = p_val2

    # Test Evaluation:
    p_test1 = clf1.predict_proba(prep_inn1.transform(test_inn1))[:, 1]
    p_test2 = clf2.predict_proba(prep_inn2.transform(test_inn2))[:, 1]

    test_preds_combined = np.empty(len(test_df), dtype=float)
    test_preds_combined[test_df["current_innings"] == 1] = p_test1
    test_preds_combined[test_df["current_innings"] == 2] = p_test2

    val_metrics = compute_standard_metrics(val_df[TARGET_COL].values, val_preds_combined)
    test_metrics = compute_standard_metrics(test_df[TARGET_COL].values, test_preds_combined)

    res = {
        "architecture": "Two-Model (Innings 1 + Innings 2)",
        "feature_set": feature_set,
        "model": algorithm,
        "train_time_sec": total_train_time,
        "inference_time_ms_per_sample": 0.0,
        "val_log_loss": val_metrics["log_loss"],
        "val_brier": val_metrics["brier_score"],
        "val_ece": val_metrics["ece"],
        "val_accuracy": val_metrics["accuracy"],
        "val_roc_auc": val_metrics["roc_auc"],
        "test_log_loss": test_metrics["log_loss"],
        "test_brier": test_metrics["brier_score"],
        "test_ece": test_metrics["ece"],
        "test_accuracy": test_metrics["accuracy"],
        "test_roc_auc": test_metrics["roc_auc"],
        "val_predictions": val_preds_combined,
        "test_predictions": test_preds_combined,
    }

    artifacts = {
        "prep_inn1": prep_inn1,
        "prep_inn2": prep_inn2,
        "model_inn1": clf1,
        "model_inn2": clf2,
    }

    print(
        f"  [TWO-MODEL ARCH] Set {feature_set} [{algorithm}]: "
        f"Val LogLoss={val_metrics['log_loss']:.4f} | Val Brier={val_metrics['brier_score']:.4f} | "
        f"Val ECE={val_metrics['ece']:.4f} | Test LogLoss={test_metrics['log_loss']:.4f}"
    )

    return res, artifacts
