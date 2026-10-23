"""
Model Training, Evaluation, and Artifact Generation Script.
Implements:
1. Logistic Regression (Baseline)
2. Random Forest (Comparison)
3. XGBoost (Primary Model)

Follows strict chronological split and no-leakage preprocessor fitting.
"""

import os
import time
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance

from src.model_training.model_utils import (
    load_dataset,
    temporal_train_val_test_split,
    build_preprocessor,
    get_feature_group_mapping,
    FEATURE_COLS,
    TARGET_COL,
)
from src.model_training.evaluate_models import compute_metrics


def aggregate_importance_by_group(
    raw_importances: Dict[str, float],
    feature_group_mapping: Dict[str, str],
) -> Dict[str, float]:
    """Aggregate one-hot encoded feature importances back to high-level feature groups."""
    group_imp: Dict[str, float] = {}
    for feat_name, imp in raw_importances.items():
        grp = feature_group_mapping.get(feat_name, feat_name)
        group_imp[grp] = group_imp.get(grp, 0.0) + float(imp)
    # Sort descending
    return dict(sorted(group_imp.items(), key=lambda item: item[1], reverse=True))


def run_training_pipeline() -> Dict[str, Any]:
    """Execute complete end-to-end training and evaluation pipeline."""
    print("=" * 70)
    print("STARTING PHASE 3: PRE-MATCH MODEL TRAINING & EVALUATION")
    print("=" * 70)

    # 1. Load data
    dataset_path = "data/processed/pre_match_dataset.csv"
    print(f"\n[1] Loading dataset from '{dataset_path}'...")
    df = load_dataset(dataset_path)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns.")

    # 2. Chronological Split
    print("\n[2] Executing temporal split...")
    train_df, val_df, test_df = temporal_train_val_test_split(df)
    print(f"  Training set   (2008-2021): {len(train_df)} matches (Wins: {train_df[TARGET_COL].sum()}, Losses: {len(train_df) - train_df[TARGET_COL].sum()})")
    print(f"  Validation set (2022-2023): {len(val_df)} matches (Wins: {val_df[TARGET_COL].sum()}, Losses: {len(val_df) - val_df[TARGET_COL].sum()})")
    print(f"  Final Test set (2024-2026): {len(test_df)} matches (Wins: {test_df[TARGET_COL].sum()}, Losses: {len(test_df) - test_df[TARGET_COL].sum()})")

    X_train_raw = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].values
    X_val_raw = val_df[FEATURE_COLS]
    y_val = val_df[TARGET_COL].values
    X_test_raw = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL].values

    # 3. Fit Preprocessor ONLY on Training Split
    print("\n[3] Fitting ColumnTransformer strictly on Training set...")
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train_raw)

    feature_names_out = list(preprocessor.get_feature_names_out())
    feature_group_map = get_feature_group_mapping(feature_names_out)
    print(f"  Fitted on {len(X_train_raw)} training rows.")
    print(f"  Total transformed features: {len(feature_names_out)} (OHE categories + numericals)")

    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # 4. Train Model 1: Logistic Regression (Baseline)
    print("\n[4] Training Model 1: Logistic Regression (Baseline)...")
    lr_params = {
        "C": 1.0,
        "max_iter": 2500,
        "solver": "lbfgs",
        "random_state": 42,
    }
    lr_model = LogisticRegression(**lr_params)
    t0 = time.perf_counter()
    lr_model.fit(X_train, y_train)
    lr_train_time = time.perf_counter() - t0
    print(f"  Logistic Regression trained in {lr_train_time:.4f}s.")

    # 5. Train Model 2: Random Forest (Comparison)
    print("\n[5] Training Model 2: Random Forest (Comparison)...")
    rf_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
    }
    rf_model = RandomForestClassifier(**rf_params)
    t0 = time.perf_counter()
    rf_model.fit(X_train, y_train)
    rf_train_time = time.perf_counter() - t0
    print(f"  Random Forest trained in {rf_train_time:.4f}s.")

    # 6. Train Model 3: XGBoost (Primary Model)
    print("\n[6] Training Model 3: XGBoost (Primary Model)...")
    xgb_params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    xgb_model = XGBClassifier(**xgb_params)
    t0 = time.perf_counter()
    xgb_model.fit(X_train, y_train)
    xgb_train_time = time.perf_counter() - t0
    print(f"  XGBoost trained in {xgb_train_time:.4f}s.")

    # 7. Evaluate on Validation and Test sets
    print("\n[7] Computing Validation and Test Metrics...")
    models = {
        "logistic_regression": (lr_model, lr_params, lr_train_time),
        "random_forest": (rf_model, rf_params, rf_train_time),
        "xgboost": (xgb_model, xgb_params, xgb_train_time),
    }

    evaluation_results = {}
    for name, (model, params, tr_time) in models.items():
        val_probs = model.predict_proba(X_val)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]

        val_metrics = compute_metrics(y_val, val_probs)
        test_metrics = compute_metrics(y_test, test_probs)

        evaluation_results[name] = {
            "params": params,
            "train_time_sec": round(tr_time, 4),
            "validation": val_metrics,
            "test": test_metrics,
        }

    # 8. Feature Importance Analysis
    print("\n[8] Analyzing Feature Importance...")
    # RF Feature Importance
    rf_raw_imp = dict(zip(feature_names_out, rf_model.feature_importances_))
    rf_grp_imp = aggregate_importance_by_group(rf_raw_imp, feature_group_map)

    # XGBoost Gain Importance
    xgb_booster = xgb_model.get_booster()
    # Map f0, f1... to real feature names
    xgb_score_gain = xgb_booster.get_score(importance_type="gain")
    # Booster feature names might be f0, f1 or actual names depending on array/df
    xgb_raw_gain = {}
    for k, v in xgb_score_gain.items():
        if k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            feat_name = feature_names_out[idx]
        else:
            feat_name = k
        xgb_raw_gain[feat_name] = v

    # Fill in zero for features not split on
    for f in feature_names_out:
        if f not in xgb_raw_gain:
            xgb_raw_gain[f] = 0.0

    xgb_grp_gain = aggregate_importance_by_group(xgb_raw_gain, feature_group_map)

    # XGBoost Permutation Importance (on validation set)
    perm_res = permutation_importance(
        xgb_model, X_val, y_val, n_repeats=10, random_state=42, scoring="neg_log_loss"
    )
    xgb_raw_perm = dict(zip(feature_names_out, perm_res.importances_mean))
    xgb_grp_perm = aggregate_importance_by_group(xgb_raw_perm, feature_group_map)

    feature_importances = {
        "rf_raw_importance": rf_raw_imp,
        "rf_group_importance": rf_grp_imp,
        "xgb_raw_gain": xgb_raw_gain,
        "xgb_group_gain": xgb_grp_gain,
        "xgb_raw_permutation": xgb_raw_perm,
        "xgb_group_permutation": xgb_grp_perm,
    }

    # 9. Save Artifacts
    print("\n[9] Persisting Model & Preprocessor Artifacts...")
    prep_path = os.path.join(models_dir, "preprocessor.joblib")
    lr_path = os.path.join(models_dir, "logistic_regression.joblib")
    rf_path = os.path.join(models_dir, "random_forest.joblib")
    xgb_joblib_path = os.path.join(models_dir, "xgboost_model.joblib")
    xgb_json_path = os.path.join(models_dir, "xgboost_model.json")

    joblib.dump(preprocessor, prep_path)
    joblib.dump(lr_model, lr_path)
    joblib.dump(rf_model, rf_path)
    joblib.dump(xgb_model, xgb_joblib_path)
    xgb_model.save_model(xgb_json_path)

    print(f"  Saved preprocessor: {prep_path}")
    print(f"  Saved Logistic Regression: {lr_path}")
    print(f"  Saved Random Forest: {rf_path}")
    print(f"  Saved XGBoost (joblib): {xgb_joblib_path}")
    print(f"  Saved XGBoost (native JSON): {xgb_json_path}")

    # 10. Sanity Checks
    print("\n[10] Running Mandatory Sanity Checks...")
    checks_passed = True

    # Chronological checks
    assert train_df["date"].max() < val_df["date"].min(), "Train/Val temporal boundary violation!"
    assert val_df["date"].max() < test_df["date"].min(), "Val/Test temporal boundary violation!"
    print("  [Pass] Chronological ordering: Train < Val < Test.")

    # Artifact reload test
    loaded_prep = joblib.load(prep_path)
    loaded_xgb = joblib.load(xgb_joblib_path)
    print("  [Pass] Artifact reload test successful.")

    # Single-Match Inference on held-out test match
    print("\n[11] Running Single-Match Inference Verification...")
    test_sample = test_df.iloc[-1]  # Most recent match in test set
    sample_raw = test_sample[FEATURE_COLS].to_frame().T
    sample_transformed = loaded_prep.transform(sample_raw)
    sample_probs = loaded_xgb.predict_proba(sample_transformed)[0]

    prob_team_a = float(sample_probs[1])
    prob_team_b = float(sample_probs[0])
    actual_winner = test_sample["team_a"] if test_sample[TARGET_COL] == 1 else test_sample["team_b"]

    print("-" * 50)
    print(f"Match ID:                          {test_sample['match_id']}")
    print(f"Season:                            {test_sample['season']}")
    print(f"Date:                              {test_sample['date'].strftime('%Y-%m-%d')}")
    print(f"Team A:                            {test_sample['team_a']}")
    print(f"Team B:                            {test_sample['team_b']}")
    print(f"Venue:                             {test_sample['venue']}")
    print(f"Predicted Team A Win Probability:  {prob_team_a:.4f} ({prob_team_a*100:.2f}%)")
    print(f"Predicted Team B Win Probability:  {prob_team_b:.4f} ({prob_team_b*100:.2f}%)")
    print(f"Probability Sum:                   {prob_team_a + prob_team_b:.6f}")
    print(f"Actual Target (1=Team A Win):      {test_sample[TARGET_COL]}")
    print(f"Actual Match Winner:               {actual_winner}")
    print("-" * 50)

    assert abs((prob_team_a + prob_team_b) - 1.0) < 1e-5, "Probabilities do not sum to 1.0!"
    print("  [Pass] Single-match inference probabilities valid and sum to 1.0.")

    return {
        "evaluation_results": evaluation_results,
        "feature_importances": feature_importances,
        "single_match_test": {
            "match_id": int(test_sample["match_id"]),
            "season": int(test_sample["season"]),
            "date": test_sample["date"].strftime("%Y-%m-%d"),
            "team_a": test_sample["team_a"],
            "team_b": test_sample["team_b"],
            "venue": test_sample["venue"],
            "prob_team_a": round(prob_team_a, 4),
            "prob_team_b": round(prob_team_b, 4),
            "prob_sum": round(prob_team_a + prob_team_b, 6),
            "actual_target": int(test_sample[TARGET_COL]),
            "actual_winner": actual_winner,
        },
        "train_df_len": len(train_df),
        "val_df_len": len(val_df),
        "test_df_len": len(test_df),
        "total_transformed_features": len(feature_names_out),
    }


if __name__ == "__main__":
    results = run_training_pipeline()
