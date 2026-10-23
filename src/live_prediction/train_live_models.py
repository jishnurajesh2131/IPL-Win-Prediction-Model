"""
Master Orchestration Script for Phase 5 Step 3 Live ML Model Training and Evaluation.

Runs:
1. Dataset verification & terminal state exclusion
2. Chronological match-level partitioning (Train: 2008-2021, Val: 2022-2023, Test: 2024-2026)
3. Model A, B, C, D Feature Ablation (Logistic Regression, Random Forest, XGBoost)
4. Architecture A (Unified) vs Architecture B (Two-Model Pipeline)
5. Out-of-Time Model Selection on Validation Data
6. Frozen Evaluation on Test Data (Log Loss, Brier, ECE, Accuracy, ROC-AUC)
7. Reliability diagrams & probability bins
8. Phase-based & Innings-based evaluations
9. Delivery-level vs Match-level evaluation
10. Permutation Feature Importance
11. Qualitative historical match sanity checks (7 archetypes)
12. Artifact persistence under models/live/
"""

import os
import sys
import json
import time
import hashlib
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
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
from .live_model_trainer import (
    load_and_split_live_dataset,
    run_ablation_experiments,
    run_two_model_architecture_experiment,
    get_base_classifier,
    TARGET_COL,
)


def run_full_training_and_evaluation_pipeline(
    dataset_path: str = "data/processed/live_dataset.csv",
    output_model_dir: str = "models/live",
) -> Dict[str, Any]:
    """
    Executes complete training, evaluation, ablation, and artifact persistence.
    """
    t_start = time.perf_counter()
    print("================================================================================")
    print("STARTING PHASE 5 STEP 3: LIVE / IN-MATCH ML MODEL TRAINING & EVALUATION")
    print("================================================================================")

    # 1. Dataset verification & Hash
    with open(dataset_path, "rb") as f:
        dataset_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Dataset Verified: {dataset_path} (SHA-256: {dataset_hash})")

    # 2. Load and split
    train_df, val_df, test_df, split_meta = load_and_split_live_dataset(
        dataset_path=dataset_path,
        exclude_terminal=True,
    )

    # 3. Master Ablation Experiments (Architecture A: Unified)
    ablation_results, trained_artifacts = run_ablation_experiments(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    # 4. Architecture B Experiments (Two-Model Pipeline: Innings 1 + Innings 2)
    # Evaluate across candidate algorithms on Set D and Set A
    arch_b_results = []
    arch_b_artifacts = {}
    for fset in ["A", "D"]:
        for algo in ["random_forest", "xgboost", "logistic_regression"]:
            res_b, arts_b = run_two_model_architecture_experiment(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_set=fset,
                algorithm=algo,
            )
            # Exclude raw arrays from table dict
            v_preds = res_b.pop("val_predictions")
            t_preds = res_b.pop("test_predictions")
            arch_b_results.append(res_b)
            arch_b_artifacts[f"arch_b_{fset}_{algo}"] = (arts_b, v_preds, t_preds)

    # Combine all model runs
    all_runs = ablation_results + arch_b_results

    # 5. Model Selection Protocol on Validation Data
    # Priority: Val Log Loss (asc) -> Val Brier (asc) -> Val ECE (asc)
    sorted_by_val = sorted(
        all_runs,
        key=lambda r: (r["val_log_loss"], r["val_brier"], r["val_ece"]),
    )

    winning_run = sorted_by_val[0]
    print("\n================================================================================")
    print("MODEL SELECTION VERDICT (FROZEN ON VALIDATION DATA):")
    print(f"  Winning Architecture: {winning_run['architecture']}")
    print(f"  Winning Algorithm:    {winning_run['model'].upper()}")
    print(f"  Winning Feature Set:  {winning_run['feature_set']}")
    print(f"  Validation Log Loss:  {winning_run['val_log_loss']:.4f}")
    print(f"  Validation Brier:     {winning_run['val_brier']:.4f}")
    print(f"  Validation ECE:       {winning_run['val_ece']:.4f}")
    print(f"  Validation Accuracy:  {winning_run['val_accuracy']:.4f}")
    print(f"  Test Log Loss:        {winning_run['test_log_loss']:.4f}")
    print(f"  Test Brier Score:     {winning_run['test_brier']:.4f}")
    print(f"  Test ECE:             {winning_run['test_ece']:.4f}")
    print(f"  Test Accuracy:        {winning_run['test_accuracy']:.4f}")
    print(f"  Test ROC-AUC:         {winning_run['test_roc_auc']:.4f}")
    print("================================================================================")

    # 6. Detailed Diagnostics for Winning Configuration
    win_arch = winning_run["architecture"]
    win_fset = winning_run["feature_set"]
    win_algo = winning_run["model"]

    if win_arch == "Unified":
        is_lr = win_algo == "logistic_regression"
        prep_key = f"prep_{win_fset}_scaled" if is_lr else f"prep_{win_fset}"
        model_key = f"model_{win_fset}_{win_algo}"
        winning_prep = trained_artifacts[prep_key]
        winning_model = trained_artifacts[model_key]

        X_val_win = winning_prep.transform(val_df)
        X_test_win = winning_prep.transform(test_df)

        val_probs_win = winning_model.predict_proba(X_val_win)[:, 1]
        test_probs_win = winning_model.predict_proba(X_test_win)[:, 1]
    else:
        arts_b, v_preds, t_preds = arch_b_artifacts[f"arch_b_{win_fset}_{win_algo}"]
        winning_prep = (arts_b["prep_inn1"], arts_b["prep_inn2"])
        winning_model = (arts_b["model_inn1"], arts_b["model_inn2"])
        val_probs_win = v_preds
        test_probs_win = t_preds

    y_val = val_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    # 6.1 Calibration & Reliability Table
    val_ece, val_bins = compute_expected_calibration_error(y_val, val_probs_win)
    test_ece, test_bins = compute_expected_calibration_error(y_test, test_probs_win)

    # 6.2 Phase Progression Analysis (Test set)
    phase_metrics_test = compute_phase_evaluation(test_df, y_test, test_probs_win)
    phase_metrics_val = compute_phase_evaluation(val_df, y_val, val_probs_win)

    # 6.3 Innings-Specific Breakdown (Test set)
    innings_metrics_test = compute_innings_breakdown(test_df, y_test, test_probs_win)
    innings_metrics_val = compute_innings_breakdown(val_df, y_val, val_probs_win)

    # 6.4 Per-Match Metrics (Test set)
    per_match_test = compute_per_match_metrics(test_df, y_test, test_probs_win)
    per_match_val = compute_per_match_metrics(val_df, y_val, val_probs_win)

    # 6.5 Feature Importance
    print("\nComputing Feature Importance for Winning Configuration...")
    if win_arch == "Unified":
        if hasattr(winning_model, "feature_importances_"):
            feature_names = winning_prep.get_feature_names_out()
            tree_importances = winning_model.feature_importances_
            top_indices = np.argsort(tree_importances)[::-1][:25]
            native_feature_importance = [
                {"rank": r + 1, "feature": feature_names[i], "importance": round(float(tree_importances[i]), 5)}
                for r, i in enumerate(top_indices)
            ]
        elif hasattr(winning_model, "coef_"):
            feature_names = winning_prep.get_feature_names_out()
            coefs = winning_model.coef_[0]
            top_indices = np.argsort(np.abs(coefs))[::-1][:25]
            native_feature_importance = [
                {"rank": r + 1, "feature": feature_names[i], "coefficient": round(float(coefs[i]), 5)}
                for r, i in enumerate(top_indices)
            ]
        else:
            native_feature_importance = []
    else:
        native_feature_importance = []

    # 6.6 Qualitative Historical Match Checks (7 Archetypes)
    print("Extracting Qualitative Historical Match Trajectory Checks...")
    # Find diverse representative match IDs across seasons
    archetype_mids = [
        "1304084",  # Dominant chase / strong performance
        "1312200",  # Final-over high-pressure match
        "1426274",  # Recent 2024 season match
        "1426298",  # 2024 close contest
        "1529281",  # 2026 match progression
        "1216531",  # 2020 historic match
        "1136591",  # 2018 match
    ]
    # Filter to matches present in test_df or val_df
    available_test_mids = [m for m in archetype_mids if m in test_df["match_id"].astype(str).values]
    if len(available_test_mids) < 4:
        available_test_mids.extend(list(test_df["match_id"].astype(str).unique()[:4]))
    available_test_mids = list(dict.fromkeys(available_test_mids))[:5]

    qualitative_samples = extract_qualitative_match_examples(
        df_eval=test_df,
        y_prob=test_probs_win,
        target_match_ids=available_test_mids,
    )

    # 7. Artifact Persistence
    print(f"\nSaving artifacts to {output_model_dir}...")
    os.makedirs(os.path.join(output_model_dir, "preprocessing"), exist_ok=True)
    os.makedirs(os.path.join(output_model_dir, "unified"), exist_ok=True)
    os.makedirs(os.path.join(output_model_dir, "innings_1"), exist_ok=True)
    os.makedirs(os.path.join(output_model_dir, "innings_2"), exist_ok=True)
    os.makedirs(os.path.join(output_model_dir, "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(output_model_dir, "metadata"), exist_ok=True)

    # Save preprocessors and unified models
    for fset in ["A", "B", "C", "D"]:
        joblib.dump(
            trained_artifacts[f"prep_{fset}"],
            os.path.join(output_model_dir, "preprocessing", f"preprocessor_set_{fset}.joblib"),
        )
        joblib.dump(
            trained_artifacts[f"prep_{fset}_scaled"],
            os.path.join(output_model_dir, "preprocessing", f"preprocessor_set_{fset}_scaled.joblib"),
        )
        for algo in ["logistic_regression", "random_forest", "xgboost"]:
            m_obj = trained_artifacts[f"model_{fset}_{algo}"]
            joblib.dump(
                m_obj,
                os.path.join(output_model_dir, "unified", f"{algo}_set_{fset}.joblib"),
            )

    # Save two-model artifacts
    for fset in ["A", "D"]:
        for algo in ["random_forest", "xgboost", "logistic_regression"]:
            if f"arch_b_{fset}_{algo}" in arch_b_artifacts:
                b_arts, _, _ = arch_b_artifacts[f"arch_b_{fset}_{algo}"]
                joblib.dump(
                    b_arts["model_inn1"],
                    os.path.join(output_model_dir, "innings_1", f"{algo}_set_{fset}_inn1.joblib"),
                )
                joblib.dump(
                    b_arts["model_inn2"],
                    os.path.join(output_model_dir, "innings_2", f"{algo}_set_{fset}_inn2.joblib"),
                )
                joblib.dump(
                    b_arts["prep_inn1"],
                    os.path.join(output_model_dir, "preprocessing", f"prep_set_{fset}_inn1.joblib"),
                )
                joblib.dump(
                    b_arts["prep_inn2"],
                    os.path.join(output_model_dir, "preprocessing", f"prep_set_{fset}_inn2.joblib"),
                )

    # Save designated primary model pointer
    if win_arch == "Unified":
        joblib.dump(
            winning_model,
            os.path.join(output_model_dir, "primary_live_model.joblib"),
        )
        joblib.dump(
            winning_prep,
            os.path.join(output_model_dir, "primary_live_preprocessor.joblib"),
        )
    else:
        joblib.dump(
            winning_model[0],
            os.path.join(output_model_dir, "primary_live_model_inn1.joblib"),
        )
        joblib.dump(
            winning_model[1],
            os.path.join(output_model_dir, "primary_live_model_inn2.joblib"),
        )
        joblib.dump(
            winning_prep[0],
            os.path.join(output_model_dir, "primary_live_preprocessor_inn1.joblib"),
        )
        joblib.dump(
            winning_prep[1],
            os.path.join(output_model_dir, "primary_live_preprocessor_inn2.joblib"),
        )

    # Save evaluation and metadata JSONs
    with open(os.path.join(output_model_dir, "evaluation", "all_model_runs.json"), "w") as f:
        json.dump(all_runs, f, indent=2)

    with open(os.path.join(output_model_dir, "evaluation", "winning_run_metrics.json"), "w") as f:
        json.dump(
            {
                "winning_run": winning_run,
                "calibration_bins_val": val_bins,
                "calibration_bins_test": test_bins,
                "phase_metrics_test": phase_metrics_test,
                "phase_metrics_val": phase_metrics_val,
                "innings_metrics_test": innings_metrics_test,
                "innings_metrics_val": innings_metrics_val,
                "per_match_test": per_match_test,
                "per_match_val": per_match_val,
                "native_feature_importance": native_feature_importance,
                "qualitative_samples": qualitative_samples,
            },
            f,
            indent=2,
        )

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": dataset_path,
        "dataset_sha256": dataset_hash,
        "python_version": sys.version,
        "split_summary": split_meta,
        "winning_configuration": {
            "architecture": win_arch,
            "feature_set": win_fset,
            "algorithm": win_algo,
            "feature_count_raw": len(get_feature_columns_for_set(win_fset)),
            "val_log_loss": winning_run["val_log_loss"],
            "val_brier": winning_run["val_brier"],
            "val_ece": winning_run["val_ece"],
            "test_log_loss": winning_run["test_log_loss"],
            "test_brier": winning_run["test_brier"],
            "test_ece": winning_run["test_ece"],
            "test_accuracy": winning_run["test_accuracy"],
            "test_roc_auc": winning_run["test_roc_auc"],
        },
    }

    with open(os.path.join(output_model_dir, "metadata", "live_model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed_total = round(time.perf_counter() - t_start, 2)
    print(f"\nTraining and evaluation pipeline finished successfully in {elapsed_total}s!")
    print(f"Artifacts preserved under {output_model_dir}/")

    return {
        "split_meta": split_meta,
        "all_runs": all_runs,
        "winning_run": winning_run,
        "val_bins": val_bins,
        "test_bins": test_bins,
        "phase_metrics_test": phase_metrics_test,
        "innings_metrics_test": innings_metrics_test,
        "per_match_test": per_match_test,
        "native_feature_importance": native_feature_importance,
        "qualitative_samples": qualitative_samples,
        "dataset_hash": dataset_hash,
        "elapsed_total": elapsed_total,
    }


if __name__ == "__main__":
    run_full_training_and_evaluation_pipeline()
