"""
Inspect All Model Training Results, Metrics, Calibration Tables, and Feature Importance.
"""

from src.model_training.train_models import run_training_pipeline
import pandas as pd

res = run_training_pipeline()
eval_res = res["evaluation_results"]

print("\n" + "=" * 80)
print("VALIDATION SET METRICS (2022-2023, N=147)")
print("=" * 80)
val_rows = []
for model_name, data in eval_res.items():
    vm = data["validation"]
    val_rows.append({
        "Model": model_name,
        "Accuracy": vm["accuracy"],
        "Precision": vm["precision"],
        "Recall": vm["recall"],
        "F1-Score": vm["f1_score"],
        "Log Loss": vm["log_loss"],
        "Brier Score": vm["brier_score"],
        "ECE": vm["ece"],
        "MCE": vm["mce"],
        "Train Time (s)": data["train_time_sec"],
    })
print(pd.DataFrame(val_rows).to_string(index=False))

print("\n" + "=" * 80)
print("FINAL TEST SET METRICS (2024-2026, N=215)")
print("=" * 80)
test_rows = []
for model_name, data in eval_res.items():
    tm = data["test"]
    test_rows.append({
        "Model": model_name,
        "Accuracy": tm["accuracy"],
        "Precision": tm["precision"],
        "Recall": tm["recall"],
        "F1-Score": tm["f1_score"],
        "Log Loss": tm["log_loss"],
        "Brier Score": tm["brier_score"],
        "ECE": tm["ece"],
        "MCE": tm["mce"],
    })
print(pd.DataFrame(test_rows).to_string(index=False))

print("\n" + "=" * 80)
print("CALIBRATION BINS — FINAL TEST SET")
print("=" * 80)
for model_name, data in eval_res.items():
    print(f"\nModel: {model_name}")
    print(f"Posture: {data['test']['calibration_posture']}")
    print(data["test"]["calibration_bins"].to_string(index=False))

print("\n" + "=" * 80)
print("FEATURE IMPORTANCES (AGGREGATED)")
print("=" * 80)
fi = res["feature_importances"]

print("\n--- Random Forest (MDI) Top 15 Groups ---")
for k, v in list(fi["rf_group_importance"].items())[:15]:
    print(f"  {k:30s}: {v:.4f}")

print("\n--- XGBoost (Gain) Top 15 Groups ---")
for k, v in list(fi["xgb_group_gain"].items())[:15]:
    print(f"  {k:30s}: {v:.4f}")

print("\n--- XGBoost (Permutation on Validation) Top 15 Groups ---")
for k, v in list(fi["xgb_group_permutation"].items())[:15]:
    print(f"  {k:30s}: {v:.4f}")
