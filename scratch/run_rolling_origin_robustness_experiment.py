"""
Final Robustness Experiment: Rolling-Origin Pre-Match Validation (2008-2023)
Rigorous historical cross-era comparison between Baseline RF v1 and Model D1 (Core Dynamic RF).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.live_prediction.walk_forward_pre_match import (
    CATEGORICAL_COLS,
    PRE_MATCH_FEATURE_COLS,
    TARGET_COL,
)

CORE_DYNAMIC_FEATURES = [
    "team_a_form_last_5",
    "team_b_form_last_5",
    "form_diff_last_5",
    "team_a_batting_rpo_last_5",
    "team_b_batting_rpo_last_5",
    "batting_rpo_diff_last_5",
    "team_a_bowling_econ_last_5",
    "team_b_bowling_econ_last_5",
    "bowling_econ_diff_last_5",
]


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


def compute_sharpness_diagnostics(probs: np.ndarray) -> Dict[str, float]:
    """Compute probability distribution and sharpness diagnostics."""
    return {
        "mean_p": round(float(np.mean(probs)), 4),
        "std_p": round(float(np.std(probs)), 4),
        "min_p": round(float(np.min(probs)), 4),
        "max_p": round(float(np.max(probs)), 4),
        "pct_outside_45_55": round(float(np.mean((probs < 0.45) | (probs > 0.55)) * 100), 2),
        "pct_gt_60": round(float(np.mean(probs > 0.60) * 100), 2),
        "pct_lt_40": round(float(np.mean(probs < 0.40) * 100), 2),
    }


def define_windows() -> List[Dict[str, Any]]:
    """Define the 4 exact historical rolling-origin windows."""
    return [
        {
            "window_id": 1,
            "name": "Window 1 (2017-2018)",
            "era_desc": "Pre-2019 IPL environment",
            "train_seasons": ["2007/08", "2009", "2009/10", "2011", "2012", "2013", "2014", "2015", "2016"],
            "train_years_label": "2008–2016",
            "val_seasons": ["2017", "2018"],
            "val_years_label": "2017–2018",
        },
        {
            "window_id": 2,
            "name": "Window 2 (2019-2020)",
            "era_desc": "2019–2020 environment",
            "train_seasons": ["2007/08", "2009", "2009/10", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018"],
            "train_years_label": "2008–2018",
            "val_seasons": ["2019", "2020/21"],
            "val_years_label": "2019–2020",
        },
        {
            "window_id": 3,
            "name": "Window 3 (2021)",
            "era_desc": "2021 split-season environment",
            "train_seasons": ["2007/08", "2009", "2009/10", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020/21"],
            "train_years_label": "2008–2020",
            "val_seasons": ["2021"],
            "val_years_label": "2021",
        },
        {
            "window_id": 4,
            "name": "Window 4 (2022-2023)",
            "era_desc": "2022–2023 post-major structural change environment (Mega Auction / Expansion)",
            "train_seasons": ["2007/08", "2009", "2009/10", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020/21", "2021"],
            "train_years_label": "2008–2021",
            "val_seasons": ["2022", "2023"],
            "val_years_label": "2022–2023",
        },
    ]


def run_single_experiment_pass(df_pre: pd.DataFrame, windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute one full pass of the rolling-origin benchmark."""
    window_results = []
    window_predictions = {}

    for win in windows:
        w_id = win["window_id"]
        w_name = win["name"]
        tr_seasons = win["train_seasons"]
        va_seasons = win["val_seasons"]

        train_mask = df_pre["season"].isin(tr_seasons)
        val_mask = df_pre["season"].isin(va_seasons)

        df_tr = df_pre[train_mask].copy()
        df_va = df_pre[val_mask].copy()

        # =====================================================================
        # DATA INTEGRITY AND SPLIT CONTAMINATION ASSERTIONS
        # =====================================================================
        # 1. No overlap between train and validation matches
        train_ids = set(df_tr["match_id"])
        val_ids = set(df_va["match_id"])
        assert len(train_ids.intersection(val_ids)) == 0, f"Contamination: Window {w_id} has overlap between train and val matches!"

        # 2. Strict chronological boundary
        max_train_date = df_tr["date_p"].max()
        min_val_date = df_va["date_p"].min()
        assert max_train_date < min_val_date, f"Contamination: Window {w_id} train date ({max_train_date}) >= val date ({min_val_date})!"

        # 3. Sealed test check: zero matches from 2024-2026
        forbidden_seasons = {"2024", "2025", "2026"}
        assert len(set(df_tr["season"]).intersection(forbidden_seasons)) == 0, f"Contamination: Window {w_id} train has 2024-2026 matches!"
        assert len(set(df_va["season"]).intersection(forbidden_seasons)) == 0, f"Contamination: Window {w_id} val has 2024-2026 matches!"

        y_train = df_tr[TARGET_COL].values
        y_val = df_va[TARGET_COL].values
        n_val = len(y_val)

        # =====================================================================
        # MODEL A: BASELINE RANDOM FOREST V1
        # Preprocessor fitted strictly on training data
        # =====================================================================
        preproc_base = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS)],
            remainder="passthrough",
        )
        X_tr_base = preproc_base.fit_transform(df_tr[PRE_MATCH_FEATURE_COLS])
        X_va_base = preproc_base.transform(df_va[PRE_MATCH_FEATURE_COLS])

        rf_base = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,
        )
        rf_base.fit(X_tr_base, y_train)
        p_base = rf_base.predict_proba(X_va_base)[:, 1]

        # =====================================================================
        # MODEL D1: CORE DYNAMIC RANDOM FOREST
        # Preprocessor fitted strictly on training data
        # =====================================================================
        preproc_d1 = StandardScaler()
        X_tr_d1 = preproc_d1.fit_transform(df_tr[CORE_DYNAMIC_FEATURES])
        X_va_d1 = preproc_d1.transform(df_va[CORE_DYNAMIC_FEATURES])

        rf_d1 = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,
        )
        rf_d1.fit(X_tr_d1, y_train)
        p_d1 = rf_d1.predict_proba(X_va_d1)[:, 1]

        window_predictions[f"w{w_id}_base"] = p_base
        window_predictions[f"w{w_id}_d1"] = p_d1

        # =====================================================================
        # EVALUATE METRICS FOR BOTH MODELS
        # =====================================================================
        models_eval = [
            ("Baseline RF v1", p_base),
            ("Model D1 (Core Dynamic RF)", p_d1),
        ]

        for mname, probs in models_eval:
            preds = (probs >= 0.50).astype(int)
            acc = accuracy_score(y_val, preds)
            ll = log_loss(y_val, np.clip(probs, 1e-15, 1.0 - 1e-15))
            brier = brier_score_loss(y_val, probs)
            ece, mce = compute_ece_mce(probs, y_val)
            try:
                auc = roc_auc_score(y_val, probs)
            except ValueError:
                auc = 0.50

            sharp = compute_sharpness_diagnostics(probs)
            n_correct = int(np.sum(preds == y_val))
            n_incorrect = int(n_val - n_correct)

            rec = {
                "window_id": w_id,
                "window_name": w_name,
                "train_years": win["train_years_label"],
                "val_years": win["val_years_label"],
                "era_desc": win["era_desc"],
                "model": mname,
                "matches": n_val,
                "correct": n_correct,
                "incorrect": n_incorrect,
                "accuracy": round(float(acc), 4),
                "log_loss": round(float(ll), 4),
                "brier": round(float(brier), 4),
                "ece": ece,
                "auc": round(float(auc), 4),
                "std_p": sharp["std_p"],
                "min_p": sharp["min_p"],
                "max_p": sharp["max_p"],
                "pct_outside_45_55": sharp["pct_outside_45_55"],
                "pct_gt_60": sharp["pct_gt_60"],
                "pct_lt_40": sharp["pct_lt_40"],
            }
            window_results.append(rec)

    return {
        "results": window_results,
        "predictions": window_predictions,
    }


def plot_robustness_figures(df_res: pd.DataFrame, output_dir: str):
    """Generate simple, publication-suitable matplotlib visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    windows = [1, 2, 3, 4]
    window_labels = [
        "W1\n(2017-18)",
        "W2\n(2019-20)",
        "W3\n(2021)",
        "W4\n(2022-23)",
    ]

    base_rows = df_res[df_res["model"] == "Baseline RF v1"].sort_values(by="window_id")
    d1_rows = df_res[df_res["model"] == "Model D1 (Core Dynamic RF)"].sort_values(by="window_id")

    metrics_config = [
        ("log_loss", "Log Loss (Lower is Better)", "log_loss_by_window.png", True),
        ("brier", "Brier Score (Lower is Better)", "brier_by_window.png", True),
        ("accuracy", "Accuracy (Higher is Better)", "accuracy_by_window.png", False),
        ("auc", "ROC-AUC (Higher is Better)", "auc_by_window.png", False),
    ]

    x = np.arange(len(windows))
    width = 0.35

    # Individual plots
    for col, title, fname, lower_is_better in metrics_config:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
        
        base_vals = base_rows[col].values
        d1_vals = d1_rows[col].values

        rects1 = ax.bar(x - width/2, base_vals, width, label="Baseline RF v1", color="#3b82f6", alpha=0.9, edgecolor="black", linewidth=0.8)
        rects2 = ax.bar(x + width/2, d1_vals, width, label="Model D1 (Core Dynamic)", color="#10b981", alpha=0.9, edgecolor="black", linewidth=0.8)

        ax.set_ylabel(col.replace("_", " ").title())
        ax.set_title(f"Rolling-Origin Robustness: {title}", fontsize=11, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(window_labels, fontsize=10)
        ax.legend(frameon=True, loc="best")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Annotate values
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.4f}" if col != "accuracy" else f"{h*100:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.4f}" if col != "accuracy" else f"{h*100:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

        # Padding
        min_y = min(np.min(base_vals), np.min(d1_vals))
        max_y = max(np.max(base_vals), np.max(d1_vals))
        ax.set_ylim(max(0.0, min_y * 0.85), max_y * 1.12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

    # Combined 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    axes = axes.flatten()

    for idx, (col, title, _, _) in enumerate(metrics_config):
        ax = axes[idx]
        base_vals = base_rows[col].values
        d1_vals = d1_rows[col].values

        rects1 = ax.bar(x - width/2, base_vals, width, label="Baseline RF v1", color="#3b82f6", alpha=0.9, edgecolor="black", linewidth=0.8)
        rects2 = ax.bar(x + width/2, d1_vals, width, label="Model D1 (Core Dynamic)", color="#10b981", alpha=0.9, edgecolor="black", linewidth=0.8)

        ax.set_ylabel(col.replace("_", " ").title())
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(window_labels, fontsize=9)
        ax.legend(frameon=True, fontsize=8, loc="best")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

        min_y = min(np.min(base_vals), np.min(d1_vals))
        max_y = max(np.max(base_vals), np.max(d1_vals))
        ax.set_ylim(max(0.0, min_y * 0.85), max_y * 1.15)

    plt.suptitle("IPL Pre-Match T0 Model Robustness across Rolling-Origin Historical Windows", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rolling_origin_combined_4panel.png"))
    plt.close()


def run_experiment():
    print("=" * 85)
    print("FINAL ROBUSTNESS EXPERIMENT: ROLLING-ORIGIN PRE-MATCH VALIDATION")
    print("Evaluation across 4 temporal windows (2008-2023) with Locked 2024-2026 Test Protected")
    print("=" * 85)

    # 1. Load data
    pre_match_path = "data/processed/pre_match_dataset.csv"
    df_pre = pd.read_csv(pre_match_path)
    df_pre["date_p"] = pd.to_datetime(df_pre["date"])
    df_pre = df_pre.sort_values(by=["date_p", "match_id"]).reset_index(drop=True)

    # Filter strictly to matches through 2023. Sealed test 2024-2026 is NEVER touched!
    sealed_test_seasons = ["2024", "2025", "2026"]
    df_pre_eval = df_pre[~df_pre["season"].isin(sealed_test_seasons)].copy().reset_index(drop=True)
    print(f"Total historical matches available for rolling origin (2008-2023): {len(df_pre_eval)}")
    print(f"Locked 2024-2026 matches untouched: {len(df_pre) - len(df_pre_eval)}")

    windows = define_windows()

    # 2. Run Pass 1
    print("\n--- EXECUTING RUN 1 ---")
    run1 = run_single_experiment_pass(df_pre_eval, windows)
    df_res1 = pd.DataFrame(run1["results"])

    # 3. Run Pass 2 for strict reproducibility verification
    print("\n--- EXECUTING RUN 2 (REPRODUCIBILITY CHECK) ---")
    run2 = run_single_experiment_pass(df_pre_eval, windows)
    df_res2 = pd.DataFrame(run2["results"])

    # Verify Bitwise / Numerical Reproducibility
    max_prob_diff = 0.0
    for k in run1["predictions"].keys():
        diff = float(np.max(np.abs(run1["predictions"][k] - run2["predictions"][k])))
        if diff > max_prob_diff:
            max_prob_diff = diff

    metric_cols = ["accuracy", "log_loss", "brier", "ece", "auc", "std_p"]
    max_metric_diff = 0.0
    for col in metric_cols:
        diff_m = float(np.max(np.abs(df_res1[col].values - df_res2[col].values)))
        if diff_m > max_metric_diff:
            max_metric_diff = diff_m

    print(f"\nReproducibility Audit:")
    print(f"  Max Probability Difference between runs: {max_prob_diff:.10f}")
    print(f"  Max Metric Difference between runs:      {max_metric_diff:.10f}")
    assert max_prob_diff == 0.0, "Reproducibility failure: predictions differed across runs!"
    assert max_metric_diff == 0.0, "Reproducibility failure: metrics differed across runs!"
    print("  Status: 100% BITWISE IDENTICAL ACROSS RUNS (PASS)")

    # 4. Generate Main Table
    print("\n" + "=" * 85)
    print("MAIN RESULTS TABLE (SECTION 8)")
    print("=" * 85)
    table_cols = [
        "window_id", "train_years", "val_years", "model", "matches",
        "accuracy", "log_loss", "brier", "ece", "auc",
        "std_p", "min_p", "max_p", "pct_outside_45_55"
    ]
    df_main_table = df_res1[table_cols].copy()
    print(df_main_table.to_string(index=False))

    # 5. Head-to-Head Comparison
    print("\n" + "=" * 85)
    print("HEAD-TO-HEAD COMPARISON TABLE (SECTION 9)")
    print("=" * 85)

    head_to_head = []
    metric_wins = {
        "log_loss": 0,
        "brier": 0,
        "ece": 0,
        "accuracy": 0,
        "auc": 0,
    }

    for w in [1, 2, 3, 4]:
        sub = df_res1[df_res1["window_id"] == w]
        base = sub[sub["model"] == "Baseline RF v1"].iloc[0]
        d1 = sub[sub["model"] == "Model D1 (Core Dynamic RF)"].iloc[0]

        diff_acc = round(d1["accuracy"] - base["accuracy"], 4)
        diff_ll = round(base["log_loss"] - d1["log_loss"], 4)      # positive means D1 has lower log loss (wins)
        diff_br = round(base["brier"] - d1["brier"], 4)            # positive means D1 has lower brier (wins)
        diff_ece = round(base["ece"] - d1["ece"], 4)               # positive means D1 has lower ECE (wins)
        diff_auc = round(d1["auc"] - base["auc"], 4)               # positive means D1 has higher AUC (wins)

        w_acc = "D1" if diff_acc > 0 else ("Base" if diff_acc < 0 else "Tie")
        w_ll = "D1" if diff_ll > 0 else ("Base" if diff_ll < 0 else "Tie")
        w_br = "D1" if diff_br > 0 else ("Base" if diff_br < 0 else "Tie")
        w_ece = "D1" if diff_ece > 0 else ("Base" if diff_ece < 0 else "Tie")
        w_auc = "D1" if diff_auc > 0 else ("Base" if diff_auc < 0 else "Tie")

        if w_ll == "D1": metric_wins["log_loss"] += 1
        if w_br == "D1": metric_wins["brier"] += 1
        if w_ece == "D1": metric_wins["ece"] += 1
        if w_acc == "D1": metric_wins["accuracy"] += 1
        if w_auc == "D1": metric_wins["auc"] += 1

        head_to_head.append({
            "window": f"Window {w}",
            "val_era": str(d1["val_years"]),
            "matches": int(d1["matches"]),
            "acc_diff (D1-Base)": f"{diff_acc:+.4f} ({w_acc})",
            "ll_diff (Base-D1)": f"{diff_ll:+.4f} ({w_ll})",
            "brier_diff (Base-D1)": f"{diff_br:+.4f} ({w_br})",
            "ece_diff (Base-D1)": f"{diff_ece:+.4f} ({w_ece})",
            "auc_diff (D1-Base)": f"{diff_auc:+.4f} ({w_auc})",
        })

    df_h2h = pd.DataFrame(head_to_head)
    print(df_h2h.to_string(index=False))

    # 6. Robustness Score (Section 10)
    print("\n" + "=" * 85)
    print("ROBUSTNESS SCORE SUMMARY (SECTION 10)")
    print("=" * 85)
    print(f"  Log Loss:  D1 wins {metric_wins['log_loss']} / 4 windows")
    print(f"  Brier:     D1 wins {metric_wins['brier']} / 4 windows")
    print(f"  ECE:       D1 wins {metric_wins['ece']} / 4 windows")
    print(f"  Accuracy:  D1 wins {metric_wins['accuracy']} / 4 windows")
    print(f"  AUC:       D1 wins {metric_wins['auc']} / 4 windows")

    total_wins = sum(metric_wins.values())
    total_comparisons = 4 * 5
    print(f"  Total Primary Metric Wins: {total_wins} / {total_comparisons} ({total_wins/total_comparisons*100:.1f}%)")

    # Classification & Probability Quality Robustness Assessment
    # Decision logic based on section 16:
    # ROBUST D1: D1 demonstrates consistent advantages across multiple rolling validation windows (wins 3 or 4)
    # PARTIALLY ROBUST D1: D1 wins multiple windows but not consistently (wins 2)
    # ERA-SPECIFIC D1: D1's advantage is concentrated primarily in 2022-2023 (wins only 1, in window 4)
    # BASELINE MORE ROBUST: Baseline wins most rolling windows (D1 wins 0 or 1)

    ll_wins = metric_wins["log_loss"]
    acc_wins = metric_wins["accuracy"]

    # Classify probability quality
    if ll_wins >= 3:
        prob_quality_status = "ROBUST"
    elif ll_wins == 2:
        prob_quality_status = "PARTIALLY ROBUST"
    elif ll_wins == 1 and df_res1[(df_res1["window_id"]==4) & (df_res1["model"]=="Model D1 (Core Dynamic RF)")]["log_loss"].values[0] < df_res1[(df_res1["window_id"]==4) & (df_res1["model"]=="Baseline RF v1")]["log_loss"].values[0]:
        prob_quality_status = "ERA-SPECIFIC"
    else:
        prob_quality_status = "WEAKER"

    # Classify classification
    if acc_wins >= 3:
        class_status = "ROBUST"
    elif acc_wins == 2:
        class_status = "PARTIALLY ROBUST"
    elif acc_wins == 1:
        class_status = "ERA-SPECIFIC"
    else:
        class_status = "WEAKER"

    # Overall scientific conclusion
    if total_wins >= 14:
        final_conclusion = "ROBUST D1: D1 demonstrates cross-era robustness across the evaluated historical validation windows."
    elif total_wins >= 9:
        final_conclusion = "PARTIALLY ROBUST D1: D1 demonstrates partial temporal robustness, with performance advantages that vary by historical era."
    elif total_wins <= 5 and metric_wins["log_loss"] == 1:
        final_conclusion = "ERA-SPECIFIC D1: D1's observed advantage is largely era-specific and does not consistently generalize across earlier IPL periods."
    else:
        final_conclusion = "BASELINE MORE ROBUST: The baseline demonstrates stronger temporal robustness than D1 across the evaluated historical windows."

    print(f"\nD1 Classification Robustness:    {class_status}")
    print(f"D1 Probability Quality Status:   {prob_quality_status}")
    print(f"Final Scientific Conclusion:     {final_conclusion}")

    # 7. Generate Visualizations (Section 12)
    fig_dir = "docs/figures/pre_match_robustness"
    print(f"\nGenerating publication figures into: {fig_dir} ...")
    plot_robustness_figures(df_res1, fig_dir)
    print("Figures generated successfully.")

    # 8. Save structured results JSON
    save_data = {
        "experiment": "Rolling-Origin Pre-Match Validation",
        "dataset_scope": "2008-2023 (Matches through 2023 only; 2024-2026 locked test protected)",
        "reproducibility": {
            "max_prob_diff": max_prob_diff,
            "max_metric_diff": max_metric_diff,
            "status": "PASS",
        },
        "windows": windows,
        "main_results": df_res1.to_dict(orient="records"),
        "head_to_head": head_to_head,
        "metric_wins": metric_wins,
        "total_wins": total_wins,
        "total_comparisons": total_comparisons,
        "classification_status": class_status,
        "probability_quality_status": prob_quality_status,
        "final_conclusion": final_conclusion,
    }

    with open("scratch/rolling_origin_robustness_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print("Structured results saved to scratch/rolling_origin_robustness_results.json")

    # 9. Format Terminal Summary (Section 20)
    print("\n" + "=" * 60)
    print("ROLLING-ORIGIN ROBUSTNESS EXPERIMENT COMPLETE")
    print("=" * 60)

    for w in [1, 2, 3, 4]:
        sub = df_res1[df_res1["window_id"] == w]
        b_row = sub[sub["model"] == "Baseline RF v1"].iloc[0]
        d_row = sub[sub["model"] == "Model D1 (Core Dynamic RF)"].iloc[0]
        print(f"\nWindow {w} ({b_row['val_years']}, N={b_row['matches']}):")
        print(f"Baseline: Acc={b_row['accuracy']*100:.2f}%, LL={b_row['log_loss']:.4f}, Brier={b_row['brier']:.4f}, ECE={b_row['ece']:.4f}, AUC={b_row['auc']:.4f}")
        print(f"D1:       Acc={d_row['accuracy']*100:.2f}%, LL={d_row['log_loss']:.4f}, Brier={d_row['brier']:.4f}, ECE={d_row['ece']:.4f}, AUC={d_row['auc']:.4f}")

    print("\n" + "-" * 60)
    print("D1 PRIMARY METRIC WINS")
    print("-" * 60)
    print(f"Log Loss: {metric_wins['log_loss']} / 4")
    print(f"Brier:    {metric_wins['brier']} / 4")
    print(f"ECE:      {metric_wins['ece']} / 4")
    print(f"Accuracy: {metric_wins['accuracy']} / 4")
    print(f"AUC:      {metric_wins['auc']} / 4")

    print("\n" + "-" * 60)
    print("TEMPORAL ROBUSTNESS")
    print("-" * 60)
    print(f"D1 classification:\n{class_status}")
    print(f"\nD1 probability quality:\n{prob_quality_status}")

    print("\n" + "-" * 60)
    print("LEAKAGE AUDIT")
    print("-" * 60)
    print("PASS")

    print("\n" + "-" * 60)
    print("REPRODUCIBILITY")
    print("-" * 60)
    print("PASS")

    print("\n" + "-" * 60)
    print("LOCKED TEST")
    print("-" * 60)
    print("UNTOUCHED")

    print("\n" + "-" * 60)
    print("FINAL CONCLUSION")
    print("-" * 60)
    print(final_conclusion)


if __name__ == "__main__":
    run_experiment()
