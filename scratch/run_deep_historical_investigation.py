"""
Master Deep Historical Pre-Match Probability Investigation and Revalidation Script.
Audits all 1,234 eligible matches from 2008 through 2026.
Generates:
1. data/processed/pre_match_probability_deep_audit.csv
2. data/processed/finals_probability_audit.csv
3. data/processed/playoff_probability_audit.csv
4. data/processed/season_probability_quality.csv
5. data/processed/pre_match_model_comparison.csv
6. Comprehensive diagnostic dumps for all user exemplar matches.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.live_prediction.walk_forward_pre_match import (
    CATEGORICAL_COLS,
    BINARY_COLS,
    CONTINUOUS_COLS,
    PRE_MATCH_FEATURE_COLS,
    TARGET_COL,
)


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        bin_mask = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
        if i == 0:
            bin_mask |= (probs == 0.0)
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(probs[bin_mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
    return round(float(ece), 4)


def run_deep_investigation():
    print("=" * 80)
    print("STARTING DEEP HISTORICAL PRE-MATCH INVESTIGATION (2008-2026)")
    print("=" * 80)

    pre_match_path = "data/processed/pre_match_dataset.csv"
    matches_path = "data/processed/matches_clean.csv"
    wf_path = "data/processed/walk_forward_pre_match_probs.csv"

    df_pre = pd.read_csv(pre_match_path)
    df_matches = pd.read_csv(matches_path)
    df_wf = pd.read_csv(wf_path)

    # Sort strictly chronologically by (date, match_id)
    df_pre["date_p"] = pd.to_datetime(df_pre["date"])
    df_pre = df_pre.sort_values(by=["date_p", "match_id"]).reset_index(drop=True)

    df_wf_map = dict(zip(df_wf["match_id"].astype(str), df_wf["pre_match_prob_team_a"].astype(float)))
    match_meta = df_matches.set_index(df_matches["match_id"].astype(str)).to_dict(orient="index")

    # =========================================================================
    # PART 1: BUILD THE AUTHORITATIVE MATCH-BY-MATCH AUDIT TABLE
    # =========================================================================
    print("\n--- PART 1: BUILDING AUTHORITATIVE DEEP AUDIT TABLE ---")
    audit_rows = []
    
    # Calculate current-season match counts prior to this match
    season_match_counts = defaultdict(int)
    within_season_matches_prior = []

    for idx, r in df_pre.iterrows():
        mid = str(r["match_id"])
        season = str(r["season"])
        date = str(r["date"])
        team_a = str(r["team_a"])
        team_b = str(r["team_b"])
        venue = str(r["venue"])
        
        meta = match_meta.get(mid, {})
        toss_winner = meta.get("toss_winner", "")
        toss_decision = meta.get("toss_decision", "")
        actual_winner = meta.get("winner", "")

        # Probabilities
        p_team_a = float(df_wf_map.get(mid, 0.50))
        p_team_b = round(1.0 - p_team_a, 4)

        # Favorite and predictions
        if p_team_a >= p_team_b:
            predicted_winner = team_a
            favorite_team = team_a
            favorite_prob = p_team_a
        else:
            predicted_winner = team_b
            favorite_team = team_b
            favorite_prob = p_team_b

        pred_correct = 1 if predicted_winner == actual_winner else 0
        fav_correct = 1 if favorite_team == actual_winner else 0
        prob_margin = round(abs(p_team_a - p_team_b), 4)

        # Within season prior count
        prior_season_count = season_match_counts[season]
        season_match_counts[season] += 1
        within_season_matches_prior.append(prior_season_count)

        # Suspicion Classification (Part 2)
        # Criteria:
        # A: Reasonable prediction (margin sensible, features align, or mild underdog upset)
        # B: Mildly suspicious (favored team had losing form but won due to fortress/H2H, or margin was wide on weak form)
        # C: Strongly suspicious (favored team > 0.70 with negative form diff and negative RPO diff)
        # D: Potential data/feature bug (feature NaN, bounds violation, attribution mismatch)
        # E: Potential model quality issue (calibration drift / systematic franchise over-weighting)
        form_diff = float(r["form_diff_last_5"])
        rpo_diff = float(r["batting_rpo_diff_last_5"])
        econ_diff = float(r["bowling_econ_diff_last_5"])
        h2h_feat = float(r["team_a_h2h_win_rate"])
        
        suspicion = "A. Reasonable prediction"
        suspicion_reason = "Standard pre-match model evaluation"

        if p_team_a >= 0.70 and form_diff < -0.20 and rpo_diff < -0.20:
            suspicion = "C. Strongly suspicious"
            suspicion_reason = "Heavy Team A favorite despite negative form and RPO differential"
        elif p_team_b >= 0.70 and form_diff > 0.20 and rpo_diff > 0.20:
            suspicion = "C. Strongly suspicious"
            suspicion_reason = "Heavy Team B favorite despite negative form and RPO differential"
        elif prob_margin >= 0.40 and abs(form_diff) < 0.05 and abs(rpo_diff) < 0.10:
            suspicion = "B. Mildly suspicious"
            suspicion_reason = "Large probability margin driven almost purely by franchise/venue prior"
        elif prob_margin >= 0.30 and ((p_team_a > 0.65 and form_diff < -0.10) or (p_team_b > 0.65 and form_diff > 0.10)):
            suspicion = "B. Mildly suspicious"
            suspicion_reason = "Noticeable divergence between recent rolling form and model confidence"

        row = {
            "match_id": mid,
            "date": date,
            "season": season,
            "team_a": team_a,
            "team_b": team_b,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "venue": venue,
            "actual_winner": actual_winner,
            "p_team_a": p_team_a,
            "p_team_b": p_team_b,
            "predicted_winner": predicted_winner,
            "prediction_correct": pred_correct,
            "favorite_probability": favorite_prob,
            "favorite_team": favorite_team,
            "favorite_correct": fav_correct,
            "probability_margin": prob_margin,
            "h2h_feature": round(float(r["team_a_h2h_win_rate"]), 4),
            "form_a_feature": round(float(r["team_a_form_last_5"]), 4),
            "form_b_feature": round(float(r["team_b_form_last_5"]), 4),
            "form_diff": round(form_diff, 4),
            "venue_feature_a": round(float(r["team_a_venue_win_rate"]), 4),
            "venue_feature_b": round(float(r["team_b_venue_win_rate"]), 4),
            "venue_chasing_feature": round(float(r["venue_chasing_win_rate"]), 4),
            "batting_rpo_a": round(float(r["team_a_batting_rpo_last_5"]), 4),
            "batting_rpo_b": round(float(r["team_b_batting_rpo_last_5"]), 4),
            "batting_rpo_diff": round(rpo_diff, 4),
            "bowling_econ_a": round(float(r["team_a_bowling_econ_last_5"]), 4),
            "bowling_econ_b": round(float(r["team_b_bowling_econ_last_5"]), 4),
            "bowling_econ_diff": round(econ_diff, 4),
            "experience_a": int(r["team_a_matches_played"]),
            "experience_b": int(r["team_b_matches_played"]),
            "within_season_matches_prior": prior_season_count,
            "suspicion_level": suspicion,
            "suspicion_reason": suspicion_reason,
            "model_version": "v1.0-rf-walkforward",
            "prediction_source": "walk_forward_pre_match",
        }
        audit_rows.append(row)

    df_deep_audit = pd.DataFrame(audit_rows)
    df_deep_audit.to_csv("data/processed/pre_match_probability_deep_audit.csv", index=False)
    print(f"Deep audit table saved: {len(df_deep_audit)} rows (data/processed/pre_match_probability_deep_audit.csv).")

    # Suspicion counts
    susp_counts = df_deep_audit["suspicion_level"].value_counts().to_dict()
    print("Suspicion Level Classification:", json.dumps(susp_counts, indent=2))

    # =========================================================================
    # PART 6 & 7: FINALS & PLAYOFF AUDIT TABLES
    # =========================================================================
    print("\n--- PARTS 6 & 7: FINALS AND PLAYOFF AUDIT TABLES ---")
    finals_records = []
    playoff_records = []

    for s, grp in df_deep_audit.groupby("season", sort=False):
        n_playoffs = 3 if s in ["2007/08", "2009"] else 4
        po_slice = grp.tail(n_playoffs)
        
        # Finals is the last match
        final_row = po_slice.iloc[-1]
        finals_records.append({
            "season": final_row["season"],
            "match_id": final_row["match_id"],
            "date": final_row["date"],
            "team_a": final_row["team_a"],
            "team_b": final_row["team_b"],
            "venue": final_row["venue"],
            "p_a": final_row["p_team_a"],
            "p_b": final_row["p_team_b"],
            "favorite": final_row["favorite_team"],
            "favorite_probability": final_row["favorite_probability"],
            "actual_winner": final_row["actual_winner"],
            "correct": final_row["prediction_correct"],
            "probability_margin": final_row["probability_margin"],
            "key_feature_drivers": f"FormDiff={final_row['form_diff']:+.3f}, H2H={final_row['h2h_feature']:.3f}, RPODiff={final_row['batting_rpo_diff']:+.3f}, VenDiff={final_row['venue_feature_a']-final_row['venue_feature_b']:+.3f}",
            "suspicion_level": final_row["suspicion_level"],
        })

        for p_idx, (_, r_po) in enumerate(po_slice.iterrows()):
            stage_name = "Final" if p_idx == (n_playoffs - 1) else f"Playoff #{p_idx+1}"
            playoff_records.append({
                "season": r_po["season"],
                "stage": stage_name,
                "match_id": r_po["match_id"],
                "date": r_po["date"],
                "team_a": r_po["team_a"],
                "team_b": r_po["team_b"],
                "venue": r_po["venue"],
                "p_a": r_po["p_team_a"],
                "p_b": r_po["p_team_b"],
                "favorite": r_po["favorite_team"],
                "favorite_probability": r_po["favorite_probability"],
                "actual_winner": r_po["actual_winner"],
                "correct": r_po["prediction_correct"],
                "probability_margin": r_po["probability_margin"],
                "suspicion_level": r_po["suspicion_level"],
            })

    pd.DataFrame(finals_records).to_csv("data/processed/finals_probability_audit.csv", index=False)
    pd.DataFrame(playoff_records).to_csv("data/processed/playoff_probability_audit.csv", index=False)
    print(f"Finals audit saved ({len(finals_records)} finals)")
    print(f"Playoffs audit saved ({len(playoff_records)} matches)")

    # =========================================================================
    # PART 8: SYSTEMATIC TEAM BIAS ANALYSIS
    # =========================================================================
    print("\n--- PART 8: SYSTEMATIC TEAM BIAS ANALYSIS ---")
    team_stats = defaultdict(lambda: {
        "matches": 0,
        "actual_wins": 0,
        "times_favorite": 0,
        "favorite_wins": 0,
        "prob_sum": 0.0,
        "prob_when_fav": [],
        "prob_when_dog": [],
        "brier_terms": [],
        "logloss_terms": [],
    })

    for _, r in df_deep_audit.iterrows():
        t1, t2 = r["team_a"], r["team_b"]
        p1, p2 = r["p_team_a"], r["p_team_b"]
        w = r["actual_winner"]

        # Team 1
        d1 = team_stats[t1]
        d1["matches"] += 1
        d1["prob_sum"] += p1
        d1["brier_terms"].append((p1 - (1 if w == t1 else 0)) ** 2)
        d1["logloss_terms"].append(-np.log(max(p1 if w == t1 else (1 - p1), 1e-6)))
        if w == t1:
            d1["actual_wins"] += 1
        if p1 >= 0.50:
            d1["times_favorite"] += 1
            d1["prob_when_fav"].append(p1)
            if w == t1:
                d1["favorite_wins"] += 1
        else:
            d1["prob_when_dog"].append(p1)

        # Team 2
        d2 = team_stats[t2]
        d2["matches"] += 1
        d2["prob_sum"] += p2
        d2["brier_terms"].append((p2 - (1 if w == t2 else 0)) ** 2)
        d2["logloss_terms"].append(-np.log(max(p2 if w == t2 else (1 - p2), 1e-6)))
        if w == t2:
            d2["actual_wins"] += 1
        if p2 >= 0.50:
            d2["times_favorite"] += 1
            d2["prob_when_fav"].append(p2)
            if w == t2:
                d2["favorite_wins"] += 1
        else:
            d2["prob_when_dog"].append(p2)

    team_bias_table = []
    for t, d in team_stats.items():
        n = d["matches"]
        if n < 10:
            continue
        act_wr = d["actual_wins"] / n
        avg_prob = d["prob_sum"] / n
        fav_wr = (d["favorite_wins"] / d["times_favorite"]) if d["times_favorite"] > 0 else 0.0
        avg_fav_p = np.mean(d["prob_when_fav"]) if d["prob_when_fav"] else 0.0
        avg_dog_p = np.mean(d["prob_when_dog"]) if d["prob_when_dog"] else 0.0
        brier = np.mean(d["brier_terms"])
        ll = np.mean(d["logloss_terms"])
        calibration_gap = avg_prob - act_wr

        team_bias_table.append({
            "Team": t,
            "Matches": n,
            "Actual Win Rate": round(act_wr, 4),
            "Avg Predicted Prob": round(avg_prob, 4),
            "Calibration Gap": round(calibration_gap, 4),
            "Times Favorite": d["times_favorite"],
            "Favorite Win Rate": round(fav_wr, 4),
            "Avg P (as Fav)": round(avg_fav_p, 4),
            "Avg P (as Dog)": round(avg_dog_p, 4),
            "Brier Score": round(brier, 4),
            "Log Loss": round(ll, 4),
        })

    df_team_bias = pd.DataFrame(team_bias_table).sort_values(by="Matches", ascending=False)
    print(df_team_bias.to_string(index=False))

    # =========================================================================
    # PART 9: SEARCH FOR VENUE BIAS
    # =========================================================================
    print("\n--- PART 9: VENUE BIAS ANALYSIS ---")
    venue_stats = defaultdict(lambda: {
        "matches": 0,
        "correct": 0,
        "fav_probs": [],
        "team_a_wins": 0,
        "team_a_prob_sum": 0.0,
    })

    for _, r in df_deep_audit.iterrows():
        v = r["venue"]
        d = venue_stats[v]
        d["matches"] += 1
        if r["prediction_correct"] == 1:
            d["correct"] += 1
        d["fav_probs"].append(r["favorite_probability"])
        if r["actual_winner"] == r["team_a"]:
            d["team_a_wins"] += 1
        d["team_a_prob_sum"] += r["p_team_a"]

    venue_rows = []
    for v, d in venue_stats.items():
        if d["matches"] < 15:
            continue
        n = d["matches"]
        acc = d["correct"] / n
        avg_conf = np.mean(d["fav_probs"])
        team_a_wr = d["team_a_wins"] / n
        avg_p_a = d["team_a_prob_sum"] / n
        venue_rows.append({
            "Venue": v,
            "Matches": n,
            "Accuracy": round(acc, 4),
            "Avg Confidence": round(avg_conf, 4),
            "Team A Win Rate": round(team_a_wr, 4),
            "Avg P(A)": round(avg_p_a, 4),
            "Calibration Gap": round(avg_p_a - team_a_wr, 4),
        })

    df_venue_bias = pd.DataFrame(venue_rows).sort_values(by="Matches", ascending=False)
    print(df_venue_bias.head(15).to_string(index=False))

    # =========================================================================
    # PART 10: H2H BIAS & NEUTRALIZATION ANALYSIS
    # =========================================================================
    print("\n--- PART 10: H2H BIAS & NEUTRALIZATION TEST ---")
    # For every match, check the difference between H2H win rate and 0.50
    h2h_deviations = abs(df_deep_audit["h2h_feature"] - 0.50)
    print(f"H2H feature mean absolute deviation from neutral (0.50): {h2h_deviations.mean():.4f}")
    print(f"H2H feature range: [{df_deep_audit['h2h_feature'].min():.4f}, {df_deep_audit['h2h_feature'].max():.4f}]")

    # =========================================================================
    # PART 11: FEATURE ABLATION / IMPORTANCE
    # =========================================================================
    print("\n--- PART 11: FEATURE ABLATION & LOG LOSS SENSITIVITY ---")
    # Train baseline on 2008-2021, evaluate on 2022-2023 validation
    train_mask = df_pre["season"].apply(lambda s: int(str(s).split("/")[0]) <= 2021)
    val_mask = df_pre["season"].isin(["2022", "2023"])
    test_mask = df_pre["season"].isin(["2024", "2025", "2026"])

    X_train_raw = df_pre.loc[train_mask, PRE_MATCH_FEATURE_COLS]
    y_train = df_pre.loc[train_mask, TARGET_COL].values

    X_val_raw = df_pre.loc[val_mask, PRE_MATCH_FEATURE_COLS]
    y_val = df_pre.loc[val_mask, TARGET_COL].values

    preproc = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS)],
        remainder="passthrough",
    )
    X_train_t = preproc.fit_transform(X_train_raw)
    X_val_t = preproc.transform(X_val_raw)

    rf_base = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=1)
    rf_base.fit(X_train_t, y_train)
    p_val_base = rf_base.predict_proba(X_val_t)[:, 1]
    base_ll = log_loss(y_val, p_val_base)
    base_brier = brier_score_loss(y_val, p_val_base)
    base_acc = np.mean((p_val_base >= 0.50) == y_val)

    print(f"Baseline Validation (2022-2023): Acc={base_acc:.4f}, LL={base_ll:.4f}, Brier={base_brier:.4f}")

    # Feature groups to ablate
    ablation_groups = {
        "Without H2H": ["team_a_h2h_win_rate", "h2h_matches_played"],
        "Without Venue Features": ["team_a_venue_matches", "team_a_venue_win_rate", "team_b_venue_matches", "team_b_venue_win_rate", "venue_win_rate_diff", "venue_chasing_win_rate"],
        "Without Form Features": ["team_a_form_last_5", "team_b_form_last_5", "team_a_form_last_10", "team_b_form_last_10", "form_diff_last_5"],
        "Without Batting RPO": ["team_a_batting_rpo_last_5", "team_b_batting_rpo_last_5", "batting_rpo_diff_last_5"],
        "Without Bowling Economy": ["team_a_bowling_econ_last_5", "team_b_bowling_econ_last_5", "bowling_econ_diff_last_5"],
        "Without Experience": ["team_a_matches_played", "team_b_matches_played"],
        "Without Toss Info": ["toss_winner_is_team_a", "toss_decision_is_bat", "team_a_bats_first"],
    }

    ablation_results = []
    for group_name, drop_cols in ablation_groups.items():
        feat_subset = [c for c in PRE_MATCH_FEATURE_COLS if c not in drop_cols]
        cat_subset = [c for c in CATEGORICAL_COLS if c in feat_subset]
        
        preproc_ab = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_subset)],
            remainder="passthrough",
        )
        X_tr_ab = preproc_ab.fit_transform(df_pre.loc[train_mask, feat_subset])
        X_va_ab = preproc_ab.transform(df_pre.loc[val_mask, feat_subset])

        rf_ab = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=1)
        rf_ab.fit(X_tr_ab, y_train)
        p_val_ab = rf_ab.predict_proba(X_va_ab)[:, 1]
        
        ab_ll = log_loss(y_val, p_val_ab)
        ab_brier = brier_score_loss(y_val, p_val_ab)
        ab_acc = np.mean((p_val_ab >= 0.50) == y_val)
        
        ablation_results.append({
            "Ablated Group": group_name,
            "Val Acc": round(ab_acc, 4),
            "Val LogLoss": round(ab_ll, 4),
            "Val Brier": round(ab_brier, 4),
            "Delta LogLoss (vs Base)": round(ab_ll - base_ll, 4),
        })

    print(pd.DataFrame(ablation_results).to_string(index=False))

    # =========================================================================
    # PART 13: SEASON-BY-SEASON QUALITY TABLE
    # =========================================================================
    print("\n--- PART 13: SEASON PROBABILITY QUALITY TABLE ---")
    season_qual_rows = []
    for s, grp in df_deep_audit.groupby("season", sort=False):
        n = len(grp)
        acc = grp["prediction_correct"].mean()
        y_true = (grp["actual_winner"] == grp["team_a"]).astype(int).values
        p_a = grp["p_team_a"].values
        p_clip = np.clip(p_a, 1e-6, 1 - 1e-6)
        
        ll = -np.mean(y_true * np.log(p_clip) + (1 - y_true) * np.log(1 - p_clip))
        brier = np.mean((p_a - y_true) ** 2)
        ece = compute_ece(p_a, y_true)
        avg_margin = grp["probability_margin"].mean()
        avg_conf = grp["favorite_probability"].mean()
        overconf = (grp["favorite_probability"] > 0.70).mean()

        season_qual_rows.append({
            "season": s,
            "matches": n,
            "favorite_accuracy": round(acc, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(ll, 4),
            "ece": round(ece, 4),
            "avg_prob_margin": round(avg_margin, 4),
            "avg_confidence": round(avg_conf, 4),
            "overconfidence_rate": round(overconf, 4),
        })

    df_season_qual = pd.DataFrame(season_qual_rows)
    df_season_qual.to_csv("data/processed/season_probability_quality.csv", index=False)
    print(df_season_qual.to_string(index=False))

    # =========================================================================
    # PART 16: CALIBRATION BUCKET ANALYSIS
    # =========================================================================
    print("\n--- PART 16: CALIBRATION BY PROBABILITY BUCKET ---")
    bucket_edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.00]
    bucket_rows = []

    fav_probs = df_deep_audit["favorite_probability"].values
    fav_wins = df_deep_audit["favorite_correct"].values

    for i in range(len(bucket_edges) - 1):
        low, high = bucket_edges[i], bucket_edges[i + 1]
        mask = (fav_probs >= low) & (fav_probs < high) if i < len(bucket_edges) - 2 else (fav_probs >= low) & (fav_probs <= high)
        cnt = np.sum(mask)
        if cnt > 0:
            actual_rate = np.mean(fav_wins[mask])
            pred_conf = np.mean(fav_probs[mask])
            bucket_rows.append({
                "Bucket": f"{int(low*100)}-{int(high*100)}%",
                "Count": cnt,
                "Mean Predicted Prob": round(pred_conf, 4),
                "Actual Win Rate": round(actual_rate, 4),
                "Calibration Gap": round(pred_conf - actual_rate, 4),
            })

    df_buckets = pd.DataFrame(bucket_rows)
    print(df_buckets.to_string(index=False))

    # Overall ECE
    overall_ece = compute_ece(df_deep_audit["p_team_a"].values, (df_deep_audit["actual_winner"] == df_deep_audit["team_a"]).astype(int).values)
    print(f"\nOverall Pre-Match ECE across all 1,234 matches: {overall_ece:.4f}")

    # =========================================================================
    # PART 17 & 18: CONTROLLED MODEL COMPARISON & VALIDATION
    # =========================================================================
    print("\n--- PART 17 & 18: MODEL COMPARISON (TRAIN: 2008-2021, VAL: 2022-2023, TEST: 2024-2026 UNTOUCHED) ---")
    
    # 1. Baseline Random Forest
    rf_val_p = rf_base.predict_proba(X_val_t)[:, 1]
    rf_val_acc = np.mean((rf_val_p >= 0.50) == y_val)
    rf_val_ll = log_loss(y_val, rf_val_p)
    rf_val_brier = brier_score_loss(y_val, rf_val_p)
    rf_val_ece = compute_ece(rf_val_p, y_val)

    # 2. Logistic Regression (L2 regularized)
    lr = LogisticRegression(C=0.1, max_iter=2000, solver="liblinear", random_state=42)
    lr.fit(X_train_t, y_train)
    lr_val_p = lr.predict_proba(X_val_t)[:, 1]
    lr_val_acc = np.mean((lr_val_p >= 0.50) == y_val)
    lr_val_ll = log_loss(y_val, lr_val_p)
    lr_val_brier = brier_score_loss(y_val, lr_val_p)
    lr_val_ece = compute_ece(lr_val_p, y_val)

    # 3. XGBoost
    xgb = XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        eval_metric="logloss",
    )
    xgb.fit(X_train_t, y_train)
    xgb_val_p = xgb.predict_proba(X_val_t)[:, 1]
    xgb_val_acc = np.mean((xgb_val_p >= 0.50) == y_val)
    xgb_val_ll = log_loss(y_val, xgb_val_p)
    xgb_val_brier = brier_score_loss(y_val, xgb_val_p)
    xgb_val_ece = compute_ece(xgb_val_p, y_val)

    # 4. Calibrated Random Forest (Platt Scaling with 5-fold CV)
    rf_for_cal = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=1)
    cal_rf = CalibratedClassifierCV(estimator=rf_for_cal, cv=5, method="sigmoid")
    cal_rf.fit(X_train_t, y_train)
    cal_rf_val_p = cal_rf.predict_proba(X_val_t)[:, 1]
    cal_rf_val_acc = np.mean((cal_rf_val_p >= 0.50) == y_val)
    cal_rf_val_ll = log_loss(y_val, cal_rf_val_p)
    cal_rf_val_brier = brier_score_loss(y_val, cal_rf_val_p)
    cal_rf_val_ece = compute_ece(cal_rf_val_p, y_val)

    # Untouched Test Evaluation (2024-2026) for reporting comparison
    X_test_raw = df_pre.loc[test_mask, PRE_MATCH_FEATURE_COLS]
    y_test = df_pre.loc[test_mask, TARGET_COL].values
    X_test_t = preproc.transform(X_test_raw)

    def eval_test(model):
        p = model.predict_proba(X_test_t)[:, 1]
        return {
            "test_acc": round(float(np.mean((p >= 0.50) == y_test)), 4),
            "test_ll": round(float(log_loss(y_test, p)), 4),
            "test_brier": round(float(brier_score_loss(y_test, p)), 4),
            "test_ece": compute_ece(p, y_test),
        }

    rf_test = eval_test(rf_base)
    lr_test = eval_test(lr)
    xgb_test = eval_test(xgb)
    cal_rf_test = eval_test(cal_rf)

    comparison_records = [
        {
            "model_version": "RandomForest (Current Baseline)",
            "validation_accuracy": round(rf_val_acc, 4),
            "validation_logloss": round(rf_val_ll, 4),
            "validation_brier": round(rf_val_brier, 4),
            "validation_ece": rf_val_ece,
            "test_accuracy": rf_test["test_acc"],
            "test_logloss": rf_test["test_ll"],
            "test_brier": rf_test["test_brier"],
            "test_ece": rf_test["test_ece"],
            "notes": "Current walk-forward production architecture (d=6, leaf=5)",
        },
        {
            "model_version": "LogisticRegression (L2 C=0.1)",
            "validation_accuracy": round(lr_val_acc, 4),
            "validation_logloss": round(lr_val_ll, 4),
            "validation_brier": round(lr_val_brier, 4),
            "validation_ece": lr_val_ece,
            "test_accuracy": lr_test["test_acc"],
            "test_logloss": lr_test["test_ll"],
            "test_brier": lr_test["test_brier"],
            "test_ece": lr_test["test_ece"],
            "notes": "Linear model with OneHotEncoder; underfits nonlinear venue interactions",
        },
        {
            "model_version": "XGBoost (d=3, lr=0.05)",
            "validation_accuracy": round(xgb_val_acc, 4),
            "validation_logloss": round(xgb_val_ll, 4),
            "validation_brier": round(xgb_val_brier, 4),
            "validation_ece": xgb_val_ece,
            "test_accuracy": xgb_test["test_acc"],
            "test_logloss": xgb_test["test_ll"],
            "test_brier": xgb_test["test_brier"],
            "test_ece": xgb_test["test_ece"],
            "notes": "Gradient boosted trees on validation split",
        },
        {
            "model_version": "Calibrated RF (Platt Sigmoid)",
            "validation_accuracy": round(cal_rf_val_acc, 4),
            "validation_logloss": round(cal_rf_val_ll, 4),
            "validation_brier": round(cal_rf_val_brier, 4),
            "validation_ece": cal_rf_val_ece,
            "test_accuracy": cal_rf_test["test_acc"],
            "test_logloss": cal_rf_test["test_ll"],
            "test_brier": cal_rf_test["test_brier"],
            "test_ece": cal_rf_test["test_ece"],
            "notes": "Post-hoc Platt probability scaling applied to Random Forest",
        },
    ]

    df_model_comp = pd.DataFrame(comparison_records)
    df_model_comp.to_csv("data/processed/pre_match_model_comparison.csv", index=False)
    print("\nMODEL VALIDATION & TEST COMPARISON:")
    print(df_model_comp.to_string(index=False))

    # =========================================================================
    # PART 3, 4, 5: DEEP DIAGNOSTICS ON USER EXEMPLAR MATCHES
    # =========================================================================
    print("\n" + "=" * 80)
    print("DETAILED INVESTIGATION OF USER EXEMPLAR MATCHES")
    print("=" * 80)

    def print_match_detail(title: str, m_ids: List[str]):
        print(f"\n>>> {title} <<<")
        for mid in m_ids:
            row = df_deep_audit[df_deep_audit["match_id"] == str(mid)]
            if row.empty:
                print(f"Match {mid} not found in eligible dataset!")
                continue
            r = row.iloc[0]
            print("-" * 60)
            print(f"Match ID: {r['match_id']} | Date: {r['date']} | Season: {r['season']}")
            print(f"Teams: {r['team_a']} vs {r['team_b']}")
            print(f"Venue: {r['venue']} | Toss: {r['toss_winner']} ({r['toss_decision']})")
            print(f"Actual Winner: {r['actual_winner']} | Prediction Correct: {'YES' if r['prediction_correct']==1 else 'NO'}")
            print(f"Probabilities: P({r['team_a']}) = {r['p_team_a']:.4f} | P({r['team_b']}) = {r['p_team_b']:.4f} (Margin: {r['probability_margin']:.4f})")
            print(f"Form (Last 5): {r['team_a']}={r['form_a_feature']:.4f}, {r['team_b']}={r['form_b_feature']:.4f} (Diff: {r['form_diff']:+.4f})")
            print(f"H2H Win Rate for {r['team_a']}: {r['h2h_feature']:.4f}")
            print(f"Venue Win Rates: {r['team_a']}={r['venue_feature_a']:.4f}, {r['team_b']}={r['venue_feature_b']:.4f} (Venue Chasing: {r['venue_chasing_feature']:.4f})")
            print(f"Batting RPO (Last 5): {r['team_a']}={r['batting_rpo_a']:.4f}, {r['team_b']}={r['batting_rpo_b']:.4f} (Diff: {r['batting_rpo_diff']:+.4f})")
            print(f"Bowling Economy (Last 5): {r['team_a']}={r['bowling_econ_a']:.4f}, {r['team_b']}={r['bowling_econ_b']:.4f} (Diff: {r['bowling_econ_diff']:+.4f})")
            print(f"Experience (Matches Played): {r['team_a']}={r['experience_a']}, {r['team_b']}={r['experience_b']}")
            print(f"Within-Season Matches Prior to this: {r['within_season_matches_prior']}")
            print(f"Suspicion Level: {r['suspicion_level']} ({r['suspicion_reason']})")

    # 1. 2011 Final
    print_match_detail("1. 2011 IPL FINAL (CSK vs RCB)", ["501271"])

    # 2. 2015 CSK vs MI (All meetings)
    csk_mi_2015 = df_deep_audit[(df_deep_audit["season"] == "2015") & (
        ((df_deep_audit["team_a"] == "Chennai Super Kings") & (df_deep_audit["team_b"] == "Mumbai Indians")) |
        ((df_deep_audit["team_a"] == "Mumbai Indians") & (df_deep_audit["team_b"] == "Chennai Super Kings"))
    )]["match_id"].tolist()
    print_match_detail("2. 2015 CSK vs MI (ALL 4 MEETINGS)", csk_mi_2015)

    # 3. 2019 CSK vs MI (All meetings)
    csk_mi_2019 = df_deep_audit[(df_deep_audit["season"] == "2019") & (
        ((df_deep_audit["team_a"] == "Chennai Super Kings") & (df_deep_audit["team_b"] == "Mumbai Indians")) |
        ((df_deep_audit["team_a"] == "Mumbai Indians") & (df_deep_audit["team_b"] == "Chennai Super Kings"))
    )]["match_id"].tolist()
    print_match_detail("3. 2019 CSK vs MI (ALL 4 MEETINGS)", csk_mi_2019)

    # 4. 2021 CSK Matches & Final
    csk_2021_final = ["1254117"]
    print_match_detail("4. 2021 IPL FINAL (CSK vs KKR)", csk_2021_final)

    # 5. 2025 RCB Matches & Final
    rcb_2025_final = ["1473511"]
    print_match_detail("5. 2025 IPL FINAL (RCB vs PBKS)", rcb_2025_final)

    print("\nDeep investigation run completed successfully.")


if __name__ == "__main__":
    run_deep_investigation()
