"""
Match-Level Chronological Walk-Forward Pre-Match Probability Generator.

Simulates expanding historical predictions strictly out-of-sample:
For each match M:
- Training set = eligible matches strictly before M ((date, match_id) < (M.date, M.match_id)).
- Preprocessor (OneHotEncoder) is fitted strictly on this historical subset.
- Random Forest (n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=42) is fitted on this historical subset.
- Generates out-of-sample predict_proba() for match M.
- For cold-start matches (first 40 matches of season 2008), assigns neutral prior 0.50.

Guarantees 100% temporal isolation:
- Zero self-inclusion
- Zero future match leakage
- Zero future preprocessing fitting
"""

import os
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


CATEGORICAL_COLS = ["team_a", "team_b", "venue", "city"]
BINARY_COLS = ["toss_winner_is_team_a", "toss_decision_is_bat", "team_a_bats_first"]
CONTINUOUS_COLS = [
    "team_a_form_last_5",
    "team_b_form_last_5",
    "team_a_form_last_10",
    "team_b_form_last_10",
    "form_diff_last_5",
    "team_a_matches_played",
    "team_b_matches_played",
    "h2h_matches_played",
    "team_a_h2h_win_rate",
    "team_a_venue_matches",
    "team_a_venue_win_rate",
    "team_b_venue_matches",
    "team_b_venue_win_rate",
    "venue_win_rate_diff",
    "venue_chasing_win_rate",
    "team_a_batting_rpo_last_5",
    "team_b_batting_rpo_last_5",
    "batting_rpo_diff_last_5",
    "team_a_bowling_econ_last_5",
    "team_b_bowling_econ_last_5",
    "bowling_econ_diff_last_5",
]

PRE_MATCH_FEATURE_COLS = CATEGORICAL_COLS + BINARY_COLS + CONTINUOUS_COLS
TARGET_COL = "target_team_a_win"


def predict_single_match_walk_forward(
    df: pd.DataFrame,
    match_idx: int,
    min_training_matches: int = 40,
) -> float:
    """
    Fits preprocessing and Random Forest strictly on matches [0 .. match_idx - 1]
    and predicts match_idx.
    """
    if match_idx < min_training_matches:
        return 0.50

    X_train_raw = df.loc[: match_idx - 1, PRE_MATCH_FEATURE_COLS]
    y_train = df.loc[: match_idx - 1, TARGET_COL].values

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            )
        ],
        remainder="passthrough",
    )
    X_train_trans = preprocessor.fit_transform(X_train_raw)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1,
    )
    rf.fit(X_train_trans, y_train)

    X_curr_raw = df.loc[[match_idx], PRE_MATCH_FEATURE_COLS]
    X_curr_trans = preprocessor.transform(X_curr_raw)
    prob_raw = rf.predict_proba(X_curr_trans)[0, 1]
    return round(float(prob_raw), 4)


def get_training_boundary_for_match(
    df: pd.DataFrame,
    match_idx: int,
) -> Dict[str, Any]:
    """
    Inspects the exact training boundary for match_idx.
    """
    curr_match_id = str(df.loc[match_idx, "match_id"])
    curr_date = str(df.loc[match_idx, "date"])
    curr_season = str(df.loc[match_idx, "season"])

    if match_idx == 0:
        return {
            "match_id": curr_match_id,
            "date": curr_date,
            "season": curr_season,
            "historical_training_match_count": 0,
            "training_match_ids": [],
            "training_max_date": None,
            "training_max_match_id": None,
        }

    train_subset = df.loc[: match_idx - 1]
    return {
        "match_id": curr_match_id,
        "date": curr_date,
        "season": curr_season,
        "historical_training_match_count": len(train_subset),
        "training_match_ids": [str(m) for m in train_subset["match_id"].tolist()],
        "training_max_date": str(train_subset.iloc[-1]["date"]),
        "training_max_match_id": str(train_subset.iloc[-1]["match_id"]),
    }


def generate_walk_forward_probabilities(
    pre_match_csv: str = "data/processed/pre_match_dataset.csv",
    output_audit_csv: str = "data/processed/walk_forward_pre_match_probs.csv",
    min_training_matches: int = 40,
) -> pd.DataFrame:
    """
    Executes match-level expanding walk-forward pre-match win probability estimation.
    """
    print(f"Loading pre-match dataset from {pre_match_csv}...")
    df = pd.read_csv(pre_match_csv)
    df["date_parsed"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["date_parsed", "match_id"]).reset_index(drop=True)

    n_matches = len(df)
    print(f"Total eligible pre-match rows to evaluate: {n_matches}")
    print(f"Cold-start threshold: first {min_training_matches} matches assigned 0.50 neutral prior.")

    results: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    for i in range(n_matches):
        mid = str(df.loc[i, "match_id"])
        season = str(df.loc[i, "season"])
        date_str = str(df.loc[i, "date"])
        team_a = df.loc[i, "team_a"]
        team_b = df.loc[i, "team_b"]
        actual_target = int(df.loc[i, TARGET_COL])

        if i < min_training_matches:
            prob = 0.50
            cold_start = 1
            earliest_train = ""
            latest_train = ""
            train_count = 0
        else:
            cold_start = 0
            # Historical training subset strictly BEFORE match i
            X_train_raw = df.loc[: i - 1, PRE_MATCH_FEATURE_COLS]
            y_train = df.loc[: i - 1, TARGET_COL].values
            train_count = len(X_train_raw)
            earliest_train = str(df.loc[0, "match_id"])
            latest_train = str(df.loc[i - 1, "match_id"])

            # Fit preprocessor strictly on historical training subset
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        CATEGORICAL_COLS,
                    )
                ],
                remainder="passthrough",
            )
            X_train_trans = preprocessor.fit_transform(X_train_raw)

            # Fit pre-match Random Forest strictly on historical training subset
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=1,
            )
            rf.fit(X_train_trans, y_train)

            # Predict current match i
            X_curr_raw = df.loc[[i], PRE_MATCH_FEATURE_COLS]
            X_curr_trans = preprocessor.transform(X_curr_raw)
            prob_raw = rf.predict_proba(X_curr_trans)[0, 1]
            prob = round(float(prob_raw), 4)

        results.append(
            {
                "match_id": mid,
                "season": season,
                "date": date_str,
                "team_a": team_a,
                "team_b": team_b,
                "target_team_a_win": actual_target,
                "pre_match_prob_team_a": prob,
                "cold_start_flag": cold_start,
                "historical_training_match_count": train_count,
                "earliest_training_match": earliest_train,
                "latest_training_match": latest_train,
            }
        )

        if (i + 1) % 100 == 0 or (i + 1) == n_matches:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            rem = (n_matches - (i + 1)) / max(0.01, rate)
            print(
                f"  Walk-forward progress: {i + 1}/{n_matches} matches ({prob:.4f} at match {mid}) "
                f"[{elapsed:.1f}s elapsed, ~{rem:.1f}s remaining]"
            )

    out_df = pd.DataFrame(results)
    if output_audit_csv:
        os.makedirs(os.path.dirname(output_audit_csv), exist_ok=True)
        out_df.to_csv(output_audit_csv, index=False)
        print(f"Saved walk-forward pre-match audit table to {output_audit_csv}")

    return out_df


def load_walk_forward_prob_map(
    audit_csv: str = "data/processed/walk_forward_pre_match_probs.csv",
) -> Dict[str, float]:
    """
    Load precomputed leak-free pre-match probability map: match_id (str) -> probability (float).
    """
    df = pd.read_csv(audit_csv)
    return dict(zip(df["match_id"].astype(str), df["pre_match_prob_team_a"].astype(float)))


if __name__ == "__main__":
    generate_walk_forward_probabilities()
