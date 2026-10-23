"""
Model Utilities: Data Loading, Temporal Splitting, and Preprocessing Pipelines.
Authoritative source: PRE_MATCH_FEATURE_SPECIFICATION_V2.md
"""

from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Feature group definitions
METADATA_COLS: List[str] = ["match_id", "season", "date"]

CATEGORICAL_COLS: List[str] = ["team_a", "team_b", "venue", "city"]

BINARY_NUMERICAL_COLS: List[str] = [
    "toss_winner_is_team_a",
    "toss_decision_is_bat",
    "team_a_bats_first",
]

CONTINUOUS_NUMERICAL_COLS: List[str] = [
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

NUMERICAL_COLS: List[str] = BINARY_NUMERICAL_COLS + CONTINUOUS_NUMERICAL_COLS

FEATURE_COLS: List[str] = CATEGORICAL_COLS + NUMERICAL_COLS

TARGET_COL: str = "target_team_a_win"


def load_dataset(dataset_path: str = "data/processed/pre_match_dataset.csv") -> pd.DataFrame:
    """Load pre-match dataset and sort chronologically by date and match_id."""
    df = pd.read_csv(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["date", "match_id"]).reset_index(drop=True)
    return df


TRAIN_SEASONS: List[str] = [
    "2007/08",
    "2008",
    "2009",
    "2009/10",
    "2010",
    "2011",
    "2012",
    "2013",
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
    "2020/21",
    "2021",
]
VAL_SEASONS: List[str] = ["2022", "2023"]
TEST_SEASONS: List[str] = ["2024", "2025", "2026"]


def temporal_train_val_test_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset chronologically according to project specification:
    - Training: 2008–2021 (872 rows)
    - Validation: 2022–2023 (147 rows)
    - Final Test: 2024–2026 (215 rows)
    """
    train_df = df[df["season"].astype(str).isin(TRAIN_SEASONS)].copy()
    val_df = df[df["season"].astype(str).isin(VAL_SEASONS)].copy()
    test_df = df[df["season"].astype(str).isin(TEST_SEASONS)].copy()

    # Validation check
    total_len = len(train_df) + len(val_df) + len(test_df)
    if total_len != len(df):
        raise ValueError(
            f"Temporal split partition mismatch: train={len(train_df)}, "
            f"val={len(val_df)}, test={len(test_df)}, total={len(df)}"
        )

    return train_df, val_df, test_df


def build_preprocessor() -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer for pre-match features:
    - OneHotEncoder on categorical features (handle_unknown='ignore', sparse_output=False)
    - Passthrough on numerical features
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
            (
                "num",
                "passthrough",
                NUMERICAL_COLS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def get_feature_group_mapping(feature_names_out: List[str]) -> Dict[str, str]:
    """
    Map one-hot encoded and numerical transformed feature names back to their original feature group.
    e.g.:
    cat__team_a_Chennai Super Kings -> team_a
    cat__venue_Wankhede Stadium -> venue
    num__form_diff_last_5 -> form_diff_last_5
    """
    mapping = {}
    for name in feature_names_out:
        if name.startswith("cat__"):
            clean = name[5:]  # strip 'cat__'
            for orig_col in CATEGORICAL_COLS:
                if clean.startswith(orig_col + "_"):
                    mapping[name] = orig_col
                    break
            if name not in mapping:
                mapping[name] = "categorical_other"
        elif name.startswith("num__"):
            mapping[name] = name[5:]
        else:
            mapping[name] = name
    return mapping
