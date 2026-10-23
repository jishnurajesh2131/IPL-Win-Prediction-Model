"""
Live Model Preprocessor and Feature Pipeline.

Provides reproducible, leakage-free feature selection and preprocessing for:
- Model A: Match State Baseline (30 features)
- Model B: Model A + Current Player (44 features)
- Model C: Model B + Historical Player (50 features)
- Model D: Model C + Matchup Bayes m=24 (54 features)

Enforces non-negotiable physical guards:
- Clipping required_run_rate to [0.0, 36.0]
- Clipping rrr_crr_differential to [-50.0, 50.0]
- Clamping overs_completed to [0.0, 20.0]
- Fitting OneHotEncoder strictly on training split
- Optional StandardScaler for Logistic Regression
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_COLS = ["team_a", "team_b", "venue", "city"]

BINARY_MATCH_COLS = [
    "toss_winner_is_team_a",
    "toss_decision_is_bat",
    "team_a_bats_first",
]

MACRO_STATE_COLS = [
    "current_innings",
    "is_team_a_batting",
    "current_innings_runs",
    "current_innings_wickets",
    "wickets_remaining",
    "legal_balls_bowled",
    "legal_balls_remaining",
    "overs_completed",
    "current_run_rate",
    "target_runs",
    "runs_needed",
    "required_run_rate",
    "rrr_crr_differential",
    "runs_last_6_balls",
    "wickets_last_6_balls",
    "runs_last_12_balls",
    "wickets_last_12_balls",
    "dot_ball_ratio_last_12",
    "boundary_ratio_last_12",
    "current_partnership_runs",
    "current_partnership_balls",
    "current_partnership_run_rate",
    "pre_match_prob_team_a",
]

CURRENT_PLAYER_COLS = [
    "batter_runs_current_inn",
    "batter_balls_current_inn",
    "batter_strike_rate_current",
    "batter_boundaries_current",
    "batter_dots_current",
    "batter_dot_ratio_current",
    "bowler_legal_balls_current",
    "bowler_overs_current",
    "bowler_runs_conceded_current",
    "bowler_wickets_current",
    "bowler_economy_current",
    "bowler_dots_current",
    "bowler_boundaries_conceded_current",
    "bowler_dot_ratio_current",
]

HISTORICAL_PLAYER_COLS = [
    "historical_batter_strike_rate",
    "historical_batter_average",
    "historical_batter_balls_faced",
    "historical_bowler_economy",
    "historical_bowler_strike_rate",
    "historical_bowler_balls_bowled",
]

MATCHUP_COLS = [
    "matchup_balls_prior",
    "matchup_runs_prior",
    "matchup_wickets_prior",
    "matchup_strike_rate_shrunk",
]


def get_feature_columns_for_set(feature_set: str = "D") -> List[str]:
    """
    Returns ordered list of raw feature columns for requested ablation set:
    - 'A': Match State Baseline (30 cols)
    - 'B': Model A + Current Player (44 cols)
    - 'C': Model B + Historical Player (50 cols)
    - 'D': Model C + Matchup (54 cols)
    """
    base = CATEGORICAL_COLS + BINARY_MATCH_COLS + MACRO_STATE_COLS
    if feature_set.upper() == "A":
        return base
    elif feature_set.upper() == "B":
        return base + CURRENT_PLAYER_COLS
    elif feature_set.upper() == "C":
        return base + CURRENT_PLAYER_COLS + HISTORICAL_PLAYER_COLS
    elif feature_set.upper() == "D":
        return base + CURRENT_PLAYER_COLS + HISTORICAL_PLAYER_COLS + MATCHUP_COLS
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}. Must be 'A', 'B', 'C', or 'D'.")


def clip_numerical_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Apply mandatory physical boundary clipping to numerical features:
    - required_run_rate: [0.0, 36.0]
    - rrr_crr_differential: [-50.0, 50.0]
    - overs_completed: [0.0, 20.0]
    Returns a copy with clipped values.
    """
    df = df_in.copy()
    if "required_run_rate" in df.columns:
        df["required_run_rate"] = df["required_run_rate"].clip(lower=0.0, upper=36.0)
    if "rrr_crr_differential" in df.columns:
        df["rrr_crr_differential"] = df["rrr_crr_differential"].clip(lower=-50.0, upper=50.0)
    if "overs_completed" in df.columns:
        df["overs_completed"] = df["overs_completed"].clip(lower=0.0, upper=20.0)
    return df


class LiveFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    Encapsulates complete preprocessing pipeline for live models:
    - Clips numerical columns
    - Encodes categorical columns with OneHotEncoder(handle_unknown='ignore')
    - Optionally scales numerical columns with StandardScaler (for Logistic Regression)
    - Fully compatible with scikit-learn
    """

    def __init__(self, feature_set: str = "D", scale_numerical: bool = False):
        self.feature_set = feature_set.upper()
        self.scale_numerical = scale_numerical
        self.categorical_cols = CATEGORICAL_COLS
        self.all_feature_cols = get_feature_columns_for_set(self.feature_set)
        self.numerical_cols = [c for c in self.all_feature_cols if c not in self.categorical_cols]

        self.ohe: Optional[OneHotEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_out_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit preprocessor strictly on training data."""
        # Ensure all columns are present
        X_sub = X[self.all_feature_cols]

        # 1. Fit OneHotEncoder strictly on categorical columns
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ohe.fit(X_sub[self.categorical_cols])

        # 2. Fit StandardScaler if requested on clipped numerical columns
        if self.scale_numerical:
            X_num_clipped = clip_numerical_features(X_sub[self.numerical_cols])
            self.scaler = StandardScaler()
            self.scaler.fit(X_num_clipped)
        else:
            self.scaler = None

        # Build feature names
        cat_feature_names = list(self.ohe.get_feature_names_out(self.categorical_cols))
        self.feature_names_out_ = cat_feature_names + self.numerical_cols

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features into numerical design matrix."""
        if self.ohe is None:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")

        X_sub = X[self.all_feature_cols]

        # 1. Transform categorical columns
        X_cat_trans = self.ohe.transform(X_sub[self.categorical_cols])

        # 2. Clip and transform numerical columns
        X_num_clipped = clip_numerical_features(X_sub[self.numerical_cols])
        if self.scale_numerical and self.scaler is not None:
            X_num_trans = self.scaler.transform(X_num_clipped)
        else:
            X_num_trans = X_num_clipped.values.astype(np.float64)

        # 3. Concatenate horizontally: [cat_encoded, numerical]
        return np.hstack([X_cat_trans, X_num_trans])

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return list of transformed feature names."""
        if self.feature_names_out_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return list(self.feature_names_out_)
