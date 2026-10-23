"""
Feature Builder for Live Simulator.

Assembles the exact 54 raw predictor columns for Feature Set D in the precise order
and data representation expected by LiveFeaturePreprocessor.
"""

from typing import Dict, Any, List
import pandas as pd
from src.live_prediction.live_model_preprocessor import (
    get_feature_columns_for_set,
    CATEGORICAL_COLS,
    BINARY_MATCH_COLS,
    MACRO_STATE_COLS,
    CURRENT_PLAYER_COLS,
    HISTORICAL_PLAYER_COLS,
    MATCHUP_COLS,
)


FEATURE_SET_D_COLUMNS = get_feature_columns_for_set("D")


class LiveFeatureBuilder:
    """
    Constructs a model-facing pandas DataFrame row from current match and player states.
    """

    def __init__(
        self,
        team_a: str,
        team_b: str,
        venue: str,
        city: str,
        toss_winner_is_team_a: int,
        toss_decision_is_bat: int,
        team_a_bats_first: int,
    ):
        self.match_context = {
            "team_a": team_a,
            "team_b": team_b,
            "venue": venue,
            "city": city,
            "toss_winner_is_team_a": int(toss_winner_is_team_a),
            "toss_decision_is_bat": int(toss_decision_is_bat),
            "team_a_bats_first": int(team_a_bats_first),
        }
        self.expected_columns = FEATURE_SET_D_COLUMNS

    def build_feature_row(
        self,
        match_features: Dict[str, Any],
        current_player_features: Dict[str, Any],
        historical_batter_features: Dict[str, float],
        historical_bowler_features: Dict[str, float],
        matchup_features: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Combines contextual, macro-state, player, and matchup features into a single-row
        DataFrame strictly aligned with Feature Set D.
        """
        row_dict: Dict[str, Any] = {}

        # 1. Match context (Categoricals & Binary)
        row_dict.update(self.match_context)

        # 2. Macro match state
        for col in MACRO_STATE_COLS:
            if col not in match_features:
                raise KeyError(f"Missing required macro state feature: {col}")
            row_dict[col] = match_features[col]

        # 3. Current player micro-state
        for col in CURRENT_PLAYER_COLS:
            if col not in current_player_features:
                raise KeyError(f"Missing required current player feature: {col}")
            row_dict[col] = current_player_features[col]

        # 4. Historical player priors
        for col in ["historical_batter_strike_rate", "historical_batter_average", "historical_batter_balls_faced"]:
            if col not in historical_batter_features:
                raise KeyError(f"Missing required historical batter feature: {col}")
            row_dict[col] = historical_batter_features[col]

        for col in ["historical_bowler_economy", "historical_bowler_strike_rate", "historical_bowler_balls_bowled"]:
            if col not in historical_bowler_features:
                raise KeyError(f"Missing required historical bowler feature: {col}")
            row_dict[col] = historical_bowler_features[col]

        # 5. Batter-vs-bowler matchups
        for col in MATCHUP_COLS:
            if col not in matchup_features:
                raise KeyError(f"Missing required matchup feature: {col}")
            row_dict[col] = matchup_features[col]

        # Construct single-row DataFrame ordered by expected columns
        df_row = pd.DataFrame([row_dict], columns=self.expected_columns)
        return df_row
