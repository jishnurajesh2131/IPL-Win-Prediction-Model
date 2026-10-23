"""
Model Runner and Inference Routing for Live Simulator.

Loads and routes deliveries to the authoritative Two-Model XGBoost (Feature Set D) pipeline:
- Innings 1: primary_live_model_inn1.joblib + primary_live_preprocessor_inn1.joblib
- Innings 2: primary_live_model_inn2.joblib + primary_live_preprocessor_inn2.joblib

Handles deterministic terminal outcomes:
- Team A Win: P(A) = 1.0, P(B) = 0.0
- Team B Win: P(A) = 0.0, P(B) = 1.0
- Regulation Tie: P(A) = 0.50, P(B) = 0.50
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Optional


DEFAULT_INN1_MODEL = "models/live/primary_live_model_inn1.joblib"
DEFAULT_INN2_MODEL = "models/live/primary_live_model_inn2.joblib"
DEFAULT_INN1_PREP = "models/live/primary_live_preprocessor_inn1.joblib"
DEFAULT_INN2_PREP = "models/live/primary_live_preprocessor_inn2.joblib"


class LiveModelRunner:
    """
    Encapsulates Two-Model pipeline inference with strict innings routing and terminal handling.
    """

    def __init__(
        self,
        model_inn1_path: str = DEFAULT_INN1_MODEL,
        model_inn2_path: str = DEFAULT_INN2_MODEL,
        prep_inn1_path: str = DEFAULT_INN1_PREP,
        prep_inn2_path: str = DEFAULT_INN2_PREP,
    ):
        self._verify_file_exists(model_inn1_path, "Innings 1 Model")
        self._verify_file_exists(model_inn2_path, "Innings 2 Model")
        self._verify_file_exists(prep_inn1_path, "Innings 1 Preprocessor")
        self._verify_file_exists(prep_inn2_path, "Innings 2 Preprocessor")

        self.model_inn1 = joblib.load(model_inn1_path)
        self.model_inn2 = joblib.load(model_inn2_path)
        self.prep_inn1 = joblib.load(prep_inn1_path)
        self.prep_inn2 = joblib.load(prep_inn2_path)

        # Sanity check feature dimensions
        n_feats_inn1 = len(self.prep_inn1.get_feature_names_out())
        n_feats_inn2 = len(self.prep_inn2.get_feature_names_out())
        if n_feats_inn1 != 141 or n_feats_inn2 != 141:
            raise ValueError(
                f"Unexpected feature count in preprocessors: inn1={n_feats_inn1}, inn2={n_feats_inn2}. Expected 141."
            )

    @staticmethod
    def _verify_file_exists(path: str, label: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} artifact not found at: {path}")

    def predict_probability(
        self,
        df_row: pd.DataFrame,
        innings: int,
        is_terminal: bool = False,
        terminal_winner: Optional[str] = None,
        team_a: Optional[str] = None,
        is_tie: bool = False,
    ) -> Tuple[float, float, str]:
        """
        Calculates P(Team A Win) and P(Team B Win) for the current state.
        
        Returns:
            Tuple[float, float, str]: (p_team_a, p_team_b, prediction_type)
        """
        # 1. Deterministic terminal handling
        if is_terminal:
            if is_tie:
                return 0.50, 0.50, "terminal_tie"
            elif terminal_winner == team_a:
                return 1.0, 0.0, "terminal_team_a_win"
            else:
                return 0.0, 1.0, "terminal_team_b_win"

        # 2. In-match active prediction via appropriate innings model
        if innings == 1:
            X_trans = self.prep_inn1.transform(df_row)
            prob_raw = self.model_inn1.predict_proba(X_trans)[0, 1]
            pred_type = "model_inn1"
        elif innings == 2:
            X_trans = self.prep_inn2.transform(df_row)
            prob_raw = self.model_inn2.predict_proba(X_trans)[0, 1]
            pred_type = "model_inn2"
        else:
            raise ValueError(f"Invalid innings: {innings}. Must be 1 or 2.")

        p_team_a = float(np.clip(prob_raw, 0.0, 1.0))
        p_team_a = round(p_team_a, 4)
        p_team_b = round(1.0 - p_team_a, 4)

        return p_team_a, p_team_b, pred_type
