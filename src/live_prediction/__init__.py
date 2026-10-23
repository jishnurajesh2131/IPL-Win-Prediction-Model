"""
Live Prediction Package.

Provides state tracking, live feature engineering, historical player priors,
matchup tracking, and the end-to-end dataset generation pipeline for live in-match
win probability prediction.
"""

from .live_state_builder import LiveMatchStateMachine
from .match_state_features import extract_match_state_features
from .player_features import HistoricalPlayerTracker, extract_current_player_features
from .matchup_features import HistoricalMatchupTracker, extract_matchup_features
from .live_dataset_builder import generate_live_dataset
from .live_validation import validate_live_dataset

__all__ = [
    "LiveMatchStateMachine",
    "extract_match_state_features",
    "HistoricalPlayerTracker",
    "extract_current_player_features",
    "HistoricalMatchupTracker",
    "extract_matchup_features",
    "generate_live_dataset",
    "validate_live_dataset",
]
