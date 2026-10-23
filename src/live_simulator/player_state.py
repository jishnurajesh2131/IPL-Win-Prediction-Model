"""
Player State Management for Live Simulator.

Handles:
1. Current-match active batter & bowler micro-figures (runs, balls, SR, boundaries, dots, econ).
2. Historical player priors (< t) using Empirical Bayesian shrinkage from prior matches.
"""

from typing import Dict, Any, List, Optional
from src.live_prediction.player_features import (
    HistoricalPlayerTracker,
    extract_current_player_features,
)


class PlayerStateManager:
    """
    Manages in-match player figures and prior historical player statistics.
    """

    def __init__(self, historical_tracker: Optional[HistoricalPlayerTracker] = None):
        self.historical_tracker = historical_tracker or HistoricalPlayerTracker()

    def get_current_player_features(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract in-match micro-features for the active striker and active bowler.
        """
        return extract_current_player_features(raw_state=raw_state)

    def get_historical_batter_stats(self, batter: str) -> Dict[str, float]:
        """
        Query Empirical Bayes shrunk historical batting statistics prior to match start.
        """
        return self.historical_tracker.get_historical_batter_stats(batter)

    def get_historical_bowler_stats(self, bowler: str) -> Dict[str, float]:
        """
        Query Empirical Bayes shrunk historical bowling statistics prior to match start.
        """
        return self.historical_tracker.get_historical_bowler_stats(bowler)
