"""
Batter-vs-Bowler Matchup State Management for Live Simulator.

Tracks head-to-head encounters strictly chronologically (< t) with
Empirical Bayesian shrinkage (m = 24.0 balls).
"""

from typing import Dict, Any, Optional
from src.live_prediction.matchup_features import (
    HistoricalMatchupTracker,
    extract_matchup_features,
)


class MatchupStateManager:
    """
    Manages batter-vs-bowler head-to-head interaction state prior to match start.
    """

    def __init__(self, historical_tracker: Optional[HistoricalMatchupTracker] = None):
        self.historical_tracker = historical_tracker or HistoricalMatchupTracker()

    def get_matchup_features(
        self,
        striker: str,
        bowler: str,
        historical_batter_sr: float,
    ) -> Dict[str, Any]:
        """
        Query Empirical Bayes shrunk head-to-head statistics prior to match start.
        """
        return extract_matchup_features(
            tracker=self.historical_tracker,
            striker=striker,
            bowler=bowler,
            historical_batter_sr=historical_batter_sr,
        )
