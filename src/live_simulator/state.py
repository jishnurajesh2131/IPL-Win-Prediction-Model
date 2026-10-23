"""
Live Match and Innings State Management for Live Simulator.

Wraps and coordinates ball-by-ball match state tracking:
- Scores, wickets, legal balls, overs completed
- Innings 1 total and Innings 2 chase dynamics (target, runs needed, RRR)
- Active partnership dynamics (runs, balls, run rate)
- Rolling 6-ball and 12-ball momentum buffers
"""

from typing import Dict, Any, Optional
from src.live_prediction.live_state_builder import LiveMatchStateMachine
from src.live_prediction.match_state_features import extract_match_state_features


class MatchStateManager:
    """
    Manages the sequential ball-by-ball physical match state.
    Guarantees strict temporal order and exact parity with live_dataset.csv.
    """

    def __init__(self, match_id: str, team_a: str, team_b: str, pre_match_prob_team_a: float):
        self.match_id = str(match_id)
        self.team_a = team_a
        self.team_b = team_b
        self.pre_match_prob_team_a = pre_match_prob_team_a

        # Core state machine
        self.state_machine = LiveMatchStateMachine(
            match_id=self.match_id,
            team_a=self.team_a,
            team_b=self.team_b,
        )

        self.last_raw_state: Optional[Dict[str, Any]] = None
        self.last_match_features: Optional[Dict[str, Any]] = None

    def process_delivery(self, delivery: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single delivery event, updates internal state, and returns
        extracted match state features.
        """
        raw_state = self.state_machine.update_with_delivery(delivery)
        self.last_raw_state = raw_state

        match_features = extract_match_state_features(
            raw_state=raw_state,
            team_a=self.team_a,
            pre_match_prob_team_a=self.pre_match_prob_team_a,
        )
        self.last_match_features = match_features
        return match_features

    @property
    def current_innings(self) -> int:
        return self.state_machine.current_innings

    @property
    def target_runs(self) -> int:
        return self.state_machine.target_runs

    def get_raw_state(self) -> Optional[Dict[str, Any]]:
        return self.last_raw_state

    def get_match_features(self) -> Optional[Dict[str, Any]]:
        return self.last_match_features

    def get_batting_figures(self, innings: Optional[int] = None) -> list:
        return self.state_machine.get_batting_figures(innings)

    def get_bowling_figures(self, innings: Optional[int] = None) -> list:
        return self.state_machine.get_bowling_figures(innings)

    def get_batting_team(self, innings: Optional[int] = None) -> str:
        inn = innings or self.current_innings
        return self.state_machine.batting_teams.get(inn, "")

    def get_bowling_team(self, innings: Optional[int] = None) -> str:
        inn = innings or self.current_innings
        return self.state_machine.bowling_teams.get(inn, "")

