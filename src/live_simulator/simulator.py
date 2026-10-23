"""
Master Sequential Replay Engine for Live Simulator.

Replays historical IPL matches delivery-by-delivery:
- Reconstructs instantaneous physical and player states without lookahead leakage.
- Establishes T0 pre-match baseline with leakage-safe walk-forward probability anchor.
- Handles wides/no-balls as distinct prediction points without advancing legal ball count.
- Transitions seamlessly across Innings 1 and Innings 2.
- Evaluates active deliveries using Two-Model XGBoost (Set D) and handles terminal states deterministically.
"""

import os
import csv
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd

from src.live_prediction.live_dataset_builder import (
    load_deliveries_by_match,
    HistoricalPlayerTracker,
    HistoricalMatchupTracker,
)
from src.live_prediction.walk_forward_pre_match import load_walk_forward_prob_map

from .state import MatchStateManager
from .player_state import PlayerStateManager
from .matchup_state import MatchupStateManager
from .feature_builder import LiveFeatureBuilder
from .model_runner import LiveModelRunner


@dataclass
class SimulationEvent:
    event_number: int
    innings: int
    ball: float
    delivery_index: int
    is_legal: bool
    score: int
    wickets: int
    legal_balls_bowled: int
    overs_completed: float
    current_run_rate: float
    target_runs: int
    runs_needed: int
    required_run_rate: float
    rrr_crr_differential: float
    striker: str
    non_striker: str
    bowler: str
    team_a_win_probability: float
    team_b_win_probability: float
    prediction_type: str
    feature_snapshot: Optional[Dict[str, Any]] = None
    batting_team: str = ""
    bowling_team: str = ""
    batting: List[Dict[str, Any]] = field(default_factory=list)
    bowling: List[Dict[str, Any]] = field(default_factory=list)
    innings_1: Optional[Dict[str, Any]] = None
    innings_2: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_number": self.event_number,
            "innings": self.innings,
            "ball": self.ball,
            "delivery_index": self.delivery_index,
            "is_legal": self.is_legal,
            "score": self.score,
            "wickets": self.wickets,
            "legal_balls_bowled": self.legal_balls_bowled,
            "overs_completed": self.overs_completed,
            "current_run_rate": self.current_run_rate,
            "target_runs": self.target_runs,
            "runs_needed": self.runs_needed,
            "required_run_rate": self.required_run_rate,
            "rrr_crr_differential": self.rrr_crr_differential,
            "striker": self.striker,
            "non_striker": self.non_striker,
            "bowler": self.bowler,
            "team_a_win_probability": self.team_a_win_probability,
            "team_b_win_probability": self.team_b_win_probability,
            "prediction_type": self.prediction_type,
            "batting_team": self.batting_team,
            "bowling_team": self.bowling_team,
            "batting": self.batting,
            "bowling": self.bowling,
            "innings_1": self.innings_1,
            "innings_2": self.innings_2,
        }


@dataclass
class SimulationResult:
    match_id: str
    season: str
    date: str
    venue: str
    city: str
    team_a: str
    team_b: str
    toss_winner: str
    toss_decision: str
    winner: str
    events: List[SimulationEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "season": self.season,
            "date": self.date,
            "venue": self.venue,
            "city": self.city,
            "team_a": self.team_a,
            "team_b": self.team_b,
            "toss_winner": self.toss_winner,
            "toss_decision": self.toss_decision,
            "winner": self.winner,
            "total_events": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }


class MatchSimulator:
    """
    Orchestrates delivery-by-delivery simulation of a historical match.
    """

    def __init__(
        self,
        matches_csv: str = "data/processed/matches_clean.csv",
        deliveries_csv: str = "data/processed/deliveries_clean.csv",
        walk_forward_csv: str = "data/processed/walk_forward_pre_match_probs.csv",
        model_runner: Optional[LiveModelRunner] = None,
    ):
        self.matches_csv = matches_csv
        self.deliveries_csv = deliveries_csv
        self.walk_forward_csv = walk_forward_csv
        self.model_runner = model_runner or LiveModelRunner()

        self._all_matches: Optional[List[Dict[str, Any]]] = None
        self._all_deliveries: Optional[Dict[str, List[Dict[str, Any]]]] = None
        self._pre_match_probs: Optional[Dict[str, float]] = None

    def _ensure_data_loaded(self):
        if self._all_matches is None:
            with open(self.matches_csv, "r", encoding="utf-8") as f:
                self._all_matches = list(csv.DictReader(f))
            # Sort chronologically
            self._all_matches.sort(key=lambda m: (m["date"], int(m["match_id"])))

        if self._all_deliveries is None:
            self._all_deliveries = load_deliveries_by_match(self.deliveries_csv)

        if self._pre_match_probs is None:
            if os.path.exists(self.walk_forward_csv):
                self._pre_match_probs = load_walk_forward_prob_map(self.walk_forward_csv)
            else:
                self._pre_match_probs = {}

    def get_match_metadata(self, match_id: str) -> Dict[str, Any]:
        self._ensure_data_loaded()
        for m in self._all_matches:
            if str(m["match_id"]) == str(match_id):
                return m
        raise KeyError(f"Match ID '{match_id}' not found in {self.matches_csv}")

    def build_trackers_for_match(
        self, target_match_id: str
    ) -> Tuple[HistoricalPlayerTracker, HistoricalMatchupTracker]:
        """
        Builds historical player and matchup trackers strictly using eligible matches
        completed before target_match_id.
        """
        self._ensure_data_loaded()
        p_tracker = HistoricalPlayerTracker()
        m_tracker = HistoricalMatchupTracker()

        target_str = str(target_match_id)
        for m in self._all_matches:
            mid = str(m["match_id"])
            if mid == target_str:
                break
            # Only include valid matches
            if m["match_result_type"] != "no_result" and m["method"] != "D/L":
                dels = self._all_deliveries.get(mid, [])
                if dels:
                    p_tracker.update_with_match(dels)
                    m_tracker.update_with_match(dels)

        return p_tracker, m_tracker

    def simulate(
        self,
        match_id: str,
        custom_p_tracker: Optional[HistoricalPlayerTracker] = None,
        custom_m_tracker: Optional[HistoricalMatchupTracker] = None,
        custom_pre_match_prob: Optional[float] = None,
        custom_deliveries: Optional[List[Dict[str, Any]]] = None,
    ) -> SimulationResult:
        """
        Replays the full match delivery by delivery.
        """
        self._ensure_data_loaded()
        m_meta = self.get_match_metadata(match_id)

        mid = str(m_meta["match_id"])
        season = m_meta["season"]
        date = m_meta["date"]
        venue = m_meta["venue"]
        city = m_meta["city"]
        team_a = m_meta["team_1"]
        team_b = m_meta["team_2"]
        winner = m_meta["winner"]
        toss_winner = m_meta["toss_winner"]
        toss_decision = m_meta["toss_decision"]

        toss_winner_is_team_a = 1 if toss_winner == team_a else 0
        toss_decision_is_bat = 1 if toss_decision == "bat" else 0

        # Retrieve deliveries
        if custom_deliveries is not None:
            deliveries = custom_deliveries
        else:
            deliveries = self._all_deliveries.get(mid, [])

        if not deliveries:
            raise ValueError(f"No deliveries found for match ID {mid}")

        first_batting_team = deliveries[0]["batting_team"]
        team_a_bats_first = 1 if first_batting_team == team_a else 0

        # Pre-match probability anchor
        if custom_pre_match_prob is not None:
            p_anchor = custom_pre_match_prob
        else:
            p_anchor = self._pre_match_probs.get(mid, 0.50)

        # Historical trackers (< t)
        if custom_p_tracker is not None and custom_m_tracker is not None:
            p_tracker = custom_p_tracker
            m_tracker = custom_m_tracker
        else:
            p_tracker, m_tracker = self.build_trackers_for_match(mid)

        # State Managers
        state_mgr = MatchStateManager(
            match_id=mid,
            team_a=team_a,
            team_b=team_b,
            pre_match_prob_team_a=p_anchor,
        )
        player_mgr = PlayerStateManager(historical_tracker=p_tracker)
        matchup_mgr = MatchupStateManager(historical_tracker=m_tracker)

        feature_builder = LiveFeatureBuilder(
            team_a=team_a,
            team_b=team_b,
            venue=venue,
            city=city,
            toss_winner_is_team_a=toss_winner_is_team_a,
            toss_decision_is_bat=toss_decision_is_bat,
            team_a_bats_first=team_a_bats_first,
        )

        sim_result = SimulationResult(
            match_id=mid,
            season=season,
            date=date,
            venue=venue,
            city=city,
            team_a=team_a,
            team_b=team_b,
            toss_winner=toss_winner,
            toss_decision=toss_decision,
            winner=winner,
        )

        # T0 Event (Before ball 1)
        p_a_t0 = round(float(p_anchor), 4)
        p_b_t0 = round(1.0 - p_a_t0, 4)
        first_bat_team = deliveries[0]["batting_team"]
        first_bowl_team = deliveries[0]["bowling_team"]
        first_striker = str(deliveries[0].get("striker", ""))
        first_non_striker = str(deliveries[0].get("non_striker", ""))
        first_bowler = str(deliveries[0].get("bowler", ""))

        t0_batting = []
        if first_striker:
            t0_batting.append({
                "player_name": first_striker,
                "runs": 0,
                "balls": 0,
                "fours": 0,
                "sixes": 0,
                "strike_rate": 0.0,
                "is_striker": True,
            })
        if first_non_striker and first_non_striker != first_striker:
            t0_batting.append({
                "player_name": first_non_striker,
                "runs": 0,
                "balls": 0,
                "fours": 0,
                "sixes": 0,
                "strike_rate": 0.0,
                "is_striker": False,
            })

        t0_bowling = []
        if first_bowler:
            t0_bowling.append({
                "player_name": first_bowler,
                "overs": "0.0",
                "maidens": 0,
                "runs": 0,
                "wickets": 0,
                "economy": 0.0,
                "is_current_bowler": True,
            })

        t0_innings_1 = {
            "innings": 1,
            "team": first_bat_team,
            "score": 0,
            "wickets": 0,
            "overs": "0.0",
            "legal_balls": 0,
            "batting": t0_batting,
            "bowling": t0_bowling,
        }
        t0_innings_2 = {
            "innings": 2,
            "team": first_bowl_team,
            "score": 0,
            "wickets": 0,
            "overs": "0.0",
            "legal_balls": 0,
            "batting": [],
            "bowling": [],
        }

        t0_event = SimulationEvent(
            event_number=0,
            innings=1,
            ball=0.0,
            delivery_index=-1,
            is_legal=True,
            score=0,
            wickets=0,
            legal_balls_bowled=0,
            overs_completed=0.0,
            current_run_rate=0.0,
            target_runs=0,
            runs_needed=0,
            required_run_rate=0.0,
            rrr_crr_differential=0.0,
            striker=first_striker,
            non_striker=first_non_striker,
            bowler=first_bowler,
            team_a_win_probability=p_a_t0,
            team_b_win_probability=p_b_t0,
            prediction_type="pre_match",
            batting_team=first_bat_team,
            bowling_team=first_bowl_team,
            batting=t0_batting,
            bowling=t0_bowling,
            innings_1=t0_innings_1,
            innings_2=t0_innings_2,
        )
        sim_result.events.append(t0_event)

        total_deliveries = len(deliveries)
        event_num = 1

        for idx, d in enumerate(deliveries):
            is_last_delivery = (idx == total_deliveries - 1)
            inn = int(d["innings"])

            # Process state update
            match_feats = state_mgr.process_delivery(d)
            raw_state = state_mgr.get_raw_state()

            curr_player_feats = player_mgr.get_current_player_features(raw_state)
            striker = raw_state["striker"]
            bowler = raw_state["bowler"]

            hist_batter = player_mgr.get_historical_batter_stats(striker)
            hist_bowler = player_mgr.get_historical_bowler_stats(bowler)

            matchup_feats = matchup_mgr.get_matchup_features(
                striker=striker,
                bowler=bowler,
                historical_batter_sr=hist_batter["historical_batter_strike_rate"],
            )

            # Build model design row
            df_row = feature_builder.build_feature_row(
                match_features=match_feats,
                current_player_features=curr_player_feats,
                historical_batter_features=hist_batter,
                historical_bowler_features=hist_bowler,
                matchup_features=matchup_feats,
            )

            # Check for mechanical terminal conclusion
            # Follow Step 3 policy: on final delivery of Innings 2, match has ended
            is_terminal = False
            is_tie = False
            terminal_winner = None

            if inn == 2:
                runs_needed = match_feats["runs_needed"]
                wickets_rem = match_feats["wickets_remaining"]
                balls_rem = match_feats["legal_balls_remaining"]

                chasing_team = raw_state["batting_team"]
                defending_team = raw_state["bowling_team"]

                if runs_needed == 0:
                    # Chaser achieved target
                    is_terminal = True
                    terminal_winner = chasing_team
                elif runs_needed > 1 and (balls_rem == 0 or wickets_rem == 0):
                    # Defender successfully defended
                    is_terminal = True
                    terminal_winner = defending_team
                elif runs_needed == 1 and (balls_rem == 0 or wickets_rem == 0):
                    # Regulation tie
                    is_terminal = True
                    is_tie = True
                elif is_last_delivery and custom_deliveries is None:
                    # Full match ended on this delivery
                    is_terminal = True
                    terminal_winner = winner

            p_a, p_b, pred_type = self.model_runner.predict_probability(
                df_row=df_row,
                innings=inn,
                is_terminal=is_terminal,
                terminal_winner=terminal_winner,
                team_a=team_a,
                is_tie=is_tie,
            )

            if inn == 1:
                lb1 = raw_state["legal_balls_bowled"]
                inn1_scorecard = {
                    "innings": 1,
                    "team": raw_state["batting_team"],
                    "score": raw_state["current_innings_runs"],
                    "wickets": raw_state["current_innings_wickets"],
                    "overs": f"{lb1 // 6}.{lb1 % 6}",
                    "legal_balls": lb1,
                    "batting": state_mgr.get_batting_figures(1),
                    "bowling": state_mgr.get_bowling_figures(1),
                }
                inn2_scorecard = {
                    "innings": 2,
                    "team": raw_state["bowling_team"],
                    "score": 0,
                    "wickets": 0,
                    "overs": "0.0",
                    "legal_balls": 0,
                    "batting": [],
                    "bowling": [],
                }
            else:
                lb1 = state_mgr.state_machine.innings_legal_balls.get(1, 0)
                inn1_scorecard = {
                    "innings": 1,
                    "team": state_mgr.get_batting_team(1) or first_bat_team,
                    "score": state_mgr.state_machine.innings_runs.get(1, 0),
                    "wickets": state_mgr.state_machine.innings_wickets.get(1, 0),
                    "overs": f"{lb1 // 6}.{lb1 % 6}",
                    "legal_balls": lb1,
                    "batting": state_mgr.get_batting_figures(1),
                    "bowling": state_mgr.get_bowling_figures(1),
                }
                lb2 = raw_state["legal_balls_bowled"]
                inn2_scorecard = {
                    "innings": 2,
                    "team": raw_state["batting_team"],
                    "score": raw_state["current_innings_runs"],
                    "wickets": raw_state["current_innings_wickets"],
                    "overs": f"{lb2 // 6}.{lb2 % 6}",
                    "legal_balls": lb2,
                    "batting": state_mgr.get_batting_figures(2),
                    "bowling": state_mgr.get_bowling_figures(2),
                }

            event = SimulationEvent(
                event_number=event_num,
                innings=inn,
                ball=raw_state["ball"],
                delivery_index=idx,
                is_legal=raw_state["is_legal"],
                score=raw_state["current_innings_runs"],
                wickets=raw_state["current_innings_wickets"],
                legal_balls_bowled=raw_state["legal_balls_bowled"],
                overs_completed=match_feats["overs_completed"],
                current_run_rate=match_feats["current_run_rate"],
                target_runs=match_feats["target_runs"],
                runs_needed=match_feats["runs_needed"],
                required_run_rate=match_feats["required_run_rate"],
                rrr_crr_differential=match_feats["rrr_crr_differential"],
                striker=striker,
                non_striker=raw_state["non_striker"],
                bowler=bowler,
                team_a_win_probability=p_a,
                team_b_win_probability=p_b,
                prediction_type=pred_type,
                feature_snapshot=df_row.iloc[0].to_dict(),
                batting_team=raw_state["batting_team"],
                bowling_team=raw_state["bowling_team"],
                batting=raw_state.get("batting", []),
                bowling=raw_state.get("bowling", []),
                innings_1=inn1_scorecard,
                innings_2=inn2_scorecard,
            )
            sim_result.events.append(event)
            event_num += 1

        return sim_result
