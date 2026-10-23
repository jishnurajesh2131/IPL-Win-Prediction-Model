"""
In-memory session state management for live historical match simulations.
"""

import uuid
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from src.live_simulator.simulator import MatchSimulator, SimulationResult, SimulationEvent
from src.backend.schemas.prediction import PredictionResponse, ProbabilityPoint
from src.backend.schemas.simulation import (
    MatchStateSnapshot,
    DeliverySnapshot,
    BattingFigure,
    BowlingFigure,
    InningsScorecard,
)


class SessionStatus(str, Enum):
    READY = "ready"
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"


class SimulationSession:
    """
    Encapsulates state, event stream, and trajectory pointer for a single match simulation.
    """

    def __init__(
        self,
        session_id: str,
        match_id: str,
        simulation_result: SimulationResult,
        raw_deliveries: List[Dict[str, Any]],
    ):
        self.session_id = session_id
        self.match_id = match_id
        self.simulation_result = simulation_result
        self.raw_deliveries = raw_deliveries
        self.total_events = len(simulation_result.events)

        self.current_event_index = 0  # Starts at T0 (pre-match anchor)
        self.status = SessionStatus.READY
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at

    @property
    def team_a(self) -> str:
        return self.simulation_result.team_a

    @property
    def team_b(self) -> str:
        return self.simulation_result.team_b

    @property
    def winner(self) -> str:
        return self.simulation_result.winner

    def get_current_event(self) -> SimulationEvent:
        return self.simulation_result.events[self.current_event_index]

    def advance_step(self) -> SimulationEvent:
        """
        Advances exactly one delivery event along the simulation timeline.
        """
        if self.status == SessionStatus.COMPLETED or self.current_event_index >= self.total_events - 1:
            raise ValueError("Simulation session has already completed.")

        self.current_event_index += 1
        self.updated_at = datetime.now(timezone.utc)

        if self.current_event_index == self.total_events - 1:
            self.status = SessionStatus.COMPLETED
        else:
            self.status = SessionStatus.ACTIVE

        return self.get_current_event()

    def previous_step(self) -> SimulationEvent:
        """
        Moves the simulation pointer exactly one recorded delivery event backward.
        """
        if self.current_event_index <= 0:
            raise ValueError("Cannot step backward: Simulation is already at T0 pre-match anchor.")

        self.current_event_index -= 1
        self.updated_at = datetime.now(timezone.utc)

        if self.current_event_index == 0:
            self.status = SessionStatus.READY
        else:
            self.status = SessionStatus.ACTIVE

        return self.get_current_event()

    def seek_to_index(self, index: int) -> SimulationEvent:
        """
        Directly navigates replay state to any target event index [0, total_events - 1].
        """
        if index < 0 or index >= self.total_events:
            raise ValueError(f"Event index {index} out of valid bounds [0, {self.total_events - 1}].")

        self.current_event_index = index
        self.updated_at = datetime.now(timezone.utc)

        if self.current_event_index == 0:
            self.status = SessionStatus.READY
        elif self.current_event_index == self.total_events - 1:
            self.status = SessionStatus.COMPLETED
        else:
            self.status = SessionStatus.ACTIVE

        return self.get_current_event()

    def run_to_end(self) -> Tuple[int, SimulationEvent]:
        """
        Runs remaining delivery events until match conclusion.
        """
        start_idx = self.current_event_index
        end_idx = self.total_events - 1
        events_processed = end_idx - start_idx

        self.current_event_index = end_idx
        self.status = SessionStatus.COMPLETED
        self.updated_at = datetime.now(timezone.utc)

        return events_processed, self.get_current_event()

    def get_probability_history(self, include_all: bool = False) -> List[ProbabilityPoint]:
        """
        Returns probability points traversed up to the current event index, or full trajectory if include_all is True.
        """
        limit = self.total_events if include_all else self.current_event_index + 1
        points: List[ProbabilityPoint] = []
        for idx in range(limit):
            e = self.simulation_result.events[idx]
            points.append(
                ProbabilityPoint(
                    event_index=e.event_number,
                    innings=e.innings,
                    ball=round(float(e.ball), 1),
                    overs_completed=e.overs_completed,
                    score=e.score,
                    wickets=e.wickets,
                    team_1_probability=e.team_a_win_probability,
                    team_2_probability=e.team_b_win_probability,
                    prediction_type=e.prediction_type,
                    is_legal=e.is_legal,
                )
            )
        return points

    def to_prediction_response(self, event: Optional[SimulationEvent] = None) -> PredictionResponse:
        e = event or self.get_current_event()
        return PredictionResponse(
            team_1=self.team_a,
            team_2=self.team_b,
            team_1_probability=round(float(e.team_a_win_probability), 4),
            team_2_probability=round(float(e.team_b_win_probability), 4),
            prediction_type=e.prediction_type,
        )

    def _parse_scorecard(self, sc_dict: Optional[Dict[str, Any]]) -> Optional[InningsScorecard]:
        if not sc_dict:
            return None
        bat_figs = [
            b if isinstance(b, BattingFigure) else BattingFigure(**b)
            for b in sc_dict.get("batting", [])
        ]
        bowl_figs = [
            b if isinstance(b, BowlingFigure) else BowlingFigure(**b)
            for b in sc_dict.get("bowling", [])
        ]
        return InningsScorecard(
            innings=sc_dict.get("innings", 1),
            team=sc_dict.get("team"),
            score=sc_dict.get("score", 0),
            wickets=sc_dict.get("wickets", 0),
            overs=sc_dict.get("overs", "0.0"),
            legal_balls=sc_dict.get("legal_balls", 0),
            batting=bat_figs,
            bowling=bowl_figs,
        )

    def to_state_snapshot(self, event: Optional[SimulationEvent] = None) -> MatchStateSnapshot:
        e = event or self.get_current_event()

        # Format overs as standard cricket string (e.g. 15.3)
        completed_overs = int(e.legal_balls_bowled // 6)
        remaining_balls = int(e.legal_balls_bowled % 6)
        overs_str = f"{completed_overs}.{remaining_balls}"

        target = e.target_runs if e.innings == 2 and e.target_runs > 0 else None
        needed = e.runs_needed if e.innings == 2 and e.target_runs > 0 else None
        rrr = e.required_run_rate if e.innings == 2 and e.target_runs > 0 else None
        diff = e.rrr_crr_differential if e.innings == 2 and e.target_runs > 0 else None

        batting_figures = [
            b if isinstance(b, BattingFigure) else BattingFigure(**b)
            for b in getattr(e, "batting", [])
        ]
        bowling_figures = [
            b if isinstance(b, BowlingFigure) else BowlingFigure(**b)
            for b in getattr(e, "bowling", [])
        ]

        inn1 = self._parse_scorecard(getattr(e, "innings_1", None))
        inn2 = self._parse_scorecard(getattr(e, "innings_2", None))

        return MatchStateSnapshot(
            innings=e.innings,
            score=e.score,
            wickets=e.wickets,
            legal_balls=e.legal_balls_bowled,
            overs=overs_str,
            overs_completed=round(float(e.overs_completed), 2),
            current_run_rate=round(float(e.current_run_rate), 2),
            target_runs=target,
            runs_needed=needed,
            required_run_rate=round(float(rrr), 2) if rrr is not None else None,
            rrr_crr_differential=round(float(diff), 2) if diff is not None else None,
            striker=e.striker or "N/A",
            non_striker=e.non_striker or "N/A",
            bowler=e.bowler or "N/A",
            batting_team=getattr(e, "batting_team", None),
            bowling_team=getattr(e, "bowling_team", None),
            batting=batting_figures,
            bowling=bowling_figures,
            innings_1=inn1,
            innings_2=inn2,
        )

    def to_delivery_snapshot(self, event: Optional[SimulationEvent] = None) -> Optional[DeliverySnapshot]:
        e = event or self.get_current_event()
        if e.event_number == 0:
            return None  # T0 pre-match anchor has no delivery

        d_idx = e.delivery_index
        d = self.raw_deliveries[d_idx] if (0 <= d_idx < len(self.raw_deliveries)) else {}

        # Authoritative delivery identity strictly derived from selected replay event e
        auth_ball = round(float(e.ball), 1)
        over_num = int(auth_ball)
        ball_in_over = int(round((auth_ball * 10) % 10))
        # Enforce cricket sanity: ball in over is 1..6
        if ball_in_over < 1 or ball_in_over > 6:
            ball_in_over = min(6, max(1, ball_in_over))
            auth_ball = round(over_num + (ball_in_over / 10.0), 1)

        display_ball_str = f"{auth_ball:.1f}"

        return DeliverySnapshot(
            innings=e.innings,
            ball=auth_ball,
            over=over_num,
            ball_in_over=ball_in_over,
            display_ball=display_ball_str,
            event_index=e.event_number,
            is_legal=e.is_legal,
            batting_team=str(d.get("batting_team", getattr(e, "batting_team", "") or "")),
            bowling_team=str(d.get("bowling_team", getattr(e, "bowling_team", "") or "")),
            striker=str(d.get("striker", e.striker or "")),
            non_striker=str(d.get("non_striker", e.non_striker or "")),
            bowler=str(d.get("bowler", e.bowler or "")),
            runs_off_bat=int(d.get("runs_off_bat", 0)),
            extras=int(d.get("extras", 0)),
            wides=int(d.get("wides", 0)),
            noballs=int(d.get("noballs", 0)),
            wicket_type=d.get("wicket_type") if d.get("wicket_type") else None,
            player_dismissed=d.get("player_dismissed") if d.get("player_dismissed") else None,
        )


class SessionManager:
    """
    Thread-safe in-memory session manager.
    """

    def __init__(self):
        self._sessions: Dict[str, SimulationSession] = {}
        self._lock = threading.Lock()

    def create_session(self, match_id: str, simulator: MatchSimulator) -> SimulationSession:
        """
        Creates and stores a new simulation session for the requested match ID.
        """
        simulator._ensure_data_loaded()

        # Run simulation to prepare event trajectory
        sim_res = simulator.simulate(match_id)
        raw_deliveries = simulator._all_deliveries.get(str(match_id), [])

        session_id = str(uuid.uuid4())
        session = SimulationSession(
            session_id=session_id,
            match_id=str(match_id),
            simulation_result=sim_res,
            raw_deliveries=raw_deliveries,
        )

        with self._lock:
            self._sessions[session_id] = session

        return session

    def get_session(self, session_id: str) -> SimulationSession:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Simulation session '{session_id}' not found.")
            return self._sessions[session_id]

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def list_sessions(self) -> List[str]:
        with self._lock:
            return list(self._sessions.keys())

    def clear(self):
        with self._lock:
            self._sessions.clear()


# Global singleton instance
session_manager = SessionManager()
