"""
Simulation orchestration service connecting FastAPI endpoints to MatchSimulator and SessionManager.
"""

from typing import Optional

from src.live_simulator.simulator import MatchSimulator
from src.backend.config import MATCHES_CSV, DELIVERIES_CSV, WALK_FORWARD_CSV
from src.backend.state.session_manager import SessionManager, session_manager
from src.backend.schemas.simulation import (
    SimulationCreateResponse,
    SimulationStepResponse,
    SimulationRunResponse,
)
from src.backend.schemas.prediction import ProbabilityHistoryResponse
from src.backend.services.match_intelligence import match_intelligence_service


class SimulationService:
    """
    Coordinates simulation creation, single-step advancing, and trajectory querying.
    """

    def __init__(
        self,
        simulator: Optional[MatchSimulator] = None,
        mgr: Optional[SessionManager] = None,
    ):
        self.simulator = simulator or MatchSimulator(
            matches_csv=MATCHES_CSV,
            deliveries_csv=DELIVERIES_CSV,
            walk_forward_csv=WALK_FORWARD_CSV,
        )
        self.session_manager = mgr or session_manager

    def create_simulation(self, match_id: str) -> SimulationCreateResponse:
        """
        Initializes a new historical replay session for the given match ID.
        """
        # Validate match exists
        self.simulator._ensure_data_loaded()
        self.simulator.get_match_metadata(match_id)

        session = self.session_manager.create_session(match_id, self.simulator)
        t0_event = session.get_current_event()

        return SimulationCreateResponse(
            session_id=session.session_id,
            match_id=session.match_id,
            status=session.status.value,
            current_event_index=session.current_event_index,
            total_events=session.total_events,
            current_prediction=session.to_prediction_response(t0_event),
        )

    def get_simulation_state(self, session_id: str) -> SimulationStepResponse:
        """
        Returns the current state snapshot without advancing the session.
        """
        session = self.session_manager.get_session(session_id)
        event = session.get_current_event()

        intelligence = match_intelligence_service.calculate_intelligence(
            events=session.simulation_result.events,
            current_index=session.current_event_index,
            match_id=session.match_id,
            team_a=session.team_a,
            team_b=session.team_b,
            model_runner=self.simulator.model_runner
        )

        return SimulationStepResponse(
            session_id=session.session_id,
            match_id=session.match_id,
            event_index=session.current_event_index,
            total_events=session.total_events,
            status=session.status.value,
            delivery=session.to_delivery_snapshot(event),
            state=session.to_state_snapshot(event),
            prediction=session.to_prediction_response(event),
            match_intelligence=intelligence,
        )

    def advance_simulation(self, session_id: str) -> SimulationStepResponse:
        """
        Advances the simulation by exactly one delivery event.
        """
        session = self.session_manager.get_session(session_id)
        event = session.advance_step()

        intelligence = match_intelligence_service.calculate_intelligence(
            events=session.simulation_result.events,
            current_index=session.current_event_index,
            match_id=session.match_id,
            team_a=session.team_a,
            team_b=session.team_b,
            model_runner=self.simulator.model_runner
        )

        return SimulationStepResponse(
            session_id=session.session_id,
            match_id=session.match_id,
            event_index=session.current_event_index,
            total_events=session.total_events,
            status=session.status.value,
            delivery=session.to_delivery_snapshot(event),
            state=session.to_state_snapshot(event),
            prediction=session.to_prediction_response(event),
            match_intelligence=intelligence,
        )

    def previous_simulation(self, session_id: str) -> SimulationStepResponse:
        """
        Steps the simulation backward by exactly one delivery event.
        """
        session = self.session_manager.get_session(session_id)
        event = session.previous_step()

        intelligence = match_intelligence_service.calculate_intelligence(
            events=session.simulation_result.events,
            current_index=session.current_event_index,
            match_id=session.match_id,
            team_a=session.team_a,
            team_b=session.team_b,
            model_runner=self.simulator.model_runner,
        )

        return SimulationStepResponse(
            session_id=session.session_id,
            match_id=session.match_id,
            event_index=session.current_event_index,
            total_events=session.total_events,
            status=session.status.value,
            delivery=session.to_delivery_snapshot(event),
            state=session.to_state_snapshot(event),
            prediction=session.to_prediction_response(event),
            match_intelligence=intelligence,
        )

    def seek_simulation(self, session_id: str, event_index: int) -> SimulationStepResponse:
        """
        Navigates the simulation state directly to any recorded event index.
        """
        session = self.session_manager.get_session(session_id)
        event = session.seek_to_index(event_index)

        intelligence = match_intelligence_service.calculate_intelligence(
            events=session.simulation_result.events,
            current_index=session.current_event_index,
            match_id=session.match_id,
            team_a=session.team_a,
            team_b=session.team_b,
            model_runner=self.simulator.model_runner,
        )

        return SimulationStepResponse(
            session_id=session.session_id,
            match_id=session.match_id,
            event_index=session.current_event_index,
            total_events=session.total_events,
            status=session.status.value,
            delivery=session.to_delivery_snapshot(event),
            state=session.to_state_snapshot(event),
            prediction=session.to_prediction_response(event),
            match_intelligence=intelligence,
        )

    def get_probabilities(self, session_id: str, include_all: bool = False) -> ProbabilityHistoryResponse:
        """
        Returns the probability history from event 0 up to current position, or all events if include_all is True.
        """
        session = self.session_manager.get_session(session_id)
        points = session.get_probability_history(include_all=include_all)

        return ProbabilityHistoryResponse(
            session_id=session.session_id,
            match_id=session.match_id,
            total_points=len(points),
            points=points,
        )

    def run_simulation(self, session_id: str, include_history: bool = True) -> SimulationRunResponse:
        """
        Fast-forwards the simulation through all remaining events to conclusion.
        """
        session = self.session_manager.get_session(session_id)
        events_processed, final_event = session.run_to_end()

        history = session.get_probability_history() if include_history else None

        intelligence = match_intelligence_service.calculate_intelligence(
            events=session.simulation_result.events,
            current_index=session.current_event_index,
            match_id=session.match_id,
            team_a=session.team_a,
            team_b=session.team_b,
            model_runner=self.simulator.model_runner,
        )

        return SimulationRunResponse(
            session_id=session.session_id,
            match_id=session.match_id,
            status=session.status.value,
            events_processed=events_processed,
            total_events=session.total_events,
            winner=session.winner,
            final_state=session.to_state_snapshot(final_event),
            final_prediction=session.to_prediction_response(final_event),
            delivery=session.to_delivery_snapshot(final_event),
            match_intelligence=intelligence,
            history=history,
        )


# Global singleton instance
simulation_service = SimulationService()
