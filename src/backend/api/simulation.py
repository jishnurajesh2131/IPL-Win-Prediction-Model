"""
Simulation session and replay execution API routes.
"""

from fastapi import APIRouter, Query, HTTPException, status

from src.backend.schemas.simulation import (
    SimulationCreateRequest,
    SimulationCreateResponse,
    SimulationStepResponse,
    SimulationRunResponse,
    SimulationSeekRequest,
)
from src.backend.schemas.prediction import ProbabilityHistoryResponse
from src.backend.services.simulation_service import simulation_service

router = APIRouter(prefix="/simulations", tags=["Simulation"])


@router.post(
    "",
    response_model=SimulationCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Simulation Session",
    description="Initializes a new delivery-by-delivery simulation session for a match, establishing the T0 pre-match anchor.",
)
def create_simulation(request: SimulationCreateRequest):
    try:
        return simulation_service.create_simulation(request.match_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Match ID '{request.match_id}' not found in database.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize simulation: {str(e)}",
        )


@router.get(
    "/{session_id}",
    response_model=SimulationStepResponse,
    summary="Get Simulation Current State",
    description="Returns the current match state, last delivery, and win probability snapshot without advancing the session.",
)
def get_simulation_state(session_id: str):
    try:
        return simulation_service.get_simulation_state(session_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation session '{session_id}' not found.",
        )


@router.post(
    "/{session_id}/next",
    response_model=SimulationStepResponse,
    summary="Advance Simulation by One Delivery",
    description="Advances the simulation by exactly one delivery event, updates match state, and computes new win probabilities.",
)
def advance_simulation(session_id: str):
    try:
        return simulation_service.advance_simulation(session_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation session '{session_id}' not found.",
        )
    except ValueError as ve:
        # Match already completed
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(ve),
        )


@router.post(
    "/{session_id}/previous",
    response_model=SimulationStepResponse,
    summary="Step Simulation Backward by One Delivery",
    description="Steps the replay backward by exactly one delivery event without lookahead.",
)
def previous_simulation(session_id: str):
    try:
        return simulation_service.previous_simulation(session_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation session '{session_id}' not found.",
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(ve),
        )


@router.post(
    "/{session_id}/seek",
    response_model=SimulationStepResponse,
    summary="Seek Simulation to Exact Event Index",
    description="Navigates replay directly to any event index [0, total_events - 1].",
)
def seek_simulation(session_id: str, request: SimulationSeekRequest):
    try:
        return simulation_service.seek_simulation(session_id, request.event_index)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation session '{session_id}' not found.",
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )


@router.get(
    "/{session_id}/probabilities",
    response_model=ProbabilityHistoryResponse,
    summary="Get Probability Trajectory History",
    description="Returns the ordered sequence of probability predictions from event 0 up to current position, or all events if all=True.",
)
def get_probabilities(
    session_id: str,
    all: bool = Query(False, description="Whether to return full trajectory for the entire match"),
):
    try:
        return simulation_service.get_probabilities(session_id, include_all=all)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation session '{session_id}' not found.",
        )


@router.post(
    "/{session_id}/run",
    response_model=SimulationRunResponse,
    summary="Run Simulation to Conclusion",
    description="Fast-forwards through all remaining delivery events until the match reaches terminal completion.",
)
def run_simulation(
    session_id: str,
    include_history: bool = Query(True, description="Whether to include full trajectory history in response"),
):
    try:
        return simulation_service.run_simulation(session_id, include_history=include_history)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation session '{session_id}' not found.",
        )
