"""
Prediction and probability history schema models.
"""

from typing import List
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    team_1: str = Field(..., description="Team 1 name")
    team_2: str = Field(..., description="Team 2 name")
    team_1_probability: float = Field(..., ge=0.0, le=1.0, description="Win probability for Team 1 [0.0, 1.0]")
    team_2_probability: float = Field(..., ge=0.0, le=1.0, description="Win probability for Team 2 [0.0, 1.0]")
    prediction_type: str = Field(
        ...,
        description="Source of prediction: 'pre_match', 'in_play_xgb', 'terminal_team_a_win', 'terminal_team_b_win', 'terminal_tie'",
    )


class ProbabilityPoint(BaseModel):
    event_index: int = Field(..., description="Sequential event index (0 = T0 pre-match)")
    innings: int = Field(..., description="Current innings (1 or 2)")
    ball: float = Field(..., description="Cricsheet ball format (e.g. 15.3)")
    overs_completed: float = Field(..., description="Completed legal overs")
    score: int = Field(..., description="Current total runs in this innings")
    wickets: int = Field(..., description="Current wickets down in this innings")
    team_1_probability: float = Field(..., ge=0.0, le=1.0, description="Win probability for Team 1")
    team_2_probability: float = Field(..., ge=0.0, le=1.0, description="Win probability for Team 2")
    prediction_type: str = Field(..., description="Prediction engine classification")
    is_legal: bool = Field(True, description="Whether delivery was legal (not wide, not noball)")


class ProbabilityHistoryResponse(BaseModel):
    session_id: str = Field(..., description="Simulation session identifier")
    match_id: str = Field(..., description="Simulated match identifier")
    total_points: int = Field(..., description="Number of probability checkpoints recorded so far")
    points: List[ProbabilityPoint] = Field(..., description="Ordered list of historical probability points")
