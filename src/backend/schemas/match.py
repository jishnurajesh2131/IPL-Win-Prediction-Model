"""
Match schemas for listing and detailed metadata views.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class MatchSummary(BaseModel):
    match_id: str = Field(..., description="Unique match identifier")
    season: str = Field(..., description="IPL season/year (e.g. '2024')")
    date: str = Field(..., description="Match start date (YYYY-MM-DD)")
    team_1: str = Field(..., description="Team 1 canonical name")
    team_2: str = Field(..., description="Team 2 canonical name")
    venue: str = Field(..., description="Canonical stadium venue name")
    city: str = Field(..., description="Host city")
    winner: Optional[str] = Field(None, description="Match winner (None if no result)")
    match_result_type: str = Field(..., description="Outcome category: 'normal', 'super_over', 'no_result'")


class MatchDetails(MatchSummary):
    toss_winner: str = Field(..., description="Franchise that won the toss")
    toss_decision: str = Field(..., description="Toss choice: 'bat' or 'field'")
    method: Optional[str] = Field(None, description="Result method if curtailed (e.g. 'D/L')")
    is_super_over: bool = Field(False, description="Whether the match was decided in a Super Over")
    is_dl_method: bool = Field(False, description="Whether Duckworth-Lewis-Stern method was applied")
    total_deliveries: Optional[int] = Field(None, description="Total recorded delivery events if available")


class MatchListResponse(BaseModel):
    matches: List[MatchSummary] = Field(..., description="Paginated list of match summaries")
    total: int = Field(..., description="Total count of matches matching query filters")
    limit: int = Field(..., description="Pagination limit")
    offset: int = Field(..., description="Pagination offset")
