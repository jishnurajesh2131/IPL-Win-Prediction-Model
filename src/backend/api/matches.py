"""
Match catalog discovery and retrieval API routes.
"""

from typing import Optional, List
from fastapi import APIRouter, Query, HTTPException, status

from src.backend.schemas.match import MatchListResponse, MatchDetails
from src.backend.services.match_service import match_service

router = APIRouter(prefix="/matches", tags=["Matches"])


@router.get(
    "",
    response_model=MatchListResponse,
    summary="List Historical Matches",
    description="Retrieves a paginated catalog of historical IPL matches with optional season and team filtering.",
)
def list_matches(
    season: Optional[str] = Query(None, description="Filter by IPL season (e.g. '2024')"),
    team: Optional[str] = Query(None, description="Filter by team name or substring (e.g. 'KKR' or 'Kolkata')"),
    limit: int = Query(50, ge=1, le=200, description="Number of matches per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    summaries, total = match_service.list_matches(
        season=season,
        team=team,
        limit=limit,
        offset=offset,
    )
    return MatchListResponse(
        matches=summaries,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/seasons",
    response_model=List[str],
    summary="List Available Seasons",
    description="Returns the list of available IPL seasons in descending order.",
)
def list_seasons():
    return match_service.get_seasons()


@router.get(
    "/teams",
    response_model=List[str],
    summary="List Available Teams",
    description="Returns the list of participating teams, optionally filtered by season.",
)
def list_teams(
    season: Optional[str] = Query(None, description="Optional IPL season to filter participating teams (e.g. '2016')")
):
    return match_service.get_teams_for_season(season=season)


@router.get(
    "/{match_id}",
    response_model=MatchDetails,
    summary="Get Match Details",
    description="Retrieves comprehensive metadata for a specific match ID.",
)
def get_match(match_id: str):
    try:
        return match_service.get_match(match_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Match ID '{match_id}' not found.",
        )
