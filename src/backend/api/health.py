"""
Health check API route.
"""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.backend.schemas.health import HealthResponse
from src.backend.services.prediction_service import prediction_service
from src.backend.services.match_service import match_service

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Verifies the operational status of the backend API, model artifact availability, and dataset access.",
)
def get_health():
    total_matches = match_service.total_count()
    health = prediction_service.check_health(total_matches=total_matches)

    http_status = status.HTTP_200_OK if health.status == "ok" else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=http_status, content=health.model_dump())
