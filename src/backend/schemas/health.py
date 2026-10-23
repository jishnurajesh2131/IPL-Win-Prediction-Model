"""
Health check schema models.
"""

from typing import Dict
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status: 'ok' or 'degraded'")
    service: str = Field(..., description="Service identifier")
    version: str = Field(..., description="Backend API version")
    ml_status: str = Field(..., description="ML engine status: 'loaded' or 'error'")
    model_artifacts: Dict[str, bool] = Field(..., description="Availability map of required frozen model artifacts")
    matches_available: int = Field(..., description="Number of historical matches available in database")
