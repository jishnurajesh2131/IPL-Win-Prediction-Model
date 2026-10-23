"""
API Endpoints for Prediction Intelligence:
- Reliability Auditor
- Model Failure Observatory
- IPL Era-Shift Monitor
"""

from typing import Optional
from fastapi import APIRouter, Query, status

from src.backend.schemas.intelligence import (
    ReliabilityAuditorResponse,
    FailureObservatoryResponse,
    EraShiftMonitorResponse,
    IntelligenceSummaryResponse,
)
from src.backend.services.prediction_intelligence_service import (
    prediction_intelligence_service,
)

router = APIRouter(prefix="/intelligence", tags=["Prediction Intelligence"])


@router.get(
    "/summary",
    response_model=IntelligenceSummaryResponse,
    summary="Prediction Intelligence Executive Summary",
    description="Returns high-level reliability metrics, sample sizes, and era shift findings.",
)
def get_intelligence_summary():
    return prediction_intelligence_service.get_intelligence_summary()


@router.get(
    "/reliability",
    response_model=ReliabilityAuditorResponse,
    summary="Prediction Reliability Auditor",
    description="Evaluates calibration, Brier score, log loss, and ECE against actual match outcomes.",
)
def get_reliability_audit(
    model_type: str = Query(
        "live",
        description="Model to audit: 'live' (Two-Model live simulator) or 'pre_match' (T0 walk-forward model)",
    ),
    split: str = Query(
        "test",
        description="Data split to evaluate: 'test' (2024-2026 held-out test), 'val' (validation period), or 'walk_forward' (all seasons)",
    ),
    season: Optional[str] = Query(
        None,
        description="Optional filter by IPL season (e.g., '2024', '2023')",
    ),
):
    return prediction_intelligence_service.get_reliability_audit(
        model_type=model_type,
        split=split,
        season=season,
    )


@router.get(
    "/failures",
    response_model=FailureObservatoryResponse,
    summary="Model Failure Observatory",
    description="Uncovers recurring historical error patterns by venue, season, and match context with statistical rigor.",
)
def get_failure_observatory():
    return prediction_intelligence_service.get_failure_observatory()


@router.get(
    "/era-shifts",
    response_model=EraShiftMonitorResponse,
    summary="IPL Era-Shift Monitor",
    description="Tracks chronological evolution of scoring distributions and model stability across IPL eras.",
)
def get_era_shift_monitor():
    return prediction_intelligence_service.get_era_shift_monitor()
