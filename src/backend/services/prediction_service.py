"""
Model artifact verification and health monitoring service.
"""

import os
from typing import Dict, Tuple

from src.backend.config import (
    LIVE_MODEL_INN1,
    LIVE_MODEL_INN2,
    LIVE_PREP_INN1,
    LIVE_PREP_INN2,
    MATCHES_CSV,
    DELIVERIES_CSV,
    WALK_FORWARD_CSV,
    PROJECT_VERSION,
)
from src.backend.schemas.health import HealthResponse


class PredictionService:
    """
    Validates availability of required frozen ML artifacts and datasets.
    """

    def check_health(self, total_matches: int = 0) -> HealthResponse:
        artifacts = {
            "live_model_inn1": os.path.exists(LIVE_MODEL_INN1),
            "live_model_inn2": os.path.exists(LIVE_MODEL_INN2),
            "live_preprocessor_inn1": os.path.exists(LIVE_PREP_INN1),
            "live_preprocessor_inn2": os.path.exists(LIVE_PREP_INN2),
            "matches_csv": os.path.exists(MATCHES_CSV),
            "deliveries_csv": os.path.exists(DELIVERIES_CSV),
            "walk_forward_csv": os.path.exists(WALK_FORWARD_CSV),
        }

        all_models_present = (
            artifacts["live_model_inn1"]
            and artifacts["live_model_inn2"]
            and artifacts["live_preprocessor_inn1"]
            and artifacts["live_preprocessor_inn2"]
        )

        all_datasets_present = (
            artifacts["matches_csv"]
            and artifacts["deliveries_csv"]
            and artifacts["walk_forward_csv"]
        )

        overall_ok = all_models_present and all_datasets_present

        return HealthResponse(
            status="ok" if overall_ok else "degraded",
            service="ipl-live-analytics-backend",
            version=PROJECT_VERSION,
            ml_status="loaded" if all_models_present else "error",
            model_artifacts=artifacts,
            matches_available=total_matches,
        )


# Global singleton instance
prediction_service = PredictionService()
