"""
Backend configuration settings and path management.
"""

import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data paths
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "data" / "processed"))
MATCHES_CSV = os.getenv("MATCHES_CSV", str(Path(DATA_DIR) / "matches_clean.csv"))
DELIVERIES_CSV = os.getenv("DELIVERIES_CSV", str(Path(DATA_DIR) / "deliveries_clean.csv"))
PRE_MATCH_DATASET_CSV = os.getenv("PRE_MATCH_DATASET_CSV", str(Path(DATA_DIR) / "pre_match_dataset.csv"))
LIVE_DATASET_CSV = os.getenv("LIVE_DATASET_CSV", str(Path(DATA_DIR) / "live_dataset.csv"))
WALK_FORWARD_CSV = os.getenv("WALK_FORWARD_CSV", str(Path(DATA_DIR) / "walk_forward_pre_match_probs.csv"))

# Model paths
MODELS_DIR = os.getenv("MODELS_DIR", str(BASE_DIR / "models"))
LIVE_MODELS_DIR = os.getenv("LIVE_MODELS_DIR", str(Path(MODELS_DIR) / "live"))
LIVE_MODEL_INN1 = str(Path(LIVE_MODELS_DIR) / "primary_live_model_inn1.joblib")
LIVE_MODEL_INN2 = str(Path(LIVE_MODELS_DIR) / "primary_live_model_inn2.joblib")
LIVE_PREP_INN1 = str(Path(LIVE_MODELS_DIR) / "primary_live_preprocessor_inn1.joblib")
LIVE_PREP_INN2 = str(Path(LIVE_MODELS_DIR) / "primary_live_preprocessor_inn2.joblib")

# Output directory for simulation artifacts
SIMULATION_OUTPUT_DIR = os.getenv("SIMULATION_OUTPUT_DIR", str(Path(DATA_DIR) / "simulations"))

# Server configuration
API_PREFIX = "/api/v1"
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Service metadata
PROJECT_TITLE = "IPL Win Probability & Live Match Analytics API"
PROJECT_DESCRIPTION = (
    "Local REST API providing delivery-by-delivery historical match simulation "
    "and live winning probability prediction using the frozen Two-Model XGBoost engine."
)
PROJECT_VERSION = "1.0.0"
