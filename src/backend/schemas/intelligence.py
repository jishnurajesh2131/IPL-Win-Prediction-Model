"""
Pydantic Schemas for Prediction Intelligence Module:
1. Prediction Reliability Auditor
2. Model Failure Observatory
3. IPL Era-Shift Monitor
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# --- Reliability Auditor Schemas ---

class CalibrationBin(BaseModel):
    bin_index: int
    bin_range: str
    count: int
    mean_predicted_prob: float
    observed_frequency: Optional[float] = None
    calibration_error: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


class PhaseReliabilityMetric(BaseModel):
    phase: str
    rows: int
    matches: int
    log_loss: float
    brier_score: float
    ece: float
    accuracy: float
    roc_auc: Optional[float] = None


class InningsReliabilityMetric(BaseModel):
    innings: str
    rows: int
    matches: int
    log_loss: float
    brier_score: float
    ece: float
    accuracy: float
    roc_auc: Optional[float] = None


class ReliabilityAuditorResponse(BaseModel):
    model_type: str = Field(..., description="'live' or 'pre_match'")
    split: str = Field(..., description="'test', 'val', or 'walk_forward'")
    evaluation_matches: int
    evaluation_predictions: int
    brier_score: float
    log_loss: float
    ece: float
    accuracy: float
    roc_auc: Optional[float] = None
    calibration_bins: List[CalibrationBin]
    phase_metrics: Optional[List[PhaseReliabilityMetric]] = None
    innings_metrics: Optional[List[InningsReliabilityMetric]] = None
    methodology_notes: str
    clustering_caveat: Optional[str] = None
    temporal_isolation_verified: bool = True
    insufficient_data: bool = False
    status_message: Optional[str] = None


# --- Model Failure Observatory Schemas ---

class ErrorSubgroupPattern(BaseModel):
    dimension: str
    subgroup: str
    total_matches: int
    total_errors: int
    error_rate: float
    accuracy: float
    baseline_error_rate: float
    difference_from_baseline: float
    ci_lower: float
    ci_upper: float
    high_confidence_errors: int
    high_confidence_error_rate: float
    p_value: Optional[float] = None
    p_value_adjusted: Optional[float] = None
    comparison_group: str = "Rest of matches (non-overlapping)"
    status: str = Field(..., description="'validated', 'exploratory', or 'insufficient_evidence'")
    caveat: str


class FailureObservatoryResponse(BaseModel):
    total_matches_audited: int
    total_errors_audited: int
    overall_error_rate: float
    baseline_accuracy: float
    subgroups: List[ErrorSubgroupPattern]
    methodology: str
    non_causal_statement: str


# --- IPL Era-Shift Monitor Schemas ---

class EraDefinition(BaseModel):
    era_name: str
    season_range: str
    description: str
    avg_first_innings_score: float
    avg_run_rate: float
    brier_score: float
    accuracy: float
    matches: int


class SeasonMetrics(BaseModel):
    season: str
    matches: int
    avg_first_innings_score: float
    avg_run_rate: float
    scores_200_plus_count: int
    scores_200_plus_pct: float
    favorite_accuracy: float
    brier_score: float
    log_loss: float
    ece: float
    overconfidence_rate: float


class ShiftTestResult(BaseModel):
    comparison_name: str
    metric: str
    baseline_era: str
    recent_era: str
    baseline_mean: float
    recent_mean: float
    absolute_change: float
    percentage_change: float
    p_value: Optional[float] = None
    is_statistically_significant: bool
    status: str = Field(..., description="'statistically_significant', 'provisional', or 'inconclusive'")
    analysis_type: str = Field(..., description="'scoring_distribution_shift' or 'model_performance_drift'")
    notes: str


class EraShiftMonitorResponse(BaseModel):
    eras: List[EraDefinition]
    seasons: List[SeasonMetrics]
    shift_tests: List[ShiftTestResult]
    impact_player_rule_era_start: str = "2023"
    methodology: str
    limitations: List[str]


# --- Summary / Metadata Schema ---

class IntelligenceSummaryResponse(BaseModel):
    live_model_test_brier: float
    live_model_test_log_loss: float
    live_model_test_ece: float
    live_model_test_accuracy: float
    live_model_test_matches: int
    live_model_test_deliveries: int
    pre_match_walkforward_matches: int
    pre_match_walkforward_brier: float
    pre_match_walkforward_accuracy: float
    total_seasons_monitored: int
    era_shift_detected: bool
    era_shift_summary: str
    temporal_isolation_guarantee: str
