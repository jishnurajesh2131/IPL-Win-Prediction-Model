export interface CalibrationBin {
  bin_index: number;
  bin_range: string;
  count: number;
  mean_predicted_prob: number;
  observed_frequency?: number | null;
  calibration_error?: number | null;
  ci_lower?: number | null;
  ci_upper?: number | null;
}

export interface PhaseReliabilityMetric {
  phase: string;
  rows: number;
  matches: number;
  log_loss: number;
  brier_score: number;
  ece: number;
  accuracy: number;
  roc_auc?: number | null;
}

export interface InningsReliabilityMetric {
  innings: string;
  rows: number;
  matches: number;
  log_loss: number;
  brier_score: number;
  ece: number;
  accuracy: number;
  roc_auc?: number | null;
}

export interface ReliabilityAuditorData {
  model_type: 'live' | 'pre_match';
  split: 'test' | 'val' | 'walk_forward';
  evaluation_matches: number;
  evaluation_predictions: number;
  brier_score: number;
  log_loss: number;
  ece: number;
  accuracy: number;
  roc_auc?: number | null;
  calibration_bins: CalibrationBin[];
  phase_metrics?: PhaseReliabilityMetric[] | null;
  innings_metrics?: InningsReliabilityMetric[] | null;
  methodology_notes: string;
  clustering_caveat?: string | null;
  temporal_isolation_verified: boolean;
  insufficient_data: boolean;
  status_message?: string | null;
}

export interface ErrorSubgroupPattern {
  dimension: string;
  subgroup: string;
  total_matches: number;
  total_errors: number;
  error_rate: number;
  accuracy: number;
  baseline_error_rate: number;
  difference_from_baseline: number;
  ci_lower: number;
  ci_upper: number;
  high_confidence_errors: number;
  high_confidence_error_rate: number;
  p_value?: number | null;
  p_value_adjusted?: number | null;
  comparison_group?: string;
  status: 'validated' | 'exploratory' | 'insufficient_evidence';
  caveat: string;
}

export interface FailureObservatoryData {
  total_matches_audited: number;
  total_errors_audited: number;
  overall_error_rate: number;
  baseline_accuracy: number;
  subgroups: ErrorSubgroupPattern[];
  methodology: string;
  non_causal_statement: string;
}

export interface EraDefinition {
  era_name: string;
  season_range: string;
  description: string;
  avg_first_innings_score: number;
  avg_run_rate: number;
  brier_score: number;
  accuracy: number;
  matches: number;
}

export interface SeasonMetrics {
  season: string;
  matches: number;
  avg_first_innings_score: number;
  avg_run_rate: number;
  scores_200_plus_count: number;
  scores_200_plus_pct: number;
  favorite_accuracy: number;
  brier_score: number;
  log_loss: number;
  ece: number;
  overconfidence_rate: number;
}

export interface ShiftTestResult {
  comparison_name: string;
  metric: string;
  baseline_era: string;
  recent_era: string;
  baseline_mean: number;
  recent_mean: number;
  absolute_change: number;
  percentage_change: number;
  p_value?: number | null;
  is_statistically_significant: boolean;
  status: 'statistically_significant' | 'provisional' | 'inconclusive';
  analysis_type: 'scoring_distribution_shift' | 'model_performance_drift';
  notes: string;
}

export interface EraShiftMonitorData {
  eras: EraDefinition[];
  seasons: SeasonMetrics[];
  shift_tests: ShiftTestResult[];
  impact_player_rule_era_start: string;
  methodology: string;
  limitations: string[];
}

export interface IntelligenceSummaryData {
  live_model_test_brier: number;
  live_model_test_log_loss: number;
  live_model_test_ece: number;
  live_model_test_accuracy: number;
  live_model_test_matches: number;
  live_model_test_deliveries: number;
  pre_match_walkforward_matches: number;
  pre_match_walkforward_brier: number;
  pre_match_walkforward_accuracy: number;
  total_seasons_monitored: number;
  era_shift_detected: boolean;
  era_shift_summary: string;
  temporal_isolation_guarantee: string;
}
