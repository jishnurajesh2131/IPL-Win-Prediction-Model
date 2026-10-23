import { PredictionResponse, ProbabilityPoint } from './prediction';

export interface BattingFigure {
  player_name: string;
  dismissal?: string;
  runs: number;
  balls: number;
  fours: number;
  sixes: number;
  strike_rate: number;
  is_striker: boolean;
}

export interface BowlingFigure {
  player_name: string;
  overs: string;
  maidens: number;
  runs: number;
  wickets: number;
  economy: number;
  is_current_bowler: boolean;
}

export interface InningsScorecard {
  innings: number;
  team?: string | null;
  score: number;
  wickets: number;
  overs: string;
  legal_balls: number;
  batting: BattingFigure[];
  bowling: BowlingFigure[];
}

export interface MatchStateSnapshot {
  innings: number;
  score: number;
  wickets: number;
  legal_balls: number;
  overs: string;
  overs_completed: number;
  current_run_rate: number;
  target_runs: number | null;
  runs_needed: number | null;
  required_run_rate: number | null;
  rrr_crr_differential: number | null;
  striker: string;
  non_striker: string;
  bowler: string;
  batting_team?: string;
  bowling_team?: string;
  batting?: BattingFigure[];
  bowling?: BowlingFigure[];
  innings_1?: InningsScorecard | null;
  innings_2?: InningsScorecard | null;
}

export interface DeliverySnapshot {
  innings: number;
  ball: number;
  over?: number;
  ball_in_over?: number;
  display_ball?: string;
  event_index?: number;
  is_legal: boolean;
  batting_team: string;
  bowling_team: string;
  striker: string;
  non_striker: string;
  bowler: string;
  runs_off_bat: number;
  extras: number;
  wides: number;
  noballs: number;
  wicket_type: string | null;
  player_dismissed: string | null;
}

export interface SimulationCreateRequest {
  match_id: string;
}

export interface SimulationCreateResponse {
  session_id: string;
  match_id: string;
  status: 'ready' | 'active' | 'completed' | 'error' | string;
  current_event_index: number;
  total_events: number;
  current_prediction: PredictionResponse;
}

export interface SimulationStepResponse {
  session_id: string;
  match_id: string;
  event_index: number;
  total_events: number;
  status: 'ready' | 'active' | 'completed' | 'error' | string;
  delivery: DeliverySnapshot | null;
  state: MatchStateSnapshot;
  prediction: PredictionResponse;
  match_intelligence?: MatchIntelligenceSnapshot | null;
}

export interface SimulationRunResponse {
  session_id: string;
  match_id: string;
  status: 'completed' | string;
  events_processed: number;
  total_events: number;
  winner: string;
  final_state: MatchStateSnapshot;
  final_prediction: PredictionResponse;
  delivery?: DeliverySnapshot | null;
  match_intelligence?: MatchIntelligenceSnapshot | null;
  history?: ProbabilityPoint[] | null;
}

export interface HealthResponse {
  status: 'ok' | 'degraded';
  service: string;
  version: string;
  ml_status: 'loaded' | 'error';
  model_artifacts: Record<string, boolean>;
  matches_available: number;
}

export interface MatchIntelligenceSnapshot {
  current_team?: string | null;
  current_win_probability?: number | null;
  trend_probability: number | null;
  trend_team: string | null;
  volatility: number | null;
  criticality: number | null;
  turning_point_over: string | null;
  turning_point_event: string | null;
  turning_point_wpa: number | null;
  turning_point_team: string | null;
  influential_player: string | null;
  influential_player_wpa: number | null;
  historical_similarity: string | null;
}
