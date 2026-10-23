import { BattingFigure, BowlingFigure } from './simulation';

export interface LiveMatchSummary {
  provider_match_id: string;
  name: string;
  match_type: string;
  status_text: string;
  match_started: boolean;
  match_ended: boolean;
  date: string;
  teams: string[];
  venue: string;
  toss_winner?: string | null;
  toss_decision?: string | null;
  has_squad: boolean;
  bbb_enabled: boolean;
}

export interface LiveSimulationResponse {
  session_id: string;
  match_id: string;
  provider: string;
  status: string;
  team_a: string;
  team_b: string;
  venue: string;
  city: string;
  current_innings: number;
  score: number;
  wickets: number;
  legal_balls: number;
  overs: string;
  current_run_rate: number;
  target_runs: number;
  runs_needed: number;
  required_run_rate: number;
  striker: string;
  non_striker: string;
  bowler: string;
  team_a_win_probability: number;
  team_b_win_probability: number;
  prediction_type: string;
  last_delivery?: {
    ball: number;
    innings: number;
    runs: number;
    delivery_type: string;
    batsman: string;
    bowler: string;
    text: string;
  } | null;
  total_events: number;
  last_updated: string;
  last_updated_timestamp: number;
  is_delayed: boolean;
  dls_active: boolean;
  dls_supported: boolean;
  dls_message: string;
  validation?: {
    is_matched: boolean;
    discrepancies: string[];
  } | null;
  batting_team?: string;
  bowling_team?: string;
  batting?: BattingFigure[];
  bowling?: BowlingFigure[];
}

export interface LiveProbabilityPoint {
  event_number: number;
  innings: number;
  ball: number;
  score: number;
  wickets: number;
  team_a_win_probability: number;
  team_b_win_probability: number;
  prediction_type: string;
  delivery_type: string;
}

export interface LiveProbabilityHistoryResponse {
  session_id: string;
  match_id: string;
  team_a: string;
  team_b: string;
  total_points: number;
  probabilities: LiveProbabilityPoint[];
}
