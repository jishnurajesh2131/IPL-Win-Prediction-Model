export interface PredictionResponse {
  team_1: string;
  team_2: string;
  team_1_probability: number;
  team_2_probability: number;
  prediction_type: string;
}

export interface ProbabilityPoint {
  event_index: number;
  innings: number;
  ball: number;
  overs_completed: number;
  score: number;
  wickets: number;
  team_1_probability: number;
  team_2_probability: number;
  prediction_type: string;
  is_legal?: boolean;
}

export interface ProbabilityHistoryResponse {
  session_id: string;
  match_id: string;
  total_points: number;
  points: ProbabilityPoint[];
}
