export interface MatchSummary {
  match_id: string;
  season: string;
  date: string;
  team_1: string;
  team_2: string;
  venue: string;
  city: string;
  winner: string | null;
  match_result_type: 'normal' | 'super_over' | 'no_result' | string;
}

export interface MatchDetails extends MatchSummary {
  toss_winner: string;
  toss_decision: 'bat' | 'field' | string;
  method: string | null;
  is_super_over: boolean;
  is_dl_method: boolean;
  total_deliveries?: number | null;
}

export interface MatchListResponse {
  matches: MatchSummary[];
  total: number;
  limit: number;
  offset: number;
}
