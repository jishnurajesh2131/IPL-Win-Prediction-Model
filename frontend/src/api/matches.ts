import { request } from './client';
import { MatchListResponse, MatchDetails } from '../types/match';

export async function fetchMatches(
  season?: string,
  team?: string,
  limit: number = 50,
  offset: number = 0
): Promise<MatchListResponse> {
  const params = new URLSearchParams();
  if (season) params.append('season', season);
  if (team) params.append('team', team);
  params.append('limit', limit.toString());
  params.append('offset', offset.toString());

  return request<MatchListResponse>(`/matches?${params.toString()}`);
}

export async function fetchMatchDetails(matchId: string): Promise<MatchDetails> {
  return request<MatchDetails>(`/matches/${encodeURIComponent(matchId)}`);
}

export async function fetchSeasons(): Promise<string[]> {
  return request<string[]>('/matches/seasons');
}

export async function fetchTeams(season?: string): Promise<string[]> {
  const url = season ? `/matches/teams?season=${encodeURIComponent(season)}` : '/matches/teams';
  return request<string[]>(url);
}
