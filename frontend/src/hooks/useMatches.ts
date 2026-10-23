import { useState, useEffect, useCallback } from 'react';
import { MatchSummary, MatchDetails } from '../types/match';
import { fetchMatches, fetchMatchDetails, fetchSeasons, fetchTeams } from '../api/matches';

const DEFAULT_SEASONS = [
  '2024', '2023', '2022', '2021', '2020/21', '2019', '2018',
  '2017', '2016', '2015', '2014', '2013', '2012', '2011',
  '2009/10', '2009', '2007/08'
];

export function useMatches(initialSeason: string = '2023') {
  const [matches, setMatches] = useState<MatchSummary[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [selectedSeason, setSelectedSeason] = useState<string>(initialSeason);
  const [selectedTeam, setSelectedTeam] = useState<string>('');
  const [selectedMatch, setSelectedMatch] = useState<MatchDetails | null>(null);
  const [availableSeasons, setAvailableSeasons] = useState<string[]>(DEFAULT_SEASONS);
  const [availableTeams, setAvailableTeams] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingDetails, setLoadingDetails] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Load available seasons once on mount
  useEffect(() => {
    fetchSeasons()
      .then((seasons) => {
        if (seasons && seasons.length > 0) {
          setAvailableSeasons(seasons);
        }
      })
      .catch(() => {
        // Fallback to DEFAULT_SEASONS on error
      });
  }, []);

  // Load season-aware teams whenever selectedSeason changes
  useEffect(() => {
    fetchTeams(selectedSeason || undefined)
      .then((teams) => {
        if (teams) {
          setAvailableTeams(teams);
          // If the currently selected team is not in the new season's teams, reset it
          setSelectedTeam((prev) => (prev && !teams.includes(prev) ? '' : prev));
        }
      })
      .catch(() => {
        // Fallback: keep previous teams or empty
      });
  }, [selectedSeason]);

  const loadMatches = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetchMatches(selectedSeason || undefined, selectedTeam || undefined, 100, 0);
      setMatches(resp.matches);
      setTotal(resp.total);
    } catch (err: any) {
      setError(err.message || 'Failed to load matches.');
    } finally {
      setLoading(false);
    }
  }, [selectedSeason, selectedTeam]);

  useEffect(() => {
    loadMatches();
  }, [loadMatches]);

  const selectMatch = async (matchId: string) => {
    setLoadingDetails(true);
    setError(null);
    try {
      const details = await fetchMatchDetails(matchId);
      setSelectedMatch(details);
      return details;
    } catch (err: any) {
      setError(err.message || 'Failed to load match details.');
      return null;
    } finally {
      setLoadingDetails(false);
    }
  };

  const clearSelectedMatch = () => {
    setSelectedMatch(null);
  };

  return {
    matches,
    total,
    selectedSeason,
    setSelectedSeason,
    selectedTeam,
    setSelectedTeam,
    availableSeasons,
    availableTeams,
    selectedMatch,
    selectMatch,
    clearSelectedMatch,
    loading,
    loadingDetails,
    error,
    refreshMatches: loadMatches,
  };
}
