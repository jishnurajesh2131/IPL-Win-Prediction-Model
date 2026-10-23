"""
Stateful Historical Statistics Tracker for Pre-Match Features.

Maintains rolling and expanding historical statistics strictly chronologically.
Guarantees zero future leakage by enforcing:
1. compute_pre_match_features(...) queries state BEFORE match outcome is incorporated.
2. update_with_match_result(...) updates state strictly AFTER features have been computed.
"""

from collections import defaultdict
from typing import Dict, Any, List, Optional


class HistoricalStateTracker:
    """
    Chronological state tracker for team form, head-to-head records,
    venue performance, batting run rate, and bowling economy.
    """

    def __init__(self, default_league_rpo: float = 8.20):
        self.default_league_rpo = default_league_rpo

        # 1. Team completed matches: team -> list of bool (True if won, False if lost)
        self.team_history: Dict[str, List[bool]] = defaultdict(list)

        # 2. Head-to-Head: (team_a, team_b) sorted tuple -> list of winner name
        self.h2h_history: Dict[tuple, List[str]] = defaultdict(list)

        # 3. Venue history: (team, venue) -> list of bool (won/lost)
        self.team_venue_history: Dict[tuple, List[bool]] = defaultdict(list)

        # 4. Venue chasing history: venue -> list of bool (chasing team won)
        self.venue_chasing_history: Dict[str, List[bool]] = defaultdict(list)

        # 5. Expanding league chase stats
        self.league_chase_wins = 0
        self.league_total_matches = 0

        # 6. Batting strength: team -> list of {'runs': int, 'legal_balls': int}
        self.team_batting_history: Dict[str, List[Dict[str, int]]] = defaultdict(list)

        # 7. Bowling strength: team -> list of {'runs_conceded': int, 'legal_balls_bowled': int, 'wickets': int}
        self.team_bowling_history: Dict[str, List[Dict[str, int]]] = defaultdict(list)

        # 8. Expanding league totals for RPO / Economy prior
        self.league_total_runs = 0
        self.league_total_legal_balls = 0

    @property
    def expanding_league_rpo(self) -> float:
        """Expanding league run rate (runs per over) prior to current match."""
        if self.league_total_legal_balls < 120:
            return self.default_league_rpo
        total_overs = self.league_total_legal_balls / 6.0
        return self.league_total_runs / total_overs

    @property
    def expanding_league_chase_rate(self) -> float:
        """Expanding league chasing win percentage prior to current match."""
        if self.league_total_matches < 10:
            return 0.50
        return self.league_chase_wins / self.league_total_matches

    def _get_smoothed_form(self, team: str, window: int) -> float:
        """
        Calculate rolling win rate with empirical Bayesian smoothing towards 0.50 prior.
        """
        history = self.team_history[team]
        recent = history[-window:] if len(history) >= window else history
        observed_wins = sum(1 for w in recent if w)
        n = len(recent)
        # Smoothing formula: (wins + m * p0) / (n + m), with m=2, p0=0.50
        return round((observed_wins + 1.0) / (n + 2.0), 4)

    def _get_overall_win_rate(self, team: str) -> float:
        """Cumulative historical win rate of a team prior to current match."""
        history = self.team_history[team]
        if not history:
            return 0.50
        wins = sum(1 for w in history if w)
        return (wins + 1.0) / (len(history) + 2.0)

    def _get_smoothed_venue_win_rate(self, team: str, venue: str) -> float:
        """
        Calculate venue win rate smoothed towards the team's overall historical win rate.
        """
        key = (team, venue)
        venue_records = self.team_venue_history[key]
        n = len(venue_records)
        prior_p0 = self._get_overall_win_rate(team)
        m = 3.0  # prior weight of 3 matches
        wins = sum(1 for w in venue_records if w)
        return round((wins + m * prior_p0) / (n + m), 4)

    def _get_venue_chasing_win_rate(self, venue: str) -> float:
        """
        Calculate ground-level chasing win rate smoothed towards expanding league chase rate.
        """
        records = self.venue_chasing_history[venue]
        n = len(records)
        p0 = self.expanding_league_chase_rate
        m = 5.0  # prior weight of 5 matches
        wins = sum(1 for w in records if w)
        return round((wins + m * p0) / (n + m), 4)

    def _get_batting_rpo(self, team: str, window: int = 5) -> float:
        """
        Calculate rolling batting runs per over with shrinkage towards league RPO.
        """
        history = self.team_batting_history[team]
        recent = history[-window:] if len(history) >= window else history

        observed_runs = sum(inn["runs"] for inn in recent)
        observed_overs = sum(inn["legal_balls"] for inn in recent) / 6.0

        p0_rpo = self.expanding_league_rpo
        # Prior equivalent to m=2 innings of 20 overs (40 overs)
        m_overs = 40.0
        prior_runs = p0_rpo * m_overs

        smoothed_rpo = (observed_runs + prior_runs) / (observed_overs + m_overs)
        return round(smoothed_rpo, 4)

    def _get_bowling_economy(self, team: str, window: int = 5) -> float:
        """
        Calculate rolling bowling economy (runs conceded per over) with shrinkage.
        """
        history = self.team_bowling_history[team]
        recent = history[-window:] if len(history) >= window else history

        observed_runs = sum(inn["runs_conceded"] for inn in recent)
        observed_overs = sum(inn["legal_balls_bowled"] for inn in recent) / 6.0

        p0_econ = self.expanding_league_rpo  # league runs conceded == league runs scored
        m_overs = 40.0
        prior_runs = p0_econ * m_overs

        smoothed_econ = (observed_runs + prior_runs) / (observed_overs + m_overs)
        return round(smoothed_econ, 4)

    def compute_pre_match_features(
        self,
        team_a: str,
        team_b: str,
        venue: str,
        city: str,
        toss_winner: str,
        toss_decision: str,
    ) -> Dict[str, Any]:
        """
        Compute all pre-match features available at T_0 (post-toss, pre-first ball).
        Uses ONLY information stored prior to this match.
        """
        # Toss features
        toss_winner_is_team_a = 1 if toss_winner == team_a else 0
        toss_decision_is_bat = 1 if toss_decision == "bat" else 0

        # Team A bats first if:
        # (Team A won toss and chose bat) OR (Team B won toss and chose field)
        team_a_bats_first = 1 if (
            (toss_winner_is_team_a == 1 and toss_decision_is_bat == 1)
            or (toss_winner_is_team_a == 0 and toss_decision_is_bat == 0)
        ) else 0

        # Form features
        team_a_form_last_5 = self._get_smoothed_form(team_a, 5)
        team_b_form_last_5 = self._get_smoothed_form(team_b, 5)
        team_a_form_last_10 = self._get_smoothed_form(team_a, 10)
        team_b_form_last_10 = self._get_smoothed_form(team_b, 10)
        form_diff_last_5 = round(team_a_form_last_5 - team_b_form_last_5, 4)

        team_a_matches_played = len(self.team_history[team_a])
        team_b_matches_played = len(self.team_history[team_b])

        # Head-to-head features
        h2h_key = tuple(sorted([team_a, team_b]))
        h2h_records = self.h2h_history[h2h_key]
        h2h_matches_played = len(h2h_records)
        team_a_h2h_wins = sum(1 for w in h2h_records if w == team_a)
        team_a_h2h_win_rate = round((team_a_h2h_wins + 1.0) / (h2h_matches_played + 2.0), 4)

        # Venue features
        team_a_venue_matches = len(self.team_venue_history[(team_a, venue)])
        team_a_venue_win_rate = self._get_smoothed_venue_win_rate(team_a, venue)

        team_b_venue_matches = len(self.team_venue_history[(team_b, venue)])
        team_b_venue_win_rate = self._get_smoothed_venue_win_rate(team_b, venue)

        venue_win_rate_diff = round(team_a_venue_win_rate - team_b_venue_win_rate, 4)
        venue_chasing_win_rate = self._get_venue_chasing_win_rate(venue)

        # Batting and bowling strength
        team_a_batting_rpo_last_5 = self._get_batting_rpo(team_a, 5)
        team_b_batting_rpo_last_5 = self._get_batting_rpo(team_b, 5)
        batting_rpo_diff_last_5 = round(team_a_batting_rpo_last_5 - team_b_batting_rpo_last_5, 4)

        team_a_bowling_econ_last_5 = self._get_bowling_economy(team_a, 5)
        team_b_bowling_econ_last_5 = self._get_bowling_economy(team_b, 5)
        # Note: In bowling, lower economy is better.
        # bowling_econ_diff > 0 means Team A concedes fewer runs than Team B (Team A advantage)
        bowling_econ_diff_last_5 = round(team_b_bowling_econ_last_5 - team_a_bowling_econ_last_5, 4)

        return {
            "team_a": team_a,
            "team_b": team_b,
            "venue": venue,
            "city": city,
            "toss_winner_is_team_a": toss_winner_is_team_a,
            "toss_decision_is_bat": toss_decision_is_bat,
            "team_a_bats_first": team_a_bats_first,
            "team_a_form_last_5": team_a_form_last_5,
            "team_b_form_last_5": team_b_form_last_5,
            "team_a_form_last_10": team_a_form_last_10,
            "team_b_form_last_10": team_b_form_last_10,
            "form_diff_last_5": form_diff_last_5,
            "team_a_matches_played": team_a_matches_played,
            "team_b_matches_played": team_b_matches_played,
            "h2h_matches_played": h2h_matches_played,
            "team_a_h2h_win_rate": team_a_h2h_win_rate,
            "team_a_venue_matches": team_a_venue_matches,
            "team_a_venue_win_rate": team_a_venue_win_rate,
            "team_b_venue_matches": team_b_venue_matches,
            "team_b_venue_win_rate": team_b_venue_win_rate,
            "venue_win_rate_diff": venue_win_rate_diff,
            "venue_chasing_win_rate": venue_chasing_win_rate,
            "team_a_batting_rpo_last_5": team_a_batting_rpo_last_5,
            "team_b_batting_rpo_last_5": team_b_batting_rpo_last_5,
            "batting_rpo_diff_last_5": batting_rpo_diff_last_5,
            "team_a_bowling_econ_last_5": team_a_bowling_econ_last_5,
            "team_b_bowling_econ_last_5": team_b_bowling_econ_last_5,
            "bowling_econ_diff_last_5": bowling_econ_diff_last_5,
        }

    def update_with_match_result(
        self,
        team_a: str,
        team_b: str,
        winner: str,
        venue: str,
        innings_stats: Optional[Dict[str, Any]] = None,
        chasing_team_won: bool = False,
    ):
        """
        Update historical state strictly AFTER feature row has been computed.
        """
        # 1. Update team history
        team_a_won = (winner == team_a)
        team_b_won = (winner == team_b)

        self.team_history[team_a].append(team_a_won)
        self.team_history[team_b].append(team_b_won)

        # 2. Update head-to-head
        h2h_key = tuple(sorted([team_a, team_b]))
        self.h2h_history[h2h_key].append(winner)

        # 3. Update venue history
        self.team_venue_history[(team_a, venue)].append(team_a_won)
        self.team_venue_history[(team_b, venue)].append(team_b_won)

        # 4. Update venue chasing history
        self.venue_chasing_history[venue].append(chasing_team_won)
        self.league_total_matches += 1
        if chasing_team_won:
            self.league_chase_wins += 1

        # 5. Update batting and bowling delivery totals
        if innings_stats:
            for inn_data in innings_stats:
                batting_team = inn_data.get("batting_team")
                bowling_team = inn_data.get("bowling_team")
                runs = inn_data.get("runs", 0)
                legal_balls = inn_data.get("legal_balls", 0)
                wickets = inn_data.get("wickets", 0)

                if batting_team:
                    self.team_batting_history[batting_team].append({
                        "runs": runs,
                        "legal_balls": legal_balls,
                    })
                if bowling_team:
                    self.team_bowling_history[bowling_team].append({
                        "runs_conceded": runs,
                        "legal_balls_bowled": legal_balls,
                        "wickets": wickets,
                    })

                self.league_total_runs += runs
                self.league_total_legal_balls += legal_balls
