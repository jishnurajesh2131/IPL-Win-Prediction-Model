"""
Player Feature Extractor & Historical Player State Tracker.

Computes:
1. Current-match active batter & active bowler figures.
2. Chronological historical player priors (< t) using Empirical Bayesian shrinkage.
"""

from typing import Dict, Any
from collections import defaultdict


class HistoricalPlayerTracker:
    """
    Chronological historical tracker for player career performance.
    Guarantees strict anti-leakage: stats for match M use only matches completed BEFORE match M.
    """

    def __init__(self):
        # Career batting: batter -> {'runs': int, 'balls': int, 'outs': int}
        self.batting_careers: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"runs": 0, "balls": 0, "outs": 0}
        )

        # Career bowling: bowler -> {'balls': int, 'runs_conceded': int, 'wickets': int}
        self.bowling_careers: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"balls": 0, "runs_conceded": 0, "wickets": 0}
        )

        # Expanding league totals
        self.league_bat_runs = 0
        self.league_bat_balls = 0
        self.league_bat_outs = 0

        self.league_bowl_runs = 0
        self.league_bowl_balls = 0
        self.league_bowl_wickets = 0

    @property
    def league_strike_rate(self) -> float:
        if self.league_bat_balls < 300:
            return 130.0
        return (self.league_bat_runs / self.league_bat_balls) * 100.0

    @property
    def league_batting_average(self) -> float:
        if self.league_bat_outs < 20:
            return 26.0
        return self.league_bat_runs / max(1, self.league_bat_outs)

    @property
    def league_economy(self) -> float:
        if self.league_bowl_balls < 300:
            return 8.20
        total_overs = self.league_bowl_balls / 6.0
        return self.league_bowl_runs / max(0.1667, total_overs)

    @property
    def league_bowling_strike_rate(self) -> float:
        if self.league_bowl_wickets < 20:
            return 20.0
        return self.league_bowl_balls / max(1, self.league_bowl_wickets)

    def get_historical_batter_stats(self, batter: str) -> Dict[str, float]:
        """
        Query Empirical Bayes shrunk historical batting statistics prior to current match.
        """
        record = self.batting_careers[batter]
        runs = record["runs"]
        balls = record["balls"]
        outs = record["outs"]

        # 1. Shrunk Strike Rate (m = 60 balls prior)
        m_sr = 60.0
        league_sr = self.league_strike_rate
        # league_sr is runs per 100 balls; prior runs in m balls = m * (league_sr / 100)
        smoothed_sr = ((runs + m_sr * (league_sr / 100.0)) / (balls + m_sr)) * 100.0

        # 2. Shrunk Batting Average (m = 5 dismissals prior)
        m_avg = 5.0
        league_avg = self.league_batting_average
        smoothed_avg = (runs + m_avg * league_avg) / (outs + m_avg)

        return {
            "historical_batter_strike_rate": round(smoothed_sr, 2),
            "historical_batter_average": round(smoothed_avg, 2),
            "historical_batter_balls_faced": balls,
        }

    def get_historical_bowler_stats(self, bowler: str) -> Dict[str, float]:
        """
        Query Empirical Bayes shrunk historical bowling statistics prior to current match.
        """
        record = self.bowling_careers[bowler]
        balls = record["balls"]
        runs_conceded = record["runs_conceded"]
        wickets = record["wickets"]
        overs = balls / 6.0

        # 1. Shrunk Economy Rate (m = 10 overs / 60 balls prior)
        m_econ = 10.0  # overs
        league_econ = self.league_economy
        smoothed_econ = (runs_conceded + m_econ * league_econ) / (overs + m_econ)

        # 2. Shrunk Bowling Strike Rate (m = 5 wickets prior)
        m_wickets = 5.0
        league_bowl_sr = self.league_bowling_strike_rate
        smoothed_bowl_sr = (balls + m_wickets * league_bowl_sr) / (wickets + m_wickets)

        return {
            "historical_bowler_economy": round(smoothed_econ, 2),
            "historical_bowler_strike_rate": round(smoothed_bowl_sr, 2),
            "historical_bowler_balls_bowled": balls,
        }

    def update_with_match(self, match_deliveries: list):
        """
        Update historical player totals strictly AFTER all deliveries of a match have concluded.
        """
        bowler_wicket_types = {
            "caught",
            "bowled",
            "lbw",
            "caught and bowled",
            "stumped",
            "hit wicket",
        }

        # Track per-player match increments
        match_batters = defaultdict(lambda: {"runs": 0, "balls": 0, "outs": 0})
        match_bowlers = defaultdict(lambda: {"balls": 0, "runs_conceded": 0, "wickets": 0})

        for d in match_deliveries:
            inn = int(d["innings"])
            if inn not in (1, 2):
                continue  # exclude Super Overs from standard 20-over priors

            striker = d["striker"]
            bowler = d["bowler"]
            runs_off_bat = int(d["runs_off_bat"])
            extras = int(d["extras"])
            wides = int(d["wides"])
            noballs = int(d["noballs"])
            is_legal = (wides == 0 and noballs == 0)

            # Batting update
            match_batters[striker]["runs"] += runs_off_bat
            if wides == 0:
                match_batters[striker]["balls"] += 1

            # Check dismissal of batter
            player_dismissed = str(d.get("player_dismissed", "") or "").strip()
            if player_dismissed and player_dismissed.lower() != "nan":
                match_batters[player_dismissed]["outs"] += 1

            # Bowling update
            if is_legal:
                match_bowlers[bowler]["balls"] += 1
            # Conceded by bowler: bat runs + wides + noballs
            match_bowlers[bowler]["runs_conceded"] += (runs_off_bat + wides + noballs)

            w_type = str(d.get("wicket_type", "") or "").strip()
            if w_type in bowler_wicket_types:
                match_bowlers[bowler]["wickets"] += 1

        # Apply match increments to career totals
        for b, data in match_batters.items():
            self.batting_careers[b]["runs"] += data["runs"]
            self.batting_careers[b]["balls"] += data["balls"]
            self.batting_careers[b]["outs"] += data["outs"]
            self.league_bat_runs += data["runs"]
            self.league_bat_balls += data["balls"]
            self.league_bat_outs += data["outs"]

        for bow, data in match_bowlers.items():
            self.bowling_careers[bow]["balls"] += data["balls"]
            self.bowling_careers[bow]["runs_conceded"] += data["runs_conceded"]
            self.bowling_careers[bow]["wickets"] += data["wickets"]
            self.league_bowl_balls += data["balls"]
            self.league_bowl_runs += data["runs_conceded"]
            self.league_bowl_wickets += data["wickets"]


def extract_current_player_features(raw_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts in-match micro-performance metrics for the active striker and bowler.
    """
    b_runs = raw_state["batter_runs_current_inn"]
    b_balls = raw_state["batter_balls_current_inn"]
    b_dots = raw_state["batter_dots_current"]
    b_boundaries = raw_state["batter_boundaries_current"]

    b_sr = round((b_runs / max(1, b_balls)) * 100.0, 2)
    b_dot_ratio = round(b_dots / max(1, b_balls), 4)

    bow_balls = raw_state["bowler_legal_balls_current"]
    bow_overs = round(bow_balls / 6.0, 4)
    bow_runs = raw_state["bowler_runs_conceded_current"]
    bow_wkts = raw_state["bowler_wickets_current"]
    bow_dots = raw_state["bowler_dots_current"]
    bow_boundaries = raw_state["bowler_boundaries_conceded_current"]

    bow_econ = round(bow_runs / max(0.1667, bow_overs), 2)
    bow_dot_ratio = round(bow_dots / max(1, bow_balls), 4)

    return {
        # Active batter in-match state
        "batter_runs_current_inn": b_runs,
        "batter_balls_current_inn": b_balls,
        "batter_strike_rate_current": b_sr,
        "batter_boundaries_current": b_boundaries,
        "batter_dots_current": b_dots,
        "batter_dot_ratio_current": b_dot_ratio,
        # Active bowler in-match state
        "bowler_legal_balls_current": bow_balls,
        "bowler_overs_current": bow_overs,
        "bowler_runs_conceded_current": bow_runs,
        "bowler_wickets_current": bow_wkts,
        "bowler_economy_current": bow_econ,
        "bowler_dots_current": bow_dots,
        "bowler_boundaries_conceded_current": bow_boundaries,
        "bowler_dot_ratio_current": bow_dot_ratio,
    }
