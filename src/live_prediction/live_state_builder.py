"""
Live Match State Machine.

Maintains cumulative ball-by-ball match state strictly chronologically:
- Scores, wickets, legal balls, and overs
- Batter and bowler in-match performance
- Active partnership dynamics
- Rolling 6-ball and 12-ball momentum windows
- First innings total and second innings chase targets

Guarantees zero future leakage: state at delivery k depends ONLY on deliveries <= k.
"""

from typing import Dict, Any, List, Optional
from collections import deque

from src.utils.cricket_format import get_authoritative_delivery_ball


BOWLER_WICKET_TYPES = {
    "caught",
    "bowled",
    "lbw",
    "caught and bowled",
    "stumped",
    "hit wicket",
}


class LiveMatchStateMachine:
    """
    State machine tracking the progression of an individual IPL match across Innings 1 and 2.
    """

    def __init__(self, match_id: str, team_a: str, team_b: str):
        self.match_id = match_id
        self.team_a = team_a
        self.team_b = team_b

        # Innings tracking
        self.current_innings = 1
        self.innings_1_final_runs = 0
        self.target_runs = 0

        # Innings 1 & 2 cumulative counters
        # Key: innings int (1 or 2)
        self.innings_runs = {1: 0, 2: 0}
        self.innings_wickets = {1: 0, 2: 0}
        self.innings_legal_balls = {1: 0, 2: 0}
        self.innings_delivery_count = {1: 0, 2: 0}
        self.batting_teams = {1: "", 2: ""}
        self.bowling_teams = {1: "", 2: ""}

        # Current partnership (resets after a wicket falls)
        self.partnership_runs = 0
        self.partnership_balls = 0
        self._reset_partnership_next_ball = False

        # Rolling delivery history for current innings:
        # stores tuple: (runs_total, is_wicket, is_dot, is_boundary)
        self.rolling_deliveries: deque = deque(maxlen=12)

        # In-match batter stats: (innings, batter_name) -> dict
        # keys: runs, legal_balls, fours, sixes, dots
        self.batter_stats: Dict[tuple, Dict[str, int]] = {}

        # In-match bowler stats: (bowler_name) -> dict
        # keys: legal_balls, runs_conceded, wickets, maidens, dots, fours_conceded, sixes_conceded
        self.bowler_stats: Dict[str, Dict[str, int]] = {}

        # Innings bowler stats: (innings, bowler_name) -> dict
        self.inn_bowler_stats: Dict[tuple, Dict[str, int]] = {}

        # Batting and bowling appearance orders per innings
        self.batting_order: Dict[int, List[str]] = {1: [], 2: []}
        self.bowling_order: Dict[int, List[str]] = {1: [], 2: []}

        # Active player tracking
        self.current_striker = ""
        self.current_non_striker = ""
        self.current_bowler = ""

        # Over-level tracking for maidens: (innings, over_idx) -> {legal_balls, bowler_runs, bowler}
        self.over_tracking: Dict[tuple, Dict[str, Any]] = {}
        self.maiden_overs: set = set()

    def _get_or_create_batter(self, innings: int, batter: str) -> Dict[str, Any]:
        key = (innings, batter)
        if key not in self.batter_stats:
            self.batter_stats[key] = {
                "runs": 0,
                "legal_balls": 0,
                "fours": 0,
                "sixes": 0,
                "dots": 0,
                "dismissal": "not out",
            }
        return self.batter_stats[key]

    def _get_or_create_bowler(self, bowler: str, innings: Optional[int] = None) -> Dict[str, int]:
        if bowler not in self.bowler_stats:
            self.bowler_stats[bowler] = {
                "legal_balls": 0,
                "runs_conceded": 0,
                "wickets": 0,
                "maidens": 0,
                "dots": 0,
                "fours_conceded": 0,
                "sixes_conceded": 0,
            }
        elif "maidens" not in self.bowler_stats[bowler]:
            self.bowler_stats[bowler]["maidens"] = 0

        inn = innings or self.current_innings
        key = (inn, bowler)
        if key not in self.inn_bowler_stats:
            self.inn_bowler_stats[key] = {
                "legal_balls": 0,
                "runs_conceded": 0,
                "wickets": 0,
                "maidens": 0,
                "dots": 0,
                "fours_conceded": 0,
                "sixes_conceded": 0,
            }
        elif "maidens" not in self.inn_bowler_stats[key]:
            self.inn_bowler_stats[key]["maidens"] = 0

        return self.bowler_stats[bowler]

    def switch_innings(self, new_innings: int):
        """Transition from Innings 1 to Innings 2."""
        if new_innings == 2 and self.current_innings == 1:
            self.innings_1_final_runs = self.innings_runs[1]
            self.target_runs = self.innings_1_final_runs + 1
            self.current_innings = 2
            self.partnership_runs = 0
            self.partnership_balls = 0
            self._reset_partnership_next_ball = False
            self.rolling_deliveries.clear()

    def update_with_delivery(self, delivery: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single completed delivery and return the instantaneous post-delivery state.
        
        Information returned represents the match state IMMEDIATELY AFTER delivery k.
        """
        inn = int(delivery["innings"])
        if inn != self.current_innings:
            self.switch_innings(inn)

        bat_team = delivery["batting_team"]
        bowl_team = delivery["bowling_team"]
        self.batting_teams[inn] = bat_team
        self.bowling_teams[inn] = bowl_team

        striker = delivery["striker"]
        non_striker = delivery["non_striker"]
        bowler = delivery["bowler"]

        runs_off_bat = int(delivery["runs_off_bat"])
        extras = int(delivery["extras"])
        total_runs = runs_off_bat + extras

        wides = int(delivery["wides"])
        noballs = int(delivery["noballs"])
        is_legal = (wides == 0 and noballs == 0)

        wicket_type = str(delivery.get("wicket_type", "") or "").strip()
        player_dismissed = str(delivery.get("player_dismissed", "") or "").strip()
        is_wicket = (wicket_type != "" and wicket_type.lower() != "nan")
        is_bowler_wicket = is_wicket and (wicket_type in BOWLER_WICKET_TYPES)

        # Handle partnership reset if a wicket fell on previous delivery
        if self._reset_partnership_next_ball:
            self.partnership_runs = 0
            self.partnership_balls = 0
            self._reset_partnership_next_ball = False

        # 1. Update cumulative innings counters
        self.innings_runs[inn] += total_runs
        if is_wicket:
            self.innings_wickets[inn] += 1
            if player_dismissed:
                p_entry = self._get_or_create_batter(inn, player_dismissed)
                p_entry["dismissal"] = wicket_type if wicket_type else "out"
        if is_legal:
            self.innings_legal_balls[inn] += 1
        self.innings_delivery_count[inn] += 1

        # 2. Update partnership
        self.partnership_runs += total_runs
        if is_legal:
            self.partnership_balls += 1

        if is_wicket:
            # Mark partnership to reset on the NEXT delivery
            self._reset_partnership_next_ball = True

        # Track active players
        self.current_striker = striker
        self.current_non_striker = non_striker
        self.current_bowler = bowler

        if striker and striker not in self.batting_order[inn]:
            self.batting_order[inn].append(striker)
        if non_striker and non_striker not in self.batting_order[inn]:
            self.batting_order[inn].append(non_striker)
        if bowler and bowler not in self.bowling_order[inn]:
            self.bowling_order[inn].append(bowler)

        # 3. Update active batter figures (current innings)
        b_entry = self._get_or_create_batter(inn, striker)
        if non_striker:
            self._get_or_create_batter(inn, non_striker)
        b_entry["runs"] += runs_off_bat
        # Batter balls faced: legal deliveries + noballs (wides excluded)
        if wides == 0:
            b_entry["legal_balls"] += 1
        if runs_off_bat == 4:
            b_entry["fours"] += 1
        elif runs_off_bat == 6:
            b_entry["sixes"] += 1
        if is_legal and runs_off_bat == 0:
            b_entry["dots"] += 1

        # 4. Update active bowler figures (current match & innings)
        bow_entry = self._get_or_create_bowler(bowler, inn)
        inn_bow_entry = self.inn_bowler_stats[(inn, bowler)]

        if is_legal:
            bow_entry["legal_balls"] += 1
            inn_bow_entry["legal_balls"] += 1
        # Bowler runs conceded: bat runs + bowler extras (wides and noballs)
        bowler_runs_conceded = runs_off_bat + wides + noballs
        bow_entry["runs_conceded"] += bowler_runs_conceded
        inn_bow_entry["runs_conceded"] += bowler_runs_conceded
        if is_bowler_wicket:
            bow_entry["wickets"] += 1
            inn_bow_entry["wickets"] += 1
        if total_runs == 0:
            bow_entry["dots"] += 1
            inn_bow_entry["dots"] += 1
        if runs_off_bat == 4:
            bow_entry["fours_conceded"] += 1
            inn_bow_entry["fours_conceded"] += 1
        elif runs_off_bat == 6:
            bow_entry["sixes_conceded"] += 1
            inn_bow_entry["sixes_conceded"] += 1

        # Maiden over tracking
        over_idx = int(float(delivery.get("actual_delivery", delivery["ball"])))
        over_key = (inn, over_idx)
        if over_key not in self.over_tracking:
            self.over_tracking[over_key] = {
                "legal_balls": 0,
                "bowler_runs": 0,
                "bowler": bowler,
            }
        if is_legal:
            self.over_tracking[over_key]["legal_balls"] += 1
        self.over_tracking[over_key]["bowler_runs"] += bowler_runs_conceded

        if self.over_tracking[over_key]["legal_balls"] == 6:
            if self.over_tracking[over_key]["bowler_runs"] == 0 and over_key not in self.maiden_overs:
                self.maiden_overs.add(over_key)
                bow_entry["maidens"] += 1
                inn_bow_entry["maidens"] += 1

        # 5. Update rolling momentum buffer
        is_dot = (total_runs == 0)
        is_boundary = (runs_off_bat in (4, 6))
        self.rolling_deliveries.append((total_runs, 1 if is_wicket else 0, is_dot, is_boundary))

        # Compute rolling window statistics
        rolling_list = list(self.rolling_deliveries)
        last_6 = rolling_list[-6:] if len(rolling_list) >= 6 else rolling_list
        last_12 = rolling_list

        runs_last_6 = sum(item[0] for item in last_6)
        wickets_last_6 = sum(item[1] for item in last_6)
        runs_last_12 = sum(item[0] for item in last_12)
        wickets_last_12 = sum(item[1] for item in last_12)

        n_12 = len(last_12)
        dot_ratio_12 = (sum(1 for item in last_12 if item[2]) / n_12) if n_12 > 0 else 0.0
        boundary_ratio_12 = (sum(1 for item in last_12 if item[3]) / n_12) if n_12 > 0 else 0.0

        # Snapshot of instantaneous state
        return {
            "innings": inn,
            "batting_team": bat_team,
            "bowling_team": bowl_team,
            "striker": striker,
            "non_striker": non_striker,
            "bowler": bowler,
            "delivery_index": self.innings_delivery_count[inn],
            "ball": get_authoritative_delivery_ball(delivery, self.innings_legal_balls[inn], is_legal),
            "is_legal": is_legal,
            # Innings totals
            "current_innings_runs": self.innings_runs[inn],
            "current_innings_wickets": self.innings_wickets[inn],
            "legal_balls_bowled": self.innings_legal_balls[inn],
            "target_runs": self.target_runs,
            # Active partnership
            "partnership_runs": self.partnership_runs,
            "partnership_balls": self.partnership_balls,
            # Rolling momentum
            "runs_last_6": runs_last_6,
            "wickets_last_6": wickets_last_6,
            "runs_last_12": runs_last_12,
            "wickets_last_12": wickets_last_12,
            "dot_ball_ratio_last_12": round(dot_ratio_12, 4),
            "boundary_ratio_last_12": round(boundary_ratio_12, 4),
            # Batter current figures
            "batter_runs_current_inn": b_entry["runs"],
            "batter_balls_current_inn": b_entry["legal_balls"],
            "batter_boundaries_current": b_entry["fours"] + b_entry["sixes"],
            "batter_dots_current": b_entry["dots"],
            # Bowler current figures
            "bowler_legal_balls_current": bow_entry["legal_balls"],
            "bowler_runs_conceded_current": bow_entry["runs_conceded"],
            "bowler_wickets_current": bow_entry["wickets"],
            "bowler_dots_current": bow_entry["dots"],
            "bowler_boundaries_conceded_current": bow_entry["fours_conceded"] + bow_entry["sixes_conceded"],
            # Complete batting and bowling tables for current innings
            "batting": self.get_batting_figures(inn),
            "bowling": self.get_bowling_figures(inn),
        }

    def get_batting_figures(self, innings: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Returns the ordered list of batter figures for the requested or current innings.
        """
        inn = innings if innings is not None else self.current_innings
        figures = []
        for b_name in self.batting_order.get(inn, []):
            b_data = self.batter_stats.get((inn, b_name), {
                "runs": 0, "legal_balls": 0, "fours": 0, "sixes": 0, "dots": 0
            })
            runs = int(b_data["runs"])
            balls = int(b_data["legal_balls"])
            fours = int(b_data["fours"])
            sixes = int(b_data["sixes"])
            sr = round((runs / balls) * 100.0, 2) if balls > 0 else 0.0
            is_striker = bool(b_name == self.current_striker and inn == self.current_innings)
            figures.append({
                "player_name": b_name,
                "dismissal": b_data.get("dismissal", "not out"),
                "runs": runs,
                "balls": balls,
                "fours": fours,
                "sixes": sixes,
                "strike_rate": sr,
                "is_striker": is_striker,
            })
        return figures

    def get_bowling_figures(self, innings: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Returns the ordered list of bowler figures for the requested or current innings.
        """
        inn = innings if innings is not None else self.current_innings
        figures = []
        for bow_name in self.bowling_order.get(inn, []):
            bow_data = self.inn_bowler_stats.get((inn, bow_name)) or self.bowler_stats.get(bow_name, {
                "legal_balls": 0, "runs_conceded": 0, "wickets": 0, "maidens": 0
            })
            legal_balls = int(bow_data["legal_balls"])
            completed_overs = legal_balls // 6
            rem_balls = legal_balls % 6
            overs_str = f"{completed_overs}.{rem_balls}"
            maidens = int(bow_data.get("maidens", 0))
            runs = int(bow_data["runs_conceded"])
            wickets = int(bow_data["wickets"])
            economy = round(runs / (legal_balls / 6.0), 2) if legal_balls > 0 else 0.0
            is_cur_bowler = bool(bow_name == self.current_bowler and inn == self.current_innings)
            figures.append({
                "player_name": bow_name,
                "overs": overs_str,
                "maidens": maidens,
                "runs": runs,
                "wickets": wickets,
                "economy": economy,
                "is_current_bowler": is_cur_bowler,
            })
        return figures

