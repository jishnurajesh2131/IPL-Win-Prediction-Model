import os
import pandas as pd
from typing import List, Optional
from src.live_simulator.simulator import SimulationEvent
from src.backend.schemas.simulation import MatchIntelligenceSnapshot
from src.live_simulator.model_runner import LiveModelRunner
from src.backend.config import LIVE_DATASET_CSV


class MatchIntelligenceService:
    def __init__(self):
        self._live_dataset: Optional[pd.DataFrame] = None
        self._load_live_dataset()

    def _load_live_dataset(self):
        try:
            if os.path.exists(LIVE_DATASET_CSV):
                cols = [
                    "current_innings",
                    "wickets_remaining",
                    "legal_balls_remaining",
                    "runs_needed",
                    "current_innings_runs",
                    "target_runs",
                    "target_team_a_win",
                    "is_team_a_batting",
                ]
                self._live_dataset = pd.read_csv(LIVE_DATASET_CSV, usecols=cols)
            else:
                print(f"MatchIntelligenceService: Dataset file not found at {LIVE_DATASET_CSV}")
        except Exception as e:
            print(f"MatchIntelligenceService: Could not load live dataset for similarities: {e}")

    def calculate_intelligence(
        self,
        events: List[SimulationEvent],
        current_index: int,
        match_id: str,
        team_a: str,
        team_b: str,
        model_runner: Optional[LiveModelRunner] = None,
    ) -> MatchIntelligenceSnapshot:
        # Default empty snapshot if no events
        if not events or current_index < 0:
            return MatchIntelligenceSnapshot()

        current_event = events[current_index]

        # 1. Current Win Probability (Leading Team from existing prediction engine)
        if current_event.team_a_win_probability >= current_event.team_b_win_probability:
            current_team = team_a
            current_prob = round(current_event.team_a_win_probability * 100, 1)
        else:
            current_team = team_b
            current_prob = round(current_event.team_b_win_probability * 100, 1)

        # 2. Probability Trend (last 12 legal balls / ~2 overs)
        # If fewer than 2 overs (12 legal balls) of prediction history exist: "Awaiting sufficient history"
        trend_prob: Optional[float] = None
        trend_team: Optional[str] = None

        if current_event.legal_balls_bowled >= 12 and current_index > 0:
            target_legal_balls = current_event.legal_balls_bowled - 12
            past_event = events[0]
            for i in range(current_index, -1, -1):
                if events[i].legal_balls_bowled <= target_legal_balls:
                    past_event = events[i]
                    break
            p_a_diff = current_event.team_a_win_probability - past_event.team_a_win_probability
            trend_prob = round(abs(p_a_diff) * 100, 1)
            trend_team = team_a if p_a_diff >= 0 else team_b

        # 3. Match Volatility (sum of absolute diffs so far, normalized to 0-100)
        # If insufficient history (< 6 transitions / 1 over): "Awaiting sufficient history"
        volatility_score: Optional[float] = None
        if current_index >= 6:
            volatility_sum = sum(
                abs(events[i].team_a_win_probability - events[i - 1].team_a_win_probability)
                for i in range(1, current_index + 1)
            )
            volatility_score = round(min(100.0, (volatility_sum / 2.5) * 100), 1)

        # 4. Criticality of Next Ball
        criticality_score: Optional[float] = None
        if model_runner and current_event.feature_snapshot and current_event.innings in [1, 2]:
            try:
                base_row = pd.DataFrame([current_event.feature_snapshot])
                probs = []
                for outcome in ["dot", "1", "2", "4", "6", "wicket"]:
                    row = base_row.copy()
                    row["legal_balls_bowled"] += 1
                    row["legal_balls_remaining"] -= 1

                    if outcome == "wicket":
                        row["current_innings_wickets"] += 1
                        row["wickets_remaining"] -= 1
                        row["wickets_last_6_balls"] += 1
                        row["wickets_last_12_balls"] += 1
                    elif outcome != "dot":
                        runs = int(outcome)
                        row["current_innings_runs"] += runs
                        if current_event.innings == 2:
                            row["runs_needed"] -= runs
                        row["runs_last_6_balls"] += runs
                        row["runs_last_12_balls"] += runs

                    p_a, _, _ = model_runner.predict_probability(
                        df_row=row,
                        innings=current_event.innings,
                        is_terminal=False,
                        team_a=team_a,
                    )
                    probs.append(p_a)

                swing = (max(probs) - min(probs)) * 100
                criticality_score = round(min(100.0, (swing / 15.0) * 100), 1)
            except Exception:
                criticality_score = 0.0

        # 5. Biggest Turning Point (Delivery with largest absolute WPA)
        # Available once there is delivery history (current_index >= 1)
        max_wpa = 0.0
        turning_point_over: Optional[str] = None
        turning_point_event: Optional[str] = None
        turning_point_team: Optional[str] = None
        turning_point_wpa: Optional[float] = None

        player_wpa: dict[str, float] = {}

        if current_index >= 1:
            for i in range(1, current_index + 1):
                prev_e = events[i - 1]
                curr_e = events[i]

                diff_a = curr_e.team_a_win_probability - prev_e.team_a_win_probability

                # Check turning point
                if abs(diff_a) > abs(max_wpa):
                    max_wpa = diff_a
                    turning_point_over = f"{curr_e.ball}"
                    if diff_a > 0:
                        turning_point_team = team_a
                        turning_point_wpa = round(diff_a * 100, 1)
                    else:
                        turning_point_team = team_b
                        turning_point_wpa = round(abs(diff_a) * 100, 1)

                    runs = curr_e.score - prev_e.score
                    wicks = curr_e.wickets - prev_e.wickets
                    if wicks > 0:
                        turning_point_event = "WICKET"
                    elif runs == 6:
                        turning_point_event = "SIX"
                    elif runs == 4:
                        turning_point_event = "FOUR"
                    elif runs == 0:
                        turning_point_event = "DOT"
                    elif runs == 1:
                        turning_point_event = "1 RUN"
                    else:
                        turning_point_event = f"{runs} RUNS"

                # Accumulate delivery-level WPA by player
                batting_team = curr_e.batting_team
                wpa_batting = diff_a if batting_team == team_a else -diff_a

                if curr_e.striker:
                    player_wpa[curr_e.striker] = player_wpa.get(curr_e.striker, 0.0) + wpa_batting
                if curr_e.bowler:
                    player_wpa[curr_e.bowler] = player_wpa.get(curr_e.bowler, 0.0) - wpa_batting

        # 6. Most Influential Player
        influential_player: Optional[str] = None
        influential_player_wpa: Optional[float] = None
        if player_wpa:
            best_p = max(player_wpa.keys(), key=lambda k: player_wpa[k])
            influential_player = best_p
            influential_player_wpa = round(player_wpa[best_p] * 100, 1)

        # 7. Similar Historical States
        hist_sim_str: Optional[str] = None
        if current_event.event_number == 0:
            hist_sim_str = "Awaiting sufficient history"
        elif self._live_dataset is not None and current_event.innings in [1, 2]:
            try:
                df = self._live_dataset
                wicks_rem = max(0, min(10, 10 - current_event.wickets))
                balls_rem = max(0, min(120, 120 - current_event.legal_balls_bowled))

                if current_event.innings == 2 and current_event.runs_needed is not None:
                    runs_need = current_event.runs_needed
                    mask = (
                        (df["current_innings"] == 2)
                        & (df["wickets_remaining"].between(max(0, wicks_rem - 1), min(10, wicks_rem + 1)))
                        & (df["legal_balls_remaining"].between(max(0, balls_rem - 6), min(120, balls_rem + 6)))
                        & (df["runs_needed"].between(max(1, runs_need - 10), runs_need + 10))
                    )
                    matches = df[mask]
                    if len(matches) < 5:
                        mask_wide = (
                            (df["current_innings"] == 2)
                            & (df["wickets_remaining"].between(max(0, wicks_rem - 2), min(10, wicks_rem + 2)))
                            & (df["legal_balls_remaining"].between(max(0, balls_rem - 12), min(120, balls_rem + 12)))
                            & (df["runs_needed"].between(max(1, runs_need - 15), runs_need + 15))
                        )
                        matches = df[mask_wide]

                    if len(matches) >= 5:
                        chase_wins = (matches["target_team_a_win"] == matches["is_team_a_batting"]).mean()
                        hist_sim_str = f"{int(round(chase_wins * 100))}% resulted in successful chases"
                    else:
                        hist_sim_str = "No sufficiently similar historical states"
                else:
                    # Innings 1: Defending
                    score = current_event.score
                    mask = (
                        (df["current_innings"] == 1)
                        & (df["wickets_remaining"].between(max(0, wicks_rem - 1), min(10, wicks_rem + 1)))
                        & (df["legal_balls_remaining"].between(max(0, balls_rem - 6), min(120, balls_rem + 6)))
                        & (df["current_innings_runs"].between(max(0, score - 10), score + 10))
                    )
                    matches = df[mask]
                    if len(matches) < 5:
                        mask_wide = (
                            (df["current_innings"] == 1)
                            & (df["wickets_remaining"].between(max(0, wicks_rem - 2), min(10, wicks_rem + 2)))
                            & (df["legal_balls_remaining"].between(max(0, balls_rem - 12), min(120, balls_rem + 12)))
                            & (df["current_innings_runs"].between(max(0, score - 15), score + 15))
                        )
                        matches = df[mask_wide]

                    if len(matches) >= 5:
                        defend_wins = (matches["target_team_a_win"] == matches["is_team_a_batting"]).mean()
                        hist_sim_str = f"{int(round(defend_wins * 100))}% of teams defended this state"
                    else:
                        hist_sim_str = "No sufficiently similar historical states"
            except Exception as e:
                hist_sim_str = "No sufficiently similar historical states"
        else:
            hist_sim_str = "Awaiting sufficient history"

        return MatchIntelligenceSnapshot(
            current_team=current_team,
            current_win_probability=current_prob,
            trend_probability=trend_prob,
            trend_team=trend_team,
            volatility=volatility_score,
            criticality=criticality_score,
            turning_point_over=turning_point_over,
            turning_point_event=turning_point_event,
            turning_point_wpa=turning_point_wpa,
            turning_point_team=turning_point_team,
            influential_player=influential_player,
            influential_player_wpa=influential_player_wpa,
            historical_similarity=hist_sim_str,
        )


match_intelligence_service = MatchIntelligenceService()
