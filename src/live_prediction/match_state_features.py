"""
Match State Feature Extractor.

Extracts macro match-state, dynamic scoring rates, chase metrics, rolling momentum,
and active partnership dynamics at the prediction point.
"""

from typing import Dict, Any


def extract_match_state_features(
    raw_state: Dict[str, Any],
    team_a: str,
    pre_match_prob_team_a: float,
) -> Dict[str, Any]:
    """
    Transforms instantaneous post-delivery state into standard macro match-state features.
    
    Innings 1 strictly zeroes chase-specific features (target_runs, runs_needed, required_run_rate,
    rrr_crr_differential) to prevent lookahead leakage.
    """
    inn = raw_state["innings"]
    bat_team = raw_state["batting_team"]
    is_team_a_batting = 1 if bat_team == team_a else 0

    runs = raw_state["current_innings_runs"]
    wickets = raw_state["current_innings_wickets"]
    wickets_remaining = max(0, 10 - wickets)

    legal_balls = raw_state["legal_balls_bowled"]
    legal_balls_remaining = max(0, 120 - legal_balls)
    overs_completed = round(legal_balls / 6.0, 4)

    # Current Run Rate (CRR): guarded against zero overs
    effective_overs = max(0.1667, legal_balls / 6.0)
    current_run_rate = round(runs / effective_overs, 4)

    # Chase Metrics (Innings 2 only)
    if inn == 1:
        target_runs = 0
        runs_needed = 0
        required_run_rate = 0.0
        rrr_crr_differential = 0.0
    else:
        target_runs = raw_state["target_runs"]
        runs_needed = max(0, target_runs - runs)

        if runs_needed == 0:
            required_run_rate = 0.0
        elif legal_balls_remaining > 0:
            overs_rem = legal_balls_remaining / 6.0
            required_run_rate = round(runs_needed / overs_rem, 4)
        else:
            required_run_rate = 99.0  # Capped upper bound when balls exhausted and runs still needed

        rrr_crr_differential = round(current_run_rate - required_run_rate, 4)

    # Partnership Run Rate
    part_runs = raw_state["partnership_runs"]
    part_balls = raw_state["partnership_balls"]
    part_overs = max(0.1667, part_balls / 6.0)
    current_partnership_run_rate = round(part_runs / part_overs, 4)

    return {
        "current_innings": inn,
        "is_team_a_batting": is_team_a_batting,
        "current_innings_runs": runs,
        "current_innings_wickets": wickets,
        "wickets_remaining": wickets_remaining,
        "legal_balls_bowled": legal_balls,
        "legal_balls_remaining": legal_balls_remaining,
        "overs_completed": overs_completed,
        "current_run_rate": current_run_rate,
        "target_runs": target_runs,
        "runs_needed": runs_needed,
        "required_run_rate": required_run_rate,
        "rrr_crr_differential": rrr_crr_differential,
        # Rolling momentum
        "runs_last_6_balls": raw_state["runs_last_6"],
        "wickets_last_6_balls": raw_state["wickets_last_6"],
        "runs_last_12_balls": raw_state["runs_last_12"],
        "wickets_last_12_balls": raw_state["wickets_last_12"],
        "dot_ball_ratio_last_12": raw_state["dot_ball_ratio_last_12"],
        "boundary_ratio_last_12": raw_state["boundary_ratio_last_12"],
        # Partnership dynamics
        "current_partnership_runs": part_runs,
        "current_partnership_balls": part_balls,
        "current_partnership_run_rate": current_partnership_run_rate,
        # Pre-match win prob anchor
        "pre_match_prob_team_a": round(pre_match_prob_team_a, 4),
    }
