"""
Live Dataset Builder Orchestrator.

Transforms:
- data/processed/matches_clean.csv
- data/processed/deliveries_clean.csv
into:
- data/processed/live_dataset.csv

Processes matches strictly chronologically, enforces all anti-leakage invariants,
and generates complete supervised live-state rows for model training.
"""

import os
import csv
import joblib
import pandas as pd
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from .live_state_builder import LiveMatchStateMachine
from .match_state_features import extract_match_state_features
from .player_features import HistoricalPlayerTracker, extract_current_player_features
from .matchup_features import HistoricalMatchupTracker
from .walk_forward_pre_match import (
    load_walk_forward_prob_map,
    generate_walk_forward_probabilities,
)


# Feature names definition
METADATA_COLUMNS = [
    "match_id",
    "season",
    "date",
    "venue",
    "city",
    "team_a",
    "team_b",
    "toss_winner_is_team_a",
    "toss_decision_is_bat",
    "team_a_bats_first",
    "innings",
    "delivery_index",
    "ball",
    "batting_team",
    "bowling_team",
    "striker",
    "non_striker",
    "bowler",
]

MATCH_STATE_COLUMNS = [
    "current_innings",
    "is_team_a_batting",
    "current_innings_runs",
    "current_innings_wickets",
    "wickets_remaining",
    "legal_balls_bowled",
    "legal_balls_remaining",
    "overs_completed",
    "current_run_rate",
    "target_runs",
    "runs_needed",
    "required_run_rate",
    "rrr_crr_differential",
    "runs_last_6_balls",
    "wickets_last_6_balls",
    "runs_last_12_balls",
    "wickets_last_12_balls",
    "dot_ball_ratio_last_12",
    "boundary_ratio_last_12",
    "current_partnership_runs",
    "current_partnership_balls",
    "current_partnership_run_rate",
    "pre_match_prob_team_a",
]

CURRENT_PLAYER_COLUMNS = [
    "batter_runs_current_inn",
    "batter_balls_current_inn",
    "batter_strike_rate_current",
    "batter_boundaries_current",
    "batter_dots_current",
    "batter_dot_ratio_current",
    "bowler_legal_balls_current",
    "bowler_overs_current",
    "bowler_runs_conceded_current",
    "bowler_wickets_current",
    "bowler_economy_current",
    "bowler_dots_current",
    "bowler_boundaries_conceded_current",
    "bowler_dot_ratio_current",
]

HISTORICAL_PLAYER_COLUMNS = [
    "historical_batter_strike_rate",
    "historical_batter_average",
    "historical_batter_balls_faced",
    "historical_bowler_economy",
    "historical_bowler_strike_rate",
    "historical_bowler_balls_bowled",
]

MATCHUP_COLUMNS = [
    "matchup_balls_prior",
    "matchup_runs_prior",
    "matchup_wickets_prior",
    "matchup_strike_rate_shrunk",
]

TARGET_COLUMN = ["target_team_a_win"]

ALL_COLUMNS = (
    METADATA_COLUMNS
    + MATCH_STATE_COLUMNS
    + CURRENT_PLAYER_COLUMNS
    + HISTORICAL_PLAYER_COLUMNS
    + MATCHUP_COLUMNS
    + TARGET_COLUMN
)


def compute_pre_match_probabilities(
    walk_forward_csv: str = "data/processed/walk_forward_pre_match_probs.csv",
    pre_match_csv: str = "data/processed/pre_match_dataset.csv",
) -> Dict[str, float]:
    """
    Loads leakage-safe chronological walk-forward pre-match probabilities for Team A.
    If walk-forward cache does not exist, generates expanding walk-forward simulation.
    """
    if os.path.exists(walk_forward_csv):
        print(f"Loading leakage-safe walk-forward pre-match probabilities from {walk_forward_csv}...")
        return load_walk_forward_prob_map(walk_forward_csv)
    else:
        print(f"Walk-forward pre-match probabilities not found at {walk_forward_csv}. Generating expanding walk-forward simulation...")
        df_wf = generate_walk_forward_probabilities(pre_match_csv=pre_match_csv, output_audit_csv=walk_forward_csv)
        return dict(zip(df_wf["match_id"].astype(str), df_wf["pre_match_prob_team_a"].astype(float)))


def load_deliveries_by_match(
    deliveries_path: str = "data/processed/deliveries_clean.csv",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load deliveries indexed by match_id.
    Excludes Super Over deliveries (innings 3-6).
    """
    print(f"Loading deliveries from {deliveries_path}...")
    match_deliveries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with open(deliveries_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            inn = int(r["innings"])
            # Strictly 20-over regular innings 1 and 2
            if inn in (1, 2):
                match_deliveries[r["match_id"]].append(r)

    return match_deliveries


def generate_live_dataset(
    matches_path: str = "data/processed/matches_clean.csv",
    deliveries_path: str = "data/processed/deliveries_clean.csv",
    output_path: str = "data/processed/live_dataset.csv",
    pre_match_csv: str = "data/processed/pre_match_dataset.csv",
    walk_forward_csv: str = "data/processed/walk_forward_pre_match_probs.csv",
    exclude_dl_matches: bool = True,
) -> Tuple[int, int]:
    """
    Orchestrate the chronological live dataset generation.
    Returns (total_matches_processed, total_delivery_rows_generated).
    """
    # 1. Precompute pre-match probability anchor
    pre_match_probs = compute_pre_match_probabilities(
        walk_forward_csv=walk_forward_csv,
        pre_match_csv=pre_match_csv,
    )

    # 2. Load deliveries
    deliveries_by_match = load_deliveries_by_match(deliveries_path=deliveries_path)

    # 3. Load matches and apply approved filtering
    print(f"Loading matches from {matches_path}...")
    with open(matches_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_matches = list(reader)

    eligible_matches = []
    excluded_no_result = 0
    excluded_dl = 0

    for m in all_matches:
        if m["match_result_type"] == "no_result":
            excluded_no_result += 1
            continue
        if exclude_dl_matches and m["method"] == "D/L":
            excluded_dl += 1
            continue
        eligible_matches.append(m)

    print(
        f"Matches: Total={len(all_matches)} | Eligible={len(eligible_matches)} | "
        f"Excluded No-Result={excluded_no_result} | Excluded D/L={excluded_dl}"
    )

    # Sort strictly chronologically by (date, match_id)
    eligible_matches.sort(key=lambda m: (m["date"], int(m["match_id"])))

    # 4. Initialize historical state trackers
    player_tracker = HistoricalPlayerTracker()
    matchup_tracker = HistoricalMatchupTracker()

    total_rows = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Generating live dataset to {output_path}...")
    with open(output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=ALL_COLUMNS)
        writer.writeheader()

        for m_idx, m in enumerate(eligible_matches):
            mid = m["match_id"]
            season = m["season"]
            date = m["date"]
            venue = m["venue"]
            city = m["city"]
            team_a = m["team_1"]
            team_b = m["team_2"]
            winner = m["winner"]
            target_team_a_win = 1 if winner == team_a else 0

            toss_winner = m["toss_winner"]
            toss_decision = m["toss_decision"]
            toss_winner_is_team_a = 1 if toss_winner == team_a else 0
            toss_decision_is_bat = 1 if toss_decision == "bat" else 0

            deliveries = deliveries_by_match.get(mid, [])
            if not deliveries:
                continue

            first_batting_team = deliveries[0]["batting_team"]
            team_a_bats_first = 1 if first_batting_team == team_a else 0

            # Pre-match probability anchor
            p_anchor = pre_match_probs.get(mid, 0.50)

            # Initialize state machine for this match
            state_machine = LiveMatchStateMachine(match_id=mid, team_a=team_a, team_b=team_b)

            for d in deliveries:
                raw_state = state_machine.update_with_delivery(d)

                # Extract features
                match_feats = extract_match_state_features(
                    raw_state=raw_state,
                    team_a=team_a,
                    pre_match_prob_team_a=p_anchor,
                )

                curr_player_feats = extract_current_player_features(raw_state=raw_state)

                striker = raw_state["striker"]
                bowler = raw_state["bowler"]

                hist_batter = player_tracker.get_historical_batter_stats(striker)
                hist_bowler = player_tracker.get_historical_bowler_stats(bowler)

                matchup_feats = matchup_tracker.get_matchup_features(
                    striker=striker,
                    bowler=bowler,
                    historical_batter_sr=hist_batter["historical_batter_strike_rate"],
                )

                # Assemble row
                row = {
                    # Traceability & Metadata
                    "match_id": mid,
                    "season": season,
                    "date": date,
                    "venue": venue,
                    "city": city,
                    "team_a": team_a,
                    "team_b": team_b,
                    "toss_winner_is_team_a": toss_winner_is_team_a,
                    "toss_decision_is_bat": toss_decision_is_bat,
                    "team_a_bats_first": team_a_bats_first,
                    "innings": raw_state["innings"],
                    "delivery_index": raw_state["delivery_index"],
                    "ball": raw_state["ball"],
                    "batting_team": raw_state["batting_team"],
                    "bowling_team": raw_state["bowling_team"],
                    "striker": striker,
                    "non_striker": raw_state["non_striker"],
                    "bowler": bowler,
                    # Match State (Model A)
                    **match_feats,
                    # Current Player Performance (Model B)
                    **curr_player_feats,
                    # Historical Player Priors (Model C)
                    **hist_batter,
                    **hist_bowler,
                    # Matchup Features (Model D)
                    **matchup_feats,
                    # Target
                    "target_team_a_win": target_team_a_win,
                }

                writer.writerow(row)
                total_rows += 1

            # Update historical trackers strictly AFTER all deliveries of match have completed
            player_tracker.update_with_match(deliveries)
            matchup_tracker.update_with_match(deliveries)

            if (m_idx + 1) % 200 == 0 or (m_idx + 1) == len(eligible_matches):
                print(f"  Processed {m_idx + 1}/{len(eligible_matches)} matches ({total_rows:,} rows)...")

    print(f"Live dataset generation complete! Total rows: {total_rows:,}")
    return len(eligible_matches), total_rows


if __name__ == "__main__":
    generate_live_dataset()
