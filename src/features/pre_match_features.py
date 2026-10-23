"""
Pre-Match Feature Engineering Pipeline.

Reads matches_clean.csv and deliveries_clean.csv, processes matches in strict
chronological order, computes leak-free pre-match features, and saves
data/processed/pre_match_dataset.csv.
"""

import os
import csv
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from .historical_stats import HistoricalStateTracker


BOWLER_WICKET_TYPES = {
    "caught",
    "bowled",
    "lbw",
    "caught and bowled",
    "stumped",
    "hit wicket",
}


def load_match_delivery_summaries(
    deliveries_path: str = "data/processed/deliveries_clean.csv"
) -> Dict[str, Dict[str, Any]]:
    """
    Pre-aggregate delivery logs into match-level innings summaries.
    Excludes Super Over deliveries (innings 3-6).
    """
    # match_id -> {
    #   'innings': [ {'innings': 1, 'batting_team': ..., 'bowling_team': ..., 'runs': ..., 'legal_balls': ..., 'wickets': ...}, ... ],
    #   'inn2_batting_team': ...
    # }
    match_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"innings": defaultdict(lambda: {
        "batting_team": "",
        "bowling_team": "",
        "runs": 0,
        "legal_balls": 0,
        "wickets": 0,
    }), "chasing_team": ""})

    with open(deliveries_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["is_super_over"] == "1":
                continue  # exclude Super Overs from 20-over strength metrics

            inn = int(r["innings"])
            if inn not in (1, 2):
                continue

            mid = r["match_id"]
            inn_dict = match_data[mid]["innings"][inn]

            bat_team = r["batting_team"]
            bowl_team = r["bowling_team"]
            inn_dict["batting_team"] = bat_team
            inn_dict["bowling_team"] = bowl_team

            runs = int(r["runs_off_bat"]) + int(r["extras"])
            inn_dict["runs"] += runs

            wides = int(r["wides"])
            noballs = int(r["noballs"])
            if wides == 0 and noballs == 0:
                inn_dict["legal_balls"] += 1

            w_type = r["wicket_type"].strip()
            if w_type in BOWLER_WICKET_TYPES:
                inn_dict["wickets"] += 1

            if inn == 2:
                match_data[mid]["chasing_team"] = bat_team

    # Flatten innings dict
    summaries = {}
    for mid, data in match_data.items():
        inn_list = [data["innings"][inn] for inn in sorted(data["innings"].keys()) if data["innings"][inn]["batting_team"]]
        summaries[mid] = {
            "innings": inn_list,
            "chasing_team": data["chasing_team"],
        }

    return summaries


def generate_pre_match_dataset(
    matches_path: str = "data/processed/matches_clean.csv",
    deliveries_path: str = "data/processed/deliveries_clean.csv",
    output_path: str = "data/processed/pre_match_dataset.csv",
) -> List[Dict[str, Any]]:
    """
    Generate chronological pre-match feature dataset.
    """
    print(f"Loading delivery summaries from {deliveries_path}...")
    delivery_summaries = load_match_delivery_summaries(deliveries_path)

    print(f"Loading cleaned matches from {matches_path}...")
    with open(matches_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_matches = list(reader)

    # Filter out the 9 genuine abandoned/no-result matches from supervised target
    eligible_matches = [m for m in all_matches if m["match_result_type"] != "no_result"]
    print(f"Total matches: {len(all_matches)} | Eligible decisive matches: {len(eligible_matches)}")

    # Sort strictly chronologically by (date, match_id)
    # This completely eliminates any temporal disorder across dates
    eligible_matches.sort(key=lambda m: (m["date"], int(m["match_id"])))

    tracker = HistoricalStateTracker()
    feature_rows = []

    fieldnames = [
        "match_id",
        "season",
        "date",
        "team_a",
        "team_b",
        "venue",
        "city",
        "toss_winner_is_team_a",
        "toss_decision_is_bat",
        "team_a_bats_first",
        "team_a_form_last_5",
        "team_b_form_last_5",
        "team_a_form_last_10",
        "team_b_form_last_10",
        "form_diff_last_5",
        "team_a_matches_played",
        "team_b_matches_played",
        "h2h_matches_played",
        "team_a_h2h_win_rate",
        "team_a_venue_matches",
        "team_a_venue_win_rate",
        "team_b_venue_matches",
        "team_b_venue_win_rate",
        "venue_win_rate_diff",
        "venue_chasing_win_rate",
        "team_a_batting_rpo_last_5",
        "team_b_batting_rpo_last_5",
        "batting_rpo_diff_last_5",
        "team_a_bowling_econ_last_5",
        "team_b_bowling_econ_last_5",
        "bowling_econ_diff_last_5",
        "target_team_a_win",
    ]

    print("Computing pre-match features chronologically...")
    for m in eligible_matches:
        mid = m["match_id"]
        team_a = m["team_1"]
        team_b = m["team_2"]
        venue = m["venue"]
        city = m["city"]
        toss_winner = m["toss_winner"]
        toss_decision = m["toss_decision"]
        winner = m["winner"]

        # Step 1: Compute features strictly using history prior to this match
        features = tracker.compute_pre_match_features(
            team_a=team_a,
            team_b=team_b,
            venue=venue,
            city=city,
            toss_winner=toss_winner,
            toss_decision=toss_decision,
        )

        target = 1 if winner == team_a else 0

        row = {
            "match_id": mid,
            "season": m["season"],
            "date": m["date"],
            **features,
            "target_team_a_win": target,
        }
        feature_rows.append(row)

        # Step 2: Update historical state strictly AFTER features have been computed
        deliv_summary = delivery_summaries.get(mid, {"innings": [], "chasing_team": ""})
        chasing_team = deliv_summary.get("chasing_team", "")
        chasing_team_won = (winner == chasing_team) if chasing_team else False

        tracker.update_with_match_result(
            team_a=team_a,
            team_b=team_b,
            winner=winner,
            venue=venue,
            innings_stats=deliv_summary.get("innings", []),
            chasing_team_won=chasing_team_won,
        )

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)

    print(f"Successfully generated {len(feature_rows)} pre-match feature rows to {output_path}.")
    return feature_rows


if __name__ == "__main__":
    generate_pre_match_dataset()
