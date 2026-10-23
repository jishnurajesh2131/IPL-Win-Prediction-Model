"""
Match-Level Preprocessing and Outcome Resolution Module.

Generates data/processed/matches_clean.csv from raw Cricsheet *_info.csv files.
"""

import os
import glob
import csv
from typing import List, Dict, Any, Optional

from .parser import parse_match_info_file
from .team_mapping import map_team_name
from .venue_mapping import map_venue_name, impute_city_if_missing


def resolve_match_outcome(raw_info: Dict[str, Any]) -> Dict[str, str]:
    """
    Resolve match outcome, winner, and match result type.

    Rules:
    - Normal win: match_result_type = 'normal', winner = mapped winner.
    - Tied match: match_result_type = 'super_over', winner = mapped eliminator.
    - Abandoned / No Result: match_result_type = 'no_result', winner = ''.

    Args:
        raw_info: Dictionary returned from parse_match_info_file.

    Returns:
        Dictionary with 'winner' and 'match_result_type'.
    """
    outcome = (raw_info.get("outcome") or "").strip().lower()
    raw_winner = (raw_info.get("winner") or "").strip()
    eliminator = (raw_info.get("eliminator") or "").strip()

    if outcome == "no result" or outcome == "no_result":
        return {
            "winner": "",
            "match_result_type": "no_result"
        }
    elif outcome == "tie":
        winner = map_team_name(eliminator) if eliminator else ""
        return {
            "winner": winner,
            "match_result_type": "super_over"
        }
    else:
        winner = map_team_name(raw_winner) if raw_winner else ""
        return {
            "winner": winner,
            "match_result_type": "normal"
        }


def clean_single_match(raw_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and standardize a single match record.
    """
    outcome_info = resolve_match_outcome(raw_info)
    venue = map_venue_name(raw_info.get("venue", ""))
    city = impute_city_if_missing(raw_info.get("venue", ""), raw_info.get("city", ""))

    return {
        "match_id": raw_info.get("match_id", ""),
        "season": raw_info.get("season", ""),
        "date": raw_info.get("date", ""),
        "venue": venue,
        "city": city,
        "team_1": map_team_name(raw_info.get("team_1", "")),
        "team_2": map_team_name(raw_info.get("team_2", "")),
        "toss_winner": map_team_name(raw_info.get("toss_winner", "")),
        "toss_decision": raw_info.get("toss_decision", ""),
        "winner": outcome_info["winner"],
        "match_result_type": outcome_info["match_result_type"],
        "method": raw_info.get("method", ""),
    }


def clean_all_matches(
    raw_dir: str = "data/raw/cricsheet",
    output_path: str = "data/processed/matches_clean.csv"
) -> List[Dict[str, Any]]:
    """
    Read all match-info files from raw_dir, clean them, and write matches_clean.csv.

    Args:
        raw_dir: Path to directory containing *_info.csv files.
        output_path: Path to output clean matches CSV.

    Returns:
        List of cleaned match records.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pattern = os.path.join(raw_dir, "*_info.csv")
    info_files = glob.glob(pattern)

    if not info_files:
        raise FileNotFoundError(f"No match info files found matching {pattern}")

    # Sort numerically by match_id where possible, or alphabetically
    def sort_key(fpath):
        fname = os.path.basename(fpath).replace("_info.csv", "")
        try:
            return (0, int(fname))
        except ValueError:
            return (1, fname)

    info_files.sort(key=sort_key)

    cleaned_records = []
    for fpath in info_files:
        raw_info = parse_match_info_file(fpath)
        cleaned_record = clean_single_match(raw_info)
        cleaned_records.append(cleaned_record)

    fieldnames = [
        "match_id",
        "season",
        "date",
        "venue",
        "city",
        "team_1",
        "team_2",
        "toss_winner",
        "toss_decision",
        "winner",
        "match_result_type",
        "method",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_records)

    return cleaned_records
