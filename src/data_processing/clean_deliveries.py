"""
Delivery-Level Preprocessing Module.

Generates data/processed/deliveries_clean.csv from raw Cricsheet ball-by-ball records.
"""

import os
import csv
from typing import Dict, Any, Optional

from .team_mapping import map_team_name
from .venue_mapping import map_venue_name


def parse_numeric_extra(val: Optional[str]) -> int:
    """
    Parse extra field into integer (0 if empty/missing).
    """
    if val is None or val == "":
        return 0
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def clean_delivery_row(row: Dict[str, str]) -> Dict[str, Any]:
    """
    Clean, map, and standardize a single ball-by-ball delivery row.
    """
    innings = int(row.get("innings", 1))
    is_super_over = 1 if innings >= 3 else 0

    return {
        "match_id": row.get("match_id", ""),
        "season": row.get("season", ""),
        "start_date": row.get("start_date", ""),
        "venue": map_venue_name(row.get("venue", "")),
        "innings": innings,
        "is_super_over": is_super_over,
        "ball": row.get("ball", ""),
        "actual_delivery": row.get("actual_delivery", ""),
        "batting_team": map_team_name(row.get("batting_team", "")),
        "bowling_team": map_team_name(row.get("bowling_team", "")),
        "striker": row.get("striker", ""),
        "non_striker": row.get("non_striker", ""),
        "bowler": row.get("bowler", ""),
        "runs_off_bat": int(row.get("runs_off_bat", 0)),
        "extras": int(row.get("extras", 0)),
        "wides": parse_numeric_extra(row.get("wides")),
        "noballs": parse_numeric_extra(row.get("noballs")),
        "byes": parse_numeric_extra(row.get("byes")),
        "legbyes": parse_numeric_extra(row.get("legbyes")),
        "penalty": parse_numeric_extra(row.get("penalty")),
        "non_boundary": parse_numeric_extra(row.get("non_boundary")),
        "wicket_type": row.get("wicket_type", ""),
        "player_dismissed": row.get("player_dismissed", ""),
    }


def clean_all_deliveries(
    raw_csv_path: str = "data/raw/cricsheet/all_matches.csv",
    output_path: str = "data/processed/deliveries_clean.csv"
) -> int:
    """
    Streamingly process raw delivery CSV into cleaned deliveries CSV.

    Args:
        raw_csv_path: Path to all_matches.csv.
        output_path: Path to output clean deliveries CSV.

    Returns:
        Total number of processed delivery rows.
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw deliveries file not found: {raw_csv_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "match_id",
        "season",
        "start_date",
        "venue",
        "innings",
        "is_super_over",
        "ball",
        "actual_delivery",
        "batting_team",
        "bowling_team",
        "striker",
        "non_striker",
        "bowler",
        "runs_off_bat",
        "extras",
        "wides",
        "noballs",
        "byes",
        "legbyes",
        "penalty",
        "non_boundary",
        "wicket_type",
        "player_dismissed",
    ]

    total_rows = 0
    with open(raw_csv_path, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        with open(output_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

            for raw_row in reader:
                clean_row = clean_delivery_row(raw_row)
                writer.writerow(clean_row)
                total_rows += 1

    return total_rows
