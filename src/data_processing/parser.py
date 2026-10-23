"""
Raw Cricsheet Match Info and Delivery CSV Parsers.
"""

import csv
import os
from typing import Dict, Any, Optional


def parse_match_info_file(filepath: str, match_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse a Cricsheet {match_id}_info.csv file into structured key-value attributes.

    Args:
        filepath: Absolute or relative path to the *_info.csv file.
        match_id: Optional match identifier. If not provided, inferred from filename.

    Returns:
        Dictionary containing extracted raw match attributes.
    """
    if match_id is None:
        filename = os.path.basename(filepath)
        match_id = filename.replace("_info.csv", "")

    teams = []
    dates = []
    data = {
        "match_id": match_id,
        "season": "",
        "date": "",
        "venue": "",
        "city": "",
        "team_1": "",
        "team_2": "",
        "toss_winner": "",
        "toss_decision": "",
        "winner": "",
        "outcome": "",
        "eliminator": "",
        "method": "",
    }

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2 or row[0] != "info":
                continue

            field = row[1].strip()
            value = row[2].strip() if len(row) > 2 else ""

            if field == "team":
                teams.append(value)
            elif field == "date":
                dates.append(value)
            elif field == "season":
                data["season"] = value
            elif field == "venue":
                data["venue"] = value
            elif field == "city":
                data["city"] = value
            elif field == "toss_winner":
                data["toss_winner"] = value
            elif field == "toss_decision":
                data["toss_decision"] = value
            elif field == "winner":
                data["winner"] = value
            elif field == "outcome":
                data["outcome"] = value
            elif field == "eliminator":
                data["eliminator"] = value
            elif field == "method":
                data["method"] = value

    if len(teams) >= 1:
        data["team_1"] = teams[0]
    if len(teams) >= 2:
        data["team_2"] = teams[1]

    # Use first date if multiple (reserve days)
    if dates:
        data["date"] = dates[0]

    return data
