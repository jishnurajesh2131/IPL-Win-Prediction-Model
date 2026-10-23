"""
Canonical Team Name Standardization Mapping

Standardizes franchise renamings and cosmetic name variations to modern canonical forms
while strictly preserving distinct historical and defunct franchises (e.g. Deccan Chargers != SRH).
"""

from typing import Optional

# Explicit mapping of all known raw team names to canonical representations
CANONICAL_TEAM_MAP = {
    # Rebranded Active Franchises
    "Delhi Daredevils": "Delhi Capitals",
    "Delhi Capitals": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Punjab Kings": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Royal Challengers Bengaluru": "Royal Challengers Bengaluru",

    # Plural vs Singular Rebrand for Temporary Franchise
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Rising Pune Supergiant": "Rising Pune Supergiant",

    # Unchanged Active Franchises
    "Chennai Super Kings": "Chennai Super Kings",
    "Mumbai Indians": "Mumbai Indians",
    "Kolkata Knight Riders": "Kolkata Knight Riders",
    "Rajasthan Royals": "Rajasthan Royals",
    "Sunrisers Hyderabad": "Sunrisers Hyderabad",
    "Gujarat Titans": "Gujarat Titans",
    "Lucknow Super Giants": "Lucknow Super Giants",

    # Distinct Defunct Franchises (MUST remain separate)
    "Deccan Chargers": "Deccan Chargers",
    "Pune Warriors": "Pune Warriors",
    "Gujarat Lions": "Gujarat Lions",
    "Kochi Tuskers Kerala": "Kochi Tuskers Kerala",

    # Common Franchise Abbreviations and Short Codes
    "RPS": "Rising Pune Supergiant",
    "GL": "Gujarat Lions",
    "CSK": "Chennai Super Kings",
    "RCB": "Royal Challengers Bengaluru",
    "MI": "Mumbai Indians",
    "KKR": "Kolkata Knight Riders",
    "DC": "Delhi Capitals",
    "RR": "Rajasthan Royals",
    "SRH": "Sunrisers Hyderabad",
    "PBKS": "Punjab Kings",
    "KXIP": "Punjab Kings",
    "GT": "Gujarat Titans",
    "LSG": "Lucknow Super Giants",
    "PWI": "Pune Warriors",
    "KTK": "Kochi Tuskers Kerala",
    "DCG": "Deccan Chargers",
}

# The 15 canonical team names
VALID_CANONICAL_TEAMS = {
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
    "Deccan Chargers",
    "Kochi Tuskers Kerala",
    "Pune Warriors",
    "Gujarat Lions",
    "Rising Pune Supergiant",
}

# Case-insensitive lookup dictionary
_LOWER_TEAM_MAP = {k.lower(): v for k, v in CANONICAL_TEAM_MAP.items()}


def map_team_name(name: Optional[str]) -> Optional[str]:
    """
    Map a raw team name or abbreviation to its canonical standardized name.

    Args:
        name: Raw team name string, alias, abbreviation, or None.

    Returns:
        Canonical team name, or None/empty string if input is empty/None.
    """
    if not name:
        return name
    name_clean = name.strip()
    # Try exact match first, then case-insensitive match
    if name_clean in CANONICAL_TEAM_MAP:
        return CANONICAL_TEAM_MAP[name_clean]
    return _LOWER_TEAM_MAP.get(name_clean.lower(), name_clean)
