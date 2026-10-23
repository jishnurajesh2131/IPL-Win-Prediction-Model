"""
Match metadata discovery and retrieval service.
"""

import csv
import os
from typing import List, Optional, Tuple, Dict, Any

from src.backend.config import MATCHES_CSV
from src.backend.schemas.match import MatchSummary, MatchDetails
from src.data_processing.team_mapping import map_team_name


class MatchService:
    """
    Manages historical match catalog and filtering.
    """

    def __init__(self, matches_csv: str = MATCHES_CSV):
        self.matches_csv = matches_csv
        self._matches: Optional[List[Dict[str, Any]]] = None
        self._match_lookup: Optional[Dict[str, Dict[str, Any]]] = None

    def _ensure_loaded(self):
        if self._matches is None:
            if not os.path.exists(self.matches_csv):
                raise FileNotFoundError(f"Matches CSV file not found at: {self.matches_csv}")

            with open(self.matches_csv, "r", encoding="utf-8") as f:
                records = list(csv.DictReader(f))

            # Sort chronologically (date ascending, match_id ascending)
            records.sort(key=lambda m: (m.get("date", ""), int(m.get("match_id", 0))))
            self._matches = records
            self._match_lookup = {str(m["match_id"]): m for m in records}

    def list_matches(
        self,
        season: Optional[str] = None,
        team: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[MatchSummary], int]:
        """
        Lists matches with optional filtering and pagination.
        """
        self._ensure_loaded()
        assert self._matches is not None

        filtered = self._matches

        if season:
            season_str = str(season).strip()
            filtered = [m for m in filtered if str(m.get("season", "")).strip() == season_str]

        if team:
            raw_query = team.strip().lower()
            canon = map_team_name(team.strip())
            canon_lower = canon.lower() if canon else raw_query
            filtered = [
                m for m in filtered
                if (raw_query in str(m.get("team_1", "")).lower()
                    or raw_query in str(m.get("team_2", "")).lower()
                    or canon_lower in str(m.get("team_1", "")).lower()
                    or canon_lower in str(m.get("team_2", "")).lower()
                    or str(m.get("team_1", "")).lower().startswith(raw_query)
                    or str(m.get("team_2", "")).lower().startswith(raw_query)
                    or str(m.get("team_1", "")).lower() in raw_query
                    or str(m.get("team_2", "")).lower() in raw_query)
            ]

        total = len(filtered)
        paginated = filtered[offset : offset + limit]

        summaries = [
            MatchSummary(
                match_id=str(m["match_id"]),
                season=str(m["season"]),
                date=str(m["date"]),
                team_1=str(m["team_1"]),
                team_2=str(m["team_2"]),
                venue=str(m["venue"]),
                city=str(m["city"]),
                winner=str(m["winner"]) if m.get("winner") else None,
                match_result_type=str(m.get("match_result_type", "normal")),
            )
            for m in paginated
        ]

        return summaries, total

    def get_match(self, match_id: str) -> MatchDetails:
        """
        Retrieves complete metadata for a single match ID.
        """
        self._ensure_loaded()
        assert self._match_lookup is not None

        mid_str = str(match_id).strip()
        if mid_str not in self._match_lookup:
            raise KeyError(f"Match ID '{match_id}' not found.")

        m = self._match_lookup[mid_str]
        res_type = str(m.get("match_result_type", "normal"))
        method = m.get("method") if m.get("method") else None

        return MatchDetails(
            match_id=str(m["match_id"]),
            season=str(m["season"]),
            date=str(m["date"]),
            team_1=str(m["team_1"]),
            team_2=str(m["team_2"]),
            venue=str(m["venue"]),
            city=str(m["city"]),
            winner=str(m["winner"]) if m.get("winner") else None,
            match_result_type=res_type,
            toss_winner=str(m.get("toss_winner", "")),
            toss_decision=str(m.get("toss_decision", "")),
            method=method,
            is_super_over=(res_type == "super_over"),
            is_dl_method=(method == "D/L"),
        )

    def total_count(self) -> int:
        self._ensure_loaded()
        assert self._matches is not None
        return len(self._matches)

    def get_seasons(self) -> List[str]:
        """
        Returns all unique seasons in reverse chronological order.
        """
        self._ensure_loaded()
        assert self._matches is not None
        seasons: List[str] = []
        for m in reversed(self._matches):
            s = str(m.get("season", "")).strip()
            if s and s not in seasons:
                seasons.append(s)
        return seasons

    def get_teams_for_season(self, season: Optional[str] = None) -> List[str]:
        """
        Returns the sorted list of canonical team names that participated in the given season,
        or all canonical teams across all seasons if season is omitted or empty.
        """
        self._ensure_loaded()
        assert self._matches is not None
        teams = set()
        target_season = str(season).strip() if season else None

        for m in self._matches:
            if not target_season or str(m.get("season", "")).strip() == target_season:
                t1 = str(m.get("team_1", "")).strip()
                t2 = str(m.get("team_2", "")).strip()
                if t1:
                    teams.add(t1)
                if t2:
                    teams.add(t2)
        return sorted(list(teams))


# Global singleton instance
match_service = MatchService()
