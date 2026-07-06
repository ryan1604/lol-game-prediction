"""Ingestion helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DEFAULT_HEADERS",
    "DEFAULT_LINKS_CSV_PATH",
    "DEFAULT_RAW_DATA_CSV_PATH",
    "LINKS_CSV_COLUMNS",
    "RAW_DATA_CSV_COLUMNS",
    "TEAM_ALIAS_MAP",
    "TOURNAMENT_SPLIT_MAP",
    "build_team_alias_groups",
    "build_tournament_split_mapping",
    "discover_match_links",
    "parse_gol_game_page",
    "read_match_links_csv",
    "scrape_match_links",
    "scrape_reference_matches",
    "write_team_aliases_csv",
    "write_match_links_csv",
    "write_raw_data_csv",
    "write_tournament_splits_csv",
]

_SCRAPE_EXPORTS = {
    "DEFAULT_HEADERS",
    "DEFAULT_LINKS_CSV_PATH",
    "DEFAULT_RAW_DATA_CSV_PATH",
    "LINKS_CSV_COLUMNS",
    "RAW_DATA_CSV_COLUMNS",
    "TEAM_ALIAS_MAP",
    "TOURNAMENT_SPLIT_MAP",
    "discover_match_links",
    "parse_gol_game_page",
    "read_match_links_csv",
    "scrape_match_links",
    "scrape_reference_matches",
    "write_match_links_csv",
    "write_raw_data_csv",
}

_TEAM_ALIAS_EXPORTS = {
    "build_team_alias_groups",
    "write_team_aliases_csv",
}

_TOURNAMENT_SPLIT_EXPORTS = {
    "build_tournament_split_mapping",
    "write_tournament_splits_csv",
}


def __getattr__(name: str) -> Any:
    if name in _SCRAPE_EXPORTS:
        return getattr(import_module(".scrape", __name__), name)
    if name in _TEAM_ALIAS_EXPORTS:
        return getattr(import_module(".discover_team_aliases", __name__), name)
    if name in _TOURNAMENT_SPLIT_EXPORTS:
        return getattr(import_module(".discover_tournament_splits", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
