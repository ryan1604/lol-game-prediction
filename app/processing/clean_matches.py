from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROLE_ORDER = ("top", "jungle", "mid", "bot", "support")
SIDES = ("blue", "red")
INT_REGION_ALIASES = {"2025", "2026", "FIRST", "MSI", "WORLD", "WORLDS"}
EXTRA_TEAM_ALIASES = {
    "Movistar R7": "R7",
    "Team Whales": "TSW",
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_REFERENCES_DIR = PROJECT_ROOT / "data" / "references"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "matches_clean.parquet"
DEFAULT_INVALID_PATH = PROJECT_ROOT / "data" / "processed" / "matches_invalid.csv"

REQUIRED_COLUMNS = (
    "match_id",
    "game_url",
    "match_name",
    "tournament",
    "region",
    "year",
    "split",
    "stage",
    "match_date",
    "blue_team_name",
    "red_team_name",
    "winner_side",
    "first_pick_side",
    *(f"{side}_{role}_champion" for side in SIDES for role in ROLE_ORDER),
    *(f"{side}_{role}_player" for side in SIDES for role in ROLE_ORDER),
    "players",
    "blue_bans",
    "red_bans",
)


@dataclass(frozen=True)
class CleanMatchResult:
    matches: pd.DataFrame
    invalid_rows: pd.DataFrame


def _normalize_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).upper()


def _blank(value: Any) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def _clean_region(value: Any) -> str:
    region = str(value).strip().upper()
    return "INT" if region in INT_REGION_ALIASES else region


def _load_team_aliases(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        aliases = {
            _normalize_key(row["alias_name"]): row["canonical_team_code"].strip()
            for row in csv.DictReader(handle)
        }
    aliases.update({_normalize_key(alias): code for alias, code in EXTRA_TEAM_ALIASES.items()})
    return aliases


def _load_tournament_splits(path: Path) -> dict[str, list[tuple[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: dict[str, list[tuple[str, str]]] = {}
        for row in csv.DictReader(handle):
            rows.setdefault(_normalize_key(row["region"]), []).append(
                (_normalize_key(row["raw_tournament_name"]), row["canonical_split"].strip())
            )
        return rows


def _canonical_team(value: Any, aliases: dict[str, str]) -> str:
    text = str(value).strip()
    return aliases.get(_normalize_key(text), text)


def _canonical_split(value: Any, region: str, splits: dict[str, list[tuple[str, str]]]) -> str:
    text = str(value).strip()
    normalized = _normalize_key(text)
    without_year = re.sub(r"\b20\d{2}\b", "", normalized)
    without_year = re.sub(r"\s+", " ", without_year).strip()
    for raw_name, canonical in splits.get(_normalize_key(region), []):
        if raw_name in normalized or raw_name in without_year:
            return canonical
    return text


def _parse_list(value: Any, field: str) -> list[Any]:
    if _blank(value):
        raise ValueError(f"{field} is blank")
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"{field} is not a list")
    return parsed


def _clean_row(
    row: pd.Series,
    *,
    aliases: dict[str, str],
    splits: dict[str, list[tuple[str, str]]],
) -> dict[str, Any]:
    missing = [column for column in REQUIRED_COLUMNS if column not in row.index or _blank(row[column])]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")

    winner_side = str(row["winner_side"]).strip().lower()
    first_pick_side = str(row["first_pick_side"]).strip().lower()
    if winner_side not in SIDES:
        raise ValueError("winner_side must be blue or red")
    if first_pick_side not in SIDES:
        raise ValueError("first_pick_side must be blue or red")

    match_date = pd.to_datetime(row["match_date"], format="%Y-%m-%d", errors="raise").date()
    region = _clean_region(row["region"])
    players = _parse_list(row["players"], "players")
    blue_bans = _parse_list(row["blue_bans"], "blue_bans")
    red_bans = _parse_list(row["red_bans"], "red_bans")
    if len(players) != 10:
        raise ValueError("players must contain 10 entries")

    cleaned = {column: row[column] for column in REQUIRED_COLUMNS if column not in {"players", "blue_bans", "red_bans"}}
    cleaned["match_id"] = int(row["match_id"])
    cleaned["year"] = int(row["year"])
    cleaned["region"] = region
    cleaned["match_date"] = match_date.isoformat()
    cleaned["winner_side"] = winner_side
    cleaned["first_pick_side"] = first_pick_side
    cleaned["blue_team_name"] = _canonical_team(row["blue_team_name"], aliases)
    cleaned["red_team_name"] = _canonical_team(row["red_team_name"], aliases)
    cleaned["split"] = _canonical_split(row["tournament"], region, splits)
    cleaned["blue_win"] = winner_side == "blue"
    cleaned["players"] = [
        {
            **player,
            "side": str(player.get("side", "")).strip().lower(),
            "role": str(player.get("role", "")).strip().lower(),
            "team_name": _canonical_team(player.get("team_name", ""), aliases),
        }
        for player in players
    ]
    cleaned["blue_bans"] = [str(value).strip() for value in blue_bans]
    cleaned["red_bans"] = [str(value).strip() for value in red_bans]
    return cleaned


def discover_raw_csvs(raw_dir: Path = DEFAULT_RAW_DIR) -> list[Path]:
    return sorted(raw_dir.glob("*_raw_data.csv"))


def build_clean_matches(
    input_paths: Iterable[Path],
    *,
    references_dir: Path = DEFAULT_REFERENCES_DIR,
) -> CleanMatchResult:
    paths = list(input_paths)
    if not paths:
        raise ValueError("No raw match CSV files provided")

    aliases = _load_team_aliases(references_dir / "team_aliases.csv")
    splits = _load_tournament_splits(references_dir / "tournament_split_mapping.csv")
    raw = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"Raw CSV missing columns: {', '.join(missing_columns)}")

    cleaned_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    seen_match_ids: set[int] = set()
    for index, row in raw.iterrows():
        try:
            cleaned = _clean_row(row, aliases=aliases, splits=splits)
            if cleaned["match_id"] in seen_match_ids:
                raise ValueError("duplicate match_id")
            seen_match_ids.add(cleaned["match_id"])
            cleaned_rows.append(cleaned)
        except Exception as exc:
            invalid = row.to_dict()
            invalid["row_number"] = index + 2
            invalid["invalid_reason"] = str(exc)
            invalid_rows.append(invalid)

    matches = pd.DataFrame(cleaned_rows)
    if not matches.empty:
        matches = matches.sort_values("match_id").reset_index(drop=True)
    invalid = pd.DataFrame(invalid_rows)
    return CleanMatchResult(matches=matches, invalid_rows=invalid)


def write_clean_matches(result: CleanMatchResult, output_path: Path, invalid_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    result.matches.to_parquet(output_path, index=False)
    result.invalid_rows.to_csv(invalid_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, dest="inputs")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--references-dir", type=Path, default=DEFAULT_REFERENCES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--invalid-output", type=Path, default=DEFAULT_INVALID_PATH)
    args = parser.parse_args()

    inputs = args.inputs if args.inputs else discover_raw_csvs(args.raw_dir)
    result = build_clean_matches(inputs, references_dir=args.references_dir)
    write_clean_matches(result, args.output, args.invalid_output)
    print(f"Wrote {len(result.matches)} clean rows to {args.output}")
    print(f"Wrote {len(result.invalid_rows)} invalid rows to {args.invalid_output}")


if __name__ == "__main__":
    main()
