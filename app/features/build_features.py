from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ROLE_ORDER = ("top", "jungle", "mid", "bot", "support")
SIDES = ("blue", "red")
HISTORY_PRIOR = 0.5
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "matches_clean.parquet"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "match_features.parquet"

BASE_COLUMNS = (
    "match_id",
    "match_date",
    "region",
    "year",
    "split",
    "stage",
    "blue_win",
    "first_pick_side",
    "blue_has_first_pick",
    "is_international",
    "blue_team_name",
    "red_team_name",
    *(f"{side}_{role}_player" for side in SIDES for role in ROLE_ORDER),
    *(f"{side}_{role}_champion" for side in SIDES for role in ROLE_ORDER),
)


def _rate(wins: int, games: int) -> float:
    return wins / games if games else HISTORY_PRIOR


def _row_date(row: pd.Series) -> date:
    return pd.Timestamp(row["match_date"]).date()


def _player_champion_stats(history: list[tuple[date, bool]], current_date: date) -> tuple[int, float]:
    cutoff = (pd.Timestamp(current_date) - pd.DateOffset(years=2)).date()
    recent = [won for played_at, won in history if cutoff <= played_at <= current_date]
    return len(recent), _rate(sum(recent), len(recent))


def build_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(columns=BASE_COLUMNS)

    sorted_matches = matches.copy()
    sorted_matches["match_date"] = pd.to_datetime(sorted_matches["match_date"])
    sorted_matches = sorted_matches.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    team_split: dict[tuple[Any, ...], dict[str, int]] = defaultdict(lambda: {"games": 0, "wins": 0})
    champion_role: dict[tuple[Any, ...], dict[str, int]] = defaultdict(lambda: {"games": 0, "wins": 0})
    player_champion: dict[tuple[str, str], list[tuple[date, bool]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    for _, row in sorted_matches.iterrows():
        current_date = _row_date(row)
        feature_row = {column: row[column] for column in BASE_COLUMNS if column in row.index}
        feature_row["match_date"] = current_date.isoformat()
        feature_row["blue_win"] = bool(row["blue_win"])
        feature_row["blue_has_first_pick"] = str(row["first_pick_side"]).lower() == "blue"
        feature_row["is_international"] = row["region"] == "INT"

        for side in SIDES:
            team = row[f"{side}_team_name"]
            stats = team_split[(row["region"], row["year"], row["split"], team)]
            feature_row[f"{side}_team_split_games_before"] = stats["games"]
            feature_row[f"{side}_team_split_win_rate_before"] = _rate(stats["wins"], stats["games"])

            for role in ROLE_ORDER:
                champion = row[f"{side}_{role}_champion"]
                player = row[f"{side}_{role}_player"]
                role_key = (row["region"], row["year"], row["split"], role, champion)
                role_stats = champion_role[role_key]
                games, win_rate = _player_champion_stats(player_champion[(player, champion)], current_date)
                feature_row[f"{side}_{role}_champion_role_games_before"] = role_stats["games"]
                feature_row[f"{side}_{role}_champion_role_win_rate_before"] = _rate(
                    role_stats["wins"], role_stats["games"]
                )
                feature_row[f"{side}_{role}_player_champion_games_last_2y"] = games
                feature_row[f"{side}_{role}_player_champion_win_rate_last_2y"] = win_rate

        feature_row["team_split_win_rate_diff"] = (
            feature_row["blue_team_split_win_rate_before"] - feature_row["red_team_split_win_rate_before"]
        )
        rows.append(feature_row)

        for side in SIDES:
            won = bool(row["blue_win"]) if side == "blue" else not bool(row["blue_win"])
            team_key = (row["region"], row["year"], row["split"], row[f"{side}_team_name"])
            team_split[team_key]["games"] += 1
            team_split[team_key]["wins"] += int(won)
            for role in ROLE_ORDER:
                champion = row[f"{side}_{role}_champion"]
                player = row[f"{side}_{role}_player"]
                role_key = (row["region"], row["year"], row["split"], role, champion)
                champion_role[role_key]["games"] += 1
                champion_role[role_key]["wins"] += int(won)
                player_champion[(player, champion)].append((current_date, won))

    return pd.DataFrame(rows)


def build_features_file(input_path: Path = DEFAULT_INPUT_PATH, output_path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    features = build_match_features(pd.read_parquet(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    features = build_features_file(args.input, args.output)
    print(f"Wrote {len(features)} feature rows to {args.output}")


if __name__ == "__main__":
    main()
