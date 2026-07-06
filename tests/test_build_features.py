from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.features.build_features import build_features_file, build_match_features


def _match(
    *,
    match_id: int,
    match_date: str,
    blue_team: str = "AAA",
    red_team: str = "BBB",
    blue_win: bool = True,
    year: int = 2025,
    split: str = "Spring",
    blue_top_player: str = "Alice",
    blue_top_champion: str = "Aatrox",
) -> dict[str, object]:
    row: dict[str, object] = {
        "match_id": match_id,
        "match_date": match_date,
        "region": "LCK",
        "year": year,
        "split": split,
        "stage": "WEEK1",
        "tournament": "LCK Spring",
        "blue_win": blue_win,
        "first_pick_side": "blue",
        "blue_team_name": blue_team,
        "red_team_name": red_team,
    }
    for side in ("blue", "red"):
        for role in ("top", "jungle", "mid", "bot", "support"):
            row[f"{side}_{role}_player"] = f"{side}_{role}_{match_id}"
            row[f"{side}_{role}_champion"] = f"{role}_{match_id}"
    row["blue_top_player"] = blue_top_player
    row["blue_top_champion"] = blue_top_champion
    return row


def test_build_match_features_uses_prior_rows_only() -> None:
    matches = pd.DataFrame(
        [
            _match(match_id=2, match_date="2025-01-01", red_team="CCC", blue_win=False),
            _match(match_id=1, match_date="2025-01-01", blue_win=True),
            _match(match_id=3, match_date="2025-01-02", blue_win=True),
        ]
    )

    features = build_match_features(matches)

    assert features["match_id"].tolist() == [1, 2, 3]
    assert "tournament" not in features.columns
    first = features.iloc[0]
    assert first["blue_team_split_games_before"] == 0
    assert first["blue_top_champion_role_games_before"] == 0
    assert first["blue_top_player_champion_games_last_2y"] == 0
    assert first["blue_top_player_champion_win_rate_last_2y"] == 0.5

    second = features.iloc[1]
    assert second["blue_team_split_games_before"] == 1
    assert second["blue_team_split_win_rate_before"] == 1.0
    assert second["blue_top_champion_role_games_before"] == 1
    assert second["blue_top_champion_role_win_rate_before"] == 1.0
    assert second["blue_top_player_champion_games_last_2y"] == 1
    assert second["blue_top_player_champion_win_rate_last_2y"] == 1.0


def test_player_champion_comfort_uses_last_two_years() -> None:
    matches = pd.DataFrame(
        [
            _match(match_id=1, match_date="2022-01-01", blue_win=True, year=2022),
            _match(match_id=2, match_date="2025-01-02", blue_win=False),
        ]
    )

    features = build_match_features(matches)

    latest = features.iloc[1]
    assert latest["blue_top_player_champion_games_last_2y"] == 0
    assert latest["blue_top_player_champion_win_rate_last_2y"] == 0.5
    assert latest["blue_top_champion_role_games_before"] == 0


def test_champion_role_history_is_region_year_split_specific() -> None:
    matches = pd.DataFrame(
        [
            _match(match_id=1, match_date="2025-01-01", split="Spring"),
            _match(match_id=2, match_date="2025-01-02", split="Summer"),
            _match(match_id=3, match_date="2025-01-03", split="Spring"),
        ]
    )

    features = build_match_features(matches)

    assert features.loc[1, "blue_top_champion_role_games_before"] == 0
    assert features.loc[2, "blue_top_champion_role_games_before"] == 1


def test_build_features_file_writes_parquet(tmp_path: Path) -> None:
    input_path = tmp_path / "matches_clean.parquet"
    output_path = tmp_path / "match_features.parquet"
    pd.DataFrame([_match(match_id=1, match_date="2025-01-01")]).to_parquet(input_path, index=False)

    features = build_features_file(input_path, output_path)

    written = pd.read_parquet(output_path)
    assert len(features) == 1
    assert written.loc[0, "match_id"] == 1
    assert written.loc[0, "blue_has_first_pick"] == True
    assert written.loc[0, "team_split_win_rate_diff"] == 0.0
