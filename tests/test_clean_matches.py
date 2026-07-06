from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.processing.clean_matches import REQUIRED_COLUMNS, build_clean_matches, discover_raw_csvs, write_clean_matches


def _write_references(root: Path) -> Path:
    references = root / "references"
    references.mkdir()
    (references / "team_aliases.csv").write_text(
        "\n".join(
            [
                "region,canonical_team_code,alias_name",
                "LCK,GEN,Gen.G eSports",
                "LCK,HLE,Hanwha Life eSports",
            ]
        ),
        encoding="utf-8",
    )
    (references / "tournament_split_mapping.csv").write_text(
        "\n".join(
            [
                "region,raw_tournament_name,canonical_split",
                "LCK,LCK Rounds 1-2,Spring",
            ]
        ),
        encoding="utf-8",
    )
    return references


def _raw_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "match_id": 1,
        "game_url": "https://gol.gg/game/stats/1/page-game/",
        "match_name": "HLE vs GEN",
        "tournament": "LCK 2025 Rounds 1-2",
        "region": "LCK",
        "year": 2025,
        "split": "LCK 2025 Rounds 1-2",
        "stage": "WEEK1",
        "match_date": "2025-04-02",
        "blue_team_name": "Hanwha Life eSports",
        "red_team_name": "Gen.G eSports",
        "winner_side": "red",
        "first_pick_side": "blue",
        "players": json.dumps(
            [
                {"side": side, "role": role, "player_name": f"{side}_{role}", "champion": role, "team_name": team}
                for side, team in (("blue", "Hanwha Life eSports"), ("red", "Gen.G eSports"))
                for role in ("top", "jungle", "mid", "bot", "support")
            ]
        ),
        "blue_bans": json.dumps(["Varus", "Skarner"]),
        "red_bans": json.dumps(["Kalista", "Yone"]),
    }
    for side in ("blue", "red"):
        for role in ("top", "jungle", "mid", "bot", "support"):
            row[f"{side}_{role}_champion"] = role.title()
            row[f"{side}_{role}_player"] = f"{side}_{role}"
    row.update(overrides)
    return row


def _write_raw_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows, columns=REQUIRED_COLUMNS).to_csv(path, index=False)


def test_build_clean_matches_parses_and_normalizes_rows(tmp_path: Path) -> None:
    references = _write_references(tmp_path)
    raw_path = tmp_path / "raw.csv"
    _write_raw_csv(raw_path, [_raw_row()])

    result = build_clean_matches([raw_path], references_dir=references)

    assert result.invalid_rows.empty
    row = result.matches.iloc[0]
    assert row["blue_team_name"] == "HLE"
    assert row["red_team_name"] == "GEN"
    assert row["split"] == "Spring"
    assert row["blue_win"] == False
    assert row["players"][0]["team_name"] == "HLE"
    assert row["red_bans"] == ["Kalista", "Yone"]


def test_build_clean_matches_normalizes_international_region_aliases(tmp_path: Path) -> None:
    references = _write_references(tmp_path)
    raw_path = tmp_path / "raw.csv"
    _write_raw_csv(raw_path, [_raw_row(region="WORLDS")])

    result = build_clean_matches([raw_path], references_dir=references)

    assert result.matches.iloc[0]["region"] == "INT"


def test_build_clean_matches_uses_extra_team_aliases(tmp_path: Path) -> None:
    references = _write_references(tmp_path)
    raw_path = tmp_path / "raw.csv"
    _write_raw_csv(raw_path, [_raw_row(blue_team_name="Movistar R7", red_team_name="Team Whales")])

    result = build_clean_matches([raw_path], references_dir=references)

    row = result.matches.iloc[0]
    assert row["blue_team_name"] == "R7"
    assert row["red_team_name"] == "TSW"


def test_discover_raw_csvs_includes_test_raw_files(tmp_path: Path) -> None:
    (tmp_path / "lcs_raw_data.csv").write_text("", encoding="utf-8")
    (tmp_path / "lcs_test_raw_data.csv").write_text("", encoding="utf-8")

    paths = discover_raw_csvs(tmp_path)

    assert [path.name for path in paths] == ["lcs_raw_data.csv", "lcs_test_raw_data.csv"]


def test_build_clean_matches_quarantines_invalid_rows(tmp_path: Path) -> None:
    references = _write_references(tmp_path)
    raw_path = tmp_path / "raw.csv"
    _write_raw_csv(raw_path, [_raw_row(winner_side="middle")])

    result = build_clean_matches([raw_path], references_dir=references)

    assert result.matches.empty
    assert result.invalid_rows.iloc[0]["invalid_reason"] == "winner_side must be blue or red"


def test_build_clean_matches_rejects_duplicate_match_ids(tmp_path: Path) -> None:
    references = _write_references(tmp_path)
    raw_path = tmp_path / "raw.csv"
    _write_raw_csv(raw_path, [_raw_row(), _raw_row(game_url="https://example.test/duplicate")])

    result = build_clean_matches([raw_path], references_dir=references)

    assert result.matches["match_id"].tolist() == [1]
    assert result.invalid_rows.iloc[0]["invalid_reason"] == "duplicate match_id"


def test_write_clean_matches_writes_parquet_and_invalid_csv(tmp_path: Path) -> None:
    references = _write_references(tmp_path)
    raw_path = tmp_path / "raw.csv"
    _write_raw_csv(raw_path, [_raw_row(), _raw_row(match_id=2, players="[]")])
    result = build_clean_matches([raw_path], references_dir=references)

    output = tmp_path / "matches_clean.parquet"
    invalid = tmp_path / "matches_invalid.csv"
    write_clean_matches(result, output, invalid)

    written = pd.read_parquet(output)
    invalid_rows = pd.read_csv(invalid)
    assert written["match_id"].tolist() == [1]
    assert invalid_rows["invalid_reason"].tolist() == ["players must contain 10 entries"]
