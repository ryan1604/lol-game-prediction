import csv
import pytest
from bs4 import BeautifulSoup

from app.ingestion.discover_team_aliases import (
    build_team_alias_rows,
    build_team_alias_groups,
    collect_observed_team_names,
    write_team_aliases_csv,
)
from app.ingestion.discover_tournament_splits import (
    build_tournament_split_groups,
    build_tournament_split_rows,
    build_tournament_split_mapping,
    discover_tournament_names,
    write_tournament_splits_csv,
)
from app.ingestion.reference_discovery import discover_tournament_pages

def test_build_team_alias_groups_filters_to_observed_names(monkeypatch) -> None:
    observed_names = {
        "T1",
        "GEN",
        "Gen.G eSports",
        "HLE",
        "KT Rolster",
        "NS",
        "DRX",
        "DWG KIA",
        "Dplus KIA",
        "KDF",
        "DN Freecs",
        "LSB",
        "FearX",
        "BNK FearX",
        "BRO",
        "OK BRION",
        "BRION",
    }
    monkeypatch.setattr(
        "app.ingestion.discover_team_aliases.collect_observed_team_names",
        lambda *args, **kwargs: observed_names,
    )

    grouped = build_team_alias_groups(region="LCK")

    assert grouped["T1"] == ["T1"]
    assert grouped["GEN"] == ["GEN", "Gen.G eSports"]
    assert grouped["DK"] == ["DWG KIA", "Dplus KIA"]
    assert grouped["DNS"] == ["KDF", "DN Freecs"]
    assert grouped["BNK"] == ["LSB", "FearX", "BNK FearX"]
    assert grouped["BRO"] == ["BRO", "OK BRION", "BRION"]

def test_build_tournament_split_mapping_normalizes_seed_tournaments(monkeypatch) -> None:
    discovered_names = [
        "LCK Spring 2024",
        "LCK Spring Playoffs 2024",
        "LCK 2025 Rounds 1-2",
        "LCK 2025 Road to MSI",
        "LCK Summer 2024",
        "LCK Summer Playoffs 2024",
        "LCK 2025 Rounds 3-5",
        "LCK 2025 Season Play-In",
        "LCK 2025 Season Playoffs",
        "LCK Regional Finals 2024",
        "LCK Regionals Finals 2022",
        "LCK Cup 2025",
    ]
    monkeypatch.setattr(
        "app.ingestion.discover_tournament_splits.discover_tournament_names",
        lambda *args, **kwargs: discovered_names,
    )

    mapping = build_tournament_split_mapping(region="LCK")

    assert mapping["LCK Cup"] == "Winter"
    assert mapping["LCK Spring"] == "Spring"
    assert mapping["LCK Spring Playoffs"] == "Spring"
    assert mapping["LCK Rounds 1-2"] == "Spring"
    assert mapping["LCK Road to MSI"] == "Spring"
    assert mapping["LCK Summer"] == "Summer"
    assert mapping["LCK Summer Playoffs"] == "Summer"
    assert mapping["LCK Rounds 3-5"] == "Summer"
    assert mapping["LCK Season Play-In"] == "Summer"
    assert mapping["LCK Season Playoffs"] == "Summer"
    assert mapping["LCK Regional Finals"] == "Summer"
    assert mapping["LCK Regionals Finals"] == "Summer"


def test_build_team_alias_groups_supports_lec(monkeypatch) -> None:
    observed_names = {
        "G2 Esports",
        "Fnatic",
        "MKOI",
        "KOI",
        "MAD Lions",
        "GIANTX",
        "KC",
    }
    monkeypatch.setattr(
        "app.ingestion.discover_team_aliases.collect_observed_team_names",
        lambda *args, **kwargs: observed_names,
    )

    grouped = build_team_alias_groups(region="LEC")

    assert grouped["G2"] == ["G2 Esports"]
    assert grouped["FNC"] == ["Fnatic"]
    assert grouped["MKOI"] == ["MKOI", "KOI", "MAD Lions"]
    assert grouped["GX"] == ["GIANTX"]
    assert grouped["KC"] == ["KC"]


def test_build_tournament_split_mapping_supports_lpl(monkeypatch) -> None:
    discovered_names = [
        "LPL Spring 2024",
        "LPL Spring Playoffs 2024",
        "LPL Summer 2024",
        "LPL Summer Playoffs 2024",
        "LPL 2025 Split 1",
        "LPL 2025 Split 1 Playoffs",
        "LPL 2025 Split 2",
        "LPL 2025 Split 3",
        "LPL 2025 Grand Finals",
        "LPL Regional Finals 2024",
    ]
    monkeypatch.setattr(
        "app.ingestion.discover_tournament_splits.discover_tournament_names",
        lambda *args, **kwargs: discovered_names,
    )

    mapping = build_tournament_split_mapping(region="LPL")

    assert mapping["LPL Spring"] == "Spring"
    assert mapping["LPL Spring Playoffs"] == "Spring"
    assert mapping["LPL Summer"] == "Summer"
    assert mapping["LPL Summer Playoffs"] == "Summer"
    assert mapping["LPL Split 1"] == "Winter"
    assert mapping["LPL Split 1 Playoffs"] == "Winter"
    assert mapping["LPL Split 2"] == "Spring"
    assert mapping["LPL Split 3"] == "Summer"
    assert mapping["LPL Grand Finals"] == "Summer"
    assert mapping["LPL Regional Finals"] == "Summer"


def test_build_tournament_split_groups_supports_lck(monkeypatch) -> None:
    discovered_names = [
        "LCK Spring 2024",
        "LCK Spring Playoffs 2024",
        "LCK Summer 2024",
        "LCK Cup 2025",
    ]
    monkeypatch.setattr(
        "app.ingestion.discover_tournament_splits.discover_tournament_names",
        lambda *args, **kwargs: discovered_names,
    )

    grouped = build_tournament_split_groups(region="LCK")

    assert grouped == {
        "Winter": ["LCK Cup"],
        "Spring": ["LCK Spring", "LCK Spring Playoffs"],
        "Summer": ["LCK Summer"],
    }


def test_build_team_alias_groups_rejects_unsupported_region() -> None:
    with pytest.raises(ValueError, match="Unsupported region"):
        build_team_alias_groups(region="PCS")


def test_collect_observed_team_names_only_uses_vs_text(monkeypatch) -> None:
    html = """
    <table>
      <tr>
        <td>AL vs FPX</td>
        <td>Anyone s Legend</td>
        <td>2-1</td>
        <td>Funplus Phoenix</td>
      </tr>
    </table>
    """
    monkeypatch.setattr(
        "app.ingestion.discover_team_aliases.discover_tournament_pages",
        lambda *args, **kwargs: [("LPL Spring 2024", "https://example.test/tournament")],
    )
    monkeypatch.setattr(
        "app.ingestion.discover_team_aliases.fetch_soup",
        lambda *args, **kwargs: BeautifulSoup(html, "html.parser"),
    )

    observed_names = collect_observed_team_names(region="LPL")

    assert observed_names == {"AL", "FPX"}


def test_collect_observed_team_names_shows_progress(monkeypatch) -> None:
    progress_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "app.ingestion.discover_team_aliases.discover_tournament_pages",
        lambda *args, **kwargs: [("LPL Spring 2024", "https://example.test/tournament")],
    )
    monkeypatch.setattr(
        "app.ingestion.discover_team_aliases.fetch_soup",
        lambda *args, **kwargs: BeautifulSoup("<table></table>", "html.parser"),
    )
    monkeypatch.setattr(
        "app.ingestion.discover_team_aliases.tqdm",
        lambda iterable, **kwargs: progress_calls.append(kwargs) or iterable,
    )

    collect_observed_team_names(region="LPL", show_progress=True)

    assert progress_calls == [
        {
            "disable": False,
            "desc": "LPL tournaments",
            "total": 1,
            "unit": "tournament",
        }
    ]

def test_discover_tournament_names_defaults_to_lck(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.ingestion.discover_tournament_splits.discover_tournament_pages",
        lambda *args, **kwargs: [("LCK Spring 2024", "https://example.test/lck")],
    )

    assert discover_tournament_names() == ["LCK Spring 2024"]


def test_discover_tournament_pages_supports_international_labels(monkeypatch) -> None:
    html = """
    <html>
      <head><title>MSI 2024 - match list</title></head>
      <body>
        <a href="../tournament/tournament-stats/2024%20Mid-Season%20Invitational/">2024 Mid-Season Invitational</a>
        <a href="../tournament/tournament-stats/MSI%202024/">MSI 2024</a>
        <a href="../tournament/tournament-stats/Worlds%20Play-In%202024/">Worlds Play-In 2024</a>
        <a href="../tournament/tournament-stats/LCK%20Spring%202024/">LCK Spring 2024</a>
        <a href="../tournament/tournament-stats/MSI%202023/">MSI 2023</a>
      </body>
    </html>
    """
    monkeypatch.setattr(
        "app.ingestion.reference_discovery.fetch_soup",
        lambda *args, **kwargs: BeautifulSoup(html, "html.parser"),
    )

    tournaments = discover_tournament_pages(
        region="INT",
        seed_urls=["https://example.test/international"],
    )

    assert tournaments == [
        (
            "2024 Mid-Season Invitational",
            "https://gol.gg/tournament/tournament-matchlist/2024%20Mid-Season%20Invitational/",
        ),
        (
            "MSI 2024",
            "https://gol.gg/tournament/tournament-matchlist/MSI%202024/",
        ),
        (
            "Worlds Play-In 2024",
            "https://gol.gg/tournament/tournament-matchlist/Worlds%20Play-In%202024/",
        ),
    ]


def test_discover_tournament_names_shows_progress(monkeypatch) -> None:
    progress_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "app.ingestion.discover_tournament_splits.discover_tournament_pages",
        lambda *args, **kwargs: [("LCK Spring 2024", "https://example.test/lck")],
    )
    monkeypatch.setattr(
        "app.ingestion.discover_tournament_splits.tqdm",
        lambda iterable, **kwargs: progress_calls.append(kwargs) or iterable,
    )

    assert discover_tournament_names(show_progress=True) == ["LCK Spring 2024"]
    assert progress_calls == [
        {
            "disable": False,
            "desc": "LCK tournaments",
            "total": 1,
            "unit": "tournament",
        }
    ]


def test_build_team_alias_rows_uses_configured_values() -> None:
    rows = build_team_alias_rows(region="LCK")

    assert rows[0] == {
        "region": "LCK",
        "canonical_team_code": "T1",
        "alias_name": "T1",
    }
    assert {
        "region": "LCK",
        "canonical_team_code": "DK",
        "alias_name": "Dplus KIA",
    } in rows


def test_write_team_aliases_csv_writes_configured_rows(tmp_path) -> None:
    output_path = tmp_path / "team_aliases.csv"

    written_path = write_team_aliases_csv(output_path, region="LCS")

    with written_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert written_path == output_path
    assert rows[0] == {
        "region": "LCS",
        "canonical_team_code": "100T",
        "alias_name": "100 Thieves",
    }
    assert {
        "region": "LCS",
        "canonical_team_code": "TL",
        "alias_name": "Team Liquid",
    } in rows


def test_write_team_aliases_csv_writes_all_regions_by_default(tmp_path) -> None:
    output_path = tmp_path / "team_aliases.csv"

    write_team_aliases_csv(output_path)

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert {row["region"] for row in rows} == {"LCK", "LPL", "LEC", "LCS"}


def test_build_tournament_split_rows_uses_configured_values() -> None:
    rows = build_tournament_split_rows(region="LPL")

    assert {
        "region": "LPL",
        "raw_tournament_name": "LPL Split 1",
        "canonical_split": "Winter",
    } in rows
    assert {
        "region": "LPL",
        "raw_tournament_name": "LPL Regional Finals",
        "canonical_split": "Summer",
    } in rows


def test_write_tournament_splits_csv_writes_configured_rows(tmp_path) -> None:
    output_path = tmp_path / "tournament_splits.csv"

    written_path = write_tournament_splits_csv(output_path, region="LEC")

    with written_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert written_path == output_path
    assert {
        "region": "LEC",
        "raw_tournament_name": "LEC Winter",
        "canonical_split": "Winter",
    } in rows
    assert {
        "region": "LEC",
        "raw_tournament_name": "LEC Season Finals",
        "canonical_split": "Summer",
    } in rows


def test_write_tournament_splits_csv_writes_all_regions_by_default(tmp_path) -> None:
    output_path = tmp_path / "tournament_splits.csv"

    write_tournament_splits_csv(output_path)

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert {row["region"] for row in rows} == {"INT", "LCK", "LPL", "LEC", "LCS"}
