import csv
import json
from unittest.mock import Mock

from bs4 import BeautifulSoup

from app.ingestion import (
    DEFAULT_HEADERS,
    TEAM_ALIAS_MAP,
    TOURNAMENT_SPLIT_MAP,
    discover_match_links,
    parse_gol_game_page,
    read_match_links_csv,
    scrape_match_links,
    scrape_reference_matches,
    write_match_links_csv,
    write_raw_data_csv,
)


SAMPLE_HTML = """
<html>
  <body>
    <div class="col-12 mt-4">
      <h1>HLE vs GEN</h1>
      <div class="row">
        <div class="col-12 col-sm-7">
          <a href="../tournament/tournament-stats/LCK%202025%20Rounds%201-2/">LCK 2025 Rounds 1-2</a>
        </div>
        <div class="col-12 col-sm-5 text-right">2025-04-02 (WEEK1)</div>
      </div>
    </div>
    <div class="col-3 text-right">v15.6</div>
    <div class="col-12 col-sm-6">
      <div class="blue-line-header"><a href="../teams/team-stats/1/">Hanwha Life eSports</a> - LOSS</div>
      <div class="row">
        <div class="col-2">Bans</div>
        <div class="col-10">
          <img alt="Varus"/><img alt="Skarner"/><img alt="Rumble"/><img alt="Rell"/><img alt="Poppy"/>
        </div>
      </div>
      <div class="row">
        <div class="col-2">Picks <img alt="First Pick"/></div>
        <div class="col-10">
          <img alt="Ambessa"/><img alt="Naafiri"/><img alt="Aurora"/><img alt="Ezreal"/><img alt="Alistar"/>
        </div>
      </div>
    </div>
    <div class="col-12 col-sm-6">
      <div class="red-line-header"><a href="../teams/team-stats/2/">Gen.G eSports</a> - WIN</div>
      <div class="row">
        <div class="col-2">Bans</div>
        <div class="col-10">
          <img alt="Kalista"/><img alt="Yone"/><img alt="Gwen"/><img alt="Ryze"/><img alt="Sylas"/>
        </div>
      </div>
      <div class="row">
        <div class="col-2">Picks</div>
        <div class="col-10">
          <img alt="Jayce"/><img alt="Vi"/><img alt="Ahri"/><img alt="Miss Fortune"/><img alt="Rakan"/>
        </div>
      </div>
    </div>
    <table class="playersInfosLine">
      <thead><tr class="blue-line-header"></tr></thead>
      <tr><td><img class="champion_icon" alt="Ambessa"/><a href="../players/player-stats/1/">Zeus</a></td></tr>
      <tr><td><img class="champion_icon" alt="Naafiri"/><a href="../players/player-stats/2/">Peanut</a></td></tr>
      <tr><td><img class="champion_icon" alt="Aurora"/><a href="../players/player-stats/3/">Zeka</a></td></tr>
      <tr><td><img class="champion_icon" alt="Ezreal"/><a href="../players/player-stats/4/">Viper</a></td></tr>
      <tr><td><img class="champion_icon" alt="Alistar"/><a href="../players/player-stats/5/">Delight</a></td></tr>
    </table>
    <table class="playersInfosLine">
      <thead><tr class="red-line-header"></tr></thead>
      <tr><td><img class="champion_icon" alt="Jayce"/><a href="../players/player-stats/6/">Kiin</a></td></tr>
      <tr><td><img class="champion_icon" alt="Vi"/><a href="../players/player-stats/7/">Canyon</a></td></tr>
      <tr><td><img class="champion_icon" alt="Ahri"/><a href="../players/player-stats/8/">Chovy</a></td></tr>
      <tr><td><img class="champion_icon" alt="Miss Fortune"/><a href="../players/player-stats/9/">Ruler</a></td></tr>
      <tr><td><img class="champion_icon" alt="Rakan"/><a href="../players/player-stats/10/">Duro</a></td></tr>
    </table>
  </body>
</html>
"""


def _mock_response() -> Mock:
    response = Mock()
    response.text = SAMPLE_HTML
    response.raise_for_status = Mock()
    return response


def test_reference_maps_are_importable() -> None:
    assert TEAM_ALIAS_MAP["LCK"]["GEN.G ESPORTS"] == "GEN"
    assert TEAM_ALIAS_MAP["LCK"]["LSB"] == "BNK"
    assert TOURNAMENT_SPLIT_MAP["LCK"]["LCK CUP"] == "Winter"
    assert TOURNAMENT_SPLIT_MAP["LCK"]["LCK REGIONAL FINALS"] == "Summer"


def test_parse_gol_game_page_for_notebook_use(monkeypatch) -> None:
    mock_get = Mock(return_value=_mock_response())
    monkeypatch.setattr("app.ingestion.scrape.requests.get", mock_get)

    parsed = parse_gol_game_page(
        "https://gol.gg/game/stats/65428/page-game/",
        include_bans=True,
    )

    assert mock_get.call_args.kwargs["headers"] == DEFAULT_HEADERS
    assert parsed["split"] == "Spring"
    assert parsed["blue_team_name"] == "HLE"
    assert parsed["red_team_name"] == "GEN"
    assert parsed["winner_side"] == "red"
    assert parsed["first_pick_side"] == "blue"
    assert parsed["blue_top_champion"] == "Ambessa"
    assert parsed["red_support_champion"] == "Rakan"
    assert parsed["blue_jungle_player"] == "Peanut"
    assert parsed["red_mid_player"] == "Chovy"
    assert parsed["blue_bans"] == ["Varus", "Skarner", "Rumble", "Rell", "Poppy"]
    assert parsed["red_bans"] == ["Kalista", "Yone", "Gwen", "Ryze", "Sylas"]


def test_discover_match_links_expands_series_links(monkeypatch) -> None:
    html_by_url = {
        "https://example.test/tournament-one/": """
        <html>
          <body>
            <a href="game/stats/111/page-game/">Series One</a>
            <a href="https://gol.gg/game/stats/222/page-summary/">Series Two</a>
          </body>
        </html>
        """,
        "https://example.test/tournament-two/": """
        <html>
          <body>
            <a href="game/stats/333/page-summary/">Series Three</a>
          </body>
        </html>
        """,
        "https://gol.gg/game/stats/111/page-game/": """
        <html>
          <body>
            <a href="/game/stats/111/page-game/">Game 1</a>
            <a href="/game/stats/112/page-game/">Game 2</a>
            <a href="/game/stats/113/page-summary/">Game 3</a>
          </body>
        </html>
        """,
        "https://gol.gg/game/stats/112/page-game/": """
        <html>
          <body>
            <a href="/game/stats/112/page-game/">Game 2</a>
            <a href="/game/stats/113/page-summary/">Game 3</a>
          </body>
        </html>
        """,
        "https://gol.gg/game/stats/113/page-game/": """
        <html>
          <body>
            <a href="/game/stats/113/page-game/">Game 3</a>
          </body>
        </html>
        """,
        "https://gol.gg/game/stats/222/page-game/": """
        <html>
          <body>
            <a href="/game/stats/222/page-summary/">Game 1</a>
            <a href="/game/stats/223/page-game/">Game 2</a>
          </body>
        </html>
        """,
        "https://gol.gg/game/stats/222/page-summary/": """
        <html>
          <body>
            <a href="/game/stats/222/page-summary/">Game 1</a>
            <a href="/game/stats/223/page-game/">Game 2</a>
          </body>
        </html>
        """,
        "https://gol.gg/game/stats/223/page-game/": """
        <html>
          <body>
            <a href="/game/stats/223/page-game/">Game 2</a>
          </body>
        </html>
        """,
        "https://gol.gg/game/stats/333/page-game/": """
        <html>
          <body>
            <a href="/game/stats/333/page-game/">Game 1</a>
          </body>
        </html>
        """,
        "https://gol.gg/game/stats/333/page-summary/": """
        <html>
          <body>
            <a href="/game/stats/333/page-game/">Game 1</a>
          </body>
        </html>
        """,
    }
    monkeypatch.setattr(
        "app.ingestion.scrape.discover_tournament_pages",
        lambda *args, **kwargs: [
            ("Tournament One", "https://example.test/tournament-one/"),
            ("Tournament Two", "https://example.test/tournament-two/"),
        ],
    )
    monkeypatch.setattr(
        "app.ingestion.scrape.fetch_soup",
        lambda url, headers=None: BeautifulSoup(html_by_url[url], "html.parser"),
    )

    discovered = discover_match_links(region="LCK")

    assert discovered == [
        "https://gol.gg/game/stats/111/page-game/",
        "https://gol.gg/game/stats/112/page-game/",
        "https://gol.gg/game/stats/113/page-game/",
        "https://gol.gg/game/stats/222/page-game/",
        "https://gol.gg/game/stats/223/page-game/",
        "https://gol.gg/game/stats/333/page-game/",
    ]


def test_discover_match_links_expands_from_original_seed_page(monkeypatch) -> None:
    html_by_url = {
        "https://example.test/tournament-one/": """
        <html>
          <body>
            <a href="https://gol.gg/game/stats/38812/page-summary/">Series One</a>
          </body>
        </html>
        """,
        "https://gol.gg/game/stats/38812/page-summary/": """
        <html>
          <body>
            <a href="/game/stats/38812/page-summary/">Game 1</a>
            <a href="/game/stats/38813/page-summary/">Game 2</a>
          </body>
        </html>
        """,
        "https://gol.gg/game/stats/38812/page-game/": """
        <html>
          <body>
            <a href="/game/stats/38812/page-game/">Game 1</a>
          </body>
        </html>
        """,
        "https://gol.gg/game/stats/38813/page-game/": """
        <html>
          <body>
            <a href="/game/stats/38813/page-game/">Game 2</a>
          </body>
        </html>
        """,
    }
    monkeypatch.setattr(
        "app.ingestion.scrape.discover_tournament_pages",
        lambda *args, **kwargs: [
            ("Tournament One", "https://example.test/tournament-one/"),
        ],
    )
    monkeypatch.setattr(
        "app.ingestion.scrape.fetch_soup",
        lambda url, headers=None: BeautifulSoup(html_by_url[url], "html.parser"),
    )

    discovered = discover_match_links(region="LCK")

    assert discovered == [
        "https://gol.gg/game/stats/38812/page-game/",
        "https://gol.gg/game/stats/38813/page-game/",
    ]


def test_write_and_read_match_links_csv(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "links.csv"
    monkeypatch.setattr(
        "app.ingestion.scrape.discover_match_links",
        lambda *args, **kwargs: [
            "https://gol.gg/game/stats/111/page-game/",
            "https://gol.gg/game/stats/112/page-game/",
        ],
    )

    written_path = write_match_links_csv(path=output_path)

    assert written_path == output_path
    assert read_match_links_csv(output_path) == [
        "https://gol.gg/game/stats/111/page-game/",
        "https://gol.gg/game/stats/112/page-game/",
    ]


def test_write_raw_data_csv_reads_links_file(tmp_path, monkeypatch) -> None:
    links_path = tmp_path / "links.csv"
    raw_data_path = tmp_path / "raw_data.csv"
    links_path.write_text("game_url\nhttps://gol.gg/game/stats/111/page-game/\n", encoding="utf-8")
    monkeypatch.setattr(
        "app.ingestion.scrape.parse_gol_game_page",
        lambda *args, **kwargs: {
            "match_id": "111",
            "game_url": "https://gol.gg/game/stats/111/page-game/",
            "match_name": "HLE vs GEN",
            "tournament": "LCK 2025 Rounds 1-2",
            "region": "LCK",
            "year": 2025,
            "split": "Spring",
            "stage": "WEEK1",
            "match_date": "2025-04-02",
            "blue_team_name": "HLE",
            "red_team_name": "GEN",
            "winner_side": "red",
            "first_pick_side": "blue",
            "blue_top_champion": "Ambessa",
            "blue_jungle_champion": "Naafiri",
            "blue_mid_champion": "Aurora",
            "blue_bot_champion": "Ezreal",
            "blue_support_champion": "Alistar",
            "red_top_champion": "Jayce",
            "red_jungle_champion": "Vi",
            "red_mid_champion": "Ahri",
            "red_bot_champion": "Miss Fortune",
            "red_support_champion": "Rakan",
            "blue_top_player": "Zeus",
            "blue_jungle_player": "Peanut",
            "blue_mid_player": "Zeka",
            "blue_bot_player": "Viper",
            "blue_support_player": "Delight",
            "red_top_player": "Kiin",
            "red_jungle_player": "Canyon",
            "red_mid_player": "Chovy",
            "red_bot_player": "Ruler",
            "red_support_player": "Duro",
            "players": [{"side": "blue", "role": "top", "player_name": "Zeus"}],
            "blue_bans": ["Varus", "Skarner"],
            "red_bans": ["Kalista", "Yone"],
        },
    )

    written_path = write_raw_data_csv(path=raw_data_path, links_path=links_path)

    with written_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert written_path == raw_data_path
    assert rows == [
        {
            "match_id": "111",
            "game_url": "https://gol.gg/game/stats/111/page-game/",
            "match_name": "HLE vs GEN",
            "tournament": "LCK 2025 Rounds 1-2",
            "region": "LCK",
            "year": "2025",
            "split": "Spring",
            "stage": "WEEK1",
            "match_date": "2025-04-02",
            "blue_team_name": "HLE",
            "red_team_name": "GEN",
            "winner_side": "red",
            "first_pick_side": "blue",
            "blue_top_champion": "Ambessa",
            "blue_jungle_champion": "Naafiri",
            "blue_mid_champion": "Aurora",
            "blue_bot_champion": "Ezreal",
            "blue_support_champion": "Alistar",
            "red_top_champion": "Jayce",
            "red_jungle_champion": "Vi",
            "red_mid_champion": "Ahri",
            "red_bot_champion": "Miss Fortune",
            "red_support_champion": "Rakan",
            "blue_top_player": "Zeus",
            "blue_jungle_player": "Peanut",
            "blue_mid_player": "Zeka",
            "blue_bot_player": "Viper",
            "blue_support_player": "Delight",
            "red_top_player": "Kiin",
            "red_jungle_player": "Canyon",
            "red_mid_player": "Chovy",
            "red_bot_player": "Ruler",
            "red_support_player": "Duro",
            "players": json.dumps([{"side": "blue", "role": "top", "player_name": "Zeus"}]),
            "blue_bans": json.dumps(["Varus", "Skarner"]),
            "red_bans": json.dumps(["Kalista", "Yone"]),
        }
    ]


def test_write_raw_data_csv_resumes_existing_output(tmp_path, monkeypatch) -> None:
    links_path = tmp_path / "links.csv"
    raw_data_path = tmp_path / "raw_data.csv"
    links_path.write_text(
        "game_url\nhttps://gol.gg/game/stats/111/page-game/\nhttps://gol.gg/game/stats/112/page-game/\n",
        encoding="utf-8",
    )
    raw_data_path.write_text(
        "match_id,game_url,match_name,tournament,region,year,split,stage,match_date,"
        "blue_team_name,red_team_name,winner_side,first_pick_side,"
        "blue_top_champion,blue_jungle_champion,blue_mid_champion,blue_bot_champion,blue_support_champion,"
        "red_top_champion,red_jungle_champion,red_mid_champion,red_bot_champion,red_support_champion,"
        "blue_top_player,blue_jungle_player,blue_mid_player,blue_bot_player,blue_support_player,"
        "red_top_player,red_jungle_player,red_mid_player,red_bot_player,red_support_player,"
        "players,blue_bans,red_bans\n"
        "111,https://gol.gg/game/stats/111/page-game/,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\n",
        encoding="utf-8",
    )
    scraped_links: list[str] = []

    def fake_parse(link: str, **kwargs) -> dict[str, str]:
        scraped_links.append(link)
        return {
            "match_id": "112",
            "game_url": link,
            "match_name": "T1 vs GEN",
        }

    monkeypatch.setattr("app.ingestion.scrape.parse_gol_game_page", fake_parse)

    write_raw_data_csv(path=raw_data_path, links_path=links_path)

    with raw_data_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert scraped_links == ["https://gol.gg/game/stats/112/page-game/"]
    assert [row["game_url"] for row in rows] == [
        "https://gol.gg/game/stats/111/page-game/",
        "https://gol.gg/game/stats/112/page-game/",
    ]


def test_scrape_reference_matches_reads_links_file(tmp_path, monkeypatch) -> None:
    links_path = tmp_path / "links.csv"
    links_path.write_text(
        "game_url\nhttps://gol.gg/game/stats/111/page-game/\nhttps://gol.gg/game/stats/112/page-game/\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.ingestion.scrape.parse_gol_game_page",
        lambda link, **kwargs: {"match_id": link.rstrip('/').split('/')[-2]},
    )

    assert scrape_reference_matches(links_path=links_path) == [
        {"match_id": "111"},
        {"match_id": "112"},
    ]


def test_scrape_match_links_shows_progress(monkeypatch) -> None:
    progress_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "app.ingestion.scrape.parse_gol_game_page",
        lambda *args, **kwargs: {"match_id": "111"},
    )
    monkeypatch.setattr(
        "app.ingestion.scrape.tqdm",
        lambda iterable, **kwargs: progress_calls.append(kwargs) or iterable,
    )

    assert scrape_match_links(["https://gol.gg/game/stats/111/page-game/"], show_progress=True) == [{"match_id": "111"}]
    assert progress_calls == [
        {
            "disable": False,
            "desc": "Scraping matches",
            "total": 1,
            "unit": "match",
        }
    ]
