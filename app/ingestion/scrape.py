from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

from .reference_discovery import DEFAULT_REGION_SEED_URLS, ROOT_URL, discover_tournament_pages, fetch_soup

ROLE_ORDER = ["top", "jungle", "mid", "bot", "support"]
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCES_DIR = PROJECT_ROOT / "data" / "references"
DEFAULT_LINKS_CSV_PATH = REFERENCES_DIR / "links.csv"
DEFAULT_RAW_DATA_CSV_PATH = REFERENCES_DIR / "raw_data.csv"
LINKS_CSV_COLUMNS = ("game_url",)
RAW_DATA_CSV_COLUMNS = (
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
    *(f"{side}_{role}_champion" for side in ("blue", "red") for role in ROLE_ORDER),
    *(f"{side}_{role}_player" for side in ("blue", "red") for role in ROLE_ORDER),
    "players",
    "blue_bans",
    "red_bans",
)
GAME_STATS_URL_RE = re.compile(r"(?:^|/)game/stats/(\d+)(?:[/?#]|$)")


def _with_progress(iterable: Iterable[Any], *, show_progress: bool, **kwargs: Any) -> Iterable[Any]:
    return tqdm(iterable, disable=not show_progress, **kwargs)


def _reference_path(filename: str) -> Path:
    path = REFERENCES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Could not find reference file: {path}")
    return path


def _normalize_alias(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).upper()


def _load_team_alias_map() -> dict[str, dict[str, str]]:
    with _reference_path("team_aliases.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        alias_map: dict[str, dict[str, str]] = {}
        for row in reader:
            region = _normalize_alias(row["region"])
            alias_map.setdefault(region, {})[_normalize_alias(row["alias_name"])] = row["canonical_team_code"]
        return alias_map


def _load_tournament_split_map() -> dict[str, dict[str, str]]:
    with _reference_path("tournament_split_mapping.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        split_map: dict[str, dict[str, str]] = {}
        for row in reader:
            region = _normalize_alias(row["region"])
            split_map.setdefault(region, {})[_normalize_alias(row["raw_tournament_name"])] = row["canonical_split"]
        return split_map


TEAM_ALIAS_MAP = _load_team_alias_map()
TOURNAMENT_SPLIT_MAP = _load_tournament_split_map()


def _canonicalize_team_name(team_name: str, region: str) -> str:
    normalized_region = _normalize_alias(region)
    return TEAM_ALIAS_MAP.get(normalized_region, {}).get(_normalize_alias(team_name), team_name)


def _canonicalize_split(tournament_name: str, region: str) -> str:
    normalized_region = _normalize_alias(region)
    region_split_map = TOURNAMENT_SPLIT_MAP.get(normalized_region, {})
    if not region_split_map:
        return tournament_name

    normalized_tournament = _normalize_alias(tournament_name)
    normalized_without_year = re.sub(r"\b20\d{2}\b", "", normalized_tournament)
    normalized_without_year = re.sub(r"\s+", " ", normalized_without_year).strip()

    for raw_name, canonical_split in region_split_map.items():
        if raw_name in normalized_tournament or raw_name in normalized_without_year:
            return canonical_split
    return tournament_name


def _extract_match_sides(match_name: str) -> dict[str, str]:
    if " vs " not in match_name:
        raise ValueError(f"Expected match name in 'X vs Y' format, got: {match_name}")

    left_side, right_side = [part.strip() for part in match_name.split(" vs ", 1)]
    return {"blue": left_side, "red": right_side}


def _flatten_role_values(side: str, values: list[str], suffix: str) -> dict[str, str | None]:
    return {
        f"{side}_{role}_{suffix}": values[index] if index < len(values) else None
        for index, role in enumerate(ROLE_ORDER)
    }


def _flatten_players(players: list[dict[str, str]]) -> dict[str, str]:
    flattened: dict[str, str] = {}
    for player in players:
        flattened[f"{player['side']}_{player['role']}_player"] = player["player_name"]
    return flattened


def _extract_side_payload(team_column: Tag, side: str, include_bans: bool = False) -> dict[str, Any]:
    header = team_column.select_one("div.blue-line-header, div.red-line-header")
    if header is None:
        raise ValueError(f"Expected team header for {side} side")

    team_link = header.select_one("a")
    if team_link is None:
        raise ValueError(f"Expected team link for {side} side")

    team_name = team_link.get_text(" ", strip=True)
    header_text = header.get_text(" ", strip=True)
    result = header_text.replace(team_name, "").replace("-", "").strip()

    picks: list[str] = []
    bans: list[str] = []
    has_first_pick = False

    for row in team_column.select("div.row"):
        label_cell = row.select_one("div.col-2")
        value_cell = row.select_one("div.col-10")
        if label_cell is None or value_cell is None:
            continue

        label_text = label_cell.get_text(" ", strip=True)
        champions = [img["alt"] for img in value_cell.select("img[alt]")]

        if label_text.startswith("Picks"):
            picks = champions
            has_first_pick = label_cell.select_one('img[alt="First Pick"]') is not None
        elif label_text.startswith("Bans"):
            bans = champions

    payload: dict[str, Any] = {
        "side": side,
        "team_name": team_name,
        "result": result,
        "has_first_pick": has_first_pick,
        "picks": picks,
    }

    if include_bans:
        payload["bans"] = bans

    return payload


def _extract_match_context(soup: BeautifulSoup, link: str) -> dict[str, Any]:
    page_header = soup.select_one("div.col-12.mt-4")
    if page_header is None:
        raise ValueError("Expected page header container")

    meta_row = page_header.select_one("div.row")
    if meta_row is None:
        raise ValueError("Expected tournament metadata row")

    left_meta = meta_row.select_one("div.col-12.col-sm-7")
    right_meta = meta_row.select_one("div.col-12.col-sm-5.text-right")
    if left_meta is None or right_meta is None:
        raise ValueError("Expected tournament metadata columns")

    tournament_link = left_meta.select_one("a")
    if tournament_link is None:
        raise ValueError("Expected tournament link in metadata")

    tournament_name = tournament_link.get_text(" ", strip=True)
    right_text = right_meta.get_text(" ", strip=True)
    region = tournament_name.split()[0]

    year_match = re.search(r"\b(20\d{2})\b", tournament_name)
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", right_text)
    stage_match = re.search(r"\(([^)]+)\)", right_text)

    match_name_node = soup.select_one("h1")
    if match_name_node is None:
        raise ValueError("Expected match title")
    match_name = match_name_node.get_text(" ", strip=True)

    return {
        "match_id": link.rstrip("/").split("/")[-2],
        "game_url": link,
        "match_name": match_name,
        "tournament": tournament_name,
        "region": region,
        "year": int(year_match.group(1)) if year_match else None,
        "split": _canonicalize_split(tournament_name, region),
        "stage": stage_match.group(1) if stage_match else None,
        "match_date": date_match.group(0) if date_match else None,
    }


def _extract_players(soup: BeautifulSoup) -> list[dict[str, str]]:
    players: list[dict[str, str]] = []

    for table in soup.select("table.playersInfosLine"):
        header_row = table.select_one("thead tr")
        header_classes = header_row.get("class", []) if header_row else []
        side = "blue" if "blue-line-header" in header_classes else "red"

        for index, row in enumerate(table.find_all("tr", recursive=False)):
            first_cell = row.find("td")
            if first_cell is None:
                continue

            champion_icon = first_cell.select_one("img.champion_icon[alt]")
            player_link = first_cell.select_one('a[href*="/players/player-stats/"]')
            if champion_icon is None or player_link is None:
                continue

            players.append(
                {
                    "side": side,
                    "role": ROLE_ORDER[index] if index < len(ROLE_ORDER) else f"slot_{index + 1}",
                    "player_name": player_link.get_text(" ", strip=True),
                    "champion": champion_icon["alt"],
                }
            )

    return players


def _serialize_csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return value


def _resolve_match_link(href: str, source_url: str) -> str:
    if GAME_STATS_URL_RE.search(href) is None:
        return urljoin(source_url, href)
    normalized_href = re.sub(r"^(?:\./|\.\./)+", "", href).lstrip("/")
    return urljoin(ROOT_URL, normalized_href)


def _canonicalize_match_link(href: str, source_url: str) -> str:
    resolved_url = _resolve_match_link(href, source_url)
    match = GAME_STATS_URL_RE.search(resolved_url)
    if match is None:
        raise ValueError(f"Unsupported match link format: {href}")
    return f"{ROOT_URL}game/stats/{match.group(1)}/page-game/"


def _iter_game_hrefs(soup: BeautifulSoup) -> Iterable[str]:
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if GAME_STATS_URL_RE.search(href) is not None:
            yield href


def _discover_seed_match_links(
    region: str,
    headers: dict[str, str] | None = None,
    show_progress: bool = False,
) -> list[str]:
    seed_links: list[str] = []
    seen_links: set[str] = set()
    tournament_pages = discover_tournament_pages(region=region, headers=headers)

    for _, tournament_url in _with_progress(
        tournament_pages,
        show_progress=show_progress,
        desc=f"{region} tournaments",
        total=len(tournament_pages),
        unit="tournament",
    ):
        soup = fetch_soup(tournament_url, headers=headers)
        for href in _iter_game_hrefs(soup):
            canonical_game_url = _canonicalize_match_link(href, tournament_url)
            if canonical_game_url in seen_links:
                continue
            seen_links.add(canonical_game_url)
            seed_links.append(_resolve_match_link(href, tournament_url))

    return seed_links


def discover_match_links(
    region: str | None = None,
    headers: dict[str, str] | None = None,
    show_progress: bool = False,
) -> list[str]:
    regions = [region] if region is not None else sorted(DEFAULT_REGION_SEED_URLS)
    seed_links: list[str] = []

    for target_region in regions:
        seed_links.extend(
            _discover_seed_match_links(
                target_region,
                headers=headers,
                show_progress=show_progress,
            )
        )

    expanded_links: list[str] = []
    seen_links: set[str] = set()
    fetched_links: set[str] = set()
    pending_links = deque(dict.fromkeys(seed_links))
    progress = tqdm(desc="Expanding matches", unit="match", disable=not show_progress)

    while pending_links:
        seed_link = pending_links.popleft()
        if seed_link in fetched_links:
            continue

        fetched_links.add(seed_link)
        soup = fetch_soup(seed_link, headers=headers)
        game_links = {
            _canonicalize_match_link(href, seed_link)
            for href in _iter_game_hrefs(soup)
        } or {_canonicalize_match_link(seed_link, seed_link)}

        for game_link in sorted(game_links):
            if game_link not in seen_links:
                seen_links.add(game_link)
                expanded_links.append(game_link)
            if game_link not in fetched_links:
                pending_links.append(game_link)

        progress.update(1)

    progress.close()

    return expanded_links


def write_match_links_csv(
    path: str | Path = DEFAULT_LINKS_CSV_PATH,
    region: str | None = None,
    headers: dict[str, str] | None = None,
    show_progress: bool = False,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    match_links = discover_match_links(region=region, headers=headers, show_progress=show_progress)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LINKS_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows({"game_url": link} for link in match_links)

    return output_path


def read_match_links_csv(path: str | Path = DEFAULT_LINKS_CSV_PATH) -> list[str]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [row["game_url"] for row in csv.DictReader(handle) if row.get("game_url")]


def parse_gol_game_page(
    link: str,
    headers: dict[str, str] | None = None,
    include_bans: bool = False,
) -> dict[str, Any]:
    time.sleep(3)
    request_headers = headers or DEFAULT_HEADERS

    response = requests.get(link, headers=request_headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    context = _extract_match_context(soup, link)

    team_columns = soup.select("div.col-12.col-sm-6")[:2]
    if len(team_columns) < 2:
        raise ValueError(f"Expected two team columns for {link}, found {len(team_columns)}")

    blue_team = _extract_side_payload(team_columns[0], side="blue", include_bans=include_bans)
    red_team = _extract_side_payload(team_columns[1], side="red", include_bans=include_bans)
    players = _extract_players(soup)
    match_sides = _extract_match_sides(context["match_name"])
    canonical_sides = {
        side: _canonicalize_team_name(team_name, context["region"])
        for side, team_name in match_sides.items()
    }

    for player in players:
        player["team_name"] = canonical_sides.get(player["side"], player["side"])

    match_record: dict[str, Any] = {
        **context,
        "blue_team_name": canonical_sides["blue"],
        "red_team_name": canonical_sides["red"],
        "winner_side": "blue" if blue_team["result"] == "WIN" else "red" if red_team["result"] == "WIN" else None,
        "first_pick_side": "blue" if blue_team["has_first_pick"] else "red" if red_team["has_first_pick"] else None,
        **_flatten_role_values("blue", blue_team["picks"], "champion"),
        **_flatten_role_values("red", red_team["picks"], "champion"),
        **_flatten_players(players),
        "players": players,
    }

    if include_bans:
        match_record["blue_bans"] = blue_team.get("bans", [])
        match_record["red_bans"] = red_team.get("bans", [])

    return match_record


def scrape_match_links(
    match_links: Iterable[str],
    headers: dict[str, str] | None = None,
    include_bans: bool = True,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    match_link_list = list(match_links)
    return [
        parse_gol_game_page(link, headers=headers, include_bans=include_bans)
        for link in _with_progress(
            match_link_list,
            show_progress=show_progress,
            desc="Scraping matches",
            total=len(match_link_list),
            unit="match",
        )
    ]


def scrape_reference_matches(
    links_path: str | Path = DEFAULT_LINKS_CSV_PATH,
    headers: dict[str, str] | None = None,
    include_bans: bool = True,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    return scrape_match_links(
        read_match_links_csv(links_path),
        headers=headers,
        include_bans=include_bans,
        show_progress=show_progress,
    )


def write_raw_data_csv(
    path: str | Path = DEFAULT_RAW_DATA_CSV_PATH,
    links_path: str | Path = DEFAULT_LINKS_CSV_PATH,
    headers: dict[str, str] | None = None,
    include_bans: bool = True,
    show_progress: bool = False,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_exists = output_path.exists() and output_path.stat().st_size > 0
    completed_urls: set[str] = set()

    if output_exists:
        with output_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "game_url" not in (reader.fieldnames or []):
                raise ValueError(f"Existing raw data CSV is missing game_url column: {output_path}")
            completed_urls = {row["game_url"] for row in reader if row.get("game_url")}

    pending_links = [
        link
        for link in read_match_links_csv(links_path)
        if link not in completed_urls
    ]

    with output_path.open("a" if output_exists else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_DATA_CSV_COLUMNS)
        if not output_exists:
            writer.writeheader()

        for link in _with_progress(
            pending_links,
            show_progress=show_progress,
            desc="Scraping matches",
            total=len(pending_links),
            unit="match",
        ):
            row = parse_gol_game_page(link, headers=headers, include_bans=include_bans)
            writer.writerow({
                column: _serialize_csv_value(row.get(column))
                for column in RAW_DATA_CSV_COLUMNS
            })
            handle.flush()

    return output_path


def discover_links_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=sorted(DEFAULT_REGION_SEED_URLS))
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_LINKS_CSV_PATH)
    args = parser.parse_args()

    output_path = write_match_links_csv(
        path=args.csv_out,
        region=args.region,
        show_progress=True,
    )
    print(output_path)


def scrape_raw_data_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--links-csv", type=Path, default=DEFAULT_LINKS_CSV_PATH)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_RAW_DATA_CSV_PATH)
    parser.add_argument("--exclude-bans", action="store_true")
    args = parser.parse_args()

    output_path = write_raw_data_csv(
        path=args.csv_out,
        links_path=args.links_csv,
        include_bans=not args.exclude_bans,
        show_progress=True,
    )
    print(output_path)


__all__ = [
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
]


if __name__ == "__main__":
    scrape_raw_data_main()
