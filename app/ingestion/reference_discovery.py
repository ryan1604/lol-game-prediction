from __future__ import annotations

import re
import time
from typing import Iterable

import requests
from bs4 import BeautifulSoup

DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0"}

ROOT_URL = "https://gol.gg/"

DEFAULT_REGION_SEED_URLS: dict[str, tuple[str, ...]] = {
    "LCK": (
        "https://gol.gg/tournament/tournament-matchlist/LCK%20Cup%202026/",
    ),
    "LPL": (
        "https://gol.gg/tournament/tournament-matchlist/LPL%202026%20Split%201/",
    ),
    "LEC": (
        "https://gol.gg/tournament/tournament-matchlist/LEC%202026%20Versus%20Season/",
    ),
    "LCS": (
        "https://gol.gg/tournament/tournament-matchlist/LCS%202026%20Lock-In/",
    ),
    "INT": (
        "https://gol.gg/tournament/tournament-matchlist/2026%20First%20Stand/",
        "https://gol.gg/tournament/tournament-matchlist/First%20Stand%202025/",
        "https://gol.gg/tournament/tournament-matchlist/2025%20Mid-Season%20Invitational/",
        "https://gol.gg/tournament/tournament-matchlist/Worlds%202025%20Play-In/",
        "https://gol.gg/tournament/tournament-matchlist/MSI%202024/",
        "https://gol.gg/tournament/tournament-matchlist/Worlds%20Play-In%202024/",
        "https://gol.gg/tournament/tournament-matchlist/MSI%202023/",
        "https://gol.gg/tournament/tournament-matchlist/Worlds%20Qualifying%20Series%202023/",
        "https://gol.gg/tournament/tournament-matchlist/MSI%202022/",
        "https://gol.gg/tournament/tournament-matchlist/World%20Championship%20Play-In%202022/"
    ),
}

INTERNATIONAL_TOURNAMENT_PATTERNS = (
    re.compile(r"\bMSI\b", re.IGNORECASE),
    re.compile(r"MID[\s-]SEASON INVITATIONAL", re.IGNORECASE),
    re.compile(r"\bFIRST STAND\b", re.IGNORECASE),
    re.compile(r"\bWORLDS?\b", re.IGNORECASE),
    re.compile(r"WORLD CHAMPIONSHIP", re.IGNORECASE),
)


def normalize_region(region: str) -> str:
    normalized_region = region.strip().upper()
    if normalized_region not in DEFAULT_REGION_SEED_URLS:
        supported = ", ".join(sorted(DEFAULT_REGION_SEED_URLS))
        raise ValueError(f"Unsupported region '{region}'. Expected one of: {supported}")
    return normalized_region


def resolve_seed_urls(region: str, seed_urls: Iterable[str] | None = None) -> tuple[str, ...]:
    if seed_urls is not None:
        return tuple(seed_urls)
    return DEFAULT_REGION_SEED_URLS[normalize_region(region)]


def fetch_soup(url: str, headers: dict[str, str] | None = None) -> BeautifulSoup:
    time.sleep(5)
    response = requests.get(url, headers=headers or DEFAULT_HEADERS, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def to_matchlist_url(href: str) -> str:
    matchlist_href = href.replace("tournament-stats", "tournament-matchlist").lstrip("./")
    while matchlist_href.startswith("../"):
        matchlist_href = matchlist_href[3:]
    return ROOT_URL + matchlist_href


def is_region_tournament_name(tournament_name: str, region: str) -> bool:
    if region == "INT":
        return any(pattern.search(tournament_name) for pattern in INTERNATIONAL_TOURNAMENT_PATTERNS)
    return tournament_name.upper().startswith(region)


def discover_tournament_pages(
    region: str,
    seed_urls: Iterable[str] | None = None,
    headers: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    normalized_region = normalize_region(region)
    tournaments: list[tuple[str, str]] = []
    seen_urls: set[str] = set()

    for seed_url in resolve_seed_urls(normalized_region, seed_urls):
        soup = fetch_soup(seed_url, headers=headers)
        page_title = soup.title.get_text(" ", strip=True) if soup.title else ""
        year_match = re.search(r"(20\d{2})", page_title)
        year = year_match.group(1) if year_match else None

        for link in soup.find_all("a", href=True):
            tournament_name = link.get_text(" ", strip=True)
            href = link["href"]
            if "tournament-stats" not in href:
                continue
            if year and year not in tournament_name:
                continue
            if not is_region_tournament_name(tournament_name, normalized_region):
                continue

            matchlist_url = to_matchlist_url(href)
            if matchlist_url in seen_urls:
                continue

            seen_urls.add(matchlist_url)
            tournaments.append((tournament_name, matchlist_url))

    return tournaments
