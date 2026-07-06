from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from .reference_discovery import (
    discover_tournament_pages,
    fetch_soup,
    normalize_region,
)

TEAM_ALIAS_GROUPS_BY_REGION: dict[str, dict[str, list[str]]] = {
    "LCK": {
        "T1": ["T1"],
        "GEN": ["GEN", "Gen.G eSports", "Gen.G"],
        "HLE": ["HLE"],
        "KT": ["KT Rolster"],
        "NS": ["NS"],
        "KRX": ["DRX", "Kiwoom DRX"],
        "DK": ["DWG KIA", "Dplus KIA"],
        "DNS": ["KDF", "DN Freecs", "DN SOOPers"],
        "BNK": ["LSB", "FearX", "BNK FearX"],
        "BRO": ["BRO", "OK BRION", "BRION"],
    },
    "LPL": {
        "AL": ["AL"],
        "BLG": ["BLG"],
        "EDG": ["EDG"],
        "FPX": ["FPX"],
        "IG": ["IG"],
        "JDG": ["JD Gaming", "JDG"],
        "LGD": ["LGD Gaming"],
        "LNG": ["LNG Esports"],
        "NIP": ["NIP"],
        "OMG": ["OMG", "Oh My God"],
        "RA": ["Rare Atom"],
        "RNG": ["RNG"],
        "TES": ["Top Esports"],
        "TT": ["TT"],
        "UP": ["Ultra Prime"],
        "V5": ["V5"],
        "WE": ["Team WE"],
        "WBG": ["WBG"],
    },
    "LEC": {
        "AST": ["Astralis"],
        "SHFT": ["Team BDS", "Shifters"],
        "FNC": ["Fnatic"],
        "G2": ["G2 Esports"],
        "GX": ["GIANTX"],
        "KC": ["KC"],
        "MKOI": ["MKOI", "KOI", "MAD", "MAD Lions", "MDK"],
        "MSF": ["MSF"],
        "RGE": ["Rogue"],
        "SK": ["SK Gaming"],
        "TH": ["TH"],
        "VIT": ["VIT"],
        "XL": ["XL"],
        "NAVI": ["NAVI"],
        "HTS": ["HTS"],
        "KCB": ["KCB"],
        "LR": ["LR", "Los Ratones"],

    },
    "LCS": {
        "100T": ["100 Thieves"],
        "C9": ["Cloud9"],
        "CLG": ["CLG"],
        "DIG": ["Dignitas"],
        "EG": ["EG"],
        "FLY": ["FlyQuest"],
        "GG": ["GG"],
        "IMT": ["Immortals"],
        "NRG": ["NRG"],
        "SR": ["SR"],
        "TL": ["Team Liquid"],
        "TSM": ["TSM"],
        "DSG": ["DSG", "Disguised"],
        "SEN": ["Sentinels"],
        "LYON": ["LYON"],
    },
}

TEAM_ALIAS_CSV_COLUMNS = ("region", "canonical_team_code", "alias_name")


def _with_progress(iterable: Iterable[Any], *, show_progress: bool, **kwargs: Any) -> Iterable[Any]:
    return tqdm(iterable, disable=not show_progress, **kwargs)


def collect_observed_team_names(
    region: str = "LCK",
    seed_urls: Iterable[str] | None = None,
    headers: dict[str, str] | None = None,
    show_progress: bool = False,
) -> set[str]:
    observed_names: set[str] = set()
    normalized_region = normalize_region(region)
    tournament_pages = discover_tournament_pages(
        region=normalized_region,
        seed_urls=seed_urls,
        headers=headers,
    )

    for _, tournament_url in _with_progress(
        tournament_pages,
        show_progress=show_progress,
        desc=f"{normalized_region} tournaments",
        total=len(tournament_pages),
        unit="tournament",
    ):
        soup = fetch_soup(tournament_url, headers=headers)
        for row in soup.select("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.select("td")]
            if len(cells) < 4 or " vs " not in cells[0]:
                continue

            left_team, right_team = [part.strip() for part in cells[0].split(" vs ", 1)]
            observed_names.update({left_team, right_team})

    return observed_names

def build_team_alias_groups(
    region: str = "LCK",
    seed_urls: Iterable[str] | None = None,
    headers: dict[str, str] | None = None,
    show_progress: bool = False,
) -> dict[str, list[str]]:
    normalized_region = normalize_region(region)
    team_alias_groups = TEAM_ALIAS_GROUPS_BY_REGION[normalized_region]
    observed_names = collect_observed_team_names(
        region=normalized_region,
        seed_urls=seed_urls,
        headers=headers,
        show_progress=show_progress,
    )
    configured_names = {
        alias for aliases in team_alias_groups.values() for alias in aliases
    }
    unmapped_names = sorted(observed_names - configured_names)
    if unmapped_names:
        raise ValueError(f"Observed unmapped {normalized_region} team names: {unmapped_names}")

    return {
        canonical_code: [alias for alias in aliases if alias in observed_names]
        for canonical_code, aliases in team_alias_groups.items()
    }


def build_team_alias_rows(region: str | None = None) -> list[dict[str, str]]:
    regions = [normalize_region(region)] if region is not None else sorted(TEAM_ALIAS_GROUPS_BY_REGION)
    rows: list[dict[str, str]] = []

    for normalized_region in regions:
        for canonical_code, aliases in TEAM_ALIAS_GROUPS_BY_REGION[normalized_region].items():
            for alias in aliases:
                rows.append(
                    {
                        "region": normalized_region,
                        "canonical_team_code": canonical_code,
                        "alias_name": alias,
                    }
                )

    return rows


def write_team_aliases_csv(path: str | Path, region: str | None = None) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEAM_ALIAS_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(build_team_alias_rows(region=region))

    return output_path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=sorted(TEAM_ALIAS_GROUPS_BY_REGION))
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()

    if args.csv_out is not None:
        output_path = write_team_aliases_csv(path=args.csv_out, region=args.region)
        print(output_path)
        return

    print(json.dumps(build_team_alias_groups(region=args.region or "LCK", show_progress=True), indent=2))


if __name__ == "__main__":
    main()
