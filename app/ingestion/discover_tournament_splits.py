from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from .reference_discovery import (
    discover_tournament_pages,
    normalize_region,
)

SPLIT_RULES_BY_REGION: dict[str, dict[str, list[str]]] = {
    "LCK": {
        "Winter": ["LCK Cup"],
        "Spring": ["LCK Spring", "LCK Spring Playoffs", "LCK Rounds 1-2", "LCK Road to MSI"],
        "Summer": ["LCK Summer", "LCK Summer Playoffs", "LCK Rounds 3-5", "LCK Season Play-In", "LCK Season Playoffs", "LCK Regionals", "LCK Regional Finals", "LCK Regionals Finals"],
    },
    "LPL": {
        "Winter": ["LPL Split 1", "LPL Split 1 Playoffs"],
        "Spring": ["LPL Spring", "LPL Spring Playoffs", "LPL Split 2 Placements", "LPL Split 2", "LPL Split 2 Playoffs"],
        "Summer": ["LPL Summer", "LPL Summer Playoffs", "LPL Split 3", "LPL Split 3 Playoffs", "LPL Grand Finals", "LPL Regionals", "LPL Regional Finals"],
    },
    "LEC": {
        "Winter": ["LEC Winter", "LEC Winter Groups", "LEC Winter Playoffs", "LEC Winter Season", "LEC Versus Season", "LEC Versus Playoffs"],
        "Spring": ["LEC Spring", "LEC Spring Groups", "LEC Spring Playoffs", "LEC Spring Season"],
        "Summer": ["LEC Summer", "LEC Summer Groups", "LEC Summer Playoffs", "LEC Summer Season", "LEC Season Finals"],
    },
    "LCS": {
        "Spring": ["LCS Spring", "LCS Spring Playoffs", "LCS Lock In", "LCS Lock-In"],
        "Summer": ["LCS Summer", "LCS Championship"],
    },
    "INT": {
        "FS": ["First Stand"],
        "MSI": ["MSI", "Mid-Season Invitational"],
        "Worlds": ["Worlds", "World Championship", "World Championship Play-In", "Worlds Main Event", "Worlds Play-In", "Worlds Qualifying Series"]
    },
}

TOURNAMENT_SPLIT_CSV_COLUMNS = ("region", "raw_tournament_name", "canonical_split")


def _with_progress(iterable: Iterable[Any], *, show_progress: bool, **kwargs: Any) -> Iterable[Any]:
    return tqdm(iterable, disable=not show_progress, **kwargs)


def normalize_tournament_label(tournament_name: str) -> str:
    without_year = re.sub(r"\b20\d{2}\b", "", tournament_name)
    normalized = re.sub(r"\s+", " ", without_year).strip()
    return normalized


def discover_tournament_names(
    region: str = "LCK",
    seed_urls: Iterable[str] | None = None,
    headers: dict[str, str] | None = None,
    show_progress: bool = False,
) -> list[str]:
    tournament_pages = discover_tournament_pages(
        region=region,
        seed_urls=seed_urls,
        headers=headers,
    )

    return [
        tournament_name
        for tournament_name, _ in _with_progress(
            tournament_pages,
            show_progress=show_progress,
            desc=f"{normalize_region(region)} tournaments",
            total=len(tournament_pages),
            unit="tournament",
        )
    ]


def map_tournament_label_to_split(label: str, region: str = "LCK") -> str:
    normalized_region = normalize_region(region)
    for canonical_split, labels in SPLIT_RULES_BY_REGION[normalized_region].items():
        if label in labels:
            return canonical_split
    raise ValueError(f"Unsupported {normalized_region} tournament label: {label}")


def build_tournament_split_mapping(
    region: str = "LCK",
    seed_urls: Iterable[str] | None = None,
    headers: dict[str, str] | None = None,
    show_progress: bool = False,
) -> dict[str, str]:
    normalized_region = normalize_region(region)
    normalized_labels = {
        normalize_tournament_label(name)
        for name in discover_tournament_names(
            region=normalized_region,
            seed_urls=seed_urls,
            headers=headers,
            show_progress=show_progress,
        )
    }

    return {
        label: map_tournament_label_to_split(label, region=normalized_region)
        for label in sorted(normalized_labels)
    }


def build_tournament_split_groups(
    region: str = "LCK",
    seed_urls: Iterable[str] | None = None,
    headers: dict[str, str] | None = None,
    show_progress: bool = False,
) -> dict[str, list[str]]:
    normalized_region = normalize_region(region)
    tournament_split_mapping = build_tournament_split_mapping(
        region=normalized_region,
        seed_urls=seed_urls,
        headers=headers,
        show_progress=show_progress,
    )
    grouped_labels: dict[str, list[str]] = {
        canonical_split: []
        for canonical_split in SPLIT_RULES_BY_REGION[normalized_region]
    }

    for label, canonical_split in tournament_split_mapping.items():
        grouped_labels[canonical_split].append(label)

    return {
        canonical_split: labels
        for canonical_split, labels in grouped_labels.items()
        if labels
    }


def build_tournament_split_rows(region: str | None = None) -> list[dict[str, str]]:
    regions = [normalize_region(region)] if region is not None else sorted(SPLIT_RULES_BY_REGION)
    rows: list[dict[str, str]] = []

    for normalized_region in regions:
        configured_labels = {
            label
            for labels in SPLIT_RULES_BY_REGION[normalized_region].values()
            for label in labels
        }

        rows.extend(
            {
                "region": normalized_region,
                "raw_tournament_name": label,
                "canonical_split": map_tournament_label_to_split(label, region=normalized_region),
            }
            for label in sorted(configured_labels)
        )

    return rows


def write_tournament_splits_csv(path: str | Path, region: str | None = None) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TOURNAMENT_SPLIT_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(build_tournament_split_rows(region=region))

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=sorted(SPLIT_RULES_BY_REGION))
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()

    if args.csv_out is not None:
        output_path = write_tournament_splits_csv(path=args.csv_out, region=args.region)
        print(output_path)
        return

    print(json.dumps(build_tournament_split_groups(region=args.region or "LCK", show_progress=True), indent=2))


if __name__ == "__main__":
    main()
