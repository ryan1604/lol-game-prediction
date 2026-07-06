from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_BOOTSTRAP_PATH = PROJECT_ROOT / "sql" / "schema" / "001_create_schemas.sql"


def test_schema_bootstrap_creates_expected_logical_schemas() -> None:
    statements = [
        line.strip()
        for line in SCHEMA_BOOTSTRAP_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert statements == [
        "CREATE SCHEMA IF NOT EXISTS raw;",
        "CREATE SCHEMA IF NOT EXISTS staging;",
        "CREATE SCHEMA IF NOT EXISTS feature;",
        "CREATE SCHEMA IF NOT EXISTS mart;",
    ]
