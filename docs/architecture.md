# LoL Game Prediction Architecture

## Overview

LoL Game Prediction is an end-to-end machine learning application that predicts the winner of a professional League of Legends match after draft is complete and before the game starts.

The system is intentionally file-first: scraped data, cleaned datasets, feature datasets, and model artifacts are stored as local CSV, Parquet, and joblib/JSON files. This keeps the project easy to run locally and simple to deploy.

## System Flow

```text
GOL.GG match pages
-> ingestion scripts
-> raw CSV data
-> cleaning pipeline
-> cleaned match parquet
-> feature builder
-> feature parquet
-> LightGBM training
-> exported model artifacts
-> FastAPI prediction service
-> React frontend
```

## Runtime Architecture

```text
Cloudflare Pages
  React + TypeScript frontend
  VITE_API_BASE_URL points to Render API

Render Web Service
  FastAPI + Uvicorn
  loads models/latest/*
  loads data/processed/matches_clean.parquet

Local artifacts
  cleaned match history
  exported model
  feature metadata
  run metadata
```

## Components

### Frontend

Location: `frontend/`

The frontend is a Vite React app. It lets users enter a completed draft by selecting match context, teams, players, champions, and first-pick side.

At startup it calls `GET /metadata` on the API to load available selector values. On submit it calls `POST /predict` and displays blue-side win probability and predicted winner.

Production deployment uses Cloudflare Pages. The frontend reads `VITE_API_BASE_URL` to call the Render API directly. Local development falls back to the Vite `/api` proxy.

### API

Location: `app/api/`

The API is a FastAPI service with three public endpoints:

- `GET /health`: service health check.
- `GET /metadata`: selector values for regions, years, splits, stages, teams, players, champions, and model version.
- `POST /predict`: completed draft prediction request.

The API is deployed to Render and started with Uvicorn bound to Render's `$PORT`. CORS is configured for the Cloudflare Pages frontend.

### Inference

Location: `app/inference/`

The predictor loads:

- `models/latest/model.joblib`
- `models/latest/preprocessor.joblib`
- `models/latest/feature_columns.json`
- `models/latest/metadata.json`
- `data/processed/matches_clean.parquet`

For each prediction, the request is converted into a synthetic match row, appended to historical cleaned matches, and passed through the same feature builder used for training. The predictor then selects the trained feature columns, applies categorical typing, and returns a probability from the LightGBM model.

### Data Collection

Location: `app/ingestion/`

Ingestion scripts discover tournament pages, discover GOL.GG match links, scrape match pages, and maintain reference mappings for team aliases and tournament splits.

Manual data collection currently starts from seed tournament match-list URLs. To refresh these manually, get the relevant tournament match links from GOL.GG and update `DEFAULT_REGION_SEED_URLS` in `app/ingestion/reference_discovery.py`.

Primary commands:

```powershell
uv run discover-team-aliases
uv run discover-tournament-splits
uv run discover-match-links
uv run scrape-raw-data
```

Raw scraped data is written as CSV before cleaning.

### Cleaning

Location: `app/processing/`

The cleaning pipeline reads raw scraped match CSVs and writes:

- `data/processed/matches_clean.parquet`
- `data/processed/matches_invalid.csv`

Cleaning responsibilities:

- parse list-like fields such as players and bans
- normalize regions, teams, splits, dates, roles, first-pick side, and winner labels
- derive `blue_win`
- reject duplicate or invalid matches
- quarantine invalid rows with reasons

### Feature Engineering

Location: `app/features/`

The feature builder reads `data/processed/matches_clean.parquet` and writes `data/features/match_features.parquet`.

Features are draft-time-safe. They use only information available before the target match starts.

Implemented feature groups:

- match context: region, year, split, stage, international flag
- draft control: first-pick side and blue first-pick flag
- team identity
- player identity
- champion picks by side and role
- team split form before the match
- champion-role history before the match
- player-champion comfort over the previous two years

Missing history falls back to `0` for counts and `0.5` for rates.

More details in [feature catalog](/docs/feature_catalog.md).

### Model Training

Location: `app/training/`

Training reads `data/features/match_features.parquet`, trains a LightGBM binary classifier, and exports model artifacts to `models/latest/`.

The chronological split is:

- train: matches before June 2025
- validation: June 2025 through December 2025
- test: 2026 matches

Training outputs:

- `model.joblib`
- `preprocessor.joblib`
- `feature_columns.json`
- `metadata.json`

MLflow logs local runs, selected parameters, metrics, and artifacts.

## Storage

The project uses file artifacts instead of a database.

```text
data/raw/          raw scraped CSV files
data/references/   team alias and tournament split mappings
data/processed/    cleaned match parquet and invalid-row report
data/features/     training-ready feature parquet
models/latest/     exported model and metadata artifacts
mlruns/            local MLflow tracking data
```

Only the runtime artifacts required by deployment need to be present in production:

```text
data/processed/matches_clean.parquet
models/latest/model.joblib
models/latest/preprocessor.joblib
models/latest/feature_columns.json
models/latest/metadata.json
```

## Deployment

### Frontend

Host: Cloudflare Pages

```text
Root directory: frontend
Build command: npm run build
Output directory: dist
Environment variable: VITE_API_BASE_URL=<Render API URL>
```

### API

Host: Render Web Service

```text
Build command: pip install uv && uv sync --frozen
Start command: uv run uvicorn app.api.main:app --host 0.0.0.0 --port $PORT
Health check: /health
```

The API service must include the runtime model artifacts and cleaned match history listed in the storage section.

## Local Operation

Install dependencies:

```powershell
uv sync
```

Build cleaned data:

```powershell
uv run build-clean-matches
```

Build features:

```powershell
uv run build-match-features
```

Train model:

```powershell
uv run train-model
```

Start API:

```powershell
uv run serve-api
```

Start frontend:

```powershell
cd frontend
npm install
npm run dev
```

Run tests:

```powershell
uv run pytest
```

## Boundaries

The current architecture does not include:

- scheduled data refresh
- scheduled retraining
- model auto-promotion
- prediction logging
- drift monitoring
- user accounts
- database-backed storage
- a full draft simulator

Those pieces can be added later if the file-first pipeline becomes a real limitation.
