# LoL Game Prediction

## Project Overview

LoL Game Prediction is a local end-to-end machine learning project for predicting the winner of a professional League of Legends match after draft is complete and before the game starts.

The project uses GOL.GG match data, cleans and normalizes it, builds draft-time-safe features, trains a LightGBM model, logs runs to MLflow, serves predictions through FastAPI, and exposes a small React frontend for submitting completed drafts.

Try it here: https://lol-game-prediction.pages.dev/

## Architecture Diagram

![Architecture Diagram](/assets/architecure_diagram.png)

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **ML:** pandas, scikit-learn, LightGBM, MLflow
- **Data:** CSV and Parquet file artifacts
- **Frontend:** Vite, React, TypeScript
- **Deployment:** Render for the API, Cloudflare Pages for the frontend
- **Tooling:** uv, pytest, npm

## Features

- Predict the likely winner of a professional League of Legends match after both teams have completed draft.
- Enter match context such as region, year, split, stage, and first-pick side.
- Select blue-side and red-side teams, players, and champions from options loaded from the model API.
- View blue-side win probability and predicted winner.

## Implemented Components

More details in [architecture](/docs/architecture.md).

### Data Collection

GOL.GG ingestion scripts discover match links, scrape raw match pages, and maintain reference mappings for team aliases and tournament splits. Raw scraped rows are saved as CSV files before downstream processing.

### Cleaning

The cleaning pipeline normalizes teams, regions, splits, dates, players, champions, bans, first-pick side, and winner labels. Invalid rows are written to `data/processed/matches_invalid.csv` instead of silently entering the training data.

### Feature Engineering

The feature builder creates draft-time-safe model inputs from cleaned matches. It includes match context, team/player/champion identity, team split form, champion-role history, and player-champion comfort without using future match data.

### Model Training

Training uses LightGBM with chronological train, validation, and test splits. The training command exports the model, preprocessing metadata, feature list, metrics, and run metadata under `models/latest/`.

### Experiment Tracking

MLflow logs local training runs, selected parameters, evaluation metrics, and model artifacts so experiments can be inspected after training.

### Prediction API

The FastAPI service exposes `GET /health`, `GET /metadata`, and `POST /predict`. It loads the exported model artifacts, builds the request-time feature row, and returns prediction results with model metadata.

The API is deployed as a Render web service using the exported `models/latest/` artifacts and `data/processed/matches_clean.parquet`. In production it is started with `uvicorn` bound to Render's `$PORT`, and CORS is configured for the Cloudflare Pages frontend.

### Frontend

The Vite React frontend loads selector metadata from the API, lets users submit completed draft scenarios, and displays prediction results in the browser.

The frontend is deployed on Cloudflare Pages. Its production build uses `VITE_API_BASE_URL` to call the Render API directly, while local development falls back to the Vite `/api` proxy.

### Tests

The test suite covers scraping helpers, cleaning, feature generation, training artifacts, inference behavior, and API responses.

## Available Commands

Install dependencies:

```powershell
uv sync
```

Discover and scrape data:

```powershell
uv run discover-team-aliases
uv run discover-tournament-splits
uv run discover-match-links
uv run scrape-raw-data
```

Build datasets:

```powershell
uv run build-clean-matches
uv run build-match-features
```

Train and serve:

```powershell
uv run train-model
uv run serve-api
```

Run tests:

```powershell
uv run pytest
```

## Folder Structure

```text
app/
  api/          FastAPI prediction service.
  features/     Draft-time-safe feature engineering.
  inference/    Request schemas, feature row building, and prediction helpers.
  ingestion/    GOL.GG scraping and reference discovery.
  processing/   Raw-to-clean match processing.
  training/     LightGBM training and MLflow logging.
data/
  processed/    Cleaned match parquet and invalid-row report.
docs/           Architecture and feature catalog notes.
frontend/       Vite React prediction UI.
models/         Exported local model artifacts.
notebooks/      Exploratory analysis and data experiments.
tests/          Automated regression tests.
```

## Run the Project Yourself

Install Python dependencies:

```powershell
uv sync
```

Build the cleaned match dataset:

```powershell
uv run build-clean-matches
```

Build the feature dataset:

```powershell
uv run build-match-features
```

Start local MLflow in one shell:

```powershell
uv run mlflow server
```

Train and log the model from another shell:

```powershell
uv run train-model
```

Inspect local MLflow runs at `http://127.0.0.1:5000`.

Start the API:

```powershell
uv run serve-api
```

Run the frontend:

```powershell
cd frontend
npm install
npm run dev
```

Run the core checks:

```powershell
uv run pytest
```

## Out of Scope

- Scheduled ingestion or retraining.
- Remote MLflow tracking, deployment, or model promotion.
- Drift dashboards and prediction monitoring UI.
- Full pick/ban simulator.
- User accounts or saved predictions.
- Complex composition features and model explainability.
