"""Shared project settings and default runtime paths."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
DB_PATH = WAREHOUSE_DIR / "world_cup.duckdb"

CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_FBREF_MANIFEST = CONFIG_DIR / "fbref_sources.json"
DEFAULT_TRANSFERMARKT_MANIFEST = CONFIG_DIR / "transfermarkt_teams.json"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "xgb_poisson_model.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
BEESWARM_PATH = FIGURES_DIR / "xgb_poisson_beeswarm.png"
ANALYTICS_EXPORT_DIR = REPORTS_DIR / "analytics"

DEFAULT_RAW_DIR = RAW_DIR
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_INGESTION_USER_AGENT = "world-cup-probability-ingestion/0.1"
DEFAULT_SOURCE_USER_AGENT = "world-cup-probability/0.1 (+https://localhost)"

DEFAULT_ITERATIONS = 100_000
DEFAULT_BATCH_SIZE = 2_500
DEFAULT_SEED = 42
