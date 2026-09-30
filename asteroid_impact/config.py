"""Configuration settings for the asteroid impact-risk project.

This module defines the project root and the locations used for raw data,
processed data, models, and generated reports. It also ensures the required
folders exist when the package is imported.
"""

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
DATASET_URI = "hf://datasets/juliensimon/sentry-impact-risk/data/sentry_impact_risk.parquet"
SAMPLE_PATH = RAW_DATA_DIR / "dataset.csv"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for directory in [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

logger.info(f"Project root: {PROJ_ROOT}")
