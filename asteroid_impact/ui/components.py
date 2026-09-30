"""Shared utility functions for all UI components."""

from pathlib import Path
import pandas as pd
import numpy as np
import tempfile

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RAW_DATASET_PATH = RAW_DIR / "dataset.csv"
PROCESSED_DATASET_PATH = PROCESSED_DIR / "dataset.csv"
FEATURES_PATH = PROCESSED_DIR / "features.csv"
LABELS_PATH = PROCESSED_DIR / "labels.csv"

MODEL_PATH = MODELS_DIR / "impact_probability_model.keras"
SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"
TRAIN_HISTORY_PATH = REPORTS_DIR / "training_history.csv"
METRICS_PATH = REPORTS_DIR / "model_metrics.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "predictions.csv"


def ensure_data_ready() -> bool:
    """Ensure the dataset, features, and labels exist, downloading when needed."""
    try:
        from asteroid_impact.dataset import download_dataset, prepare_dataset
        from asteroid_impact.plots import create_features

        if not RAW_DATASET_PATH.exists():
            download_dataset(output_path=RAW_DATASET_PATH)

        if not PROCESSED_DATASET_PATH.exists() and RAW_DATASET_PATH.exists():
            prepare_dataset(
                input_path=RAW_DATASET_PATH,
                output_path=PROCESSED_DATASET_PATH,
            )

        if not FEATURES_PATH.exists() or not LABELS_PATH.exists():
            if PROCESSED_DATASET_PATH.exists():
                create_features(
                    input_path=PROCESSED_DATASET_PATH,
                    features_path=FEATURES_PATH,
                    labels_path=LABELS_PATH,
                )
            else:
                return False
        return True
    except Exception:
        return False


def load_dataset() -> pd.DataFrame | None:
    """Load the processed dataset if available."""
    ensure_data_ready()
    if PROCESSED_DATASET_PATH.exists():
        return pd.read_csv(PROCESSED_DATASET_PATH)
    if RAW_DATASET_PATH.exists():
        return pd.read_csv(RAW_DATASET_PATH)
    return None


def get_dataset_size() -> int:
    df = load_dataset()
    if df is None:
        return 0
    return len(df)


def get_feature_count() -> int:
    df = load_dataset()
    if df is None:
        return 0
    return df.shape[1]


def get_missing_percentage() -> float:
    df = load_dataset()
    if df is None:
        return 0.0
    total_cells = df.size
    missing_cells = int(df.isna().sum().sum())
    return round((missing_cells / total_cells) * 100, 2) if total_cells else 0.0


def get_sample_data(n_rows: int = 10) -> pd.DataFrame:
    df = load_dataset()
    if df is None:
        return pd.DataFrame()
    return df.head(int(n_rows))


def download_sample_csv(rows: int = 10):
    df = load_dataset()
    if df is None:
        return None
    sample = df.head(int(rows))
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        sample.to_csv(tmp.name, index=False)
        return tmp.name