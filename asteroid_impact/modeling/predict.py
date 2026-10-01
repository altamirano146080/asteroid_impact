"""Prediction pipeline for the trained impact-probability model.

This module loads the saved model and scaler, transforms new feature values,
and writes impact probability predictions to disk.
"""

from pathlib import Path

import joblib
import pandas as pd
import tensorflow as tf
import typer
from loguru import logger

from asteroid_impact.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)

app = typer.Typer()

FEATURES_PATH = PROCESSED_DATA_DIR / "features.csv"
MODEL_PATH = MODELS_DIR / "impact_probability_model.keras"
SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"
PREDICTIONS_PATH = PROCESSED_DATA_DIR / "predictions.csv"


def predict(
    features_path: Path = FEATURES_PATH,
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
) -> pd.DataFrame:
    """Generate impact probability predictions.

    The trained model and feature scaler are loaded from disk. The input
    features are scaled before being passed to the model.

    Parameters
    ----------
    features_path : Path, default=FEATURES_PATH
        Path to the processed feature data.
    model_path : Path, default=MODEL_PATH
        Path to the trained Keras model.
    scaler_path : Path, default=SCALER_PATH
        Path to the saved feature scaler.
    predictions_path : Path, default=PREDICTIONS_PATH
        Path where predictions will be saved.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing logarithmic and original-scale predictions.
    """

    features = pd.read_csv(features_path)

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    features_scaled = scaler.transform(features)

    log_predictions = model.predict(
        features_scaled,
        verbose=0,
    ).ravel()

    predictions = pd.DataFrame(
        {
            "log_impact_probability_prediction": log_predictions,
            "impact_probability_prediction": 10**log_predictions,
        }
    )

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)

    logger.success(f"Predictions saved to {predictions_path}")

    return predictions


@app.command()
def main():
    """Generate predictions for the processed features.

    Returns
    -------
    None
        Executes the prediction pipeline.
    """

    predict()


if __name__ == "__main__":
    app()
