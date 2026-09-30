"""Model training pipeline for the impact-probability regressor.

This module builds the neural network, fits it on scaled features, evaluates
performance, and saves the trained model along with training metrics.
"""

from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import typer
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from software_development_ml_practice_1.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)

app = typer.Typer()

FEATURES_PATH = PROCESSED_DATA_DIR / "features.csv"
LABELS_PATH = PROCESSED_DATA_DIR / "labels.csv"

MODEL_PATH = MODELS_DIR / "impact_probability_model.keras"
SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"
METRICS_PATH = REPORTS_DIR / "model_metrics.csv"


def build_model(input_shape: int, learning_rate: float = 0.001) -> tf.keras.Model:
    """Create the neural network architecture.

    Parameters
    ----------
    input_shape : int
        Number of input features.
    learning_rate : float, default=0.001
        Learning rate for the optimizer.

    Returns
    -------
    tensorflow.keras.Model
        A compiled neural network model.
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_shape,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model


def train_model(
    features_path: Path = FEATURES_PATH,
    labels_path: Path = LABELS_PATH,
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    metrics_path: Path = METRICS_PATH,
    history_path: Path = REPORTS_DIR / "training_history.csv",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    sample_size: int = 0,
    progress_callback: Callable[[float, str], Any] | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and save the neural network model.

    The function splits the data, scales the features, trains the model,
    calculates evaluation metrics, and saves the model and scaler.

    Parameters
    ----------
    features_path : Path, default=FEATURES_PATH
        Path to the input feature data.
    labels_path : Path, default=LABELS_PATH
        Path to the target labels.
    model_path : Path, default=MODEL_PATH
        Path where the trained model will be saved.
    scaler_path : Path, default=SCALER_PATH
        Path where the feature scaler will be saved.
    metrics_path : Path, default=METRICS_PATH
        Path where the evaluation metrics will be saved.

    Returns
    -------
    dict
        Metrics and the Keras training history for interactive frontends.
    """
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)["log_impact_probability"]

    if sample_size > 0 and sample_size < len(features):
        sample_indices = np.random.default_rng(42).choice(
            len(features), int(sample_size), replace=False
        )
        features = features.iloc[sample_indices]
        labels = labels.iloc[sample_indices]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = build_model(
        input_shape=x_train_scaled.shape[1],
        learning_rate=learning_rate,
    )

    if progress_callback is not None:
        progress_callback(0.0, "Training model...")

    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback is not None:
                progress_callback((epoch + 1) / epochs, f"Epoch {epoch + 1}/{epochs}")

    history = model.fit(
        x_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=int(epochs),
        batch_size=int(batch_size),
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
            ),
            ProgressCallback(),
        ],
    )

    predictions = model.predict(
        x_test_scaled,
        verbose=0,
    ).ravel()

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    metrics = pd.DataFrame(
        [
            {
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        ]
    )

    metrics.to_csv(metrics_path, index=False)

    history_dataframe = pd.DataFrame(history.history)
    history_dataframe.to_csv(history_path, index=False)

    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R2: {r2:.4f}")

    logger.success(f"Model saved to {model_path}")
    logger.success(f"Scaler saved to {scaler_path}")
    logger.success(f"Metrics saved to {metrics_path}")

    if progress_callback is not None:
        progress_callback(1.0, "Training complete")

    return {
        "metrics": {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "Test samples": int(len(y_test)),
            "Epochs run": int(len(history.history["loss"])),
        },
        "history": history.history,
    }


@app.command()
def main():
    """Train and evaluate the model.

    Returns
    -------
    None
        Executes the training pipeline.
    """

    train_model()


if __name__ == "__main__":
    app()
