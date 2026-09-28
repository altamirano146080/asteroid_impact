"""Training Interface tab for the Gradio application."""

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from .components import FEATURES_PATH, LABELS_PATH, MODEL_PATH, SCALER_PATH, TRAIN_HISTORY_PATH, MODELS_DIR, REPORTS_DIR


def build_model(input_shape: int, learning_rate: float = 0.001) -> tf.keras.Model:
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


def train_custom_model(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    sample_size: int,
    progress=gr.Progress(),
):
    try:
        if not FEATURES_PATH.exists() or not LABELS_PATH.exists():
            return (
                None,
                {},
                "❌ Features or labels not found. Please make sure the dataset was prepared.",
            )

        features = pd.read_csv(FEATURES_PATH)
        labels = pd.read_csv(LABELS_PATH)["log_impact_probability"]

        if sample_size > 0 and sample_size < len(features):
            idx = np.random.choice(len(features), int(sample_size), replace=False)
            features = features.iloc[idx]
            labels = labels.iloc[idx]

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = build_model(X_train_scaled.shape[1], learning_rate)
        progress(0, desc="Training model...")
        
        history = model.fit(
            X_train_scaled,
            y_train,
            validation_split=0.2,
            epochs=int(epochs),
            batch_size=int(batch_size),
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True,
                )
            ],
        )

        predictions = model.predict(X_test_scaled, verbose=0).ravel()
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        history_df = pd.DataFrame(history.history)
        history_df.to_csv(TRAIN_HISTORY_PATH, index=False)

        metrics_dict = {
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "R²": round(float(r2), 4),
            "Test samples": int(len(y_test)),
            "Epochs run": int(len(history.history["loss"])),
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history.history["loss"], label="Train loss", linewidth=2)
        axes[0].plot(history.history["val_loss"], label="Validation loss", linewidth=2)
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(history.history["mae"], label="Train MAE", linewidth=2)
        axes[1].plot(history.history["val_mae"], label="Validation MAE", linewidth=2)
        axes[1].set_title("Training MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        plt.tight_layout()

        progress(1, desc="Training complete")

        return fig, metrics_dict, f"✅ Training completed successfully. Model saved to {MODEL_PATH}"

    except Exception as e:
        return None, {}, f"❌ Training failed: {str(e)}"


def create_training_tab() -> None:
    """Create the Training tab."""
    with gr.TabItem("🚀 Training Interface", id="training"):
        gr.Markdown("## Train the model")
        with gr.Row():
            with gr.Column():
                epochs = gr.Slider(10, 200, 50, step=10, label="Epochs")
                batch_size = gr.Slider(4, 128, 32, step=4, label="Batch Size")
                learning_rate = gr.Number(label="Learning Rate", value=0.001, precision=6)
                sample_size = gr.Slider(0, 1000, 0, step=50, label="Sample Size (0 = all)")
                train_btn = gr.Button("🎯 Start training", variant="primary")

            with gr.Column():
                status_box = gr.Textbox(label="Status", interactive=False)
                metrics_box = gr.JSON(label="Model metrics")
                training_plot = gr.Plot(label="Training history")

        train_btn.click(
            fn=train_custom_model,
            inputs=[epochs, batch_size, learning_rate, sample_size],
            outputs=[training_plot, metrics_box, status_box],
        )