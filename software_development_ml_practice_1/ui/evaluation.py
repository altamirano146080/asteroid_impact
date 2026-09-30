"""Model Evaluation tab for the Gradio application."""

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from .components import FEATURES_PATH, LABELS_PATH, MODEL_PATH, SCALER_PATH
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_COLUMNS = [
    "palermo_scale_cum",
    "v_infinity_kms",
    "n_potential_impacts",
    "palermo_scale_max",
    "diameter_km",
    "last_observation_jd",
    "absolute_magnitude",
    "torino_scale",
    "year_range_min",
    "year_range_max",
]


def load_trained_model():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None, "❌ Trained model not found. Train a model first."

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler, None
    except Exception as e:
        return None, None, f"❌ Error loading model: {str(e)}"


def plot_predictions_vs_actual() -> plt.Figure:
    model, scaler, err = load_trained_model()
    if err is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, err, ha="center", va="center")
        ax.axis("off")
        return fig

    if not FEATURES_PATH.exists() or not LABELS_PATH.exists():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No features or labels found", ha="center", va="center")
        ax.axis("off")
        return fig

    features = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)["log_impact_probability"]

    scaled = scaler.transform(features)
    pred = model.predict(scaled, verbose=0).ravel()
    residuals = labels.to_numpy() - pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(labels, pred, alpha=0.6, s=30)
    min_val = min(labels.min(), pred.min())
    max_val = max(labels.max(), pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red")
    axes[0].set_title("Predicted vs Actual")
    axes[0].set_xlabel("Actual Log10(Impact Probability)")
    axes[0].set_ylabel("Predicted Log10(Impact Probability)")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(pred, residuals, alpha=0.6, s=25)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title("Residual Plot")
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def get_evaluation_text() -> str:
    model, scaler, err = load_trained_model()
    if err is not None:
        return err

    if not FEATURES_PATH.exists() or not LABELS_PATH.exists():
        return "No features or labels available."

    features = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)["log_impact_probability"]

    scaled = scaler.transform(features)
    pred = model.predict(scaled, verbose=0).ravel()

    mae = mean_absolute_error(labels, pred)
    rmse = np.sqrt(mean_squared_error(labels, pred))
    r2 = r2_score(labels, pred)

    return (
        f"**Model performance**\n\n"
        f"- MAE: {mae:.4f}\n"
        f"- RMSE: {rmse:.4f}\n"
        f"- R²: {r2:.4f}\n\n"
        f"**Dataset info**\n"
        f"- Samples: {len(labels)}\n"
        f"- Features: {scaled.shape[1]}"
    )


def predict_single_asteroid(
    palermo_scale_cum: float,
    v_infinity_kms: float,
    n_potential_impacts: float,
    palermo_scale_max: float,
    diameter_km: float,
    last_observation_jd: float,
    absolute_magnitude: float,
    torino_scale: float,
    year_range_min: float,
    year_range_max: float,
) -> str:
    model, scaler, err = load_trained_model()
    if err:
        return err

    try:
        values = [
            palermo_scale_cum,
            v_infinity_kms,
            n_potential_impacts,
            palermo_scale_max,
            diameter_km,
            last_observation_jd,
            absolute_magnitude,
            torino_scale,
            year_range_min,
            year_range_max,
        ]
        row = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        scaled = scaler.transform(row)
        log_pred = model.predict(scaled, verbose=0)[0][0]
        prob = 10 ** log_pred
        return (
            f"**Prediction result**\n\n"
            f"- Log10(Impact Probability): {log_pred:.4f}\n"
            f"- Impact Probability: {prob:.6e}\n"
            f"- Percentage: {prob * 100:.8f}%"
        )
    except Exception as e:
        return f"❌ Prediction failed: {str(e)}"


def create_evaluation_tab() -> None:
    """Create the Model Evaluation tab."""
    with gr.TabItem("📈 Model Evaluation", id="evaluation"):
        gr.Markdown("## Evaluate the trained model")
        with gr.Row():
            with gr.Column():
                metrics_text = gr.Textbox(
                    label="Performance metrics",
                    lines=12,
                    interactive=False,
                    value=get_evaluation_text(),
                )
                reload_metrics = gr.Button("🔄 Reload metrics")
            with gr.Column():
                eval_plot = gr.Plot(label="Predictions vs Actual")

        reload_metrics.click(
            fn=get_evaluation_text,
            outputs=metrics_text,
        )

        eval_button = gr.Button("📊 Load evaluation plot", variant="primary")
        eval_button.click(
            fn=plot_predictions_vs_actual,
            outputs=eval_plot,
        )

        gr.Markdown("### Predict a new asteroid")
        with gr.Row():
            with gr.Column():
                palermo_cum = gr.Number(label="Cumulative Palermo scale", value=-2.38)
                velocity = gr.Number(label="Velocity at infinity (km/s)", value=8.42)
                potential_impacts = gr.Number(label="Potential impacts", value=4)
                palermo_max = gr.Number(label="Maximum Palermo scale", value=-2.39)
                diameter = gr.Number(label="Diameter (km)", value=0.029)
            with gr.Column():
                observation = gr.Number(label="Last observation (Julian day)", value=2454595.5)
                magnitude = gr.Number(label="Absolute magnitude", value=25.31)
                torino = gr.Number(label="Torino scale", value=0)
                year_min = gr.Number(label="Risk interval start year", value=2027)
                year_max = gr.Number(label="Risk interval end year", value=2122)
                predict_btn = gr.Button("🔮 Predict", variant="primary")

            with gr.Column():
                prediction_box = gr.Textbox(
                    label="Prediction result",
                    lines=10,
                    interactive=False,
                )

        predict_btn.click(
            fn=predict_single_asteroid,
            inputs=[
                palermo_cum,
                velocity,
                potential_impacts,
                palermo_max,
                diameter,
                observation,
                magnitude,
                torino,
                year_min,
                year_max,
            ],
            outputs=prediction_box,
        )