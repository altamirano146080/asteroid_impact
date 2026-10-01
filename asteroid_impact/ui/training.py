"""Training Interface tab for the Gradio application."""

import gradio as gr
import matplotlib.pyplot as plt

from asteroid_impact.modeling.train import train_model

from .components import ensure_data_ready


def train_custom_model(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    sample_size: int,
    progress=gr.Progress(),
):
    try:
        if not ensure_data_ready():
            return None, {}, "Data is unavailable. Check the dataset download and try again."

        result = train_model(
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            sample_size=int(sample_size),
            progress_callback=lambda value, description: progress(value, desc=description),
        )

        history = result["history"]
        metrics_dict = result["metrics"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history["loss"], label="Train loss", linewidth=2)
        axes[0].plot(history["val_loss"], label="Validation loss", linewidth=2)
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(history["mae"], label="Train MAE", linewidth=2)
        axes[1].plot(history["val_mae"], label="Validation MAE", linewidth=2)
        axes[1].set_title("Training MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        plt.tight_layout()

        return fig, metrics_dict, "Training completed successfully. The shared model artifacts were updated."

    except Exception as e:
        return None, {}, f"Training failed: {str(e)}"


def create_training_tab() -> None:
    """Create the Training tab."""
    with gr.TabItem("Training Interface", id="training"):
        gr.Markdown("## Train the model")
        with gr.Row():
            with gr.Column():
                epochs = gr.Slider(10, 200, 50, step=10, label="Epochs")
                batch_size = gr.Slider(4, 128, 32, step=4, label="Batch Size")
                learning_rate = gr.Number(label="Learning Rate", value=0.001, precision=6)
                sample_size = gr.Slider(0, 1000, 0, step=50, label="Sample Size (0 = all)")
                train_btn = gr.Button("Start training", variant="primary")

            with gr.Column():
                status_box = gr.Textbox(label="Status", interactive=False)
                metrics_box = gr.JSON(label="Model metrics")
                training_plot = gr.Plot(label="Training history")

        train_btn.click(
            fn=train_custom_model,
            inputs=[epochs, batch_size, learning_rate, sample_size],
            outputs=[training_plot, metrics_box, status_box],
        )