"""Main Gradio application entry point."""

import gradio as gr

from software_development_ml_practice_1.ui.data_exploration import create_data_exploration_tab
from software_development_ml_practice_1.ui.training import create_training_tab
from software_development_ml_practice_1.ui.evaluation import create_evaluation_tab


def create_interface() -> gr.Blocks:
    """Create the main Gradio interface with all tabs."""
    with gr.Blocks(title="Asteroid Impact Predictor", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🌍 Asteroid Impact Risk Dashboard")
        gr.Markdown("Interactive data exploration, training, and evaluation for the asteroid impact prediction model.")

        with gr.Tabs():
            create_data_exploration_tab()
            create_training_tab()
            create_evaluation_tab()

    return demo


def launch_app():
    """Launch the Gradio application."""
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)


if __name__ == "__main__":
    launch_app()