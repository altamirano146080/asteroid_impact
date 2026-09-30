import os
import socket

import gradio as gr
from asteroid_impact.ui.data_exploration import create_data_exploration_tab
from asteroid_impact.ui.training import create_training_tab
from asteroid_impact.ui.evaluation import create_evaluation_tab


def create_interface() -> gr.Blocks:
    """Create the main Gradio interface with all tabs."""
    with gr.Blocks(title="Asteroid Impact Predictor") as demo:
        gr.Markdown("# 🌍 Asteroid Impact Risk Dashboard")
        gr.Markdown("Interactive data exploration, training, and evaluation for the asteroid impact prediction model.")

        with gr.Tabs():
            create_data_exploration_tab()
            create_training_tab()
            create_evaluation_tab()

    return demo


def _get_server_port() -> int:
    """Return the configured port or an available port near the default."""
    configured_port = os.getenv("GRADIO_SERVER_PORT")
    if configured_port:
        return int(configured_port)

    default_port = 7860
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_socket:
        try:
            port_socket.bind(("127.0.0.1", default_port))
        except OSError:
            port_socket.bind(("127.0.0.1", 0))
            return int(port_socket.getsockname()[1])
    return default_port


def launch_app():
    
    demo = create_interface()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=_get_server_port(),
        share=os.getenv("GRADIO_SHARE", "true").lower() == "true",
        theme=gr.themes.Soft(),
        show_error=True,
    )

if __name__ == "__main__":
    launch_app()