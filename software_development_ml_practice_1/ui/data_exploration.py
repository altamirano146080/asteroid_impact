"""Data Exploration tab for the Gradio application."""

import gradio as gr
import numpy as np
import pandas as pd

from .components import (
    download_sample_csv,
    get_dataset_size,
    get_feature_count,
    get_missing_percentage,
    get_sample_data,
    load_dataset,
)
from .plotly_utils import (
    create_scatter_plot,
    create_histogram,
    create_box_plot,
    create_2d_histogram,
    create_line_plot,
    get_numeric_columns,
)


def generate_interactive_plot(df, plot_type, x_col, y_col, bins):
    """Generate interactive plot based on selection."""
    if df is None or df.empty:
        return None
    
    if plot_type == "Scatter Plot":
        return create_scatter_plot(df, x_col, y_col)
    elif plot_type == "2D Histogram":
        return create_2d_histogram(df, x_col, y_col, nbinsx=bins, nbinsy=bins)
    elif plot_type == "Line Plot":
        return create_line_plot(df, x_col, y_col)
    elif plot_type == "Histogram":
        return create_histogram(df, x_col, bins=bins)
    elif plot_type == "Box Plot":
        return create_box_plot(df, x_col)


def create_data_exploration_tab() -> None:
    """Create the Data Exploration tab with interactive visualizations."""
    with gr.TabItem("📊 Data Exploration", id="data-exploration"):
        gr.Markdown("""
        # 🔍 Dataset Analysis & Exploration
        
        Explore the asteroid impact risk dataset interactively. Select variables and chart type to visualize relationships in the data.
        """)
        
        # ==================== KEY METRICS ====================
        gr.Markdown("## 📌 Quick Overview")
        with gr.Row():
            with gr.Column(scale=1, min_width=150):
                gr.Number(
                    label="📦 Total Records",
                    value=get_dataset_size(),
                    precision=0,
                    interactive=False
                )
            with gr.Column(scale=1, min_width=150):
                gr.Number(
                    label="🔢 Features",
                    value=get_feature_count(),
                    precision=0,
                    interactive=False
                )
            with gr.Column(scale=1, min_width=150):
                gr.Number(
                    label="⚠️ Missing (%)",
                    value=get_missing_percentage(),
                    precision=2,
                    interactive=False
                )
        
        # ==================== INTERACTIVE ANALYSIS SECTION ====================
        gr.Markdown("## 📊 Interactive Visualization")
        gr.Markdown("Select variables and chart type to explore relationships. Hover, zoom, and pan to interact with the data!")
        
        df = load_dataset()
        numeric_cols = get_numeric_columns(df) if df is not None else []
        
        if numeric_cols:
            with gr.Row():
                with gr.Column(scale=1):
                    plot_type = gr.Dropdown(
                        choices=["Scatter Plot", "2D Histogram", "Line Plot", "Histogram", "Box Plot"],
                        value="Scatter Plot",
                        label="📊 Chart Type",
                        info="Select type of visualization"
                    )
                
                with gr.Column(scale=1):
                    x_var = gr.Dropdown(
                        choices=numeric_cols,
                        value=numeric_cols[0] if numeric_cols else None,
                        label="📈 X Variable",
                        info="First variable to analyze"
                    )
                
                with gr.Column(scale=1):
                    y_var = gr.Dropdown(
                        choices=numeric_cols,
                        value=numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0],
                        label="📊 Y Variable",
                        info="Second variable to analyze"
                    )
                
                with gr.Column(scale=1):
                    bins_slider = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=20,
                        step=5,
                        label="📏 Bins",
                        info="For histograms"
                    )
            
            # Interactive plot output
            interactive_plot = gr.Plot(label="Interactive Visualization")
            
            # Generate initial plot
            initial_plot = generate_interactive_plot(
                df, 
                "Scatter Plot", 
                numeric_cols[0] if numeric_cols else None,
                numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0],
                20
            )
            
            if initial_plot:
                interactive_plot = gr.Plot(initial_plot)
            
            # Update plot when any control changes
            def update_plot(pt, x, y, b):
                return generate_interactive_plot(df, pt, x, y, b)
            
            plot_type.change(
                fn=update_plot,
                inputs=[plot_type, x_var, y_var, bins_slider],
                outputs=interactive_plot
            )
            
            x_var.change(
                fn=update_plot,
                inputs=[plot_type, x_var, y_var, bins_slider],
                outputs=interactive_plot
            )
            
            y_var.change(
                fn=update_plot,
                inputs=[plot_type, x_var, y_var, bins_slider],
                outputs=interactive_plot
            )
            
            bins_slider.change(
                fn=update_plot,
                inputs=[plot_type, x_var, y_var, bins_slider],
                outputs=interactive_plot
            )
        
        # ==================== DATA SAMPLE ====================
        gr.Markdown("### 📥 Data Sample")
        gr.Markdown("View and download samples of the dataset.")
        
        with gr.Row():
            with gr.Column(scale=4):
                sample_rows = gr.Slider(
                    minimum=5, 
                    maximum=50, 
                    value=10, 
                    step=5, 
                    label="Rows to display",
                    info="Select how many rows to show"
                )
            with gr.Column(scale=1):
                download_btn = gr.Button("⬇️ Download", scale=1, size="lg")
                download_file = gr.File(visible=False)
                download_btn.click(
                    fn=download_sample_csv,
                    inputs=sample_rows,
                    outputs=download_file
                )
        
        sample_table = gr.Dataframe(
            value=get_sample_data(10),
            interactive=False,
            wrap=True,
            label="Dataset Preview"
        )
        
        sample_rows.change(
            fn=get_sample_data,
            inputs=sample_rows,
            outputs=sample_table
        )
