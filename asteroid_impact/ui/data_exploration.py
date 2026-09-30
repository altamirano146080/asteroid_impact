"""Data Exploration tab for the Gradio application."""

import gradio as gr
import matplotlib.pyplot as plt
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
    get_numeric_columns,
)


def plot_distribution() -> plt.Figure:
    """Plot impact probability distribution."""
    df = load_dataset()
    if df is None or "impact_probability" not in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Linear scale
    axes[0].hist(df["impact_probability"], bins=30, color="#4C78A8", edgecolor="black", alpha=0.7)
    axes[0].set_title("Impact Probability (Linear Scale)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Impact Probability")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(alpha=0.3, linestyle="--")

    # Log scale
    log_values = np.log10(df["impact_probability"])
    axes[1].hist(log_values, bins=30, color="#F58518", edgecolor="black", alpha=0.7)
    axes[1].set_title("Impact Probability (Log10 Scale)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Log10(Impact Probability)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.3, linestyle="--")

    fig.suptitle("Target Variable Distribution", fontsize=12, fontweight="bold", y=1.00)
    plt.tight_layout()
    return fig


def plot_correlations() -> plt.Figure:
    """Plot correlation heatmap of numeric features."""
    df = load_dataset()
    if df is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        return fig

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Not enough numeric columns", ha="center", va="center")
        ax.axis("off")
        return fig

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    
    ax.set_title("Feature Correlation Matrix", fontsize=11, fontweight="bold", pad=15)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)

    # Add correlation values
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            text = ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center",
                color="white" if abs(corr.iloc[i, j]) > 0.5 else "black",
                fontsize=7, fontweight="bold"
            )

    plt.colorbar(im, ax=ax, label="Correlation", shrink=0.8)
    plt.tight_layout()
    return fig


def plot_feature_distributions() -> plt.Figure:
    """Plot distributions of all features."""
    df = load_dataset()
    if df is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        return fig

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "impact_probability" in numeric_cols:
        numeric_cols.remove("impact_probability")
    if not numeric_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No feature columns available", ha="center", va="center")
        ax.axis("off")
        return fig

    n_cols = 2
    n_rows = (len(numeric_cols) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = np.array(axes).ravel()

    colors = ["#4C78A8", "#F58518", "#54A24B", "#FF9D5C", "#59A14F", "#8C564B"]

    for idx, col in enumerate(numeric_cols):
        if idx < len(axes):
            color = colors[idx % len(colors)]
            axes[idx].hist(df[col].dropna(), bins=20, color=color, edgecolor="black", alpha=0.7)
            axes[idx].set_title(f"{col}", fontsize=10, fontweight="bold")
            axes[idx].set_xlabel(col, fontsize=8)
            axes[idx].set_ylabel("Frequency", fontsize=8)
            axes[idx].grid(alpha=0.3, linestyle="--")
            axes[idx].tick_params(labelsize=8)

    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Feature Distributions", fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout()
    return fig


def generate_interactive_plot(df, plot_type, x_col, y_col, bins):
    """Generate interactive plot based on selection."""
    if df is None or df.empty:
        return None
    
    if plot_type == "Scatter Plot":
        return create_scatter_plot(df, x_col, y_col)
    elif plot_type == "2D Histogram":
        return create_2d_histogram(df, x_col, y_col, nbinsx=bins, nbinsy=bins)
    elif plot_type == "Histogram":
        return create_histogram(df, x_col, bins=bins)
    elif plot_type == "Box Plot":
        return create_box_plot(df, x_col)


def create_data_exploration_tab() -> None:
    """Create the Data Exploration tab with interactive visualizations."""
    with gr.TabItem("📊 Data Exploration", id="data-exploration"):
        gr.Markdown("""
        # 🔍 Dataset Analysis & Exploration
        
        Explore the asteroid impact risk dataset interactively.
        """)
        
        # ==================== KEY METRICS ====================
        gr.Markdown("## 📌 Quick Overview")
        with gr.Row():
            with gr.Column(scale=1, min_width=120):
                gr.Number(
                    label="📦 Total Records",
                    value=get_dataset_size(),
                    precision=0,
                    interactive=False
                )
            with gr.Column(scale=1, min_width=120):
                gr.Number(
                    label="🔢 Features",
                    value=get_feature_count(),
                    precision=0,
                    interactive=False
                )
            with gr.Column(scale=1, min_width=120):
                gr.Number(
                    label="⚠️ Missing (%)",
                    value=get_missing_percentage(),
                    precision=2,
                    interactive=False
                )
        
        # ==================== STATIC VISUALIZATIONS ====================
        # Row 1: Distribution + Correlations
        with gr.Row():
            with gr.Column():
                gr.Plot(plot_distribution(), label="Impact Probability Distribution")
            with gr.Column():
                gr.Plot(plot_correlations(), label="Feature Correlations")
        
        # Row 2: Feature Distributions
        gr.Plot(plot_feature_distributions(), label="Feature Distributions")
        
        # ==================== INTERACTIVE ANALYSIS SECTION ====================
        gr.Markdown("## 🎯 Interactive Custom Analysis")
        gr.Markdown("Customize visualizations by selecting variables and chart type below.")
        
        df = load_dataset()
        numeric_cols = get_numeric_columns(df) if df is not None else []
        
        if numeric_cols:
            # Compact controls row
            with gr.Row():
                with gr.Column(scale=1, min_width=100):
                    plot_type = gr.Dropdown(
                        choices=["Scatter Plot", "2D Histogram", "Histogram", "Box Plot"],
                        value="Scatter Plot",
                        label="Chart Type",
                        scale=1
                    )
                
                with gr.Column(scale=1, min_width=80):
                    x_var = gr.Dropdown(
                        choices=numeric_cols,
                        value=numeric_cols[0] if numeric_cols else None,
                        label="X",
                        scale=1
                    )
                
                with gr.Column(scale=1, min_width=80):
                    y_var = gr.Dropdown(
                        choices=numeric_cols,
                        value=numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0],
                        label="Y",
                        scale=1
                    )
                
                with gr.Column(scale=1, min_width=80):
                    bins_slider = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=20,
                        step=5,
                        label="Bins",
                        scale=1
                    )
            
            # 4 Interactive plots in 2x2 grid
            with gr.Row():
                with gr.Column():
                    plot1 = gr.Plot(label="Scatter Plot")
                with gr.Column():
                    plot2 = gr.Plot(label="2D Histogram")
            
            with gr.Row():
                with gr.Column():
                    plot3 = gr.Plot(label="Histogram")
                with gr.Column():
                    plot4 = gr.Plot(label="Box Plot")
            
            # Generate initial plots
            def update_all_plots(x, y, b):
                scatter = generate_interactive_plot(df, "Scatter Plot", x, y, b)
                histogram2d = generate_interactive_plot(df, "2D Histogram", x, y, b)
                histogram = generate_interactive_plot(df, "Histogram", x, y, b)
                boxplot = generate_interactive_plot(df, "Box Plot", x, y, b)
                return scatter, histogram2d, histogram, boxplot
            
            # Initial plots
            initial_x = numeric_cols[0] if numeric_cols else None
            initial_y = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            initial_plots = update_all_plots(initial_x, initial_y, 20)
            
            # Update on control change
            def on_change(x, y, b):
                return update_all_plots(x, y, b)
            
            for control in (plot_type, x_var, y_var, bins_slider):
                control.change(
                fn=on_change,
                inputs=[x_var, y_var, bins_slider],
                outputs=[plot1, plot2, plot3, plot4]
                )
        
        # ==================== DATA SAMPLE ====================
        gr.Markdown("### 📥 Data Sample")
        
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
                download_file = gr.File(
                    label="CSV ready to download",
                    interactive=False,
                )
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
