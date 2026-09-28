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


def plot_distribution() -> plt.Figure:
    """Plot impact probability distribution."""
    df = load_dataset()
    if df is None or "impact_probability" not in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    axes[0].hist(df["impact_probability"], bins=30, color="#4C78A8", edgecolor="black", alpha=0.7)
    axes[0].set_title("Impact Probability (Linear Scale)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Impact Probability")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(alpha=0.3, linestyle="--")

    # Log scale
    log_values = np.log10(df["impact_probability"])
    axes[1].hist(log_values, bins=30, color="#F58518", edgecolor="black", alpha=0.7)
    axes[1].set_title("Impact Probability (Log10 Scale)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Log10(Impact Probability)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.3, linestyle="--")

    fig.suptitle("Target Variable Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_correlations() -> plt.Figure:
    """Plot correlation heatmap of numeric features."""
    df = load_dataset()
    if df is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        return fig

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "Not enough numeric columns", ha="center", va="center")
        ax.axis("off")
        return fig

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)

    # Add correlation values
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            text = ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center",
                color="white" if abs(corr.iloc[i, j]) > 0.5 else "black",
                fontsize=8, fontweight="bold"
            )

    plt.colorbar(im, ax=ax, label="Correlation Coefficient")
    plt.tight_layout()
    return fig


def plot_feature_distributions() -> plt.Figure:
    """Plot distributions of all features."""
    df = load_dataset()
    if df is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        return fig

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "impact_probability" in numeric_cols:
        numeric_cols.remove("impact_probability")
    if not numeric_cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No feature columns available", ha="center", va="center")
        ax.axis("off")
        return fig

    n_cols = 2
    n_rows = (len(numeric_cols) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.array(axes).ravel()

    colors = ["#4C78A8", "#F58518", "#54A24B", "#FF9D5C", "#59A14F", "#8C564B"]

    for idx, col in enumerate(numeric_cols):
        if idx < len(axes):
            color = colors[idx % len(colors)]
            axes[idx].hist(df[col].dropna(), bins=20, color=color, edgecolor="black", alpha=0.7)
            axes[idx].set_title(f"{col}", fontsize=11, fontweight="bold")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Frequency")
            axes[idx].grid(alpha=0.3, linestyle="--")

    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    return fig


def get_key_statistics() -> str:
    """Get key statistics as formatted text."""
    df = load_dataset()
    if df is None:
        return "No dataset available"
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    stats = f"""
**Dataset Dimensions:** {df.shape[0]} samples × {df.shape[1]} features

**Data Quality:**
- Complete records: {len(df.dropna())} ({100 * len(df.dropna()) / len(df):.1f}%)
- Records with missing values: {len(df) - len(df.dropna())} ({100 * (1 - len(df.dropna()) / len(df)):.1f}%)

**Features:** {numeric_df.shape[1]} numeric columns

**Target Variable (Impact Probability):**
- Min: {df['impact_probability'].min():.2e}
- Max: {df['impact_probability'].max():.2e}
- Mean: {df['impact_probability'].mean():.2e}
    """
    return stats


def create_data_exploration_tab() -> None:
    """Create the Data Exploration tab."""
    with gr.TabItem("📊 Data Exploration", id="data-exploration"):
        gr.Markdown("""
        ## Dataset Overview
        Explore the asteroid impact risk dataset with interactive visualizations and statistics.
        """)
        
        # ==================== KEY METRICS ====================
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
        
        # ==================== VISUALIZATIONS ====================
        with gr.Row():
            with gr.Column():
                gr.Plot(plot_distribution(), label="Impact Probability Distribution")
            with gr.Column():
                gr.Plot(plot_correlations(), label="Feature Correlations")
        
        gr.Plot(plot_feature_distributions(), label="Feature Distributions")
        
        # ==================== KEY STATISTICS ====================
        gr.Markdown("### Dataset Summary")
        gr.Textbox(
            label="Key Statistics",
            value=get_key_statistics(),
            lines=10,
            interactive=False
        )
        
        # ==================== DATA SAMPLE ====================
        gr.Markdown("### Data Sample")
        with gr.Row():
            with gr.Column(scale=4):
                sample_rows = gr.Slider(5, 50, 10, step=5, label="Rows to display")
            with gr.Column(scale=1):
                gr.Button("⬇️ Download Sample", scale=1).click(
                    fn=download_sample_csv,
                    inputs=sample_rows,
                    outputs=gr.File()
                )
        
        sample_table = gr.Dataframe(
            value=get_sample_data(10),
            interactive=False,
            wrap=True
        )
        
        sample_rows.change(
            fn=get_sample_data,
            inputs=sample_rows,
            outputs=sample_table
        )