
"""Enhanced Data Exploration tab with interactive visualizations and analysis."""

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple

from .components import (
    download_sample_csv,
    get_dataset_size,
    get_feature_count,
    get_missing_percentage,
    get_sample_data,
    load_dataset,
)


# ================== DISTRIBUTION PLOTS ==================
def plot_distribution() -> plt.Figure:
    """Plot impact probability distribution with enhanced styling."""
    df = load_dataset()
    if df is None or "impact_probability" not in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    axes[0].hist(df["impact_probability"], bins=30, color="#4C78A8", 
                 edgecolor="black", alpha=0.7)
    axes[0].set_title("Impact Probability (Linear Scale)", 
                      fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Impact Probability")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(alpha=0.3, linestyle="--")

    # Log scale
    log_values = np.log10(df["impact_probability"])
    axes[1].hist(log_values, bins=30, color="#F58518", 
                 edgecolor="black", alpha=0.7)
    axes[1].set_title("Impact Probability (Log10 Scale)", 
                      fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Log10(Impact Probability)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.3, linestyle="--")

    fig.suptitle("Target Variable Distribution", fontsize=14, 
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_box_whisker() -> plt.Figure:
    """Plot box and whisker plot for target variable."""
    df = load_dataset()
    if df is None or "impact_probability" not in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale box plot
    axes[0].boxplot(df["impact_probability"].dropna(), vert=True)
    axes[0].set_title("Impact Probability Box Plot (Linear)", 
                      fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Impact Probability")
    axes[0].grid(alpha=0.3, linestyle="--", axis="y")
    
    # Log scale box plot
    log_values = np.log10(df["impact_probability"].dropna())
    axes[1].boxplot(log_values, vert=True)
    axes[1].set_title("Impact Probability Box Plot (Log Scale)", 
                      fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Log10(Impact Probability)")
    axes[1].grid(alpha=0.3, linestyle="--", axis="y")
    
    fig.suptitle("Distribution Analysis", fontsize=14, 
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ================== CORRELATIONS ==================
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
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    
    ax.set_title("Feature Correlation Matrix", fontsize=14, 
                 fontweight="bold", pad=20)
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


def plot_correlation_with_target() -> plt.Figure:
    """Plot correlations with target variable specifically."""
    df = load_dataset()
    if df is None or "impact_probability" not in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        return fig

    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()["impact_probability"].sort_values(ascending=False)
    
    # Remove the target variable itself
    correlations = correlations[correlations.index != "impact_probability"]
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(correlations) * 0.3)))
    colors = ["#54A24B" if x > 0 else "#FF6B6B" for x in correlations.values]
    ax.barh(range(len(correlations)), correlations.values, color=colors, edgecolor="black")
    ax.set_yticks(range(len(correlations)))
    ax.set_yticklabels(correlations.index, fontsize=10)
    ax.set_xlabel("Correlation with Impact Probability", fontsize=11, fontweight="bold")
    ax.set_title("Feature Importance: Correlation with Target", fontsize=12, fontweight="bold")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(alpha=0.3, linestyle="--", axis="x")
    
    plt.tight_layout()
    return fig


# ================== FEATURE DISTRIBUTIONS ==================
def plot_feature_distributions() -> plt.Figure:
    """Plot distributions of all numeric features."""
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
            axes[idx].hist(df[col].dropna(), bins=20, color=color, 
                          edgecolor="black", alpha=0.7)
            axes[idx].set_title(f"{col}", fontsize=11, fontweight="bold")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Frequency")
            axes[idx].grid(alpha=0.3, linestyle="--")

    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    return fig


# ================== ADVANCED STATISTICS ==================
def plot_statistical_summary() -> plt.Figure:
    """Create a visual statistical summary."""
    df = load_dataset()
    if df is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        return fig

    numeric_df = df.select_dtypes(include=[np.number])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Skewness
    skewness = numeric_df.skew().sort_values(ascending=False)
    ax = axes[0, 0]
    colors_skew = ["#FF6B6B" if x > 0 else "#4C78A8" for x in skewness.values]
    ax.barh(range(len(skewness)), skewness.values, color=colors_skew, edgecolor="black")
    ax.set_yticks(range(len(skewness)))
    ax.set_yticklabels(skewness.index, fontsize=9)
    ax.set_xlabel("Skewness", fontsize=10, fontweight="bold")
    ax.set_title("Feature Skewness", fontsize=11, fontweight="bold")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(alpha=0.3, linestyle="--", axis="x")
    
    # Kurtosis
    kurtosis = numeric_df.kurtosis().sort_values(ascending=False)
    ax = axes[0, 1]
    colors_kurt = ["#FF6B6B" if x > 3 else "#4C78A8" for x in kurtosis.values]
    ax.barh(range(len(kurtosis)), kurtosis.values, color=colors_kurt, edgecolor="black")
    ax.set_yticks(range(len(kurtosis)))
    ax.set_yticklabels(kurtosis.index, fontsize=9)
    ax.set_xlabel("Kurtosis", fontsize=10, fontweight="bold")
    ax.set_title("Feature Kurtosis", fontsize=11, fontweight="bold")
    ax.axvline(x=3, color="red", linestyle="--", linewidth=0.8, label="Normal (3)")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--", axis="x")
    
    # Missing data percentage
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    ax = axes[1, 0]
    if len(missing_pct) > 0:
        ax.barh(range(len(missing_pct)), missing_pct.values, color="#FF9D5C", edgecolor="black")
        ax.set_yticks(range(len(missing_pct)))
        ax.set_yticklabels(missing_pct.index, fontsize=9)
        ax.set_xlabel("Missing %", fontsize=10, fontweight="bold")
        ax.set_title("Missing Data by Column", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3, linestyle="--", axis="x")
    else:
        ax.text(0.5, 0.5, "No missing data", ha="center", va="center", fontsize=12)
        ax.axis("off")
    
    # Data types summary
    ax = axes[1, 1]
    dtype_counts = df.dtypes.value_counts()
    colors_dtype = ["#4C78A8", "#F58518", "#54A24B"][:len(dtype_counts)]
    wedges, texts, autotexts = ax.pie(
        dtype_counts.values, 
        labels=[str(x) for x in dtype_counts.index],
        autopct='%1.1f%%',
        colors=colors_dtype,
        startangle=90
    )
    ax.set_title("Data Types Distribution", fontsize=11, fontweight="bold")
    
    fig.suptitle("Statistical Summary", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    return fig


# ================== KEY STATISTICS ==================
def get_comprehensive_statistics() -> str:
    """Get comprehensive statistics as formatted markdown."""
    df = load_dataset()
    if df is None:
        return "No dataset available"
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate statistics
    total_rows = len(df)
    complete_rows = len(df.dropna())
    missing_rows = total_rows - complete_rows
    complete_pct = 100 * complete_rows / total_rows
    
    missing_by_col = df.isna().sum()
    missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    
    stats = f"""
## 📊 Dataset Overview
- **Total Samples:** {total_rows:,}
- **Total Features:** {df.shape[1]}
- **Numeric Features:** {numeric_df.shape[1]}
- **Categorical Features:** {df.select_dtypes(include=['object']).shape[1]}

## ✅ Data Quality
- **Complete Records:** {complete_rows:,} ({complete_pct:.1f}%)
- **Records with Missing Values:** {missing_rows:,} ({100 - complete_pct:.1f}%)
"""
    
    if len(missing_by_col) > 0:
        stats += "\n**Missing Data by Column:**\n"
        for col, count in missing_by_col.items():
            pct = 100 * count / total_rows
            stats += f"- {col}: {count} ({pct:.2f}%)\n"
    else:
        stats += "\n**✓ No missing values detected!**\n"
    
    # Target variable statistics
    if "impact_probability" in df.columns:
        target = df["impact_probability"]
        stats += f"""
## 🎯 Target Variable (Impact Probability)
- **Min:** {target.min():.2e}
- **Max:** {target.max():.2e}
- **Mean:** {target.mean():.2e}
- **Median:** {target.median():.2e}
- **Std Dev:** {target.std():.2e}
- **Skewness:** {target.skew():.4f}
- **Kurtosis:** {target.kurtosis():.4f}
"""
    
    # Feature statistics
    if numeric_df.shape[1] > 0:
        stats += f"\n## 📈 Feature Statistics\n"
        stats += f"- **Mean of Means:** {numeric_df.mean().mean():.4f}\n"
        stats += f"- **Mean of Std Devs:** {numeric_df.std().mean():.4f}\n"
    
    return stats


# ================== INTERACTIVE EXPLORATION ==================
def plot_scatter_feature_vs_target(feature_name: str = None) -> plt.Figure:
    """Create scatter plot of selected feature vs target."""
    df = load_dataset()
    if df is None or "impact_probability" not in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        return fig
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "impact_probability"]
    
    if not numeric_cols or feature_name not in numeric_cols:
        feature_name = numeric_cols[0] if numeric_cols else None
    
    if feature_name is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No features available", ha="center", va="center")
        ax.axis("off")
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[feature_name], df["impact_probability"], alpha=0.5, s=30, color="#4C78A8", edgecolors="black")
    ax.set_xlabel(feature_name, fontsize=11, fontweight="bold")
    ax.set_ylabel("Impact Probability", fontsize=11, fontweight="bold")
    ax.set_title(f"{feature_name} vs Impact Probability", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    return fig


# ================== MAIN TAB ==================
def create_data_exploration_tab() -> None:
    """Create the enhanced Data Exploration tab."""
    with gr.TabItem("📊 Data Exploration", id="data-exploration"):
        gr.Markdown("""
        # 🔍 Dataset Analysis & Exploration
        
        Explore the asteroid impact risk dataset interactively. Visualize distributions, 
        correlations, and understand your data without reading code.
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
        
        # ==================== TAB SECTIONS ====================
        with gr.Tabs():
            # Tab 1: Target Distribution
            with gr.TabItem("🎯 Target Analysis", id="target-analysis"):
                gr.Markdown("### Impact Probability Distribution")
                gr.Markdown("Compare linear and log-scale distributions to understand the target variable.")
                gr.Plot(plot_distribution(), label="Distribution")
                gr.Plot(plot_box_whisker(), label="Box Plot Analysis")
            
            # Tab 2: Feature Analysis
            with gr.TabItem("📈 Features", id="features"):
                gr.Markdown("### Feature Distributions & Statistics")
                gr.Markdown("Explore individual feature distributions to identify patterns and outliers.")
                gr.Plot(plot_feature_distributions(), label="All Features")
                gr.Plot(plot_statistical_summary(), label="Statistical Metrics")
            
            # Tab 3: Correlations
            with gr.TabItem("🔗 Correlations", id="correlations"):
                gr.Markdown("### Feature Relationships")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Full Correlation Matrix**")
                        gr.Plot(plot_correlations(), label="Correlation Matrix")
                    with gr.Column():
                        gr.Markdown("**Correlation with Target**")
                        gr.Plot(plot_correlation_with_target(), label="Target Correlation")
            
            # Tab 4: Feature-Target Relationships
            with gr.TabItem("📊 Feature vs Target", id="feature-target"):
                gr.Markdown("### Interactive Feature Analysis")
                
                df = load_dataset()
                if df is not None:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    numeric_cols = [c for c in numeric_cols if c != "impact_probability"]
                    
                    feature_selector = gr.Dropdown(
                        choices=numeric_cols,
                        value=numeric_cols[0] if numeric_cols else None,
                        label="Select Feature",
                        interactive=True
                    )
                    
                    scatter_plot = gr.Plot(
                        plot_scatter_feature_vs_target(numeric_cols[0] if numeric_cols else None),
                        label="Scatter Plot"
                    )
                    
                    feature_selector.change(
                        fn=plot_scatter_feature_vs_target,
                        inputs=feature_selector,
                        outputs=scatter_plot
                    )
            
            # Tab 5: Statistics
            with gr.TabItem("📋 Statistics", id="statistics"):
                gr.Markdown("### Comprehensive Dataset Summary")
                gr.Markdown(get_comprehensive_statistics())
        
        # ==================== DATA SAMPLE ====================
        gr.Markdown("### 📥 Data Sample")
        gr.Markdown("View and download samples of the raw dataset.")
        
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