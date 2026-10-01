"""Utility functions for creating interactive Plotly visualizations."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = None,
    color_by: str = None
) -> go.Figure:
    """Create an interactive scatter plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    x_col : str
        Column name for X axis
    y_col : str
        Column name for Y axis
    title : str, optional
        Plot title
    color_by : str, optional
        Column to color points by
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    if x_col not in df.columns or y_col not in df.columns:
        return go.Figure().add_annotation(text="Columns not found")
    
    df_clean = df[[x_col, y_col]].dropna()
    
    if color_by and color_by in df.columns:
        fig = px.scatter(
            df_clean,
            x=x_col,
            y=y_col,
            color=color_by,
            title=title or f"{x_col} vs {y_col}",
            hover_data={x_col: ":.4f", y_col: ":.4f"},
            labels={x_col: x_col, y_col: y_col}
        )
    else:
        fig = px.scatter(
            df_clean,
            x=x_col,
            y=y_col,
            title=title or f"{x_col} vs {y_col}",
            hover_data={x_col: ":.4f", y_col: ":.4f"},
            labels={x_col: x_col, y_col: y_col}
        )
        fig.update_traces(marker=dict(color="#4C78A8", size=6))
    
    fig.update_layout(
        hovermode="closest",
        height=500,
        template="plotly_white",
        font=dict(size=11),
        showlegend=True if color_by else False
    )
    
    return fig


def create_histogram(
    df: pd.DataFrame,
    col: str,
    bins: int = 30,
    title: str = None
) -> go.Figure:
    """Create an interactive histogram.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    col : str
        Column name
    bins : int
        Number of bins
    title : str, optional
        Plot title
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    if col not in df.columns:
        return go.Figure().add_annotation(text="Column not found")
    
    data = df[col].dropna()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=bins,
        marker_color="#4C78A8",
        marker_line_color="black",
        marker_line_width=1,
        opacity=0.7,
        hovertemplate="<b>Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>"
    ))
    
    fig.update_layout(
        title=title or f"Distribution of {col}",
        xaxis_title=col,
        yaxis_title="Frequency",
        height=500,
        template="plotly_white",
        font=dict(size=11),
        hovermode="x unified"
    )
    
    return fig


def create_box_plot(
    df: pd.DataFrame,
    col: str,
    title: str = None
) -> go.Figure:
    """Create an interactive box plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    col : str
        Column name
    title : str, optional
        Plot title
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    if col not in df.columns:
        return go.Figure().add_annotation(text="Column not found")
    
    data = df[col].dropna()
    
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=data,
        name=col,
        marker_color="#4C78A8",
        boxmean="sd"
    ))
    
    fig.update_layout(
        title=title or f"Box Plot: {col}",
        yaxis_title=col,
        height=500,
        template="plotly_white",
        font=dict(size=11),
        showlegend=False
    )
    
    return fig


def create_2d_histogram(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    nbinsx: int = 20,
    nbinsy: int = 20,
    title: str = None
) -> go.Figure:
    """Create an interactive 2D histogram (heatmap).
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    x_col : str
        Column name for X axis
    y_col : str
        Column name for Y axis
    nbinsx : int
        Number of bins for X
    nbinsy : int
        Number of bins for Y
    title : str, optional
        Plot title
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    if x_col not in df.columns or y_col not in df.columns:
        return go.Figure().add_annotation(text="Columns not found")
    
    df_clean = df[[x_col, y_col]].dropna()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram2d(
        x=df_clean[x_col],
        y=df_clean[y_col],
        nbinsx=nbinsx,
        nbinsy=nbinsy,
        colorscale="Blues",
        hovertemplate="<b>%{x|.2f}</b> - <b>%{y|.2f}</b><br>Count: %{z}<extra></extra>"
    ))
    
    fig.update_layout(
        title=title or f"2D Distribution: {x_col} vs {y_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=500,
        template="plotly_white",
        font=dict(size=11),
        coloraxis_colorbar=dict(title="Count")
    )
    
    return fig


def create_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = None
) -> go.Figure:
    """Create an interactive line plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    x_col : str
        Column name for X axis
    y_col : str
        Column name for Y axis
    title : str, optional
        Plot title
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    if x_col not in df.columns or y_col not in df.columns:
        return go.Figure().add_annotation(text="Columns not found")
    
    df_sorted = df[[x_col, y_col]].dropna().sort_values(x_col)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted[x_col],
        y=df_sorted[y_col],
        mode="lines+markers",
        line=dict(color="#4C78A8", width=2),
        marker=dict(size=6),
        hovertemplate="<b>%{x}</b><br>Value: %{y:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=title or f"{y_col} over {x_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=500,
        template="plotly_white",
        font=dict(size=11),
        hovermode="x unified"
    )
    
    return fig


def get_numeric_columns(df: pd.DataFrame) -> list:
    """Get list of numeric column names from dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    
    Returns
    -------
    list
        List of numeric column names
    """
    if df is None:
        return []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return sorted(numeric_cols)
