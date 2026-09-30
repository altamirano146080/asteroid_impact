"""Dataset acquisition and preprocessing utilities.

This module downloads the source dataset from Hugging Face, stores a local
sample, and cleans the data before it is used in feature engineering and model
training.
"""

from pathlib import Path
#from dataset import download_dataset
import pandas as pd
import numpy as np
import typer
from loguru import logger

from asteroid_impact.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from asteroid_impact.dataset import (
    download_dataset
)


app = typer.Typer()

DATASET_URI = (
    "hf://datasets/juliensimon/sentry-impact-risk/"
    "data/sentry_impact_risk.parquet"
)

RAW_DATA_PATH = RAW_DATA_DIR / "dataset.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "dataset.csv"


def prepare_dataset(
    input_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Clean the raw dataset and save the processed version.

    Rows containing missing values are removed before the processed dataset
    is written to disk.

    Parameters
    ----------
    input_path : Path, default=RAW_DATA_PATH
        Path to the raw dataset.
    output_path : Path, default=PROCESSED_DATA_PATH
        Path where the cleaned dataset will be saved.

    Returns
    -------
    pandas.DataFrame
        The cleaned dataset.
    """

    df = pd.read_csv(input_path)

    original_size = len(df)

    logger.info("Processing dataset...")
    df_reg = df.dropna().copy()

    print(f"Original dataset size: {len(df)}")
    print(f"Dataset size after removing missing values: {len(df_reg)}")

    print(f"Transforming target")

    df_reg["log_impact_probability"] = np.log10(
        df_reg["impact_probability"]
    )

    print("Selecting features")

    features = df_reg.select_dtypes(include="number").columns.tolist()

    features.remove("impact_probability")
    features.remove("log_impact_probability")
    
    X = df_reg[features]
    y = df_reg["log_impact_probability"]
    
    print("Selected features:")
    print(features)
    
    print(f"\nNumber of features: {X.shape[1]}")

    logger.success("Processing dataset complete.")

    cleaned_size = len(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Original dataset size: {original_size}")
    logger.info(f"Cleaned dataset size: {cleaned_size}")
    logger.success(f"Processed dataset saved to {output_path}")

    return df


@app.command()
def main(
    force_download: bool = typer.Option(
        False,
        help="Download the dataset again even if a local copy exists.",
    ),
):
    """Download and prepare the asteroid impact-risk dataset.

    Parameters
    ----------
    force_download : bool, default=False
        If True, redownload the source dataset even when a local copy already
        exists.

    Returns
    -------
    None
        This command runs the data acquisition and preprocessing pipeline.
    """

    download_dataset(force_download=force_download)
    prepare_dataset()


if __name__ == "__main__":
    app()
