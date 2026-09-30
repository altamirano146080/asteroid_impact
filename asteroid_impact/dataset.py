from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from asteroid_impact.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()

DATASET_URI = (
    "hf://datasets/juliensimon/sentry-impact-risk/"
    "data/sentry_impact_risk.parquet"
)

RAW_DATA_PATH = RAW_DATA_DIR / "dataset.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "dataset.csv"


def download_dataset(
    output_path: Path = RAW_DATA_PATH,
    sample_size: int = 500,
    force_download: bool = False,
) -> pd.DataFrame:
    """Download a sample of the asteroid impact-risk dataset.

    The dataset is downloaded from Hugging Face and stored locally as a CSV
    file. If the file already exists, the local copy is reused unless
    ``force_download`` is set to ``True``.

    Parameters
    ----------
    output_path:
        Path where the downloaded dataset will be saved.
    sample_size:
        Maximum number of rows to keep in the local sample.
    force_download:
        Whether to download the dataset again if a local copy exists.

    Returns
    -------
    pandas.DataFrame
        The downloaded dataset.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force_download:
        logger.info(f"Loading local dataset from {output_path}")
        return pd.read_csv(output_path)

    logger.info("Downloading dataset from Hugging Face...")

    dataframe = pd.read_parquet(DATASET_URI)

    dataframe = dataframe.sample(
        n=min(sample_size, len(dataframe)),
        random_state=42,
    )

    dataframe.to_csv(output_path, index=False)

    logger.success(
        f"Dataset downloaded and saved to {output_path} "
        f"with {len(dataframe)} rows."
    )

    return dataframe


def prepare_dataset(
    input_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Clean the raw dataset and save the processed version.

    Rows containing missing values are removed before the processed dataset
    is written to disk.

    Parameters
    ----------
    input_path:
        Path to the raw dataset.
    output_path:
        Path where the cleaned dataset will be saved.

    Returns
    -------
    pandas.DataFrame
        The cleaned dataset.
    """

    dataframe = pd.read_csv(input_path)

    original_size = len(dataframe)

    dataframe = dataframe.dropna().copy()

    cleaned_size = len(dataframe)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    logger.info(f"Original dataset size: {original_size}")
    logger.info(f"Cleaned dataset size: {cleaned_size}")
    logger.success(f"Processed dataset saved to {output_path}")

    return dataframe


@app.command()
def main(
    force_download: bool = typer.Option(
        False,
        help="Download the dataset again even if a local copy exists.",
    ),
):
    """
    Downloads and prepares the asteroid impact-risk dataset.
    """

    download_dataset(force_download=force_download)
    prepare_dataset()


if __name__ == "__main__":
    app()
