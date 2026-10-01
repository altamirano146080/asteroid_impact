# Software Development ML Practice 1

Machine learning project focused on asteroid impact-risk analysis. The goal is to explore a dataset of near-Earth objects, analyze their risk-related features, train a baseline predictive model, and evaluate its performance through a structured machine learning pipeline.

This project follows a Cookiecutter Data Science-style organization and includes both an exploratory notebook and modular Python scripts for data preparation, feature engineering, visualization, model training, and prediction.

## Documentation

The publish documentation is at: [https://altamirano146080.github.io/asteroid_impact/](https://altamirano146080.github.io/asteroid_impact/)
## Download package
**`uvx --from asteroid-impact-sdoml asteroid-demo`**
 
## Overview

The repository analyzes a dataset related to potential asteroid impact risk. It includes:

- dataset exploration and profiling
- missing value analysis
- feature selection and target transformation
- visualization of risk-related patterns
- baseline neural network training
- model evaluation and predictions

The model uses asteroid characteristics such as encounter velocity, absolute magnitude, diameter, Palermo scale, and potential impact dates to predict the logarithm of the impact probability.

## Data

This project uses the **`sentry-impact-risk`** dataset, which provides information on near-Earth objects and their potential collision risks.

- **Source:** The dataset is downloaded via the Hugging Face hub ([juliensimon/sentry-impact-risk](https://huggingface.co/datasets/juliensimon/sentry-impact-risk)), curated by Julien Simon. The original observational data originates from the [NASA/JPL Center for Near-Earth Object Studies (CNEOS) Sentry system](https://cneos.jpl.nasa.gov/sentry/).
- **How and why it is used:** The dataset is utilized to demonstrate a complete machine learning pipeline for regression tasks. We extract physical and kinematic features of the asteroids—such as encounter velocity, absolute magnitude, estimated diameter, and the Palermo scale—to train a baseline neural network. Because the actual impact probabilities are astronomically small, we transform the target variable to `log10(impact_probability)` to enable stable model training and evaluation.
- **License:** The dataset is provided under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. Any redistribution or reuse of this data is subject to its applicable license and attribution requirements.


## Project structure

```text
.
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
├── setup.cfg
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   │   ├── dataset.csv
│   │   ├── features.csv
│   │   ├── labels.csv
│   │   └── predictions.csv
│   └── raw/
│       └── dataset.csv
├── docs/
├── models/
│   ├── feature_scaler.joblib
│   └── impact_probability_model.keras
├── notebooks/
│   └── explore_dataset.ipynb
├── references/
├── reports/
│   ├── model_metrics.csv
│   ├── training_history.csv
│   └── figures/
│       ├── correlation_heatmap.png
│       ├── impact_probability_distribution.png
│       ├── velocity_vs_palermo.png
│       ├── magnitude_vs_palermo.png
│       └── potential_impact_timeline.png
├── asteroid_impact/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   ├── plots.py
│   └── modeling/
│       ├── __init__.py
│       ├── train.py
│       └── predict.py
└── uv.lock
```

## Repository components

### `notebooks/explore_dataset.ipynb`

Contains the complete exploratory analysis workflow, including:

- loading the dataset
- inspecting rows, columns, and data types
- checking missing values
- analyzing correlations and distributions
- visualizing asteroid risk properties
- preprocessing the data
- training and evaluating the baseline model

### `asteroid_impact/config.py`

Defines the main project directories and creates the required folders for:

- raw data
- intermediate data
- processed data
- trained models
- reports and figures

### `asteroid_impact/dataset.py`

Downloads or loads the asteroid impact-risk dataset and saves a local sample in the raw data directory. It also creates a cleaned dataset for the following pipeline steps.

### `asteroid_impact/features.py`

Creates the input features and target variable by:

- removing incomplete rows
- applying a base-10 logarithmic transformation to `impact_probability`
- selecting numerical predictive variables
- saving `features.csv` and `labels.csv`

### `asteroid_impact/plots.py`

Generates the exploratory data analysis visualizations, including:

- correlation heatmap
- impact probability distribution
- encounter velocity versus Palermo scale
- absolute magnitude versus Palermo scale
- potential impact timeline

### `asteroid_impact/modeling/train.py`

Trains and evaluates the baseline neural network model. This script:

- splits the data into training and test sets
- standardizes the input features
- trains the TensorFlow model
- calculates MAE, RMSE, and R² metrics
- saves the model, scaler, metrics, and training history

### `asteroid_impact/modeling/predict.py`

Loads the trained model and feature scaler, generates predictions for the processed features, and saves the results to a CSV file.

## Requirements

This project uses Python 3.13. The required dependencies are specified in `pyproject.toml` and managed with `uv`.

To install the dependencies, run:

```bash
uv sync
```

To create the virtual environment with Python 3.13 and install the dependencies:

```bash
make setup
```

## How to run

The project includes a Makefile with commands for each stage of the machine learning workflow.

### Install dependencies

```bash
make requirements
```

### Download and prepare the dataset

```bash
make data
```

### Generate features and labels

```bash
make features
```

### Generate exploratory plots

```bash
make plots
```

### Train the model

```bash
make train
```

### Generate predictions

```bash
make predict
```

### Run the complete pipeline

```bash
make pipeline
```

The complete pipeline runs the following steps in order:

```text
data → features → plots → train → predict
```

### Open Jupyter Lab

```bash
make notebook
```

### View all available commands

```bash
make help
```

## Code quality

To check the source code with Flake8, isort, and Black:

```bash
make lint
```

To format the source code automatically:

```bash
make format
```

To remove Python cache files:

```bash
make clean
```

## Generated outputs

The trained model and scaler are saved in:

- `models/impact_probability_model.keras`
- `models/feature_scaler.joblib`

The evaluation results are saved in:

- `reports/model_metrics.csv`
- `reports/training_history.csv`

The exploratory plots are saved in:

- `reports/figures/`

The predictions are saved in:

- `data/processed/predictions.csv`

## Notes

- The project uses a sample of up to 500 observations for the initial analysis.
- Rows with missing values are removed before model training.
- The target variable is transformed using `log10(impact_probability)` because the original probabilities are very small.
- The notebook is useful for interactive exploration and presentation.
- The Python modules provide a more maintainable and reusable version of the workflow.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Authors

- Ruth Altamirano Trujillo
- Malena Chacón Flores
- Odei Martinez de Morentin
