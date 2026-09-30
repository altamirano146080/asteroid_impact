Usage
=====

The project should be executed in the following order:

#. Download and prepare the dataset.
#. Create the features and labels.
#. Train and evaluate the model.
#. Generate predictions.

Download and prepare the dataset
--------------------------------

The ``dataset`` module downloads the dataset from Hugging Face, creates a
local sample, removes incomplete rows, and saves the cleaned dataset.

Run:

.. code-block:: bash

   python -m asteroid_impact.dataset

By default, the project downloads a sample of up to 500 rows.

To download the dataset again, use:

.. code-block:: bash

   python -m asteroid_impact.dataset --force-download

The generated files are:

* ``data/raw/dataset.csv``
* ``data/processed/dataset.csv``

Create features
---------------

The ``features`` module creates the input features and the target labels.

Run:

.. code-block:: bash

   python -m asteroid_impact.features

This command creates:

* ``data/processed/features.csv``
* ``data/processed/labels.csv``

Train the model
---------------

The ``train`` module trains and evaluates the neural network.

Run:

.. code-block:: bash

   python -m asteroid_impact.modeling.train

This command creates:

* ``models/impact_probability_model.keras``
* ``models/feature_scaler.joblib``
* ``reports/model_metrics.csv``
* ``reports/training_history.csv``

Generate predictions
--------------------

The ``predict`` module loads the trained model and generates predictions.

Run:

.. code-block:: bash

   python -m asteroid_impact.modeling.predict

The predictions are saved as a CSV file in the processed data directory.