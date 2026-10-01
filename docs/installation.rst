Installation
============

Requirements
------------

The project requires:

* Python 3.10 or higher.
* Git.
* TensorFlow.
* Pandas.
* NumPy.
* Scikit-learn.
* Joblib.
* Typer.
* Loguru.
* Sphinx, if the documentation needs to be built locally.

Installing package
--------------------
.. code-block:: bash

   uvx --from asteroid-impact-sdoml asteroid-demo

Clone the repository
--------------------

.. code-block:: bash

   git clone https://github.com/altamirano146080/software_development_ml_practice_1.git
   cd software_development_ml_practice_1

Using ``uv``
------------

The repository includes ``pyproject.toml`` and ``uv.lock``.

If ``uv`` is installed, synchronize the project dependencies with:

.. code-block:: bash

   uv sync

Activate the virtual environment if necessary:

.. code-block:: bash

   source .venv/bin/activate

Using ``venv``
--------------

Create a virtual environment:

.. code-block:: bash

   python -m venv .venv

On Linux or macOS, activate it with:

.. code-block:: bash

   source .venv/bin/activate

On Windows, activate it with:

.. code-block:: powershell

   .venv\Scripts\activate

Install the project:

.. code-block:: bash

   pip install -e .

Building the documentation
--------------------------

Move to the documentation directory:

.. code-block:: bash

   cd docs

Build the HTML documentation:

.. code-block:: bash

   make html

The generated documentation will be available at:

.. code-block:: text

   docs/_build/html/index.html
