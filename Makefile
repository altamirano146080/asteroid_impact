#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = software_development_ml_practice_1
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 software_development_ml_practice_1
	isort --check --diff software_development_ml_practice_1
	black --check software_development_ml_practice_1

## Format source code with black
.PHONY: format
format:
	isort software_development_ml_practice_1
	black software_development_ml_practice_1





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) software_development_ml_practice_1/dataset.py

## Generate the features and labels
.PHONY: features
features: data
	$(PYTHON_INTERPRETER)  software_development_ml_practice_1/features.py


## Generate exploratory data analysis plots
.PHONY: plots
plots: data
	$(PYTHON_INTERPRETER)  software_development_ml_practice_1/plots.py


## Train and evaluate the model
.PHONY: train
train: features
	$(PYTHON_INTERPRETER)  software_development_ml_practice_1/modeling/train.py


## Generate predictions with the trained model
.PHONY: predict
predict: train
	$(PYTHON_INTERPRETER)  software_development_ml_practice_1/modeling/predict.py


## Execute the complete machine learning pipeline
.PHONY: pipeline
pipeline: data features plots train predict


## Open Jupyter Lab
.PHONY: notebook
notebook: requirements
	uv run --with jupyter jupyter lab

## Build the documentation in HTML format
.PHONY: docs-html
docs-html:
	$(MAKE) -C docs html

## Launch Gradio interface
.PHONY: app
app: 
	uv run python -m software_development_ml_practice_1.app

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
