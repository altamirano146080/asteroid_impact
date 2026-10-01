#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = asteroid_impact
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
	flake8 asteroid_impact
	isort --check --diff asteroid_impact
	black --check asteroid_impact

## Format source code with black
.PHONY: format
format:
	isort asteroid_impact
	black asteroid_impact





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
	$(PYTHON_INTERPRETER) asteroid_impact/dataset.py

## Generate the features and labels
.PHONY: features
features: data
	$(PYTHON_INTERPRETER)  asteroid_impact/features.py


## Generate exploratory data analysis plots
.PHONY: plots
plots: data
	$(PYTHON_INTERPRETER)  asteroid_impact/plots.py


## Train and evaluate the model
.PHONY: train
train: features
	$(PYTHON_INTERPRETER)  asteroid_impact/modeling/train.py


## Generate predictions with the trained model
.PHONY: predict
predict: train
	$(PYTHON_INTERPRETER)  asteroid_impact/modeling/predict.py


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
	uv run python -m asteroid_impact.app

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
