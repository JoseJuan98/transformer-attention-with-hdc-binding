.PHONY: init tensorboard kill-tensorboard clean-files help install install-dev install-poetry poetry-install \
poetry-install-dev lint test run-ts clean-artifacts clean install-precommit clean-run-ts

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

install-poetry:
	pip install --no-cache-dir -U "poetry>=2.1.1" && \
	poetry config virtualenvs.in-project true

## Install dependencies
install:
	./scripts/install.sh

## Install dev dependencies
install-dev:
	./scripts/install.sh --dev

poetry-install: install-poetry
	poetry install --no-cache

install-precommit:
	poetry run pre-commit install

poetry-install-dev: install-poetry
	poetry install --no-cache --with=dev && \
	poetry run pre-commit install

## Start the tensorboard server
tensorboard:
	@tensorboard --logdir=$(ROOT_DIR)/artifacts/model/runs/ --port=6006 --load_fast=false

## Kill tensorboard server
kill-tensorboard:
	kill $(ps -e | grep 'tensorboard' | awk '{print $$1}')

## Clean logs
clean:
	@echo "Cleaning artifacts..."
	@rm -rf $(ROOT_DIR)/artifacts/model/
	@rm -rf $(ROOT_DIR)/artifacts/log/

### Delete compiled Python files
clean-files:
	@echo "Cleaning compiled Python files..."
	@find src/ -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	@find src/ -type d -name "__pycache__" -delete
	@find src/ -type d -name "build" -delete
	@find src/ -type d -name "*.egg-info" -delete
	@find src/ -type d -name ".ipynb_checkpoints" -delete

## Lint
lint:
	@poetry run pre-commit run --all-files --color always

## Test the code
test:
	@poetry run pytest

## Run experiment for time series classification
run-ts:
	@export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
	poetry run python src/experiments/time_series/train.py

## Alias for clean + run-ts
clean-run-ts: clean run-ts

DEFAULT_GOAL := help
.PHONY: help
help:
	@echo "\n$$(tput bold)Available rules:$$(tput sgr0)\n"
	@awk '/^##/{c=substr($$0,3);next}c&&/^[[:alpha:]][[:alnum:]_-]+:/{print substr($$1,1,index($$1,":")),c}1{c=0}' $(MAKEFILE_LIST) | column -s: -t
