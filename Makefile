.PHONY: init tensorboard kill-tensorboard clean-files help install install-dev install-poetry

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

## Install poetry
install-poetry:
	pip install --no-cache-dir -U "poetry>=2.1.1"

## Install dependencies
install: install-poetry
	poetry install

## Install dev dependencies
install-dev: install-poetry
	poetry install --with=dev && \
	poetry run pre-commit install

## Start the tensorboard server
tensorboard:
	@tensorboard --logdir=$(ROOT_DIR)/artifacts/model/runs/ --port=6006 --load_fast=false

## Kill tensorboard server
kill-tensorboard:
	kill $(ps -e | grep 'tensorboard' | awk '{print $1}')

## Delete compiled Python files
clean-files:
	find . | grep -E "build$|\/__pycache__$|\.pyc$|\.pyo$|\.egg-info$|\.ipynb_checkpoints" | xargs rm -rf || echo "Already clean"

## Lint
lint:
	@poetry run pre-commit run --all-files --color always

## Test the code
test:
	@poetry run pytest

## Run experiment
run-ts:
	@poetry run python src/experiments/time_series/train.py

## Clean logs
clean-logs:
	@echo "Cleaning logs..."
	@rm -rf $(ROOT_DIR)/artifacts/model/
	@rm -rf $(ROOT_DIR)/artifacts/log/

DEFAULT_GOAL := help
.PHONY: help
help:
	@echo "\n$$(tput bold)Available rules:$$(tput sgr0)\n"
	@awk '/^##/{c=substr($$0,3);next}c&&/^[[:alpha:]][[:alnum:]_-]+:/{print substr($$1,1,index($$1,":")),c}1{c=0}' $(MAKEFILE_LIST) | column -s: -t
