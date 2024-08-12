# Makefile

SHELL := /bin/bash

# Virtual environment directory
VENV := .env

# Python executable
PYTHON := $(VENV)/bin/python3

# pip executable
PIP := $(VENV)/bin/pip3

# Install dependencies
install-deps:
	@echo "Installing dependencies..."
	@python3 -m venv $(VENV)
	@$(PIP) install lark

# Freeze dependencies
freeze-deps:
	@echo "Freezing dependencies..."
	@$(PIP) freeze > requirements.txt

# Run tests
tests: parser-check

# Run parser sanity check.
parser-check:
	@echo "Running parser tests..."
	@for file in tests/programs/*.bril; do \
		filename=`basename "$$file" .bril`; \
		cat "$$file" | $(PYTHON) -m bril.bin.bril2json; \
	done
