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
		cat "$$file" | $(PYTHON) -m bril.tools.bril2json; \
	done

# Run optimizations sanity check.
optimizations-check:
	@echo "Running optimizations tests..."
	@for file in tests/transforms/dce/*.bril; do \
		input_file=$$file; \
		expected_file=$${file%.bril}.out; \
		echo "Testing for input $$input_file against $$expected_file"; \
		$(PYTHON) -m turnstile.turnstile --input $$input_file --expected $$expected_file --optimizations 'dce, lvn'; \
	done
