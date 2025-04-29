.PHONY: all test style black isort pylint flake8 pydocstyle ruff bandit autoflake pydocstringformatter unit doctest install install-dev

# Targets to help running the tests.
#
# Example
# -------
# To run the style tests
# ```
# $ make style
# ```
#
# To run the unit tests:
# ```
# $ make unit
# ```
#
# To run all the tests:
# ```
# $ make test
# ```

FILES := $(shell find . -type f -name '*.py')
SRC_FILES := $(shell find src -type f -name '*.py')

# Default target that runs the style, unit and doc tests:
all: test

# Run the style, unit and doc tests:
test: style unit

style: black isort pylint flake8 pydocstyle ruff bandit autoflake pydocstringformatter

# Check Python files using black:
black:
	@echo "Checking black..."
	python -m black --check $(FILES)
	@echo

# Sort imports in Python files using isort:
isort:
	@echo "Checking isort..."
	python -m isort  --profile=black --check $(FILES)
	@echo

# Check Python files using pylint:
pylint:
	@echo "Checking pylint..."
	python -m pylint $(FILES)
	@echo

# Check Python files using flake8:
flake8:
	@echo "Checking flake8..."
	python -m flake8 $(FILES)
	@echo

# Check Python files for docstring style using pydocstyle:
pydocstyle:
	@echo "Checking pydocstyle..."
	python -m pydocstyle --convention=numpy $(FILES)
	@echo

# Check Python files for basic syntax errors using ruff:
ruff:
	@echo "Checking ruff..."
	ruff check $(FILES)
	@echo

# Perform security checks on Python files using bandit:
bandit:
	@echo "Checking bandit..."
	bandit src/
	@echo

# Remove unused imports from Python files using autoflake:
autoflake:
	@echo "Checking autoflake..."
	autoflake $(FILES)
	@echo

# Format Python docstrings using pydocstringformatter:
pydocstringformatter:
	@echo "Checking pydocstringformatter..."
	pydocstringformatter $(FILES)
	@echo

# Run unit tests:
unit:
	@echo "Running pytest..."
	python -m pytest
	@echo

# Test doc examples:
doctest:
	echo "Running doctest with pytest..."; \
	python -m doctest -v *.md; \
	@echo

# Install package for production use:
install:
	@echo "Installing package..."
	python -m pip install .
	@echo

# Install package for development use
install-dev:
	@echo "Installing package..."
	python -m pip install -e .[dev]
	@echo
