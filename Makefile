# VacuaGym Makefile

.PHONY: help install test clean data docs

help:
	@echo "VacuaGym - Available targets:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Clean build artifacts"
	@echo "  make data       - Download and process datasets"
	@echo "  make docs       - Build documentation"

install:
	@echo "Installing dependencies..."
	# pip install -r requirements.txt

test:
	@echo "Running tests..."
	# pytest tests/

clean:
	@echo "Cleaning build artifacts..."
	# find . -type d -name "__pycache__" -exec rm -rf {} +
	# find . -type f -name "*.pyc" -delete
	# find . -type f -name "*.pyo" -delete

data:
	@echo "Data download targets will be added here..."
	# python scripts/download_ks_data.py
	# python scripts/download_cicy_data.py
	# python scripts/download_ftheory_data.py

docs:
	@echo "Documentation build targets will be added here..."
	# sphinx-build -b html docs/ docs/_build/
