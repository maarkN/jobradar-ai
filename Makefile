.PHONY: install test lint fmt typecheck check

install: ## Install deps (creates .venv via uv)
	uv sync

test: ## Run the test suite (offline, LLM mocked)
	uv run pytest

lint: ## Lint with ruff
	uv run ruff check .

fmt: ## Format with ruff
	uv run ruff format .

typecheck: ## Strict type check with mypy
	uv run mypy

check: lint typecheck test ## Everything CI runs
