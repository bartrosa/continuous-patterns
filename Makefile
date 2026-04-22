.PHONY: sync-cpu sync-cuda install-hooks lint test pre-commit-all

sync-cpu:
	uv sync --extra cpu

sync-cuda:
	uv sync --extra cuda

install-hooks:
	uv run pre-commit install --install-hooks

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

pre-commit-all:
	uv run pre-commit run --all-files
