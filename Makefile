# continuous-patterns — convenience targets
#
# See `make help` for available commands.

PYTHON ?= uv run python
CANONICAL_DIR := experiments/canonical
RESULTS_DIR := results

.PHONY: help test lint regen regen-one mini canonical inspect clean-results

help:
	@echo "continuous-patterns — available targets:"
	@echo ""
	@echo "  make test               run pytest (unit + integration)"
	@echo "  make lint               ruff check + ruff format --check"
	@echo "  make regen              regenerate ALL canonical runs (serial)"
	@echo "  make regen-one E=name   regen single experiment by name"
	@echo "                          (e.g. make regen-one E=medium_pinning)"
	@echo "  make mini               quick subset regen at shorter horizon"
	@echo "  make canonical          full regen + REGENERATION_REPORT.md"
	@echo "  make inspect E=name     inspect flux_samples diagnostic"
	@echo "  make clean-results      rm -rf results/"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON (default: uv run python)"

test:
	$(PYTHON) -m pytest -v

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

regen:
	@for yml in $(CANONICAL_DIR)/*.yaml; do \
	  name=$$(basename $$yml .yaml); \
	  echo ""; \
	  echo "=== $$name ==="; \
	  $(PYTHON) -m continuous_patterns.experiments.run \
	    --config $$yml --out-dir $(RESULTS_DIR) || exit 1; \
	done

regen-one:
	@if [ -z "$(E)" ]; then \
	  echo "Usage: make regen-one E=<experiment_name>"; \
	  echo "Available:"; \
	  ls -1 $(CANONICAL_DIR)/*.yaml | xargs -n1 basename | sed 's/.yaml$$//' | sed 's/^/  /'; \
	  exit 1; \
	fi
	$(PYTHON) -m continuous_patterns.experiments.run \
	  --config $(CANONICAL_DIR)/$(E).yaml --out-dir $(RESULTS_DIR)

mini:
	CP_REPRODUCE_MINI=1 $(PYTHON) scripts/reproduce_canonical.py

canonical:
	$(PYTHON) scripts/reproduce_canonical.py

inspect:
	@if [ -z "$(E)" ]; then \
	  echo "Usage: make inspect E=<experiment_name>"; \
	  exit 1; \
	fi
	$(PYTHON) scripts/inspect_flux_samples.py \
	  --config $(CANONICAL_DIR)/$(E).yaml

clean-results:
	rm -rf $(RESULTS_DIR)
