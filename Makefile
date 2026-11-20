.PHONY: sync check lint fmt test cov typecheck build c-parity train-smoke data firmware firmware-xor clean

IDF_EXPORT ?= $(HOME)/esp/esp-idf/export.sh
# IDF の Python venv を先に PATH に置く(システム python3 のバージョン違い対策)
IDF_PY_ENV ?= $(firstword $(wildcard $(HOME)/.espressif/python_env/idf5.5_py*_env/bin))
IDF = export PATH=$(IDF_PY_ENV):$$PATH && source $(IDF_EXPORT) >/dev/null 2>&1

sync:
	uv sync --locked

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

cov:
	uv run pytest --cov --cov-report=term --cov-report=xml

typecheck:
	uv run mypy

check: sync lint test

build:
	uv build

c-parity:
	uv run pytest -m "needs_cc" -o addopts="-ra"

train-smoke:
	uv run pytest tests/integration -o addopts="-ra -m 'not device'"

data:
	uv run --group data python scripts/fetch_data.py
	uv run gateact data-check --out results/data

firmware-xor:
	uv run python scripts/gen_firmware_model.py xor
	bash -c '$(IDF) && cd firmware/esp32s3/xor && idf.py set-target esp32s3 && idf.py build && idf.py size'

firmware:
	bash -c '$(IDF) && cd firmware/esp32s3/gateact && idf.py set-target esp32s3 && idf.py build && idf.py size'

clean:
	rm -rf dist .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml
