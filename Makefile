.PHONY: install run debug clean lint test prepare

# Pass command-line options with: make run ARGS="--input path/to/prompts.json"
ARGS ?=
GOINFRE ?= 1
GOINFRE_ROOT ?= /home/acoromin/goinfre/call_me_maybe
FLAKE8_EXCLUDE := .git,.venv,__pycache__,.mypy_cache
FLAKE8_EXCLUDE := $(FLAKE8_EXCLUDE),.pytest_cache,.ruff_cache
FLAKE8_EXCLUDE := $(FLAKE8_EXCLUDE),llm_sdk

ifeq ($(GOINFRE),1)
export UV_CACHE_DIR := $(GOINFRE_ROOT)/uv-cache
export UV_PROJECT_ENVIRONMENT := $(GOINFRE_ROOT)/.venv
export HF_HOME := $(GOINFRE_ROOT)/huggingface

prepare:
	mkdir -p "$(GOINFRE_ROOT)"
else
prepare:
	@:
endif

install: prepare
	uv sync

run: prepare
	uv run python -m src $(ARGS)

debug: prepare
	uv run python -m pdb -m src $(ARGS)

test:
	uv run python -m pytest tests/test_* -v

lint: prepare
	uv run --with flake8 flake8 . --exclude=$(FLAKE8_EXCLUDE)
	MYPYPATH=llm_sdk uv run --with mypy mypy . \
		--warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs --explicit-package-bases \
		--exclude '^llm_sdk/'

clean:
	find src tests llm_sdk -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf __pycache__ .mypy_cache .pytest_cache .ruff_cache
