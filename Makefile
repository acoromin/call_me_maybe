.PHONY: install run debug clean lint test test-model visualize prepare

# Pass command-line options with: make run ARGS="--input path/to/prompts.json"
ARGS ?=
GOINFRE ?= 0
GOINFRE_ROOT ?= /sgoinfre/students/$(USER)/call_me_maybe

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

visualize: prepare
	uv run python -m src --visualize $(ARGS)

test: prepare
	uv run python -m unittest discover -s tests -p 'test_*.py' -v

test-model: prepare
	CALLME_RUN_MODEL_TESTS=1 uv run python -m unittest \
		discover -s tests -p 'test_*.py' -v

lint: prepare
	uv run --with flake8 flake8 .
	uv run --with mypy mypy . \
		--warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs --explicit-package-bases

clean:
	find src tests llm_sdk -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf __pycache__ .mypy_cache .pytest_cache .ruff_cache
