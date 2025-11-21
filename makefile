.PHONY: install update build test lint format standards test coverage models

install:
	poetry install --with dev

update:
	poetry update

build:
	poetry build

lint:
	poetry run ruff check --fix . --config .ruff.toml

format:
	poetry run ruff format . --config .ruff.toml

standards:
	@make lint
	@echo "Code standards check complete."
	@make format
	@echo "Code formatting complete."

test:
	poetry run pytest

run:
	cd src/cezzis_com_cocktails_aisearch && uvicorn app:api --reload

coverage:
	poetry run pytest -v --cov=src/cezzis_com_cocktails_aisearch --cov-report=xml:coverage.xml --cov-report=term --junitxml=pytest-results.xml

models:
	poetry run datamodel-codegen \
		--url "https://api.cezzis.com/prd/cocktails/api-docs/v1/scalar/v1/openapi.json" \
		--input-file-type openapi --output src/cezzis_com_cocktails_aisearch/models/cocktail_models.py \
		--output-model-type pydantic_v2.BaseModel \
		--field-constraints --use-default \
		--use-schema-description \
		--target-python-version 3.12

all:
	@make install
	@make models
	@make standards
	@make coverage
	@make build