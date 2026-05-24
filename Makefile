.PHONY: api web test lint format

api:
	uvicorn tlse.api.main:app --reload

web:
	cd apps/web && npm run dev

test:
	pytest

lint:
	python -m compileall src scripts tests
	cd apps/web && npm run lint

format:
	python -m compileall src scripts tests
