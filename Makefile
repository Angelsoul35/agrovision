.PHONY: install test lint run demo eval clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=agrovision --cov-report=term-missing

lint:
	ruff check agrovision tests

run:
	uvicorn agrovision.main:app --reload --port 8000

demo:
	python -m agrovision.cli demo

eval:
	python -m agrovision.evaluation

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
