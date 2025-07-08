.PHONY: install test lint format docker-build docker-up clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=core --cov=plugins

lint:
	ruff check .

format:
	black .
	ruff check . --fix

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete