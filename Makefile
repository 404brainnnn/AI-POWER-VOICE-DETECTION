PYTHON ?= env/bin/python
PIP ?= $(PYTHON) -m pip
install:
	$(PIP) install -r requirements-core.txt
test:
	$(PYTHON) -m pytest
lint:
	$(PYTHON) -m ruff check app audio dsp models fusion evaluation tests
run:
	$(PYTHON) -m uvicorn app.main:app --reload
compose-up:
	docker compose up --build
compose-down:
	docker compose down
