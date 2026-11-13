# Owner: Rohit Gangil, Reviewer: Atharv Singhal
PYTHON ?= python

.PHONY: install test reproduce quick web demo cli edge-cases

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest

# Regenerate every result, table, figure and the report (about 2 minutes on 16 cores).
reproduce:
	$(PYTHON) -m resource_war.reproduce

# Small smoke run into experiments/results_quick/ (never touches the real results).
quick:
	$(PYTHON) -m resource_war.reproduce --quick

# Game-style demo in the browser (watch, play, explore): http://127.0.0.1:8765
web:
	$(PYTHON) -m resource_war.demo.web

# Streamlit dashboard (analysis-focused alternative)
demo:
	$(PYTHON) -m streamlit run src/resource_war/demo/app.py

cli:
	$(PYTHON) -m resource_war.demo.cli play --seed 7

edge-cases:
	$(PYTHON) -m resource_war.demo.edge_cases
