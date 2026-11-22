.PHONY: help install install-dev test experiments plan rehearse results lint clean

help:
	@echo "Q-OPT v2 - common commands"
	@echo "  make install      install pinned dependencies + package (editable)"
	@echo "  make test         unit, equivalence and regression tests"
	@echo "  make experiments  run E1-E6 at publication settings (see run_all.sh)"
	@echo "  make plan         rebuild the pre-registered hardware plan (free)"
	@echo "  make rehearse     run notebooks/qpu_run.ipynb on FakeMarrakesh (free)"
	@echo "  make results      regenerate docs/RESULTS.md from tables/"
	@echo "  make lint         ruff + black --check"

install:
	pip install -r requirements.txt
	pip install -e . --no-deps

install-dev: install
	pip install -r requirements-dev.txt

test:
	python -m pytest tests/ -q

experiments:
	bash run_all.sh

plan:
	python -c "from qopt.quantum.hardware import build_plan; build_plan('results/hardware/plan')"
	python scripts/make_qpu_notebook.py

rehearse:
	cd notebooks && jupyter nbconvert --to notebook --execute qpu_run.ipynb --output /tmp/qpu_rehearsal.ipynb --ExecutePreprocessor.timeout=3600

results:
	python scripts/summarize_results.py

lint:
	ruff check qopt/agility qopt/quantum experiments tests scripts
	black --check qopt/agility qopt/quantum experiments tests scripts

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info
