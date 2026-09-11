.PHONY: help install install-dev test test-verbose smoke quick full clean lint benchmark-pqc

help:
	@echo "Q-OPT — common commands"
	@echo ""
	@echo "  make install        Install the package and its dependencies"
	@echo "  make install-dev    Also install optional dev tools (black, ruff)"
	@echo "  make test           Run the full test suite (16 tests)"
	@echo "  make smoke          Run fast module self-checks (no pytest)"
	@echo "  make quick          Run all experiments at reduced seed count (~15-25 min)"
	@echo "  make full           Run all experiments at publication seed count (30+ min)"
	@echo "  make benchmark-pqc  Run the real (liboqs) PQC benchmark, if installed"
	@echo "  make lint           Run black --check and ruff (dev only)"
	@echo "  make clean          Remove caches and build artifacts"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev: install
	pip install -r requirements-dev.txt

test:
	python -m pytest tests/ -v

test-verbose:
	python -m pytest tests/ -vv -s

smoke:
	python -m qopt.network.topology
	python -m qopt.qkd.channel
	python -m qopt.pqc.hybrid_kdf
	python -m qopt.threat.risk_engine
	python -m qopt.optimization.qubo
	python -m qopt.optimization.qubo_to_ising
	python -m qopt.optimization.classical_baselines
	python -m qopt.optimization.milp_baseline
	python -m qopt.optimization.qaoa_qiskit
	python -m qopt.optimization.rl_baseline

quick:
	bash run_all.sh quick

full:
	bash run_all.sh full

benchmark-pqc:
	python scripts/summarize_pqc_benchmark.py --n-runs 5 --n-iters 2000

lint:
	black --check qopt/ experiments/ tests/ scripts/
	ruff check qopt/ experiments/ tests/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info
