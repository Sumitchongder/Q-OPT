"""Shared helpers for the v2 experiment scripts: paths, provenance, statistics, styling."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
TAB = ROOT / "tables"
RES = ROOT / "results"
for d in (FIG, TAB, RES / "manifest"):
    d.mkdir(parents=True, exist_ok=True)

# Colour-blind-safe palette (Okabe-Ito), fixed per method across all figures.
COLORS = {
    "Static-QKD": "#999999", "Static-PQC": "#BBBBBB", "Static-HYB": "#666666",
    "Rule-based": "#E69F00", "Separable (v0.1)": "#56B4E9", "Greedy (coupled)": "#009E73",
    "Tabu search": "#F0E442", "Simulated annealing": "#CC79A7",
    "Q-OPT v2 (exact)": "#D55E00", "Q-OPT v2 (XY-QAOA)": "#0072B2",
    "XY": "#0072B2", "PEN": "#E69F00", "W": "#999999",
}


def style():
    """
    House style. With QOPT_FIGSCALE=<f> set, every figure is created f times
    smaller and written to figures/paper/, so that at the journal text width the
    fonts appear at their nominal point size (used for the manuscript build).
    """
    import os
    import matplotlib.pyplot as plt
    scale = float(os.environ.get("QOPT_FIGSCALE", "0") or 0)
    if scale and not getattr(plt, "_qopt_scaled", False):
        _subplots, _figure = plt.subplots, plt.figure

        def subplots(*a, **k):
            if "figsize" in k:
                k["figsize"] = (k["figsize"][0] * scale, k["figsize"][1] * scale)
            return _subplots(*a, **k)

        def figure(*a, **k):
            if "figsize" in k:
                k["figsize"] = (k["figsize"][0] * scale, k["figsize"][1] * scale)
            return _figure(*a, **k)
        plt.subplots, plt.figure, plt._qopt_scaled = subplots, figure, True
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9, "font.family": "DejaVu Sans",
        "axes.labelsize": 9, "axes.titlesize": 9.5, "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "legend.frameon": False,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "lines.linewidth": 1.4, "lines.markersize": 4, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    if scale:   # journal build: fonts sized for a ~13 cm text block
        plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
                             "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
                             "lines.linewidth": 1.2, "lines.markersize": 3.2})


def legend_cols(n):
    """Legend columns: wrap long single-row legends onto two rows in paper mode."""
    import os
    return max(1, (n + 1) // 2) if os.environ.get("QOPT_FIGSCALE") else n


def savefig(fig, name):
    import os
    out = FIG / "paper" if os.environ.get("QOPT_FIGSCALE") else FIG
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")


def mean_ci(x, conf=0.95, n_boot=2000, seed=0):
    """Mean and percentile-bootstrap CI."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    if x.size == 1 or np.allclose(x, x[0]):
        return float(x.mean()), float(x.mean()), float(x.mean())
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    a = (1 - conf) / 2
    return float(x.mean()), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def paired_test(a, b):
    """Two-sided Wilcoxon signed-rank on paired samples; returns (median diff, p)."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return 0.0, 1.0
    return float(np.median(d)), float(wilcoxon(a, b, zero_method="zsplit").pvalue)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def manifest(name, args, t0, extra=None):
    import os
    if os.environ.get("QOPT_FIGSCALE"):      # paper figure rebuild: keep the original provenance
        return
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    import qiskit
    info = dict(experiment=name, args=vars(args) if hasattr(args, "__dict__") else args,
                finished=time.strftime("%Y-%m-%dT%H:%M:%S"), runtime_s=round(time.time() - t0, 1),
                python=platform.python_version(), numpy=np.__version__, qiskit=qiskit.__version__,
                platform=platform.platform(), argv=sys.argv, git_commit=commit, **(extra or {}))
    (RES / "manifest" / f"{name}.json").write_text(json.dumps(info, indent=1, default=str))
