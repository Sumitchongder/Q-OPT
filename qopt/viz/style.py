"""
Shared matplotlib styling for all Q-OPT figures, aiming for a clean
Nature/Springer-journal look: serif-adjacent sans fonts, minimal
chart junk, consistent color palette, vector-safe output (PDF+PNG).
"""
from __future__ import annotations

import matplotlib.pyplot as plt

PALETTE = {
    "Static": "#8c8c8c",
    "Greedy": "#f0a35a",
    "MILP": "#3a6b35",
    "Exact": "#3a6b35",
    "SimulatedAnnealing": "#c65b7c",
    "RL": "#7a5195",
    "QAOA(p=1)": "#2a6fdb",
    "QAOA(p=2)": "#1c4e9e",
    "QAOA(p=3)": "#12356f",
    "Q-OPT": "#d1495b",
}

FIGSIZE_SINGLE = (4.6, 3.4)
FIGSIZE_WIDE = (7.2, 3.4)


def set_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "svg.fonttype": "none",
    })


def savefig(fig, path_stem: str):
    fig.savefig(f"{path_stem}.pdf")
    fig.savefig(f"{path_stem}.png")
