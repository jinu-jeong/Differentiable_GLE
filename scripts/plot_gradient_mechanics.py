#!/usr/bin/env python3
"""Plot CO2 gradient-pathway diagnostics from saved audit arrays."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "paper_data" / "gradient_mechanics" / "co2_iter372"
OUT_DIR = ROOT / "figures" / "generated"

BLUE = "#1B6DA8"
ORANGE = "#D55E00"


def load_summary(path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items()})
    return {key: np.asarray([row[key] for row in rows], dtype=float) for key in rows[0]}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.labelsize": 9.4,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.dpi": 450,
            "figure.dpi": 180,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison = np.load(DATA_DIR / "td1000" / "arrays" / "adjoint_vs_direct_comparison.npz")
    summary = load_summary(DATA_DIR / "gradient_mechanics_summary.csv")

    direct_grad = comparison["direct_grad"]
    adjoint_grad = comparison["adjoint_grad"]
    grad_x = np.arange(direct_grad.size)
    horizons = summary["td_max"]
    direct_mem_gb = summary["direct_mem"] / 1024.0
    adjoint_mem_gb = summary["adjoint_mem"] / 1024.0

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.55, 2.45),
        constrained_layout=False,
        gridspec_kw={"wspace": 0.42},
    )
    ax_grad, ax_mem = axes

    ax_grad.plot(grad_x, direct_grad, color=BLUE, lw=1.5, label="direct")
    ax_grad.plot(grad_x, adjoint_grad, color=ORANGE, lw=1.5, label="adjoint")
    ax_grad.axhline(0.0, color="0.65", lw=0.65)
    ax_grad.set_xlabel("filter coefficient index")
    ax_grad.set_ylabel("gradient")
    ax_grad.legend(frameon=False, loc="lower left")
    despine(ax_grad)

    ax_mem.plot(horizons, direct_mem_gb, "o-", color=BLUE, lw=1.55, ms=4.2, label="direct")
    ax_mem.plot(horizons, adjoint_mem_gb, "s-", color=ORANGE, lw=1.55, ms=4.2, label="adjoint")
    ax_mem.set_xlabel("VACF objective horizon (lag steps)")
    ax_mem.set_ylabel("peak GPU memory (GB)")
    ax_mem.set_xlim(75, 1025)
    ax_mem.set_ylim(0, max(direct_mem_gb) * 1.14)
    ax_mem.legend(frameon=False, loc="upper left")
    despine(ax_mem)

    fig.savefig(OUT_DIR / "co2_gradient_mechanics.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "co2_gradient_mechanics.pdf", bbox_inches="tight")
    print(OUT_DIR / "co2_gradient_mechanics.png")
    print(OUT_DIR / "co2_gradient_mechanics.pdf")


if __name__ == "__main__":
    main()
