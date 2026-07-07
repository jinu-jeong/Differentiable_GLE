#!/usr/bin/env python3
"""Plot the fixed molecular COM pair potentials used for H2O and CO2."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "paper_data" / "supplementary" / "bulk_cg_pair_potentials.csv"
OUT_DIR = ROOT / "figures" / "generated" / "supplementary"


def main() -> None:
    rows: dict[str, list[tuple[float, float]]] = {}
    with CSV_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["system"], []).append((float(row["r_A"]), float(row["U"])))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.2, 2.45), dpi=300)
    colors = {"H2O": "#1B6DA8", "CO2": "#D55E00"}
    labels = {"H2O": r"H$_2$O", "CO2": r"CO$_2$"}
    for system in ["H2O", "CO2"]:
        data = np.asarray(rows[system], dtype=float)
        r = data[:, 0]
        u = np.clip(data[:, 1], -4.0, 8.0)
        mask = r <= 9.0
        ax.plot(r[mask], u[mask], color=colors[system], lw=1.6, label=labels[system])
    ax.axhline(0.0, color="0.65", lw=0.7)
    ax.set_xlim(0.0, 9.0)
    ax.set_ylim(-4.0, 8.0)
    ax.set_xlabel(r"$r$ (A)")
    ax.set_ylabel(r"$U(r)$")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "bulk_cg_pair_potentials.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "bulk_cg_pair_potentials.pdf", bbox_inches="tight")
    print(OUT_DIR / "bulk_cg_pair_potentials.png")
    print(OUT_DIR / "bulk_cg_pair_potentials.pdf")


if __name__ == "__main__":
    main()
