"""Generate an SI figure checking filter-to-memory FDT closure.

The figure compares the saved memory kernel/weights against the memory
recomputed from the saved colored-noise filter using the same discrete
autocorrelation convention as the simulation code.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "paper_data"
OUT_DIR = ROOT / "figures" / "generated" / "supplementary"

BOLTZMANN = 0.001987191
TIMEFACTOR = 48.88821
INTERNAL_DT_PS = 1.0 / TIMEFACTOR


@dataclass(frozen=True)
class ClosureCase:
    name: str
    filter_values: np.ndarray
    saved_memory: np.ndarray
    lag_dt: float
    x_label: str
    scale: float
    trapezoid_zero: bool = False


def one_sided_filter_acf(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    return np.asarray([np.dot(h[: h.size - lag], h[lag:]) for lag in range(h.size)])


def empirical_filter_acf(h: np.ndarray, max_lag: int, seed: int, samples: int = 1_048_576) -> np.ndarray:
    """Estimate the colored-noise covariance from synthetic white noise."""

    rng = np.random.default_rng(seed)
    h = np.asarray(h, dtype=np.float64)
    eta = rng.standard_normal(samples + h.size - 1)
    colored = np.convolve(eta, h[::-1], mode="valid")
    colored = colored - colored.mean()
    n = colored.size
    fft_n = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(colored, n=fft_n)
    corr = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_n)[:max_lag]
    counts = n - np.arange(max_lag)
    return corr / counts


def load_cases() -> list[ClosureCase]:
    co2 = np.load(
        DATA_ROOT
        / "optimization_histories/co2_symdetach_lr3e5_iter372/iteration_372.npz"
    )
    h2o = np.load(
        DATA_ROOT
        / "optimization_histories/h2o_original_fromscratch/history_arrays.npz"
    )
    return [
        ClosureCase(
            name=r"CO$_2$",
            filter_values=np.asarray(co2["filter"], dtype=np.float64),
            saved_memory=np.asarray(co2["memory_kernel"], dtype=np.float64),
            lag_dt=0.001,
            x_label=r"$\tau$ (ps)",
            scale=44.1 * INTERNAL_DT_PS / (BOLTZMANN * 300.0),
        ),
        ClosureCase(
            name=r"H$_2$O",
            filter_values=np.asarray(h2o["filters"][-1], dtype=np.float64),
            saved_memory=np.asarray(h2o["memories"][-1], dtype=np.float64),
            lag_dt=0.001,
            x_label=r"$\tau$ (ps)",
            scale=18.0 * INTERNAL_DT_PS / (BOLTZMANN * 300.0),
        ),
    ]


def closure_arrays(case: ClosureCase, seed: int) -> dict[str, np.ndarray | float]:
    racf = one_sided_filter_acf(case.filter_values)
    deterministic = racf * case.scale
    empirical = empirical_filter_acf(case.filter_values, case.filter_values.size, seed=seed) * case.scale
    if case.trapezoid_zero:
        deterministic = deterministic.copy()
        empirical = empirical.copy()
        deterministic[0] *= 0.5
        empirical[0] *= 0.5
    residual = case.saved_memory - deterministic
    return {
        "lag": np.arange(case.saved_memory.size),
        "tau": np.arange(case.saved_memory.size) * case.lag_dt,
        "saved": case.saved_memory,
        "deterministic": deterministic,
        "empirical": empirical,
        "residual": residual,
        "max_abs": float(np.max(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "rel_l2": float(np.linalg.norm(residual) / (np.linalg.norm(case.saved_memory) + 1e-30)),
    }


def write_source_data(cases: list[ClosureCase], arrays: dict[str, dict[str, np.ndarray | float]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "fdt_filter_kernel_closure_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "system",
                "lag_index",
                "tau",
                "saved_memory",
                "recomputed_from_filter",
                "empirical_noise_covariance",
                "residual_saved_minus_recomputed",
            ]
        )
        for case in cases:
            data = arrays[case.name]
            for idx, tau, saved, det, emp, res in zip(
                data["lag"],
                data["tau"],
                data["saved"],
                data["deterministic"],
                data["empirical"],
                data["residual"],
            ):
                writer.writerow([case.name, int(idx), tau, saved, det, emp, res])
    with (OUT_DIR / "fdt_filter_kernel_closure_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["system", "max_abs_residual", "rmse_residual", "relative_l2_residual"])
        for case in cases:
            data = arrays[case.name]
            writer.writerow([case.name, data["max_abs"], data["rmse"], data["rel_l2"]])


def make_plot(cases: list[ClosureCase], arrays: dict[str, dict[str, np.ndarray | float]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.3), dpi=300)
    colors = {
        "saved": "black",
        "empirical": "#0072B2",
    }
    for ax, case in zip(axes, cases):
        data = arrays[case.name]
        tau = data["tau"]
        saved = data["saved"]
        deterministic = data["deterministic"]
        empirical = data["empirical"]
        stride = max(1, len(tau) // 45)
        ax.axhline(0.0, color="0.78", lw=0.7, zorder=0)
        ax.plot(tau, saved, color=colors["saved"], lw=1.65, label="optimized kernel")
        ax.scatter(
            tau[::stride],
            empirical[::stride],
            s=7,
            color=colors["empirical"],
            alpha=0.55,
            lw=0,
            label="sampled noise cov.",
        )
        ax.set_title(case.name, pad=3)
        ax.set_xlabel(case.x_label)
        ax.set_ylabel(r"$M(\tau)$")
        rel = data["rel_l2"]
        max_abs = data["max_abs"]
        ax.text(
            0.98,
            0.95,
            f"rel. L2={rel:.1e}\nmax |res|={max_abs:.1e}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.7,
            color="0.25",
        )
        ax.legend(
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.70),
            handlelength=1.8,
            borderaxespad=0.0,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=1.1)
    fig.savefig(OUT_DIR / "fdt_filter_kernel_closure.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fdt_filter_kernel_closure.svg", bbox_inches="tight")


def main() -> None:
    cases = load_cases()
    arrays = {case.name: closure_arrays(case, seed=20260628 + idx) for idx, case in enumerate(cases)}
    write_source_data(cases, arrays)
    make_plot(cases, arrays)


if __name__ == "__main__":
    main()
