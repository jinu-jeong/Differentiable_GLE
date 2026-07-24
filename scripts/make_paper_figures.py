#!/usr/bin/env python3
"""Reproduce the main DiffGLE paper figures from compact saved arrays."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "paper_data"
OUT_DIR = ROOT / "figures" / "generated"


CO2_HIST = DATA_ROOT / "optimization_histories/co2_symdetach_lr3e5_iter372/history_arrays.npz"
CO2_PROD = DATA_ROOT / "source_data/co2_symdetach_iter372_observables.npz"
H2O_HIST = DATA_ROOT / "optimization_histories/h2o_original_fromscratch/history_arrays.npz"
H2O_PROD = DATA_ROOT / "source_data/h2o_ibi20_original_final_iter299_observables.npz"

STAR_RUN = DATA_ROOT / "optimization_histories/star_polymer_noforce_continuation_support700"
STAR_FROM_SCRATCH = DATA_ROOT / "optimization_histories/star_polymer_noforce_fromscratch_support700"
STAR_CURVES = STAR_RUN / "star_polymer_diffgle_noforce_curves.npz"
STAR_SNAP = STAR_RUN / "kernel_snapshots.npz"
STAR_VACF_REPLAY = STAR_RUN / "vacf_evolution_replay.npz"


BLACK = "black"
ORANGE = "#D55E00"
GREEN = "#146C43"
BLUE = "#1B6DA8"
BROWN = "#8C510A"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.labelsize": 9.4,
            "axes.titlesize": 9.4,
            "legend.fontsize": 7.6,
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


def sparse_indices(n: int, max_lines: int = 48) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(max_lines, n)).astype(int))


def colored_history(
    ax: plt.Axes,
    x: np.ndarray,
    ys: np.ndarray,
    iterations: np.ndarray,
    indices: np.ndarray,
    *,
    alpha: float = 0.12,
    lw: float = 0.65,
    cmap_name: str = "viridis",
) -> tuple[matplotlib.colors.Colormap, Normalize]:
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=float(np.nanmin(iterations)), vmax=float(np.nanmax(iterations)))
    for idx in indices:
        ax.plot(x, ys[idx], color=cmap(norm(iterations[idx])), lw=lw, alpha=alpha, zorder=1)
    return cmap, norm


def add_colorbar(fig: plt.Figure, ax: plt.Axes, cmap, norm, ticks) -> None:
    cax = inset_axes(
        ax,
        width="3.2%",
        height="34%",
        loc="upper right",
        bbox_to_anchor=(-0.12, -0.02, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("iteration", fontsize=6.8, labelpad=2)
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=6.3, length=2, width=0.55)
    cb.outline.set_linewidth(0.55)


def draw_iteration_colorbar(cax: plt.Axes, iterations: np.ndarray) -> None:
    """Draw a compact, robust iteration colorbar in a dedicated axis."""
    finite = iterations[np.isfinite(iterations)]
    vmin = max(0, int(np.nanmin(finite)))
    vmax = int(np.nanmax(finite))
    mid = int(round(0.5 * (vmin + vmax)))
    gradient = np.linspace(0.0, 1.0, 256)[:, None]
    cax.imshow(gradient, cmap="viridis", aspect="auto", origin="lower")
    cax.set_xticks([])
    cax.set_yticks([0, 128, 255])
    cax.set_yticklabels([f"{vmin}", f"{mid}", f"{vmax}"])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_ylabel("iteration", fontsize=6.8, labelpad=3)
    cax.tick_params(axis="y", labelsize=6.2, length=2, width=0.55, pad=1.5)
    for spine in cax.spines.values():
        spine.set_linewidth(0.55)


def add_iteration_colorbar_inset(ax: plt.Axes, iterations: np.ndarray) -> None:
    """Add a small iteration colorbar inside the memory-kernel panel."""
    finite = iterations[np.isfinite(iterations)]
    vmin = max(0, int(np.nanmin(finite)))
    vmax = int(np.nanmax(finite))
    mid = int(round(0.5 * (vmin + vmax)))
    cax = inset_axes(ax, width="3.8%", height="31%", loc="upper right", borderpad=0.42)
    gradient = np.linspace(0.0, 1.0, 256)[:, None]
    cax.imshow(gradient, cmap="viridis", aspect="auto", origin="lower")
    cax.set_xticks([])
    cax.set_yticks([0, 128, 255])
    cax.set_yticklabels([f"{vmin}", f"{mid}", f"{vmax}"])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_ylabel("iteration", fontsize=5.8, labelpad=2)
    cax.tick_params(axis="y", labelsize=5.4, length=1.8, width=0.5, pad=1.2)
    for spine in cax.spines.values():
        spine.set_linewidth(0.5)


def add_loss_inset(ax: plt.Axes, iterations: np.ndarray, loss: np.ndarray) -> None:
    """Add a compact loss-history inset to a memory-kernel panel."""
    mask = np.isfinite(iterations) & np.isfinite(loss) & (loss > 0)
    if not np.any(mask):
        return
    inset = inset_axes(
        ax,
        width="47%",
        height="31%",
        loc="upper right",
        bbox_to_anchor=(-0.10, -0.015, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.36,
    )
    inset.plot(iterations[mask], loss[mask], color=BROWN, lw=1.15)
    inset.set_yscale("log")
    inset.text(
        0.10,
        0.88,
        "loss",
        transform=inset.transAxes,
        fontsize=6.4,
        color=BROWN,
        ha="left",
        va="top",
    )
    inset.tick_params(labelsize=4.7, length=1.5, width=0.45, pad=0.8)
    inset.set_xticks([float(iterations[mask][0]), float(iterations[mask][-1])])
    inset.set_xticklabels(["0", f"{int(iterations[mask][-1])}"])
    ymin, ymax = float(np.nanmin(loss[mask])), float(np.nanmax(loss[mask]))
    inset.set_ylim(ymin * 0.75, ymax * 1.35)
    inset.set_yticks([ymin, ymax])
    inset.set_yticklabels([f"{ymin:.0e}", f"{ymax:.0e}"])
    inset.patch.set_facecolor("white")
    inset.patch.set_alpha(0.88)
    for spine in inset.spines.values():
        spine.set_linewidth(0.55)


def add_vacf_legend(ax: plt.Axes, label1: str = "AA", label2: str = "DiffGLE") -> None:
    ax.plot([0.12, 0.22], [0.87, 0.87], transform=ax.transAxes, color=BLACK, lw=1.8, clip_on=False)
    ax.text(0.235, 0.87, label1, transform=ax.transAxes, color=BLACK, fontsize=7.6, va="center")
    ax.plot([0.12, 0.22], [0.78, 0.78], transform=ax.transAxes, color=ORANGE, lw=1.9, clip_on=False)
    ax.text(0.235, 0.78, label2, transform=ax.transAxes, color=ORANGE, fontsize=7.6, va="center")


def fluid_panel(
    fig: plt.Figure,
    axes: np.ndarray,
    hist_path: Path,
    prod_path: Path,
    label: str,
    *,
    dt_ps: float,
    vacf_x_is_time: bool,
    vacf_xlim: tuple[float, float],
    vacf_ylim: tuple[float, float],
    filter_xlim: tuple[float, float],
    memory_xlim: tuple[float, float],
    memory_ylim: tuple[float, float],
    filter_ylim: tuple[float, float],
    y_labelpad: float = 2.5,
) -> None:
    hist = np.load(hist_path)
    prod = np.load(prod_path)
    iterations = hist["iterations"]
    vacf_train = hist["vacf_gle"]
    vacf_target = hist["vacf_aa"]
    filters = hist["filters"]
    memories = hist["memories"]

    final_idx = int(np.flatnonzero(iterations <= iterations.max())[-1])
    indices = sparse_indices(iterations.size)
    if vacf_x_is_time:
        vacf_x = np.arange(vacf_target.size) * dt_ps
        vacf_xlabel = "lag time (ps)"
    else:
        vacf_x = np.arange(vacf_target.size)
        vacf_xlabel = "lag step"
    filter_x = np.arange(filters.shape[1]) * dt_ps
    memory_x = np.arange(memories.shape[1]) * dt_ps

    ax_vacf, ax_filter, ax_memory = axes
    vacf_mask = (vacf_x >= vacf_xlim[0]) & (vacf_x <= vacf_xlim[1])
    colored_history(ax_vacf, vacf_x[vacf_mask], vacf_train[:, vacf_mask], iterations, indices, alpha=0.10)
    ax_vacf.plot(vacf_x[vacf_mask], vacf_target[vacf_mask], color=BLACK, lw=1.85, zorder=4)
    ax_vacf.plot(vacf_x[vacf_mask], prod["vacf_gle"][vacf_mask], color=ORANGE, lw=1.95, zorder=5)
    if "vacf_sem" in prod:
        ax_vacf.fill_between(
            vacf_x[vacf_mask],
            prod["vacf_gle"][vacf_mask] - prod["vacf_sem"][vacf_mask],
            prod["vacf_gle"][vacf_mask] + prod["vacf_sem"][vacf_mask],
            color=ORANGE,
            alpha=0.16,
            linewidth=0,
            zorder=3,
        )
    ax_vacf.axhline(0, color="0.55", lw=0.6)
    ax_vacf.set_xlim(*vacf_xlim)
    ax_vacf.set_ylim(*vacf_ylim)
    ax_vacf.set_xlabel(vacf_xlabel)
    ax_vacf.set_ylabel(f"{label}\nnormalized VACF")
    despine(ax_vacf)

    axins = inset_axes(ax_vacf, width="42%", height="38%", loc="upper right", borderpad=0.82)
    axins.plot(prod["r_aa"], prod["rdf_aa"], color=BLACK, lw=1.1)
    axins.plot(prod["r"], prod["rdf_gle"], color=ORANGE, lw=1.15)
    axins.set_xlim(0.0, 20.0)
    axins.set_ylim(-0.01, 2.75)
    axins.set_xticks([0, 10, 20])
    axins.set_yticks([0, 1, 2])
    axins.set_xlabel("r (A)", fontsize=6.4, labelpad=0.2)
    axins.set_ylabel("g(r)", fontsize=6.4, labelpad=0.2)
    axins.tick_params(labelsize=6.2, length=2, width=0.55)
    for spine in axins.spines.values():
        spine.set_linewidth(0.6)

    cmap, norm = colored_history(
        ax_filter,
        filter_x,
        filters,
        iterations,
        indices,
        alpha=0.14,
        lw=0.62,
    )
    ax_filter.plot(filter_x, filters[final_idx], color=GREEN, lw=2.0, zorder=5)
    ax_filter.axhline(0, color="0.55", lw=0.6)
    ax_filter.set_xlim(*filter_xlim)
    ax_filter.set_ylim(*filter_ylim)
    ax_filter.set_xlabel("filter lag (ps)")
    ax_filter.set_ylabel("noise filter", labelpad=y_labelpad)
    despine(ax_filter)

    colored_history(
        ax_memory,
        memory_x,
        memories,
        iterations,
        indices,
        alpha=0.14,
        lw=0.62,
    )
    ax_memory.plot(memory_x, memories[final_idx], color=BLUE, lw=2.0, zorder=5)
    ax_memory.axhline(0, color="0.55", lw=0.6)
    ax_memory.set_xlim(*memory_xlim)
    ax_memory.set_ylim(*memory_ylim)
    ax_memory.set_xlabel("memory lag (ps)")
    ax_memory.set_ylabel(r"$M(\tau)$", labelpad=y_labelpad)
    despine(ax_memory)


def make_fluid_figure() -> None:
    style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.15, 4.45),
        constrained_layout=False,
        gridspec_kw={"wspace": 0.42, "hspace": 0.42},
    )
    fluid_panel(
        fig,
        axes[0],
        CO2_HIST,
        CO2_PROD,
        "CO$_2$",
        dt_ps=0.001,
        vacf_x_is_time=True,
        vacf_xlim=(0.0, 1.0),
        vacf_ylim=(-0.4, 1.0),
        filter_xlim=(0.0, 0.2),
        memory_xlim=(0.0, 0.2),
        filter_ylim=(-0.022, 0.022),
        y_labelpad=2.5,
        memory_ylim=(-0.007, 0.0315),
    )
    fluid_panel(
        fig,
        axes[1],
        H2O_HIST,
        H2O_PROD,
        "H$_2$O",
        dt_ps=0.001,
        vacf_x_is_time=False,
        vacf_xlim=(0.0, 1000.0),
        vacf_ylim=(-0.25, 1.0),
        filter_xlim=(0.0, 1.0),
        memory_xlim=(0.0, 1.0),
        filter_ylim=(-0.008, 0.032),
        y_labelpad=2.5,
        memory_ylim=(-0.007, 0.020),
    )
    fig.savefig(OUT_DIR / "fluid_results_compact.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fluid_results_compact.pdf", bbox_inches="tight")


def make_combined_results_figure() -> None:
    """Build one manuscript figure combining fluids and star-polymer results."""
    style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7.15, 6.42),
        constrained_layout=False,
        gridspec_kw={"wspace": 0.52, "hspace": 0.50},
    )
    plot_axes = axes
    fluid_panel(
        fig,
        plot_axes[0],
        CO2_HIST,
        CO2_PROD,
        "CO$_2$",
        dt_ps=0.001,
        vacf_x_is_time=True,
        vacf_xlim=(0.0, 1.0),
        vacf_ylim=(-0.4, 1.0),
        filter_xlim=(0.0, 0.2),
        memory_xlim=(0.0, 0.2),
        filter_ylim=(-0.022, 0.022),
        memory_ylim=(-0.007, 0.0315),
        y_labelpad=1.0,
    )
    add_iteration_colorbar_inset(plot_axes[0, 2], np.load(CO2_HIST)["iterations"])
    co2_hist = np.load(CO2_HIST)
    add_loss_inset(plot_axes[0, 2], co2_hist["metrics_iteration"], co2_hist["metrics_loss"])
    fluid_panel(
        fig,
        plot_axes[1],
        H2O_HIST,
        H2O_PROD,
        "H$_2$O",
        dt_ps=0.001,
        vacf_x_is_time=True,
        vacf_xlim=(0.0, 1.0),
        vacf_ylim=(-0.25, 1.0),
        filter_xlim=(0.0, 1.0),
        memory_xlim=(0.0, 1.0),
        filter_ylim=(-0.008, 0.032),
        memory_ylim=(-0.007, 0.020),
        y_labelpad=1.0,
    )
    add_iteration_colorbar_inset(plot_axes[1, 2], np.load(H2O_HIST)["iterations"])
    h2o_hist = np.load(H2O_HIST)
    add_loss_inset(plot_axes[1, 2], h2o_hist["metrics_iteration"], h2o_hist["metrics_loss"])

    # Star-polymer row.
    curves = np.load(STAR_CURVES)
    from_snap = np.load(STAR_FROM_SCRATCH / "kernel_snapshots.npz")
    cont_snap = np.load(STAR_SNAP)
    from_replay = np.load(STAR_FROM_SCRATCH / "vacf_evolution_replay.npz")
    cont_replay = np.load(STAR_VACF_REPLAY)
    from_summary = json.loads((STAR_FROM_SCRATCH / "star_polymer_diffgle_noforce_summary.json").read_text())
    continuation_offset = int(from_summary["iterations"])

    time = curves["target_time"]
    filter_t = time[: cont_snap["filters"].shape[1]]
    memory_t = time[: cont_snap["memories"].shape[1]]
    iterations = np.concatenate(
        [
            offset_iterations(from_snap["iterations"], 0),
            offset_iterations(cont_snap["iterations"], continuation_offset, start_iteration=continuation_offset - 1),
        ]
    )
    filters = np.concatenate([from_snap["filters"], cont_snap["filters"]], axis=0)
    memories = np.concatenate([from_snap["memories"], cont_snap["memories"]], axis=0)
    final_idx = filters.shape[0] - 1
    filter_active = active_prefix(filters[final_idx])
    memory_active = active_prefix(memories[final_idx])

    replay_time = cont_replay["simulated_time"]
    replay_iterations = np.concatenate(
        [
            offset_iterations(from_replay["iterations"], 0),
            offset_iterations(cont_replay["iterations"], continuation_offset, start_iteration=continuation_offset - 1),
        ]
    )
    replay_vacf = np.concatenate([from_replay["vacf"], cont_replay["vacf"]], axis=0)

    ax_vacf, ax_filter, ax_memory = plot_axes[2]
    vacf_mask = replay_time <= 3.5
    star_history(ax_vacf, replay_time[vacf_mask], replay_vacf[:, vacf_mask], replay_iterations, alpha=0.10)
    target_mask = time <= 3.5
    ax_vacf.plot(time[target_mask], curves["target_vacf"][target_mask], color=BLACK, lw=1.85)
    ax_vacf.plot(time[target_mask], curves["diffgle_vacf"][target_mask], color=ORANGE, lw=1.95)
    ax_vacf.axhline(0, color="0.55", lw=0.6)
    ax_vacf.set_xlim(0, 3.5)
    ax_vacf.set_ylim(-0.3, 1.0)
    ax_vacf.set_xlabel("lag time (reduced)")
    ax_vacf.set_ylabel("star polymer\nnormalized VACF")
    despine(ax_vacf)

    filter_x = filter_t[filter_active]
    filter_y = filters[:, filter_active]
    star_history(ax_filter, filter_x, filter_y, iterations, alpha=0.12)
    ax_filter.plot(filter_x, filter_y[final_idx], color=GREEN, lw=2.0)
    ax_filter.axhline(0, color="0.55", lw=0.6)
    ax_filter.set_xlim(0, 2.5)
    ax_filter.set_ylim(-0.06, 0.20)
    ax_filter.set_xlabel("filter lag (reduced)")
    ax_filter.set_ylabel("noise filter", labelpad=1.0)
    despine(ax_filter)

    memory_x = memory_t[memory_active]
    memory_y = memories[:, memory_active]
    star_history(ax_memory, memory_x, memory_y, iterations, alpha=0.12)
    ax_memory.plot(memory_x, memory_y[final_idx], color=BLUE, lw=2.0)
    ax_memory.axhline(0, color="0.55", lw=0.6)
    ax_memory.set_xlim(0, 2.5)
    ax_memory.set_ylim(-0.3, 1.0)
    ax_memory.set_xlabel("memory lag (reduced)")
    ax_memory.set_ylabel(r"$M(\tau)$", labelpad=1.0)
    despine(ax_memory)
    add_iteration_colorbar_inset(ax_memory, iterations)
    from_metrics = load_metrics(STAR_FROM_SCRATCH / "optimization_metrics.csv")
    cont_metrics = load_metrics(STAR_RUN / "optimization_metrics.csv")
    star_metric_iterations = np.array(
        [row["iteration"] for row in from_metrics]
        + [row["iteration"] + continuation_offset for row in cont_metrics],
        dtype=float,
    )
    star_metric_loss = np.array([row["loss"] for row in from_metrics] + [row["loss"] for row in cont_metrics])
    add_loss_inset(ax_memory, star_metric_iterations, star_metric_loss)

    fig.savefig(OUT_DIR / "main_results_combined.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "main_results_combined.pdf", bbox_inches="tight")


def load_metrics(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def offset_iterations(iterations: np.ndarray, offset: int, start_iteration: int | None = None) -> np.ndarray:
    shifted = np.asarray(iterations, dtype=np.int64).copy()
    mask = shifted >= 0
    shifted[mask] += int(offset)
    if start_iteration is not None:
        shifted[~mask] = int(start_iteration)
    return shifted


def active_prefix(curve: np.ndarray, tol: float = 1e-12) -> slice:
    nonzero = np.flatnonzero(np.abs(curve) > tol)
    if nonzero.size == 0:
        return slice(0, curve.size)
    return slice(0, int(nonzero[-1]) + 1)


def star_history(ax, x, ys, iterations, *, alpha=0.12, lw=0.65):
    mask = iterations >= 0
    cmap = plt.get_cmap("viridis")
    norm = Normalize(float(iterations[mask].min()), float(iterations[mask].max()))
    for curve, iteration in zip(ys[mask], iterations[mask]):
        ax.plot(x, curve, color=cmap(norm(iteration)), lw=lw, alpha=alpha, zorder=1)
    return cmap, norm


def make_star_figure() -> None:
    style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curves = np.load(STAR_CURVES)
    from_snap = np.load(STAR_FROM_SCRATCH / "kernel_snapshots.npz")
    cont_snap = np.load(STAR_SNAP)
    from_replay = np.load(STAR_FROM_SCRATCH / "vacf_evolution_replay.npz")
    cont_replay = np.load(STAR_VACF_REPLAY)
    from_summary = json.loads((STAR_FROM_SCRATCH / "star_polymer_diffgle_noforce_summary.json").read_text())
    continuation_offset = int(from_summary["iterations"])

    time = curves["target_time"]
    filter_t = time[: cont_snap["filters"].shape[1]]
    memory_t = time[: cont_snap["memories"].shape[1]]
    iterations = np.concatenate(
        [
            offset_iterations(from_snap["iterations"], 0),
            offset_iterations(cont_snap["iterations"], continuation_offset, start_iteration=continuation_offset - 1),
        ]
    )
    filters = np.concatenate([from_snap["filters"], cont_snap["filters"]], axis=0)
    memories = np.concatenate([from_snap["memories"], cont_snap["memories"]], axis=0)
    final_idx = filters.shape[0] - 1
    filter_active = active_prefix(filters[final_idx])
    memory_active = active_prefix(memories[final_idx])

    replay_time = cont_replay["simulated_time"]
    replay_iterations = np.concatenate(
        [
            offset_iterations(from_replay["iterations"], 0),
            offset_iterations(cont_replay["iterations"], continuation_offset, start_iteration=continuation_offset - 1),
        ]
    )
    replay_vacf = np.concatenate([from_replay["vacf"], cont_replay["vacf"]], axis=0)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.15, 2.15),
        constrained_layout=False,
        gridspec_kw={"wspace": 0.44},
    )
    ax_vacf, ax_filter, ax_memory = axes

    vacf_mask = replay_time <= 3.5
    star_history(ax_vacf, replay_time[vacf_mask], replay_vacf[:, vacf_mask], replay_iterations, alpha=0.11)
    target_mask = time <= 3.5
    ax_vacf.plot(time[target_mask], curves["target_vacf"][target_mask], color=BLACK, lw=1.85)
    ax_vacf.plot(time[target_mask], curves["diffgle_vacf"][target_mask], color=ORANGE, lw=1.95)
    ax_vacf.axhline(0, color="0.55", lw=0.6)
    ax_vacf.set_xlim(0, 3.5)
    ax_vacf.set_ylim(-0.3, 1.0)
    ax_vacf.set_xlabel("lag time")
    ax_vacf.set_ylabel("normalized VACF")
    add_vacf_legend(ax_vacf, label1="target", label2="DiffGLE")
    despine(ax_vacf)

    filter_x = filter_t[filter_active]
    filter_y = filters[:, filter_active]
    cmap, norm = star_history(ax_filter, filter_x, filter_y, iterations, alpha=0.13)
    ax_filter.plot(filter_x, filter_y[final_idx], color=GREEN, lw=2.0)
    ax_filter.axhline(0, color="0.55", lw=0.6)
    ax_filter.set_xlim(0, 2.5)
    ax_filter.set_ylim(-0.06, 0.20)
    ax_filter.set_xlabel("filter lag time")
    ax_filter.set_ylabel("noise filter")
    despine(ax_filter)

    memory_x = memory_t[memory_active]
    memory_y = memories[:, memory_active]
    cmap_m, norm_m = star_history(ax_memory, memory_x, memory_y, iterations, alpha=0.13)
    ax_memory.plot(memory_x, memory_y[final_idx], color=BLUE, lw=2.0)
    ax_memory.axhline(0, color="0.55", lw=0.6)
    ax_memory.set_xlim(0, 2.5)
    ax_memory.set_ylim(-0.3, 1.0)
    ax_memory.set_xlabel("memory lag time")
    ax_memory.set_ylabel(r"$M(\tau)$")
    add_colorbar(fig, ax_memory, cmap_m, norm_m, [0, 500, 1000, 1499])
    despine(ax_memory)

    fig.savefig(OUT_DIR / "star_polymer_compact.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "star_polymer_compact.pdf", bbox_inches="tight")


def main() -> None:
    make_fluid_figure()
    make_combined_results_figure()
    print(OUT_DIR)


if __name__ == "__main__":
    main()
