"""Build the README optimization GIF from the bundled CO2 history arrays."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT / "paper_data/optimization_histories/co2_symdetach_lr3e5_iter372/history_arrays.npz"
OUTPUT_PATH = ROOT / "docs/assets/co2_diffgle_optimization.gif"


def frame_indices(n_items, n_frames=80):
    indices = np.unique(np.rint(np.linspace(0, n_items - 1, n_frames)).astype(int))
    if indices[-1] != n_items - 1:
        indices = np.r_[indices, n_items - 1]
    return indices


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    history = np.load(HISTORY_PATH)
    iterations = history["iterations"]
    vacf_gle = history["vacf_gle"]
    vacf_target = history["vacf_aa"]
    filters = history["filters"]
    memories = history["memories"]
    loss_iterations = history["metrics_iteration"]
    losses = history["metrics_loss"]
    rmses = history["metrics_vacf_rmse"]

    frames = frame_indices(len(iterations))
    vacf_time = np.arange(vacf_target.size) / 1000.0
    kernel_time = np.arange(memories.shape[1]) / 1000.0
    vacf_window = 500

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.linewidth": 0.8,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.8), dpi=140)
    fig.patch.set_facecolor("white")
    ax_vacf, ax_filter, ax_memory, ax_loss = axes.ravel()

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(False)

    context_ids = frame_indices(len(iterations), n_frames=28)
    for idx in context_ids:
        ax_vacf.plot(vacf_time[:vacf_window], vacf_gle[idx, :vacf_window], color="#9fb7d6", lw=0.7, alpha=0.18)
        ax_filter.plot(kernel_time, filters[idx], color="#9b7cb8", lw=0.7, alpha=0.18)
        ax_memory.plot(kernel_time, memories[idx], color="#79aeca", lw=0.7, alpha=0.18)

    ax_vacf.plot(vacf_time[:vacf_window], vacf_target[:vacf_window], color="black", lw=2.0, label="AA target")
    vacf_line, = ax_vacf.plot([], [], color="#d95f02", lw=2.2, label="DiffGLE")
    ax_vacf.set_xlim(0, 0.5)
    ax_vacf.set_ylim(-0.35, 1.05)
    ax_vacf.set_xlabel("lag / ps")
    ax_vacf.set_ylabel("normalized VACF")
    ax_vacf.legend(frameon=False, loc="upper right")

    filter_line, = ax_filter.plot([], [], color="#1b7f4b", lw=2.2)
    ax_filter.axhline(0, color="0.65", lw=0.8)
    ax_filter.set_xlim(0, 0.2)
    ax_filter.set_ylim(-0.022, 0.022)
    ax_filter.set_xlabel("filter lag / ps")
    ax_filter.set_ylabel("noise filter")

    memory_line, = ax_memory.plot([], [], color="#1f78b4", lw=2.2)
    ax_memory.axhline(0, color="0.65", lw=0.8)
    ax_memory.set_xlim(0, 0.2)
    ax_memory.set_ylim(-0.008, 0.032)
    ax_memory.set_xlabel("memory lag / ps")
    ax_memory.set_ylabel(r"$M(\tau)$")

    positive_values = np.r_[losses[np.isfinite(losses) & (losses > 0)], rmses[np.isfinite(rmses) & (rmses > 0)]]
    ax_loss.set_yscale("log")
    ax_loss.set_xlim(0, 300)
    ax_loss.set_ylim(positive_values.min() * 0.7, positive_values.max() * 1.4)
    loss_line, = ax_loss.plot([], [], color="#9a5a13", lw=1.7, label="training loss")
    rmse_line, = ax_loss.plot([], [], color="#2b6cb0", lw=1.5, label="VACF RMSE")
    loss_point, = ax_loss.plot([], [], "o", color="#9a5a13", ms=4)
    rmse_point, = ax_loss.plot([], [], "o", color="#2b6cb0", ms=4)
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("loss / RMSE")
    ax_loss.legend(frameon=False, loc="upper right")

    fig.suptitle("CO2 DiffGLE optimization from saved iteration data", y=0.985, fontweight="bold")
    iteration_label = fig.text(0.5, 0.018, "", ha="center", va="center", color="0.30", fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 0.955])

    def update(frame_number):
        saved_index = int(frames[frame_number])
        iteration = int(iterations[saved_index])
        vacf_line.set_data(vacf_time[:vacf_window], vacf_gle[saved_index, :vacf_window])
        filter_line.set_data(kernel_time, filters[saved_index])
        memory_line.set_data(kernel_time, memories[saved_index])

        history_mask = loss_iterations <= iteration
        loss_line.set_data(loss_iterations[history_mask], losses[history_mask])
        rmse_line.set_data(loss_iterations[history_mask], rmses[history_mask])
        if np.any(history_mask):
            last = np.where(history_mask)[0][-1]
            loss_point.set_data([loss_iterations[last]], [losses[last]])
            rmse_point.set_data([loss_iterations[last]], [rmses[last]])

        iteration_label.set_text(f"saved iteration {iteration}")
        return vacf_line, filter_line, memory_line, loss_line, rmse_line, loss_point, rmse_point, iteration_label

    gif = animation.FuncAnimation(fig, update, frames=len(frames), interval=95, blit=False)
    gif.save(OUTPUT_PATH, writer=animation.PillowWriter(fps=10))
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
