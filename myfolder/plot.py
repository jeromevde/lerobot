"""
plot.py — compare LIBERO-Spatial results across policies.

Reads results/*.json produced by run_pi05.py, run_smolvla.py (X-VLA), run_act.py (Pi0Fast).
Saves results/comparison.png.

Usage:
    python plot.py
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")

# Map filename stem → display label
POLICY_LABELS = {
    "pi05":    "π₀.₅",
    "xvla":    "X-VLA",
    "pi0fast": "Pi0Fast",
}


def load_results() -> dict[str, dict]:
    """Load all available policy result JSON files."""
    data = {}
    for stem, label in POLICY_LABELS.items():
        path = RESULTS_DIR / f"{stem}.json"
        if path.exists():
            data[label] = json.loads(path.read_text())
        else:
            print(f"  [missing] {path} — run the corresponding script first")
    return data


def plot_overall(ax, results: dict[str, dict]) -> None:
    """Bar chart of overall success rate per policy."""
    labels = list(results.keys())
    values = [r.get("overall", {}).get("pc_success", 0.0) for r in results.values()]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    bars = ax.bar(labels, values, color=colors[: len(labels)], width=0.5, edgecolor="white")
    ax.set_title("Overall success rate — LIBERO-Spatial", fontsize=13, fontweight="bold")
    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 105)
    ax.axhline(100, color="gray", linewidth=0.5, linestyle="--")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def plot_per_task(ax, results: dict[str, dict]) -> None:
    """Grouped bar chart of per-task success rate."""
    # Collect all task IDs encountered
    all_task_ids: set[int] = set()
    task_data: dict[str, dict[int, float]] = {}

    for label, r in results.items():
        per_task = r.get("per_task", [])
        task_data[label] = {}
        for entry in per_task:
            tid = entry.get("task_id", entry.get("task_index", -1))
            pc = entry.get("metrics", {}).get("pc_success", entry.get("pc_success", 0.0))
            task_data[label][tid] = pc
            all_task_ids.add(tid)

    if not all_task_ids:
        ax.set_visible(False)
        return

    task_ids = sorted(all_task_ids)
    n_tasks = len(task_ids)
    n_policies = len(results)
    width = 0.8 / n_policies
    x = np.arange(n_tasks)
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for i, (label, tdata) in enumerate(task_data.items()):
        values = [tdata.get(tid, 0.0) for tid in task_ids]
        offset = (i - n_policies / 2 + 0.5) * width
        ax.bar(x + offset, values, width=width * 0.9, label=label,
               color=colors[i % len(colors)], edgecolor="white")

    ax.set_title("Per-task success rate — LIBERO-Spatial", fontsize=13, fontweight="bold")
    ax.set_xlabel("Task ID")
    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Task {tid}" for tid in task_ids], rotation=45, ha="right")
    ax.legend(loc="lower right")
    ax.axhline(100, color="gray", linewidth=0.5, linestyle="--")


def main() -> None:
    print("Loading results...")
    results = load_results()

    if not results:
        print("No result files found. Run at least one of: run_pi05.py, run_smolvla.py, run_act.py")
        return

    print(f"Found results for: {', '.join(results.keys())}")
    print()
    for label, r in results.items():
        overall = r.get("overall", {}).get("pc_success", "n/a")
        print(f"  {label:12s}: {overall}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LIBERO-Spatial Policy Comparison", fontsize=15, fontweight="bold", y=1.02)

    plot_overall(axes[0], results)
    plot_per_task(axes[1], results)

    plt.tight_layout()
    out = RESULTS_DIR / "comparison.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
