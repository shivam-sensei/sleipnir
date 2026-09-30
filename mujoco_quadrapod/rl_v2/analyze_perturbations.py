"""Analyze and plot Stage 0 perturbation results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_perturbation(path: Path):
    """Load a JSON perturbation result."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data


def analyze_height_sweep(data: dict):
    """Analyze pure height sweep."""
    samples = data["samples"]
    steps_per_level = data["parameters"]["steps_per_level"]
    n_levels = len(samples) // steps_per_level

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Pure Height Sweep")

    for i, ax in enumerate(axes.flat):
        if i < 3:
            tof_key = ["tof_front", "tof_rear", "tof_left", "tof_right"][i]
            heights = []
            tofs = []
            for level in range(n_levels):
                level_samples = samples[level * steps_per_level : (level + 1) * steps_per_level]
                h = np.mean([s["debug_height"] for s in level_samples])
                t = np.mean([s[tof_key] for s in level_samples])
                heights.append(h)
                tofs.append(t)
            ax.plot(heights, tofs, "o-", linewidth=2)
            ax.set_xlabel("Body height (m)")
            ax.set_ylabel(f"{tof_key} (m)")
            ax.grid(True)
        else:
            gravity_z = [s["gravity_z"] for s in samples]
            ax.plot(gravity_z)
            ax.set_xlabel("Sample")
            ax.set_ylabel("Gravity Z")
            ax.set_ylim([-1.01, -0.99])
            ax.grid(True)

    plt.tight_layout()
    return fig


def analyze_roll_sweep(data: dict):
    """Analyze pure roll sweep."""
    samples = data["samples"]
    steps_per_level = data["parameters"]["steps_per_level"]
    n_levels = len(samples) // steps_per_level

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Pure Roll Sweep")

    roll_vals = []
    tof_front = []
    tof_rear = []
    tof_left = []
    tof_right = []
    gravity_x = []
    gravity_y = []

    for level in range(n_levels):
        level_samples = samples[level * steps_per_level : (level + 1) * steps_per_level]
        roll_vals.append(np.mean([s["debug_roll"] for s in level_samples]))
        tof_front.append(np.mean([s["tof_front"] for s in level_samples]))
        tof_rear.append(np.mean([s["tof_rear"] for s in level_samples]))
        tof_left.append(np.mean([s["tof_left"] for s in level_samples]))
        tof_right.append(np.mean([s["tof_right"] for s in level_samples]))
        gravity_x.append(np.mean([s["gravity_x"] for s in level_samples]))
        gravity_y.append(np.mean([s["gravity_y"] for s in level_samples]))

    axes[0, 0].plot(roll_vals, tof_left, "o-", label="left", linewidth=2)
    axes[0, 0].plot(roll_vals, tof_right, "s-", label="right", linewidth=2)
    axes[0, 0].set_xlabel("Body roll (rad)")
    axes[0, 0].set_ylabel("ToF (m)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(roll_vals, tof_front, "o-", label="front", linewidth=2)
    axes[0, 1].plot(roll_vals, tof_rear, "s-", label="rear", linewidth=2)
    axes[0, 1].set_xlabel("Body roll (rad)")
    axes[0, 1].set_ylabel("ToF (m)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(roll_vals, gravity_x, "o-", linewidth=2)
    axes[1, 0].set_xlabel("Body roll (rad)")
    axes[1, 0].set_ylabel("Gravity X")
    axes[1, 0].grid(True)

    axes[1, 1].plot(roll_vals, gravity_y, "o-", linewidth=2)
    axes[1, 1].set_xlabel("Body roll (rad)")
    axes[1, 1].set_ylabel("Gravity Y")
    axes[1, 1].grid(True)

    plt.tight_layout()
    return fig


def analyze_yaw_sweep(data: dict):
    """Analyze pure yaw sweep; ToF should remain constant."""
    samples = data["samples"]
    steps_per_level = data["parameters"]["steps_per_level"]
    n_levels = len(samples) // steps_per_level

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Pure Yaw Sweep (ToF should be invariant)")

    tof_front = []
    tof_rear = []
    tof_left = []
    tof_right = []

    for level in range(n_levels):
        level_samples = samples[level * steps_per_level : (level + 1) * steps_per_level]
        tof_front.append(np.mean([s["tof_front"] for s in level_samples]))
        tof_rear.append(np.mean([s["tof_rear"] for s in level_samples]))
        tof_left.append(np.mean([s["tof_left"] for s in level_samples]))
        tof_right.append(np.mean([s["tof_right"] for s in level_samples]))

    axes[0].plot(range(n_levels), tof_front, "o-", label="front", linewidth=2)
    axes[0].plot(range(n_levels), tof_rear, "s-", label="rear", linewidth=2)
    axes[0].plot(range(n_levels), tof_left, "^-", label="left", linewidth=2)
    axes[0].plot(range(n_levels), tof_right, "v-", label="right", linewidth=2)
    axes[0].set_xlabel("Yaw level")
    axes[0].set_ylabel("ToF (m)")
    axes[0].set_ylim([0.039, 0.041])
    axes[0].legend()
    axes[0].grid(True)

    tof_std = [
        np.std([tof_front[i], tof_rear[i], tof_left[i], tof_right[i]])
        for i in range(n_levels)
    ]
    axes[1].semilogy(range(n_levels), tof_std, "o-", linewidth=2)
    axes[1].set_xlabel("Yaw level")
    axes[1].set_ylabel("ToF standard deviation (m)")
    axes[1].grid(True)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("stage0_perturbations"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir / "plots"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    perturbations = {
        "pure_height": analyze_height_sweep,
        "pure_roll": analyze_roll_sweep,
        "pure_yaw": analyze_yaw_sweep,
    }

    for name, analyzer in perturbations.items():
        path = args.input_dir / f"{name}.json"
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue
        print(f"Analyzing {name}...", end=" ", flush=True)
        data = load_perturbation(path)
        fig = analyzer(data)
        output_path = args.output_dir / f"{name}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
