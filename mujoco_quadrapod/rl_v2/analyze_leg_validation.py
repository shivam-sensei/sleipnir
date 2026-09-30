"""Analyze leg-motion validation: sensor interpretability during joint actuation.

Key question: Despite joint motion, can body state (height, roll, pitch) still
be reliably inferred from sensor signals?

Metrics:
- ToF consistency: do the four channels remain coherent during motion?
- Gravity stability: does orientation estimate hold steady?
- Coupling: how strongly do joint commands correlate with sensor changes?
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_experiment(path: Path) -> dict:
    """Load a leg-validation experiment JSON."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def compute_sensor_metrics(samples: list[dict]) -> dict:
    """Compute interpretability metrics from hardware-path sensors."""
    tof_front = np.array([s["tof_front"] for s in samples])
    tof_rear = np.array([s["tof_rear"] for s in samples])
    tof_left = np.array([s["tof_left"] for s in samples])
    tof_right = np.array([s["tof_right"] for s in samples])
    gravity_x = np.array([s["gravity_x"] for s in samples])
    gravity_y = np.array([s["gravity_y"] for s in samples])
    gravity_z = np.array([s["gravity_z"] for s in samples])

    tof_all = np.column_stack([tof_front, tof_rear, tof_left, tof_right])

    return {
        "tof_front_mean": float(np.mean(tof_front)),
        "tof_front_std": float(np.std(tof_front)),
        "tof_front_range": (float(np.min(tof_front)), float(np.max(tof_front))),
        "tof_rear_mean": float(np.mean(tof_rear)),
        "tof_rear_std": float(np.std(tof_rear)),
        "tof_rear_range": (float(np.min(tof_rear)), float(np.max(tof_rear))),
        "tof_left_mean": float(np.mean(tof_left)),
        "tof_left_std": float(np.std(tof_left)),
        "tof_left_range": (float(np.min(tof_left)), float(np.max(tof_left))),
        "tof_right_mean": float(np.mean(tof_right)),
        "tof_right_std": float(np.std(tof_right)),
        "tof_right_range": (float(np.min(tof_right)), float(np.max(tof_right))),
        "tof_std_across_channels": float(np.std(tof_all)),
        "gravity_x_std": float(np.std(gravity_x)),
        "gravity_y_std": float(np.std(gravity_y)),
        "gravity_z_std": float(np.std(gravity_z)),
        "gravity_magnitude": float(np.mean(np.linalg.norm(
            np.column_stack([gravity_x, gravity_y, gravity_z]), axis=1
        ))),
    }


def plot_single_leg_test(experiment: dict, name: str, ax):
    """Plot hardware-path signals from a single-leg test."""
    samples = experiment["samples"]
    tof_all = np.array([[s["tof_front"], s["tof_rear"], s["tof_left"], s["tof_right"]]
                        for s in samples])
    gravity_all = np.array([[s["gravity_x"], s["gravity_y"], s["gravity_z"]]
                            for s in samples])
    debug_height = np.array([s["debug_height"] for s in samples])

    ax.set_title(name)
    ax.plot(tof_all, alpha=0.7, linewidth=1.5)
    ax.set_ylabel("ToF (m)")
    ax.legend(["Front", "Rear", "Left", "Right"], loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.03, 0.05])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("stage0_leg_validation"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--html-report", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir / "analysis"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    experiment_files = sorted(args.input_dir.glob("*.json"))
    if not experiment_files:
        print(f"No experiments found in {args.input_dir}")
        return

    print(f"Analyzing {len(experiment_files)} experiments...")
    metrics_summary = {}

    for path in experiment_files:
        print(f"  {path.name}...", end=" ", flush=True)
        experiment = load_experiment(path)
        metrics = compute_sensor_metrics(experiment["samples"])
        metrics_summary[experiment["name"]] = metrics
        print(f"ToF_std={metrics['tof_std_across_channels']:.6f}")

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nWrote metrics to {args.output_dir / 'metrics.json'}")

    # Generate comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Leg-Motion Validation: Sensor Interpretability")

    single_leg_tests = [
        (f"leg_fr_oscillation.json", "FR leg"),
        (f"leg_rr_oscillation.json", "RR leg"),
        (f"leg_rl_oscillation.json", "RL leg"),
        (f"leg_fl_oscillation.json", "FL leg"),
    ]

    for idx, (filename, label) in enumerate(single_leg_tests):
        if idx >= 4:
            break
        path = args.input_dir / filename
        if path.exists():
            experiment = load_experiment(path)
            plot_single_leg_test(experiment, label, axes.flat[idx])
        else:
            axes.flat[idx].text(0.5, 0.5, f"{label} not found", ha="center", va="center")
            axes.flat[idx].set_xticks([])
            axes.flat[idx].set_yticks([])

    plt.tight_layout()
    plot_path = args.output_dir / "leg_motion_tof_overlay.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Wrote plot to {plot_path}")

    # Summary interpretation
    print("\n" + "=" * 70)
    print("INTERPRETABILITY SUMMARY")
    print("=" * 70)
    for name, metrics in sorted(metrics_summary.items()):
        print(f"\n{name}:")
        print(f"  ToF std across channels: {metrics['tof_std_across_channels']:.6f} m")
        print(f"  ToF front std: {metrics['tof_front_std']:.6f} m")
        print(f"  ToF rear std: {metrics['tof_rear_std']:.6f} m")
        print(f"  ToF left std: {metrics['tof_left_std']:.6f} m")
        print(f"  ToF right std: {metrics['tof_right_std']:.6f} m")
        print(f"  Gravity std (x, y, z): ({metrics['gravity_x_std']:.4f}, " +
              f"{metrics['gravity_y_std']:.4f}, {metrics['gravity_z_std']:.4f})")

        if metrics["tof_std_across_channels"] < 0.001:
            status = "✓ ROBUST: ToF channels remain coherent"
        elif metrics["tof_std_across_channels"] < 0.003:
            status = "⚠ ACCEPTABLE: Slight coupling, likely manageable"
        else:
            status = "✗ PROBLEMATIC: Strong coupling, may confound body state"
        print(f"  {status}")


if __name__ == "__main__":
    main()
