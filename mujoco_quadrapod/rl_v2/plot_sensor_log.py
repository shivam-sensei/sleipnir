"""Plot hardware-path signals from a Stage 0 CSV trace."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sensors import TOF_NAMES


def load_trace(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("sensor log is empty")
    time = np.array([float(row["time"]) for row in rows])
    tof = np.array(
        [[float(row[f"tof_{name}"]) for name in TOF_NAMES] for row in rows]
    )
    gravity = np.array(
        [[float(row[f"gravity_{axis}"]) for axis in "xyz"] for row in rows]
    )
    gyro = np.array(
        [[float(row[f"gyro_{axis}"]) for axis in "xyz"] for row in rows]
    )
    accel = np.array(
        [[float(row[f"accel_{axis}"]) for axis in "xyz"] for row in rows]
    )
    return time, tof, gravity, gyro, accel


def plot(path: Path, output: Path | None):
    time, tof, gravity, gyro, accel = load_trace(path)
    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 10))
    axes[0].plot(time, tof)
    axes[0].set_ylabel("ToF (m)")
    axes[0].legend(TOF_NAMES)
    axes[1].plot(time, gravity)
    axes[1].set_ylabel("Gravity")
    axes[1].legend(("x", "y", "z"))
    axes[2].plot(time, gyro)
    axes[2].set_ylabel("Gyro (rad/s)")
    axes[2].legend(("x", "y", "z"))
    axes[3].plot(time, accel)
    axes[3].set_ylabel("Accel (m/s²)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(("x", "y", "z"))
    figure.tight_layout()
    if output is None:
        plt.show()
    else:
        figure.savefig(output, dpi=150)
        print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot(args.input, args.output)


if __name__ == "__main__":
    main()
