"""Stage 0 sensor validation under controlled perturbations.

No PPO, reward, or policy. Only sensor characterization.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from sensors import HardwareSensorSimulator, default_sensor_config


@dataclass
class PerturbationSweep:
    """Result of a single perturbation type."""

    name: str
    description: str
    parameters: dict
    samples: list[dict]


def _sample_state(simulator: HardwareSensorSimulator, ctrl: np.ndarray):
    """Collect one sensor sample plus debug state."""
    sample = simulator.step(ctrl)
    data = simulator.data
    base_id = simulator.model.body("base_link").id
    rotation = data.xmat[base_id].reshape(3, 3)
    gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
    roll = float(np.arctan2(gravity[1], -gravity[2]))
    pitch = float(np.arctan2(-gravity[0], -gravity[2]))
    yaw_rate = float((rotation.T @ data.qvel[3:6])[2])

    return {
        "tof_front": float(sample.tof_distance[0]),
        "tof_rear": float(sample.tof_distance[1]),
        "tof_left": float(sample.tof_distance[2]),
        "tof_right": float(sample.tof_distance[3]),
        "tof_valid": sample.tof_valid.tolist(),
        "gravity_x": float(sample.gravity_body[0]),
        "gravity_y": float(sample.gravity_body[1]),
        "gravity_z": float(sample.gravity_body[2]),
        "gyro_x": float(sample.gyro_body[0]),
        "gyro_y": float(sample.gyro_body[1]),
        "gyro_z": float(sample.gyro_body[2]),
        "accel_x": float(sample.accelerometer_body[0]),
        "accel_y": float(sample.accelerometer_body[1]),
        "accel_z": float(sample.accelerometer_body[2]),
        "debug_height": float(data.xpos[base_id, 2]),
        "debug_roll": roll,
        "debug_pitch": pitch,
        "debug_yaw_rate": yaw_rate,
    }


def run_pure_height_sweep(steps_per_level: int = 50) -> PerturbationSweep:
    """Vary body height and observe ToF response."""
    simulator = HardwareSensorSimulator(default_sensor_config())
    heights = np.linspace(0.03, 0.08, 6)
    samples = []

    for height in heights:
        simulator.reset()
        simulator.data.qpos[2] = height
        mujoco.mj_forward(simulator.model, simulator.data)
        for _ in range(steps_per_level):
            samples.append(_sample_state(simulator, np.zeros(simulator.model.nu)))

    return PerturbationSweep(
        name="pure_height",
        description="Body height sweep from 0.03 to 0.08 m",
        parameters={"height_range": [0.03, 0.08], "steps_per_level": steps_per_level},
        samples=samples,
    )


def run_pure_roll_sweep(steps_per_level: int = 50) -> PerturbationSweep:
    """Vary body roll around the X axis and observe ToF/gravity response."""
    simulator = HardwareSensorSimulator(default_sensor_config())
    rolls = np.linspace(-0.3, 0.3, 7)
    samples = []

    for roll in rolls:
        simulator.reset()
        roll_quat = np.array([np.cos(roll / 2), np.sin(roll / 2), 0, 0])
        simulator.data.qpos[3:7] = roll_quat
        mujoco.mj_forward(simulator.model, simulator.data)
        for _ in range(steps_per_level):
            samples.append(_sample_state(simulator, np.zeros(simulator.model.nu)))

    return PerturbationSweep(
        name="pure_roll",
        description="Body roll sweep from -0.3 to +0.3 rad",
        parameters={"roll_range": [-0.3, 0.3], "steps_per_level": steps_per_level},
        samples=samples,
    )


def run_pure_pitch_sweep(steps_per_level: int = 50) -> PerturbationSweep:
    """Vary body pitch around the Y axis and observe ToF/gravity response."""
    simulator = HardwareSensorSimulator(default_sensor_config())
    pitches = np.linspace(-0.3, 0.3, 7)
    samples = []

    for pitch in pitches:
        simulator.reset()
        pitch_quat = np.array([np.cos(pitch / 2), 0, np.sin(pitch / 2), 0])
        simulator.data.qpos[3:7] = pitch_quat
        mujoco.mj_forward(simulator.model, simulator.data)
        for _ in range(steps_per_level):
            samples.append(_sample_state(simulator, np.zeros(simulator.model.nu)))

    return PerturbationSweep(
        name="pure_pitch",
        description="Body pitch sweep from -0.3 to +0.3 rad",
        parameters={"pitch_range": [-0.3, 0.3], "steps_per_level": steps_per_level},
        samples=samples,
    )


def run_pure_yaw_sweep(steps_per_level: int = 50) -> PerturbationSweep:
    """Vary body yaw around the Z axis and observe that ToF should remain constant."""
    simulator = HardwareSensorSimulator(default_sensor_config())
    yaws = np.linspace(0, 2 * np.pi, 9)[:-1]
    samples = []

    for yaw in yaws:
        simulator.reset()
        yaw_quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        simulator.data.qpos[3:7] = yaw_quat
        mujoco.mj_forward(simulator.model, simulator.data)
        for _ in range(steps_per_level):
            samples.append(_sample_state(simulator, np.zeros(simulator.model.nu)))

    return PerturbationSweep(
        name="pure_yaw",
        description="Body yaw sweep from 0 to 2π; ToF should remain constant",
        parameters={"yaw_range": [0, 2 * np.pi], "steps_per_level": steps_per_level},
        samples=samples,
    )


def run_leg_movement_sweep(steps_per_level: int = 50) -> PerturbationSweep:
    """Vary joint angles while body is approximately at neutral; observe ToF/IMU."""
    simulator = HardwareSensorSimulator(default_sensor_config())
    amplitudes = np.linspace(0.0, 0.5, 6)
    samples = []

    for amplitude in amplitudes:
        simulator.reset()
        for step in range(steps_per_level):
            phase = 2.0 * np.pi * step / steps_per_level
            ctrl = amplitude * np.sin(phase) * np.ones(simulator.model.nu)
            samples.append(_sample_state(simulator, ctrl))

    return PerturbationSweep(
        name="leg_movement",
        description="Joint oscillation amplitude sweep; body approximately stationary",
        parameters={"amplitude_range": [0.0, 0.5], "steps_per_level": steps_per_level},
        samples=samples,
    )


def run_disturbance_transient(duration_steps: int = 500) -> PerturbationSweep:
    """Apply a sudden body-position or velocity change; observe gyro/accel response."""
    simulator = HardwareSensorSimulator(default_sensor_config())
    simulator.reset()
    samples = []

    for step in range(duration_steps):
        if step == 100:
            simulator.data.qvel[3:6] = np.array([0.2, 0.0, 0.1])
        samples.append(_sample_state(simulator, np.zeros(simulator.model.nu)))

    return PerturbationSweep(
        name="disturbance_transient",
        description="Angular velocity impulse at step 100; inspect gyro/accel response",
        parameters={"total_steps": duration_steps, "impulse_step": 100},
        samples=samples,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("stage0_perturbations"),
        help="Directory to write JSON perturbation results",
    )
    parser.add_argument(
        "--suite",
        nargs="+",
        default=["height", "roll", "pitch", "yaw", "legs", "transient"],
        help="Which perturbation suites to run",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    if "height" in args.suite:
        print("Running pure_height sweep...", end=" ", flush=True)
        results.append(run_pure_height_sweep())
        print("done")

    if "roll" in args.suite:
        print("Running pure_roll sweep...", end=" ", flush=True)
        results.append(run_pure_roll_sweep())
        print("done")

    if "pitch" in args.suite:
        print("Running pure_pitch sweep...", end=" ", flush=True)
        results.append(run_pure_pitch_sweep())
        print("done")

    if "yaw" in args.suite:
        print("Running pure_yaw sweep...", end=" ", flush=True)
        results.append(run_pure_yaw_sweep())
        print("done")

    if "legs" in args.suite:
        print("Running leg_movement sweep...", end=" ", flush=True)
        results.append(run_leg_movement_sweep())
        print("done")

    if "transient" in args.suite:
        print("Running disturbance_transient...", end=" ", flush=True)
        results.append(run_disturbance_transient())
        print("done")

    for result in results:
        path = args.output_dir / f"{result.name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
