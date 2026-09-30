"""Stage 0 leg-motion validation: deterministic coupling characterization.

Test each leg independently and in combination while measuring body-mounted
sensor response. Goal: Determine whether body state remains inferrable from
sensor signals during normal joint actuation.

No noise, filtering, latency, or PPO. Ideal deterministic characterization only.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from sensors import HardwareSensorSimulator, default_sensor_config


@dataclass
class LegMotionExperiment:
    """Result of a single leg-motion validation test."""

    name: str
    description: str
    parameters: dict
    samples: list[dict]


def _sample_with_debug(simulator: HardwareSensorSimulator, ctrl: np.ndarray):
    """Collect one sensor sample plus full debug state."""
    sample = simulator.step(ctrl)
    data = simulator.data
    base_id = simulator.model.body("base_link").id
    rotation = data.xmat[base_id].reshape(3, 3)
    gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
    roll = float(np.arctan2(gravity[1], -gravity[2]))
    pitch = float(np.arctan2(-gravity[0], -gravity[2]))
    yaw_rate = float((rotation.T @ data.qvel[3:6])[2])

    joint_ids = np.array(
        [simulator.model.joint(str(index)).id for index in range(1, 9)],
        dtype=np.int32,
    )

    return {
        # Hardware path: sensors only
        "tof_front": float(sample.tof_distance[0]),
        "tof_rear": float(sample.tof_distance[1]),
        "tof_left": float(sample.tof_distance[2]),
        "tof_right": float(sample.tof_distance[3]),
        "gravity_x": float(sample.gravity_body[0]),
        "gravity_y": float(sample.gravity_body[1]),
        "gravity_z": float(sample.gravity_body[2]),
        "gyro_x": float(sample.gyro_body[0]),
        "gyro_y": float(sample.gyro_body[1]),
        "gyro_z": float(sample.gyro_body[2]),
        "accel_x": float(sample.accelerometer_body[0]),
        "accel_y": float(sample.accelerometer_body[1]),
        "accel_z": float(sample.accelerometer_body[2]),
        "servo_cmd_1": float(ctrl[0]),
        "servo_cmd_2": float(ctrl[1]),
        "servo_cmd_3": float(ctrl[2]),
        "servo_cmd_4": float(ctrl[3]),
        "servo_cmd_5": float(ctrl[4]),
        "servo_cmd_6": float(ctrl[5]),
        "servo_cmd_7": float(ctrl[6]),
        "servo_cmd_8": float(ctrl[7]),
        # Debug only: true body state
        "debug_height": float(data.xpos[base_id, 2]),
        "debug_roll": roll,
        "debug_pitch": pitch,
        "debug_yaw_rate": yaw_rate,
        "debug_qpos_1": float(data.qpos[simulator.model.jnt_qposadr[joint_ids[0]]]),
        "debug_qpos_2": float(data.qpos[simulator.model.jnt_qposadr[joint_ids[1]]]),
        "debug_qpos_3": float(data.qpos[simulator.model.jnt_qposadr[joint_ids[2]]]),
        "debug_qpos_4": float(data.qpos[simulator.model.jnt_qposadr[joint_ids[3]]]),
        "debug_qpos_5": float(data.qpos[simulator.model.jnt_qposadr[joint_ids[4]]]),
        "debug_qpos_6": float(data.qpos[simulator.model.jnt_qposadr[joint_ids[5]]]),
        "debug_qpos_7": float(data.qpos[simulator.model.jnt_qposadr[joint_ids[6]]]),
        "debug_qpos_8": float(data.qpos[simulator.model.jnt_qposadr[joint_ids[7]]]),
        "debug_qvel_1": float(data.qvel[simulator.model.jnt_dofadr[joint_ids[0]]]),
        "debug_qvel_2": float(data.qvel[simulator.model.jnt_dofadr[joint_ids[1]]]),
        "debug_qvel_3": float(data.qvel[simulator.model.jnt_dofadr[joint_ids[2]]]),
        "debug_qvel_4": float(data.qvel[simulator.model.jnt_dofadr[joint_ids[3]]]),
        "debug_qvel_5": float(data.qvel[simulator.model.jnt_dofadr[joint_ids[4]]]),
        "debug_qvel_6": float(data.qvel[simulator.model.jnt_dofadr[joint_ids[5]]]),
        "debug_qvel_7": float(data.qvel[simulator.model.jnt_dofadr[joint_ids[6]]]),
        "debug_qvel_8": float(data.qvel[simulator.model.jnt_dofadr[joint_ids[7]]]),
    }


def run_single_leg_oscillation(
    leg_pair: str, amplitude: float, frequency: float, duration_steps: int = 500
) -> LegMotionExperiment:
    """Oscillate a single leg pair (e.g., 'FR' for front-right and rr for rear-right).

    Joint mapping:
      FR: joints 1, 2
      RR: joints 3, 4
      RL: joints 5, 6
      FL: joints 7, 8
    """
    simulator = HardwareSensorSimulator(default_sensor_config())
    simulator.reset()
    samples = []

    leg_map = {
        "FR": (0, 1),
        "RR": (2, 3),
        "RL": (4, 5),
        "FL": (6, 7),
    }
    if leg_pair.upper() not in leg_map:
        raise ValueError(f"Unknown leg_pair: {leg_pair}")
    joint_indices = leg_map[leg_pair.upper()]

    for step in range(duration_steps):
        phase = 2.0 * np.pi * frequency * step / duration_steps
        ctrl = np.zeros(simulator.model.nu)
        ctrl[joint_indices[0]] = amplitude * np.sin(phase)
        ctrl[joint_indices[1]] = amplitude * np.sin(phase)
        samples.append(_sample_with_debug(simulator, ctrl))

    return LegMotionExperiment(
        name=f"leg_{leg_pair.lower()}_oscillation",
        description=f"Single {leg_pair} leg oscillation at amplitude={amplitude}, freq={frequency} Hz",
        parameters={
            "leg": leg_pair.upper(),
            "amplitude": amplitude,
            "frequency": frequency,
            "duration_steps": duration_steps,
        },
        samples=samples,
    )


def run_symmetric_leg_oscillation(
    amplitude: float, frequency: float, duration_steps: int = 500
) -> LegMotionExperiment:
    """Oscillate all four legs symmetrically (all joints in phase)."""
    simulator = HardwareSensorSimulator(default_sensor_config())
    simulator.reset()
    samples = []

    for step in range(duration_steps):
        phase = 2.0 * np.pi * frequency * step / duration_steps
        ctrl = amplitude * np.sin(phase) * np.ones(simulator.model.nu)
        samples.append(_sample_with_debug(simulator, ctrl))

    return LegMotionExperiment(
        name="legs_all_symmetric",
        description=f"All legs in phase at amplitude={amplitude}, freq={frequency} Hz",
        parameters={
            "amplitude": amplitude,
            "frequency": frequency,
            "duration_steps": duration_steps,
        },
        samples=samples,
    )


def run_diagonal_leg_oscillation(
    amplitude: float, frequency: float, duration_steps: int = 500
) -> LegMotionExperiment:
    """Oscillate diagonally opposite legs in phase (FR/RL in phase, RR/FL in phase, 180° apart)."""
    simulator = HardwareSensorSimulator(default_sensor_config())
    simulator.reset()
    samples = []

    for step in range(duration_steps):
        phase = 2.0 * np.pi * frequency * step / duration_steps
        ctrl = np.zeros(simulator.model.nu)
        # FR and RL in phase
        ctrl[0] = amplitude * np.sin(phase)
        ctrl[1] = amplitude * np.sin(phase)
        ctrl[4] = amplitude * np.sin(phase)
        ctrl[5] = amplitude * np.sin(phase)
        # RR and FL 180° out of phase
        ctrl[2] = amplitude * np.sin(phase + np.pi)
        ctrl[3] = amplitude * np.sin(phase + np.pi)
        ctrl[6] = amplitude * np.sin(phase + np.pi)
        ctrl[7] = amplitude * np.sin(phase + np.pi)
        samples.append(_sample_with_debug(simulator, ctrl))

    return LegMotionExperiment(
        name="legs_diagonal",
        description=f"Diagonal legs at amplitude={amplitude}, freq={frequency} Hz",
        parameters={
            "amplitude": amplitude,
            "frequency": frequency,
            "duration_steps": duration_steps,
        },
        samples=samples,
    )


def run_stepping_gait_simulation(
    amplitude: float, frequency: float, duration_steps: int = 500
) -> LegMotionExperiment:
    """Simulate a stepping gait: front legs in phase, rear legs in phase, 180° apart (trotting pattern)."""
    simulator = HardwareSensorSimulator(default_sensor_config())
    simulator.reset()
    samples = []

    for step in range(duration_steps):
        phase = 2.0 * np.pi * frequency * step / duration_steps
        ctrl = np.zeros(simulator.model.nu)
        # Front legs (FR, FL)
        ctrl[0] = amplitude * np.sin(phase)
        ctrl[1] = amplitude * np.sin(phase)
        ctrl[6] = amplitude * np.sin(phase)
        ctrl[7] = amplitude * np.sin(phase)
        # Rear legs (RR, RL) 180° out
        ctrl[2] = amplitude * np.sin(phase + np.pi)
        ctrl[3] = amplitude * np.sin(phase + np.pi)
        ctrl[4] = amplitude * np.sin(phase + np.pi)
        ctrl[5] = amplitude * np.sin(phase + np.pi)
        samples.append(_sample_with_debug(simulator, ctrl))

    return LegMotionExperiment(
        name="legs_trotting_gait",
        description=f"Trotting gait (front/rear alternation) at amplitude={amplitude}, freq={frequency} Hz",
        parameters={
            "amplitude": amplitude,
            "frequency": frequency,
            "duration_steps": duration_steps,
        },
        samples=samples,
    )


def run_amplitude_sweep(
    amplitude_range: tuple[float, float],
    n_levels: int,
    frequency: float,
    duration_per_level: int = 200,
) -> LegMotionExperiment:
    """Sweep oscillation amplitude while keeping frequency constant."""
    simulator = HardwareSensorSimulator(default_sensor_config())
    samples = []
    amplitudes = np.linspace(amplitude_range[0], amplitude_range[1], n_levels)

    for amplitude in amplitudes:
        simulator.reset()
        for step in range(duration_per_level):
            phase = 2.0 * np.pi * frequency * step / duration_per_level
            ctrl = amplitude * np.sin(phase) * np.ones(simulator.model.nu)
            samples.append(_sample_with_debug(simulator, ctrl))

    return LegMotionExperiment(
        name="legs_amplitude_sweep",
        description=f"All-legs oscillation with amplitude sweep {amplitude_range} at {frequency} Hz",
        parameters={
            "amplitude_range": amplitude_range,
            "n_levels": n_levels,
            "frequency": frequency,
            "duration_per_level": duration_per_level,
        },
        samples=samples,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("stage0_leg_validation"),
        help="Directory to write JSON results",
    )
    parser.add_argument(
        "--suite",
        nargs="+",
        default=[
            "fr",
            "rr",
            "rl",
            "fl",
            "symmetric",
            "diagonal",
            "trotting",
            "amplitude",
        ],
        help="Which tests to run",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    if "fr" in args.suite:
        print("Running FR leg oscillation...", end=" ", flush=True)
        results.append(run_single_leg_oscillation("FR", amplitude=0.3, frequency=1.0))
        print("done")

    if "rr" in args.suite:
        print("Running RR leg oscillation...", end=" ", flush=True)
        results.append(run_single_leg_oscillation("RR", amplitude=0.3, frequency=1.0))
        print("done")

    if "rl" in args.suite:
        print("Running RL leg oscillation...", end=" ", flush=True)
        results.append(run_single_leg_oscillation("RL", amplitude=0.3, frequency=1.0))
        print("done")

    if "fl" in args.suite:
        print("Running FL leg oscillation...", end=" ", flush=True)
        results.append(run_single_leg_oscillation("FL", amplitude=0.3, frequency=1.0))
        print("done")

    if "symmetric" in args.suite:
        print("Running symmetric all-legs oscillation...", end=" ", flush=True)
        results.append(run_symmetric_leg_oscillation(amplitude=0.3, frequency=1.0))
        print("done")

    if "diagonal" in args.suite:
        print("Running diagonal leg oscillation...", end=" ", flush=True)
        results.append(run_diagonal_leg_oscillation(amplitude=0.3, frequency=1.0))
        print("done")

    if "trotting" in args.suite:
        print("Running trotting gait simulation...", end=" ", flush=True)
        results.append(run_stepping_gait_simulation(amplitude=0.3, frequency=1.0))
        print("done")

    if "amplitude" in args.suite:
        print("Running amplitude sweep...", end=" ", flush=True)
        results.append(
            run_amplitude_sweep(
                amplitude_range=(0.0, 0.5),
                n_levels=6,
                frequency=1.0,
                duration_per_level=200,
            )
        )
        print("done")

    for result in results:
        path = args.output_dir / f"{result.name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
