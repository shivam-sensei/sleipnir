"""Run deterministic Stage 0 sensor simulation and write a CSV trace."""

import argparse
import csv
from pathlib import Path

import numpy as np

from sensors import TOF_NAMES, default_sensor_config, HardwareSensorSimulator


def _field_names():
    fields = ["step", "time"]
    fields += [f"tof_{name}" for name in TOF_NAMES]
    fields += [f"tof_{name}_valid" for name in TOF_NAMES]
    fields += [f"gravity_{axis}" for axis in "xyz"]
    fields += [f"gyro_{axis}" for axis in "xyz"]
    fields += [f"accel_{axis}" for axis in "xyz"]
    fields += ["debug_height", "debug_roll", "debug_pitch", "debug_yaw_rate"]
    fields += [f"debug_qpos_{index + 1}" for index in range(8)]
    fields += [f"debug_qvel_{index + 1}" for index in range(8)]
    return fields


def _debug_values(simulator):
    data = simulator.data
    base_id = simulator.model.body("base_link").id
    rotation = data.xmat[base_id].reshape(3, 3)
    gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
    roll = float(np.arctan2(gravity[1], -gravity[2]))
    pitch = float(np.arctan2(-gravity[0], -gravity[2]))
    joint_ids = np.array(
        [
            simulator.model.joint(str(index)).id
            for index in range(1, 9)
        ],
        dtype=np.int32,
    )
    return [
        float(data.xpos[base_id, 2]),
        roll,
        pitch,
        float((rotation.T @ data.qvel[3:6])[2]),
        *data.qpos[simulator.model.jnt_qposadr[joint_ids]],
        *data.qvel[simulator.model.jnt_dofadr[joint_ids]],
    ]


def run(output: Path, steps: int):
    simulator = HardwareSensorSimulator(default_sensor_config())
    simulator.reset()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(_field_names())
        for step in range(steps):
            time = step * simulator.model.opt.timestep
            phase = 0.35 * np.sin(2.0 * np.pi * 0.5 * time)
            ctrl = np.full(simulator.model.nu, phase)
            sample = simulator.step(ctrl)
            writer.writerow(
                [
                    step,
                    time,
                    *sample.tof_distance,
                    *sample.tof_valid.astype(int),
                    *sample.gravity_body,
                    *sample.gyro_body,
                    *sample.accelerometer_body,
                    *_debug_values(simulator),
                ]
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("stage0_sensor_log.csv"))
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    run(args.output, args.steps)
    print(f"Wrote {args.steps} samples to {args.output}")


if __name__ == "__main__":
    main()
