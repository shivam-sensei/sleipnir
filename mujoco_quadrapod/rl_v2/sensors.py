"""Hardware-facing sensor simulation for V2 Stage 0.

This module intentionally has no Gymnasium, PPO, reward, or policy code.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


TOF_NAMES = ("front", "rear", "left", "right")


@dataclass(frozen=True)
class ToFMount:
    """A ToF mounting point and ray expressed in the IMU/body frame."""

    offset: tuple[float, float, float]
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)


@dataclass(frozen=True)
class SensorConfig:
    """Deterministic Stage 0 sensor configuration."""

    model_path: Path
    tof_mounts: tuple[ToFMount, ...] = (
        ToFMount((0.060, 0.0, 0.0)),
        ToFMount((-0.060, 0.0, 0.0)),
        ToFMount((0.0, 0.060, 0.0)),
        ToFMount((0.0, -0.060, 0.0)),
    )
    tof_min_range: float = 0.01
    tof_max_range: float = 2.0

    def __post_init__(self):
        if len(self.tof_mounts) != 4:
            raise ValueError("Stage 0 requires exactly four ToF mounts")
        if not 0.0 < self.tof_min_range < self.tof_max_range:
            raise ValueError("ToF range limits must be positive and ordered")


@dataclass(frozen=True)
class SensorSample:
    """Hardware-path signals plus validity metadata for one simulation step."""

    tof_distance: np.ndarray
    tof_valid: np.ndarray
    gravity_body: np.ndarray
    gyro_body: np.ndarray
    accelerometer_body: np.ndarray


class HardwareSensorSimulator:
    """Generate ToF and IMU signals without exposing simulator state."""

    def __init__(self, config: SensorConfig):
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(config.model_path))
        self.data = mujoco.MjData(self.model)
        self.imu_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )
        if self.imu_site_id < 0:
            raise ValueError("model must define an 'imu' site")
        self._previous_position = None
        self._previous_velocity = None

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        position = self.data.site_xpos[self.imu_site_id].copy()
        self._previous_position = position
        self._previous_velocity = np.zeros(3, dtype=np.float64)

    def step(self, ctrl: np.ndarray) -> SensorSample:
        ctrl = np.asarray(ctrl, dtype=np.float64)
        if ctrl.shape != (self.model.nu,) or not np.all(np.isfinite(ctrl)):
            raise ValueError(f"ctrl must contain {self.model.nu} finite values")
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        return self.read()

    def read(self) -> SensorSample:
        if self._previous_position is None:
            raise RuntimeError("reset the sensor simulator before reading")

        rotation = self.data.site_xmat[self.imu_site_id].reshape(3, 3)
        position = self.data.site_xpos[self.imu_site_id].copy()
        dt = self.model.opt.timestep
        velocity = (position - self._previous_position) / dt
        acceleration = (velocity - self._previous_velocity) / dt
        gravity_world = np.array([0.0, 0.0, -9.81])

        distances = np.full(4, np.nan, dtype=np.float64)
        valid = np.zeros(4, dtype=bool)
        for index, mount in enumerate(self.config.tof_mounts):
            offset = np.asarray(mount.offset, dtype=np.float64)
            direction = np.asarray(mount.direction, dtype=np.float64)
            direction /= np.linalg.norm(direction)
            origin = position + rotation @ offset
            ray = rotation @ direction
            if ray[2] >= -np.finfo(np.float64).eps:
                continue
            distance = -origin[2] / ray[2]
            if self.config.tof_min_range <= distance <= self.config.tof_max_range:
                distances[index] = distance
                valid[index] = True

        self._previous_position = position
        self._previous_velocity = velocity
        return SensorSample(
            tof_distance=distances,
            tof_valid=valid,
            gravity_body=rotation.T @ np.array([0.0, 0.0, -1.0]),
            gyro_body=rotation.T @ self.data.qvel[3:6],
            accelerometer_body=rotation.T @ (acceleration - gravity_world),
        )


def default_sensor_config() -> SensorConfig:
    return SensorConfig(
        model_path=Path(__file__).resolve().parent.parent / "model" / "quadrapod.xml"
    )
