#!/usr/bin/env python3
"""
Stage 0 Sensor Viewer

Visualizes the hardware sensor configuration in real-time using MuJoCo viewer:
- 4 ToF rays (front, rear, left, right) rendered as lines to ground
- IMU frame/site on body
- Live sensor values displayed as text overlay
- Debug values (height, roll, pitch, yaw rate) optional

Uses the same HardwareSensorSimulator as the logging pipeline.
Allows interactive manipulation of robot pose to test sensor responses.
"""

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sensors import HardwareSensorSimulator, SensorConfig, ToFMount, SensorSample


@dataclass
class SensorViewerConfig:
    """Configuration for sensor viewer."""
    mjcf_path: str = "model/quadrapod.xml"
    tof_ray_color: tuple = (0.2, 1.0, 0.2, 0.8)  # RGBA green
    imu_frame_color: tuple = (0.0, 0.5, 1.0, 0.8)  # RGBA blue
    tof_intersection_color: tuple = (1.0, 0.2, 0.2, 0.8)  # RGBA red
    text_size: float = 12.0
    show_debug_values: bool = True


class SensorViewer:
    """MuJoCo viewer with hardware sensor visualization overlay."""

    def __init__(self, config: SensorViewerConfig = None):
        self.config = config or SensorViewerConfig()
        
        # Load model
        mjcf_path = Path(__file__).parent.parent / self.config.mjcf_path
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        
        # Initialize sensor simulator with SensorConfig
        sensor_config = SensorConfig(model_path=mjcf_path)
        self.sensor_sim = HardwareSensorSimulator(sensor_config)
        
        # State tracking
        self.paused = False
        self.show_tof_rays = True
        self.show_imu_frame = True
        self.sample = None
        
        print("Stage 0 Sensor Viewer initialized.")
        print("  Keys:")
        print("    [space] - pause/resume")
        print("    [t]     - toggle ToF rays")
        print("    [i]     - toggle IMU frame")
        print("    [d]     - toggle debug values")
        print("    [r]     - reset to neutral pose")
        print("    [q]     - quit")

    def _step_simulation(self):
        """Step simulation if not paused."""
        if not self.paused:
            mujoco.mj_step(self.model, self.data)

    def _update_sensors(self):
        """Update sensor readings based on current viewer state."""
        # Compute sensors directly from viewer's data to stay in sync
        sensor_config = self.sensor_sim.config
        imu_site_id = self.sensor_sim.imu_site_id
        
        rotation = self.data.site_xmat[imu_site_id].reshape(3, 3)
        position = self.data.site_xpos[imu_site_id].copy()
        
        # Compute distances for each ToF mount
        distances = np.full(4, np.nan, dtype=np.float64)
        valid = np.zeros(4, dtype=bool)
        
        for index, mount in enumerate(sensor_config.tof_mounts):
            offset = np.asarray(mount.offset, dtype=np.float64)
            direction = np.asarray(mount.direction, dtype=np.float64)
            direction /= np.linalg.norm(direction)
            origin = position + rotation @ offset
            ray = rotation @ direction
            
            # Ray must point downward
            if ray[2] >= -np.finfo(np.float64).eps:
                continue
            
            distance = -origin[2] / ray[2]
            if sensor_config.tof_min_range <= distance <= sensor_config.tof_max_range:
                distances[index] = distance
                valid[index] = True
        
        # Compute IMU signals
        gravity_body = rotation.T @ np.array([0.0, 0.0, -1.0])
        gyro_body = rotation.T @ self.data.qvel[3:6]
        
        # Create a simple sample-like object
        class SimpleSensorSample:
            pass
        
        self.sample = SimpleSensorSample()
        self.sample.tof_front = distances[0] if valid[0] else np.nan
        self.sample.tof_rear = distances[1] if valid[1] else np.nan
        self.sample.tof_left = distances[2] if valid[2] else np.nan
        self.sample.tof_right = distances[3] if valid[3] else np.nan
        self.sample.gravity_x = gravity_body[0]
        self.sample.gravity_y = gravity_body[1]
        self.sample.gravity_z = gravity_body[2]
        self.sample.gyro_x = gyro_body[0]
        self.sample.gyro_y = gyro_body[1]
        self.sample.gyro_z = gyro_body[2]
        self.sample.accel_x = 0.0
        self.sample.accel_y = 0.0
        self.sample.accel_z = 0.0

    def _reset_user_scene(self, viewer):
        """Clear all custom visualization geometry."""
        viewer.user_scn.ngeom = 0


    def _add_line(self, viewer, start, end, rgba, width=0.0015):
        """Add a thin capsule between two world-space points."""
        scn = viewer.user_scn

        if scn.ngeom >= scn.maxgeom:
            return

        geom = scn.geoms[scn.ngeom]
        scn.ngeom += 1

        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([width, width, width]),
            np.zeros(3),
            np.eye(3).flatten(),
            np.asarray(rgba, dtype=np.float64),
        )

        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            width,
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )


    def _add_sphere(self, viewer, position, radius, rgba):
        """Add a sphere to the custom visualization scene."""
        scn = viewer.user_scn

        if scn.ngeom >= scn.maxgeom:
            return

        geom = scn.geoms[scn.ngeom]
        scn.ngeom += 1

        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, radius, radius]),
            np.asarray(position, dtype=np.float64),
            np.eye(3).flatten(),
            np.asarray(rgba, dtype=np.float64),
        )

    def _render_tof_rays(self, viewer):
        """Render the exact ToF rays used by HardwareSensorSimulator."""

        if self.sample is None:
            return

        imu_site_id = self.sensor_sim.imu_site_id

        # EXACT same frame used by sensors.py
        rotation = self.data.site_xmat[imu_site_id].reshape(3, 3)
        position = self.data.site_xpos[imu_site_id].copy()

        for index, mount in enumerate(self.sensor_sim.config.tof_mounts):

            if not self.sample.tof_valid[index]:
                continue

            offset = np.asarray(mount.offset, dtype=np.float64)
            direction = np.asarray(mount.direction, dtype=np.float64)

            direction /= np.linalg.norm(direction)

            # EXACT same origin calculation as sensors.py
            origin = position + rotation @ offset

            # EXACT same ray calculation as sensors.py
            ray = rotation @ direction

            distance = self.sample.tof_distance[index]

            if not np.isfinite(distance):
                continue

            # Exact measured endpoint
            intersection = origin + ray * distance

            # Draw sensor ray
            self._add_line(
                viewer,
                origin,
                intersection,
                self.config.tof_ray_color,
                width=0.001,
            )

            # Draw sensor position
            self._add_sphere(
                viewer,
                origin,
                radius=0.0025,
                rgba=self.config.tof_ray_color,
            )

            # Draw measured ground intersection
            self._add_sphere(
                viewer,
                intersection,
                radius=0.0025,
                rgba=self.config.tof_intersection_color,
            )

    def _render_imu_frame(self, viewer):
        """Render the IMU coordinate frame."""
        if not self.show_imu_frame:
            return

        body = self.data.body("base_link")
        body_pos = body.xpos.copy()
        body_rot = body.xmat.reshape(3, 3)

        frame_size = 0.03

        x_end = body_pos + body_rot[:, 0] * frame_size
        y_end = body_pos + body_rot[:, 1] * frame_size
        z_end = body_pos + body_rot[:, 2] * frame_size

        # X = red
        self._add_line(
            viewer,
            body_pos,
            x_end,
            (1.0, 0.0, 0.0, 1.0),
            width=0.0015,
        )

        # Y = green
        self._add_line(
            viewer,
            body_pos,
            y_end,
            (0.0, 1.0, 0.0, 1.0),
            width=0.0015,
        )

        # Z = blue
        self._add_line(
            viewer,
            body_pos,
            z_end,
            (0.0, 0.0, 1.0, 1.0),
            width=0.0015,
        )

        self._add_sphere(
            viewer,
            body_pos,
            radius=0.003,
            rgba=self.config.imu_frame_color,
        )

    def _format_sensor_text(self) -> str:
        """Format sensor values for text overlay."""
        if self.sample is None:
            return "Initializing sensors..."
        
        # Hardware observation values
        lines = [
            "=== HARDWARE SENSOR PATH ===",
            "",
            f"ToF Front: {self.sample.tof_front:.4f} m",
            f"ToF Rear:  {self.sample.tof_rear:.4f} m",
            f"ToF Left:  {self.sample.tof_left:.4f} m",
            f"ToF Right: {self.sample.tof_right:.4f} m",
            "",
            f"Gravity: ({self.sample.gravity_x:+.3f}, {self.sample.gravity_y:+.3f}, {self.sample.gravity_z:+.3f})",
            "",
            f"Gyro X (roll rate):  {self.sample.gyro_x:+.3f} rad/s",
            f"Gyro Y (pitch rate): {self.sample.gyro_y:+.3f} rad/s",
            f"Gyro Z (yaw rate):   {self.sample.gyro_z:+.3f} rad/s",
            "",
            f"Accel X: {self.sample.accel_x:+.3f} m/s²",
            f"Accel Y: {self.sample.accel_y:+.3f} m/s²",
            f"Accel Z: {self.sample.accel_z:+.3f} m/s²",
        ]
        
        # Optional debug values
        if self.config.show_debug_values:
            lines.extend([
                "",
                "=== DEBUG VALUES (not in PPO) ===",
                f"Body Height (from ToF mean): {(self.sample.tof_front + self.sample.tof_rear + self.sample.tof_left + self.sample.tof_right) / 4:.4f} m",
            ])
            
            # Compute roll/pitch from gravity if possible
            gravity = np.array([self.sample.gravity_x, self.sample.gravity_y, self.sample.gravity_z])
            gravity_mag = np.linalg.norm(gravity)
            if gravity_mag > 0.1:
                gravity_norm = gravity / gravity_mag
                roll = np.arctan2(gravity_norm[1], gravity_norm[2])
                pitch = np.arctan2(-gravity_norm[0], gravity_norm[2])
                lines.append(f"Roll (from gravity):  {np.degrees(roll):+.1f}°")
                lines.append(f"Pitch (from gravity): {np.degrees(pitch):+.1f}°")
                lines.append(f"Yaw rate: {self.sample.gyro_z:+.3f} rad/s")
            
            # Joint state (debug only)
            lines.append("")
            lines.append("=== JOINT STATE (debug only, not in observation) ===")
            for i in range(1, 9):
                q = self.data.qpos[i]
                qdot = self.data.qvel[i]
                lines.append(f"J{i}: q={q:+.3f} rad, qdot={qdot:+.3f} rad/s")
        
        lines.extend([
            "",
            "[space]=pause  [t]=toggle ToF  [i]=toggle IMU  [d]=debug  [r]=reset  [q]=quit"
        ])
        
        return "\n".join(lines)

    def _on_key_press(self, keycode):
        """Handle keyboard input."""
        if keycode == ord(" "):  # Space
            self.paused = not self.paused
            print(f"Paused: {self.paused}")
        elif keycode == ord("t"):  # Toggle ToF rays
            self.show_tof_rays = not self.show_tof_rays
            print(f"ToF rays: {'ON' if self.show_tof_rays else 'OFF'}")
        elif keycode == ord("i"):  # Toggle IMU frame
            self.show_imu_frame = not self.show_imu_frame
            print(f"IMU frame: {'ON' if self.show_imu_frame else 'OFF'}")
        elif keycode == ord("d"):  # Toggle debug values
            self.config.show_debug_values = not self.config.show_debug_values
            print(f"Debug values: {'ON' if self.config.show_debug_values else 'OFF'}")
        elif keycode == ord("r"):  # Reset to neutral
            self.data.qpos[:] = 0
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
            print("Reset to neutral pose")
        elif keycode == ord("q"):  # Quit
            return False
        return True

    def run(self):
        """Run the interactive MuJoCo sensor viewer."""

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:

            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

            while viewer.is_running():

                # Advance simulation
                self._step_simulation()

                # Update hardware-path sensor values
                self._update_sensors()

                # Clear previous custom geometry
                self._reset_user_scene(viewer)

                # Draw ToF rays
                if self.show_tof_rays:
                    self._render_tof_rays(viewer)

                # Draw IMU coordinate frame
                if self.show_imu_frame:
                    self._render_imu_frame(viewer)
                # Push custom scene to MuJoCo viewer
                viewer.sync()


def main():
    """Main entry point."""
    viewer = SensorViewer()
    viewer.run()


if __name__ == "__main__":
    main()
