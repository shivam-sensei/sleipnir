from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class SleipnirEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, model_path=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        if model_path is None:
            model_path = Path(__file__).resolve().parent / "quadrapod.xml"
            if not model_path.exists():
                model_path = (
                    Path(__file__).resolve().parent.parent
                    / "model"
                    / "quadrapod.xml"
                )

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.physics_dt = self.model.opt.timestep  # 0.002 s
        self.decimation = 10  # 50 Hz control rate

        # Joint Kinematics & Limits (Radians)
        self.joint_limits = {
            1: (-0.733038, 1.605703),
            2: (0.000000, 3.141593),
            3: (-1.780236, 1.082104),
            4: (0.000000, 3.141593),
            5: (-0.977384, 1.815142),
            6: (-3.141593, 0.000000),
            7: (-1.605703, 0.907571),
            8: (-3.141593, 0.000000),
        }

        # Calibrated nominal pose
        self.q_nominal = np.array(
            [0.4363, 0.0000, -0.3491, 0.0000, 0.4189, 0.0000, -0.3491, 0.0000],
            dtype=np.float32,
        )
        self.action_scale = 0.35  # Increased slightly for dynamic leg clearance
        self.h_target = 0.039  # Target base ground clearance (m)

        # Gait clock frequency (1.5 Hz trot cycle)
        self.gait_freq = 1.5

        # Reward weights
        self.w_vel = 2.0
        self.w_up = 1.0
        self.w_h = 1.0
        self.w_lateral = 0.5
        self.w_rate = 0.04
        self.w_jerk = 0.01

        # Lookup Indices
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, str(i))
            for i in range(1, 9)
        ]
        self.joint_qpos_adr = [
            self.model.jnt_qposadr[j_id] for j_id in self.joint_ids
        ]
        self.actuator_ids = [
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{i}"
            )
            for i in range(1, 9)
        ]

        self.gyro_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro"
        )
        self.tof_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            for name in ["tof_fr", "tof_rr", "tof_rl", "tof_fl"]
        ]
        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )

        # Action Space: 8 normalized joint deltas
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # Observation Space: 30 dimensions
        # [gyro(3), proj_gravity(3), tofs(4), a_prev1(8), a_prev2(8), sin_phi(1), cos_phi(1), cmd_vx(1), cmd_wz(1)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32
        )

        self.a_prev1 = np.zeros(8, dtype=np.float32)
        self.a_prev2 = np.zeros(8, dtype=np.float32)
        self.step_count = 0
        self.max_steps = 1000

        # Commanded velocities
        self.cmd_vx = 0.08
        self.cmd_wz = 0.00

    def _get_obs(self):
        # 1. Torso Gyro (rad/s)
        gyro_adr = self.model.sensor_adr[self.gyro_id]
        gyro = self.data.sensordata[gyro_adr : gyro_adr + 3].copy()

        # 2. Projected Gravity Vector
        torso_quat = self.data.qpos[3:7]
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, torso_quat)
        rot_mat = rot_mat.reshape(3, 3)
        proj_gravity = rot_mat[2, :].copy()

        # 3. 4x Base ToF Distances
        tof_dists = np.zeros(4, dtype=np.float32)
        for i, s_id in enumerate(self.tof_ids):
            adr = self.model.sensor_adr[s_id]
            val = self.data.sensordata[adr]
            tof_dists[i] = val if val > 0.0 else 0.5

        # 4. Periodic Gait Clock
        t = self.step_count * (self.physics_dt * self.decimation)
        phase = 2.0 * np.pi * self.gait_freq * t
        clock = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)

        # 5. Commands
        cmds = np.array([self.cmd_vx, self.cmd_wz], dtype=np.float32)

        obs = np.concatenate(
            [
                gyro,
                proj_gravity,
                tof_dists,
                self.a_prev1,
                self.a_prev2,
                clock,
                cmds,
            ]
        ).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        mujoco.mj_resetData(self.model, self.data)

        # Initial spawn pose with small joint pertubations
        self.data.qpos[0:3] = [0.0, 0.0, 0.06]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        jitter = self.np_random.uniform(-0.08, 0.08, size=8)
        for i, (qpos_adr, q_nom) in enumerate(
            zip(self.joint_qpos_adr, self.q_nominal)
        ):
            self.data.qpos[qpos_adr] = q_nom + jitter[i]

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.q_nominal

        self.a_prev1 = np.zeros(8, dtype=np.float32)
        self.a_prev2 = np.zeros(8, dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.step_count += 1

        # Position setpoints: q_des = q_nom + alpha * a_t
        q_des = self.q_nominal + self.action_scale * action

        for i, act_id in enumerate(self.actuator_ids):
            qmin, qmax = self.joint_limits[i + 1]
            self.data.ctrl[act_id] = np.clip(q_des[i], qmin, qmax)

        # External perturbation pushes during training
        if self.step_count % 80 == 0:
            push_force = self.np_random.uniform(-2.5, 2.5, size=3)
            self.data.xfrc_applied[self.torso_body_id, :3] = push_force
        else:
            self.data.xfrc_applied[self.torso_body_id, :] = 0.0

        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        proj_gravity = obs[3:6]
        tof_dists = obs[6:10]
        mean_tof = np.mean(tof_dists)

        # Linear velocities of the base in world frame
        torso_vx = self.data.qvel[0]
        torso_vy = self.data.qvel[1]

        # Reward terms
        # 1. Forward velocity tracking reward
        r_vel = np.exp(-((torso_vx - self.cmd_vx) ** 2) / 0.02)

        # 2. Lateral drift penalty
        r_lateral = -self.w_lateral * (torso_vy**2)

        # 3. Posture & altitude regulation
        r_up = np.exp(
            -np.linalg.norm(proj_gravity - np.array([0.0, 0.0, 1.0])) ** 2 / 0.25
        )
        r_h = np.exp(-((mean_tof - self.h_target) ** 2) / 0.015)

        # 4. Action smoothness penalties
        rate_pen = self.w_rate * np.sum((action - self.a_prev1) ** 2)
        jerk_pen = self.w_jerk * np.sum(
            (action - 2.0 * self.a_prev1 + self.a_prev2) ** 2
        )

        reward = (
            (self.w_vel * r_vel)
            + (self.w_up * r_up)
            + (self.w_h * r_h)
            + r_lateral
            - rate_pen
            - jerk_pen
        )

        self.a_prev2 = self.a_prev1.copy()
        self.a_prev1 = action.copy()

        # Termination conditions
        is_flipped = proj_gravity[2] < 0.55
        is_collapsed = mean_tof < 0.015 or self.data.qpos[2] < 0.02
        terminated = bool(is_flipped or is_collapsed)
        truncated = bool(self.step_count >= self.max_steps)

        info = {"vx": torso_vx, "mean_tof": mean_tof}
        return obs, reward, terminated, truncated, info