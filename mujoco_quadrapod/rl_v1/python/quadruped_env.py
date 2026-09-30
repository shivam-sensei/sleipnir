from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


DRIFT_STANCE_SERVO = np.array(
    [40.6, 0.0, 96.6, 0.0, 62.8, 180.0, 114.2, 180.0],
    dtype=np.float64,
)


class QuadrupedEnv(gym.Env):
    """Standing-task Gymnasium environment for the MuJoCo quadruped."""

    metadata = {"render_modes": ["human"]}
    max_action_delta = 0.05

    def __init__(
        self,
        render_mode=None,
        joint4_action_weight=1.0,
        joint4_velocity_weight=1.0,
        joint8_action_weight=1.0,
        joint8_velocity_weight=1.0,
        yaw_velocity_weight=0.1,
        leg_collision_weight=2.0,
        joint_position_weight=5.0,
        stance_target_servo=None,
    ):
        super().__init__()
        if min(
            joint4_action_weight,
            joint4_velocity_weight,
            joint8_action_weight,
            joint8_velocity_weight,
        ) <= 0:
            raise ValueError("joint-specific penalty weights must be positive")
        if yaw_velocity_weight < 0:
            raise ValueError("yaw velocity weight must be non-negative")
        if leg_collision_weight < 0:
            raise ValueError("leg collision weight must be non-negative")
        if joint_position_weight < 0:
            raise ValueError("joint position weight must be non-negative")
        model_path = (
            Path(__file__).resolve().parent.parent.parent
            / "model"
            / "quadrapod.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.n_joints = 8
        self.penalty_weights = np.ones(self.n_joints, dtype=np.float64)
        self.penalty_weights[3] = joint4_action_weight
        self.penalty_weights[7] = joint8_action_weight
        self.velocity_penalty_weights = np.ones(self.n_joints, dtype=np.float64)
        self.velocity_penalty_weights[3] = joint4_velocity_weight
        self.velocity_penalty_weights[7] = joint8_velocity_weight
        self.yaw_velocity_weight = yaw_velocity_weight
        self.leg_collision_weight = leg_collision_weight
        self.joint_position_weight = joint_position_weight

        self.joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, str(number))
                for number in range(1, self.n_joints + 1)
            ],
            dtype=np.int32,
        )
        self.actuator_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{number}"
                )
                for number in range(1, self.n_joints + 1)
            ],
            dtype=np.int32,
        )
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_addresses = self.model.jnt_dofadr[self.joint_ids]
        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        self.ground_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
        )
        self.foot_body_ids = np.array(
            [
                self.model.site_bodyid[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                ]
                for name in ("fr_foot", "rr_foot", "rl_foot", "fl_foot")
            ],
            dtype=np.int32,
        )
        leg_root_names = ("fr_arm_1", "rr_arm_1", "rl_arm_1", "fl_arm_1")
        self.leg_root_body_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in leg_root_names
            ],
            dtype=np.int32,
        )
        self.geom_leg_roots = np.full(self.model.ngeom, -1, dtype=np.int32)
        for geom_id, body_id in enumerate(self.model.geom_bodyid):
            current_body_id = int(body_id)
            while current_body_id != 0:
                root_index = np.flatnonzero(
                    self.leg_root_body_ids == current_body_id
                )
                if root_index.size:
                    self.geom_leg_roots[geom_id] = int(root_index[0])
                    break
                current_body_id = int(self.model.body_parentid[current_body_id])

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32
        )
        self.neutral_servo = np.array(
            [90.0, 0.0, 90.0, 0.0, 90.0, 180.0, 90.0, 180.0],
            dtype=np.float64,
        )
        self.neutral_qpos = self._servo_to_qpos(self.neutral_servo)
        if stance_target_servo is None:
            stance_target_servo = self.neutral_servo
        stance_target_servo = np.asarray(stance_target_servo, dtype=np.float64)
        if stance_target_servo.shape != (self.n_joints,) or not np.all(
            np.isfinite(stance_target_servo)
        ):
            raise ValueError("stance target must contain eight finite servo angles")
        self.stance_target_servo = stance_target_servo.copy()
        self.stance_target_qpos = self._servo_to_qpos(self.stance_target_servo)
        self.target_height = 0.0
        self.target_gravity = np.array([0.0, 0.0, -1.0])
        self.previous_action = np.zeros(self.n_joints, dtype=np.float64)
        self.applied_action = np.zeros(self.n_joints, dtype=np.float64)
        self.last_reward_components = {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_addresses] = self.neutral_qpos
        mujoco.mj_forward(self.model, self.data)
        self.target_height = float(self.data.xpos[self.base_body_id][2])
        self.previous_action.fill(0.0)
        self.applied_action.fill(0.0)
        self.last_reward_components = {}
        return self._get_observation(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.n_joints,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be eight finite values in [-1, 1]")
        action = np.clip(action, -1.0, 1.0)
        action = np.clip(
            action,
            self.previous_action - self.max_action_delta,
            self.previous_action + self.max_action_delta,
        )
        action = np.clip(action, -1.0, 1.0)
        self.applied_action = action.copy()
        self.data.ctrl[self.actuator_ids] = self._action_to_qpos(action)
        mujoco.mj_step(self.model, self.data)

        observation = self._get_observation()
        terminated = self._fallen()
        reward = self._standing_reward(action)
        self.previous_action[:] = action
        return observation, reward, terminated, False, self._get_info()

    def _servo_to_qpos(self, servo):
        qpos = np.empty(self.n_joints, dtype=np.float64)
        for index, joint_id in enumerate(self.joint_ids):
            qmin, qmax = self.model.jnt_range[joint_id]
            qpos[index] = qmin + servo[index] / 180.0 * (qmax - qmin)
        return qpos

    def _action_to_qpos(self, action):
        targets = np.empty(self.n_joints, dtype=np.float64)
        for index, joint_id in enumerate(self.joint_ids):
            qmin, qmax = self.model.jnt_range[joint_id]
            neutral = self.neutral_qpos[index]
            targets[index] = (
                neutral + action[index] * (qmax - neutral)
                if action[index] >= 0.0
                else neutral + action[index] * (neutral - qmin)
            )
        return targets

    def _get_observation(self):
        rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
        gravity_body = rotation.T @ np.array([0.0, 0.0, -1.0])
        return np.concatenate(
            [
                self.data.qpos[self.qpos_addresses],
                self.data.qvel[self.qvel_addresses],
                gravity_body,
                self._body_angular_velocity(),
                self.data.qvel[0:3],
                self._foot_contacts().astype(np.float64),
            ]
        ).astype(np.float32)

    def _body_angular_velocity(self):
        rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
        return rotation.T @ self.data.qvel[3:6]

    def _foot_contacts(self):
        contacts = np.zeros(4, dtype=bool)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if contact.geom1 == self.ground_geom_id:
                other_geom = contact.geom2
            elif contact.geom2 == self.ground_geom_id:
                other_geom = contact.geom1
            else:
                continue
            other_body = self.model.geom_bodyid[other_geom]
            contacts |= self.foot_body_ids == other_body
        return contacts

    def _leg_collision_count(self):
        count = 0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1_root = self.geom_leg_roots[contact.geom1]
            geom2_root = self.geom_leg_roots[contact.geom2]
            if geom1_root >= 0 and geom2_root >= 0 and geom1_root != geom2_root:
                count += 1
        return count

    def _standing_reward(self, action):
        height = self.data.xpos[self.base_body_id][2]
        rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
        gravity_body = rotation.T @ np.array([0.0, 0.0, -1.0])
        upright = max(0.0, float(np.dot(gravity_body, self.target_gravity)))
        height_score = float(np.exp(-((height - self.target_height) / 0.03) ** 2))
        contact_score = float(np.mean(self._foot_contacts()))
        action_penalty = float(np.mean(self.penalty_weights * action**2))
        action_smoothness_penalty = float(
            np.mean((action - self.previous_action) ** 2)
        )
        joint_velocity_penalty = float(
            np.mean(
                self.velocity_penalty_weights
                * self.data.qvel[self.qvel_addresses] ** 2
            )
        )
        yaw_rate = float(self._body_angular_velocity()[2])
        leg_collision_count = self._leg_collision_count()
        joint_position_penalty = float(
            np.mean(
                (self.data.qpos[self.qpos_addresses] - self.stance_target_qpos)
                ** 2
            )
        )
        reward_components = {
            "height": 0.75 * height_score,
            "upright": 1.0 * upright,
            "contacts": 1.5 * contact_score,
            "action_penalty": -0.05 * action_penalty,
            "action_smoothness_penalty": -0.2 * action_smoothness_penalty,
            "joint_position_penalty": -self.joint_position_weight
            * float(
                np.mean(
                    (self.data.qpos[self.qpos_addresses] - self.stance_target_qpos)
                    ** 2
                )
            ),
            "joint_velocity_penalty": -0.01 * joint_velocity_penalty,
            "yaw_velocity_penalty": -self.yaw_velocity_weight * yaw_rate**2,
            "leg_collision_penalty": -self.leg_collision_weight * leg_collision_count,
        }
        if self._fallen():
            reward_components["fall_penalty"] = -10.0
        else:
            reward_components["fall_penalty"] = 0.0
        self.last_reward_components = reward_components
        return float(sum(reward_components.values()))

    def _fallen(self):
        height = self.data.xpos[self.base_body_id][2]
        rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
        gravity_body = rotation.T @ np.array([0.0, 0.0, -1.0])
        return bool(height < 0.03 or gravity_body[2] > -0.35)

    def _get_info(self):
        info = {
            "base_height": float(self.data.xpos[self.base_body_id][2]),
            "foot_contacts": self._foot_contacts().copy(),
            "leg_collision_count": self._leg_collision_count(),
        }
        info["reward_components"] = dict(self.last_reward_components)
        info["applied_action"] = self.applied_action.copy()
        if self._fallen():
            info["termination_reason"] = "fallen"
        return info

    def render(self):
        raise NotImplementedError("Use python/sim.py for interactive rendering.")

    def close(self):
        pass
