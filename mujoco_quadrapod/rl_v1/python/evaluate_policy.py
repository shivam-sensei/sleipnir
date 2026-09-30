import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from quadruped_env import QuadrupedEnv


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent / "rl_v1"
DEFAULT_POLICIES = (
    PROJECT_ROOT / "artifacts" / "ppo_quadruped_standing.zip",
    PROJECT_ROOT / "artifacts" / "ppo_quadruped_drift_v1.zip",
)
HORIZON = 5000
YAW_VELOCITY_WEIGHT = 0.1


def run_episode(policy, disturbance):
    env = QuadrupedEnv(yaw_velocity_weight=YAW_VELOCITY_WEIGHT)
    observation, _ = env.reset(seed=0)
    env.data.qvel[:3] = disturbance[:3]
    env.data.qvel[3:6] = disturbance[3:6]

    positions = []
    velocities = []
    yaw_rates = []
    contacts = []
    total_reward = 0.0

    for step in range(HORIZON):
        action, _ = policy.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        positions.append(env.data.qpos[env.qpos_addresses].copy())
        velocities.append(env.data.qvel[env.qvel_addresses].copy())
        yaw_rates.append(env._body_angular_velocity()[2])
        contacts.append(info["foot_contacts"].copy())
        total_reward += reward
        if terminated or truncated:
            break

    env.close()
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    contacts = np.asarray(contacts)
    return {
        "steps": step + 1,
        "survived": not terminated,
        "return": total_reward,
        "height": info["base_height"],
        "contact_percent": contacts.mean(axis=0) * 100.0,
        "yaw_rms": float(np.sqrt(np.mean(np.square(yaw_rates)))),
        "joint_velocity_rms": np.sqrt(np.mean(np.square(velocities), axis=0)),
        "joint_position_std": np.std(positions, axis=0),
    }


def print_result(policy_path, label, result):
    print(f"\n{label}: {policy_path.name}")
    print(
        f"  steps={result['steps']} survived={result['survived']} "
        f"return={result['return']:.2f} height={result['height']:.4f}"
    )
    print("  contact percentage:", np.round(result["contact_percent"], 1))
    print(f"  yaw RMS (rad/s): {result['yaw_rms']:.4f}")
    print("  joint velocity RMS:", np.round(result["joint_velocity_rms"], 4))
    print("  joint position std:", np.round(result["joint_position_std"], 4))


def main():
    parser = argparse.ArgumentParser(
        description="Compare the standing baseline and yaw-small PPO policies."
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        type=Path,
        default=DEFAULT_POLICIES,
    )
    args = parser.parse_args()

    disturbances = {
        "nominal": np.zeros(6),
        "disturbed": np.array([1.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
        "strong disturbance": np.array([1.5, 0.0, 0.0, 0.0, 3.0, 0.0]),
    }

    for policy_path in args.policies:
        policy = PPO.load(str(policy_path), device="cpu")
        for label, disturbance in disturbances.items():
            print_result(
                policy_path,
                label,
                run_episode(policy, disturbance),
            )


if __name__ == "__main__":
    main()
