import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from quadruped_env import QuadrupedEnv


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "rl_v1"
    / "artifacts"
    / "ppo_quadruped_standing.zip"
)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained PPO policy in the MuJoCo viewer."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use the deterministic PPO action instead of sampling.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback multiplier; values above 1 slow playback.",
    )
    args = parser.parse_args()

    if args.speed <= 0:
        parser.error("--speed must be greater than zero")
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    policy = PPO.load(str(args.model), device="cpu")
    env = QuadrupedEnv(render_mode="human", yaw_velocity_weight=0.1)
    observation, _ = env.reset()

    print(f"Loaded policy: {args.model}")
    print(f"Observation shape: {observation.shape}")
    print(f"Action space: {env.action_space}")

    episode = 0
    next_report_time = 0.0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running() and episode < args.episodes:
            action, _ = policy.predict(
                observation,
                deterministic=args.deterministic,
            )
            observation, reward, terminated, truncated, info = env.step(action)
            viewer.sync()

            if env.data.time >= next_report_time:
                print(
                    f"Episode {episode + 1} | "
                    f"Time: {env.data.time:.2f} | "
                    f"Reward: {reward:.3f} | "
                    f"Height: {info['base_height']:.3f} | "
                    f"Action: {action}"
                )
                next_report_time = env.data.time + 2.0

            time.sleep(env.model.opt.timestep * args.speed)

            if terminated or truncated:
                episode += 1
                print(
                    f"Episode {episode} ended at simulation time "
                    f"{env.data.time:.2f} | "
                    f"Height: {info['base_height']:.3f} | "
                    f"Contacts: {info['foot_contacts']}"
                )
                if episode < args.episodes:
                    observation, _ = env.reset()
                    next_report_time = 0.0

    env.close()


if __name__ == "__main__":
    main()
