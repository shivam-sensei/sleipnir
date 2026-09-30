import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from quadruped_env import DRIFT_STANCE_SERVO, QuadrupedEnv


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "rl_v1"
    / "artifacts"
    / "ppo_quadruped_contact_v1"
)
def evaluate(model, episodes, disturbance):
    """Evaluate a deterministic policy, optionally perturbing base velocity."""

    returns = []
    lengths = []
    survived = 0

    for episode in range(episodes):
        env = QuadrupedEnv(yaw_velocity_weight=0.1)
        observation, _ = env.reset(seed=episode)

        if disturbance:
            env.data.qvel[0:3] = np.array([0.15, 0.0, 0.0])
            env.data.qvel[3:6] = np.array([0.0, 0.0, 0.25])

        episode_return = 0.0
        for step in range(5000):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            if terminated or truncated:
                break

        returns.append(episode_return)
        lengths.append(step + 1)
        survived += int(not terminated)
        env.close()

    return {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "survival_rate": survived / episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a PPO standing policy.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume PPO training from an existing checkpoint.",
    )
    parser.add_argument(
        "--joint4-action-weight",
        type=float,
        default=1.0,
        help="Relative action penalty weight for joint 4.",
    )
    parser.add_argument(
        "--joint4-velocity-weight",
        type=float,
        default=1.0,
        help="Relative joint-velocity penalty weight for joint 4.",
    )
    parser.add_argument(
        "--joint8-action-weight",
        type=float,
        default=1.0,
        help="Relative action penalty weight for joint 8.",
    )
    parser.add_argument(
        "--joint8-velocity-weight",
        type=float,
        default=1.0,
        help="Relative joint-velocity penalty weight for joint 8.",
    )
    parser.add_argument(
        "--yaw-velocity-weight",
        type=float,
        default=0.1,
        help="Penalty weight for body-frame yaw angular velocity.",
    )
    parser.add_argument(
        "--leg-collision-weight",
        type=float,
        default=2.0,
        help="Penalty weight per inter-leg collision contact.",
    )
    parser.add_argument(
        "--joint-position-weight",
        type=float,
        default=5.0,
        help="Penalty weight for deviation from the selected stance pose.",
    )
    parser.add_argument(
        "--stance-target",
        choices=("neutral", "drift"),
        default="neutral",
        help="Joint stance target used by the position penalty.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="cpu",
        help="Training device (default: cpu).",
    )
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. "
            "Check the NVIDIA driver and the active virtual environment."
        )
    if args.device == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

    env = QuadrupedEnv(
        joint4_action_weight=args.joint4_action_weight,
        joint4_velocity_weight=args.joint4_velocity_weight,
        joint8_action_weight=args.joint8_action_weight,
        joint8_velocity_weight=args.joint8_velocity_weight,
        yaw_velocity_weight=args.yaw_velocity_weight,
        leg_collision_weight=args.leg_collision_weight,
        joint_position_weight=args.joint_position_weight,
        stance_target_servo=(
            DRIFT_STANCE_SERVO if args.stance_target == "drift" else None
        ),
    )
    if args.resume_from is None:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            n_steps=1024,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            device=args.device,
            tensorboard_log=str(
                Path(__file__).resolve().parent.parent.parent
                / "rl_v1"
                / "artifacts"
                / "tensorboard"
            ),
        )
    else:
        model = PPO.load(str(args.resume_from), env=env, device=args.device)
        model.tensorboard_log = str(
            Path(__file__).resolve().parent.parent.parent
            / "rl_v1"
            / "artifacts"
            / "tensorboard"
        )
    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=args.resume_from is None,
        tb_log_name=args.model_path.stem,
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.model_path))
    env.close()
    print(f"Saved PPO policy to {args.model_path}.zip")

    if not args.skip_eval:
        nominal = evaluate(model, args.eval_episodes, disturbance=False)
        disturbed = evaluate(model, args.eval_episodes, disturbance=True)
        print("Nominal evaluation:", nominal)
        print("Disturbed evaluation:", disturbed)


if __name__ == "__main__":
    main()
