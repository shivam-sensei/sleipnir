import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from quadruped_env import SleipnirEnv

# Directories for checkpoints and logs
LOG_DIR = "./runs/sleipnir_walk_ppo"
CHECKPOINT_DIR = "./checkpoints_walk"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 1. Instantiate 4 parallel environments for faster training rollouts
num_envs = min(os.cpu_count() or 4, 8)
env = make_vec_env(SleipnirEnv, n_envs=num_envs)

# 2. Configure PPO hyperparameters
# A policy frequency of 50 Hz with 1024 steps per env gives a batch of 4096 steps per update.
model = PPO(
    policy="MlpPolicy",
    env=env,
    device = "cpu",
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1,
    tensorboard_log=LOG_DIR,
)

# 3. Callbacks for evaluation and saving periodic checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=25000 // num_envs,
    save_path=CHECKPOINT_DIR,
    name_prefix="sleipnir_walk_model",
)

eval_env = SleipnirEnv()
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=CHECKPOINT_DIR,
    log_path=LOG_DIR,
    eval_freq=10000 // num_envs,
    deterministic=True,
    render=False,
)

# 4. Start training for 500,000 timesteps (~15-20 minutes on a modern CPU)
print("Starting Sleipnir PPO training run...")
model.learn(
    total_timesteps=500_000,
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True,
)

# Save final policy
model.save("sleipnir_walk_final_policy")
print("Training complete! Model saved to sleipnir_final_policy.zip")