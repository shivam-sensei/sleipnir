import time
import glfw
import mujoco
import mujoco.viewer
import numpy as np
from quadruped_env import SleipnirEnv
from stable_baselines3 import PPO

MODEL_PATH = "./checkpoints_walk/best_model.zip"
print(f"Loading trained policy from {MODEL_PATH}...")
model = PPO.load(MODEL_PATH, device="cpu")

# Disable the 20-second timeout for evaluation
env = SleipnirEnv()
env.max_steps = float("inf")

obs, info = env.reset()

apply_drop = [False]
apply_push_x = [False]
apply_push_y = [False]
manual_reset = [False]


def key_callback(key):
    # L: Lift 6 cm (gentler drop test)
    if key == glfw.KEY_L:
        apply_drop[0] = True
        print("[Perturbation] Lifted 6 cm — testing drop recovery!")
    # P: Moderate push forward
    elif key == glfw.KEY_P:
        apply_push_x[0] = True
        print("[Perturbation] Forward push applied (+X)!")
    # Y: Moderate push sideways
    elif key == glfw.KEY_Y:
        apply_push_y[0] = True
        print("[Perturbation] Lateral push applied (+Y)!")
    # R: Manual reset back to center
    elif key == glfw.KEY_R:
        manual_reset[0] = True
        print("[System] Manual Reset to neutral stance.")


print("\n================ Interactive Controls ================")
print("  L     : Lift robot 6 cm to test drop catch")
print("  P     : Push forward (+X)")
print("  Y     : Push sideways (+Y)")
print("  R     : Manual reset to start")
print("  Space : Pause / Resume physics")
print("=====================================================\n")

with mujoco.viewer.launch_passive(
    env.model, env.data, key_callback=key_callback
) as viewer:
    while viewer.is_running():
        # Clear any automated environment forces so only your keys control pushes
        env.data.xfrc_applied[env.torso_body_id, :] = 0.0

        # 1. Handle manual keyboard interventions
        if manual_reset[0]:
            obs, info = env.reset()
            manual_reset[0] = False

        if apply_drop[0]:
            env.data.qpos[2] += 0.06  # 6 cm drop is within physical leg clearance
            env.data.qvel[:] = 0.0
            apply_drop[0] = False

        if apply_push_x[0]:
            env.data.qvel[0] += 0.6  # 0.6 m/s impulse
            apply_push_x[0] = False

        if apply_push_y[0]:
            env.data.qvel[1] += 0.6  # 0.6 m/s impulse
            apply_push_y[0] = False

        # 2. Policy query
        action, _states = model.predict(obs, deterministic=True)
        # action, _states = model.predict(obs, deterministic=True)
        print("Policy Action Output:", np.round(action, 3))

        # 3. Step physics
        obs, reward, terminated, truncated, info = env.step(action)

        viewer.sync()
        time.sleep(0.02)

        # Only reset if the robot is completely upside down (pitch/roll inverted)
        proj_gravity = obs[3:6]
        if proj_gravity[2] < 0.2:  # Upside down (> 78 deg tilt)
            print("[Auto-Recovery] Robot flipped over. Resetting...")
            time.sleep(0.5)
            obs, info = env.reset()
