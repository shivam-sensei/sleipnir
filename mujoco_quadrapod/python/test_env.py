from pathlib import Path
from threading import RLock
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np

# ============================================================
# Paths & Hardware Constants
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent / "quadrapod.xml"
if not MODEL_PATH.exists():
    # Fallback to model directory if running from sibling scripts folder
    MODEL_PATH = (
        Path(__file__).resolve().parent.parent / "model" / "quadrapod.xml"
    )

SERVO_TORQUE_LIMIT = 0.98  # Stall torque of MG996R at 6.0 V (N*m)
SERVO_STEP = 5.0  # Nudge step in degrees

# ============================================================
# Component Mass Model (kg)
# ============================================================

BASE_MASS = 0.150
ARM_MASS = 0.022
END_MASS = 0.032
SERVO_MASS = 0.055
BATTERY_MASS = 0.175
ELECTRONICS_MASS = 0.050

EXPECTED_TOTAL_MASS = (
    BASE_MASS
    + 4 * ARM_MASS
    + 4 * END_MASS
    + 8 * SERVO_MASS
    + BATTERY_MASS
    + ELECTRONICS_MASS
)

# Site group 3 holds the debug spheres defined in your MJCF
MASS_DEBUG_SITE_GROUP = 3
mass_debug_enabled = [False]

# ============================================================
# Joint Kinematics & Calibrated Neutral Pose
# ============================================================

JOINT_LIMITS = {
    1: (-0.733038, 1.605703),
    2: (0.000000, 3.141593),
    3: (-1.780236, 1.082104),
    4: (0.000000, 3.141593),
    5: (-0.977384, 1.815142),
    6: (-3.141593, 0.000000),
    7: (-1.605703, 0.907571),
    8: (-3.141593, 0.000000),
}

NEUTRAL_SERVO = {
    1: 90.0,
    2: 0.0,
    3: 90.0,
    4: 0.0,
    5: 90.0,
    6: 180.0,
    7: 90.0,
    8: 180.0,
}


def servo_to_mujoco(joint, servo_deg):
    """Linear mapping from 0-180 degree servo command to MuJoCo joint radians."""
    qmin, qmax = JOINT_LIMITS[joint]
    return qmin + (servo_deg / 180.0) * (qmax - qmin)


def mujoco_to_servo(joint, angle):
    """Convert MuJoCo joint radians back to 0-180 degree servo space."""
    qmin, qmax = JOINT_LIMITS[joint]
    return (angle - qmin) / (qmax - qmin) * 180.0


# ============================================================
# Model Initialization
# ============================================================

print(f"Loading MuJoCo model: {MODEL_PATH}")
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

print(
    f"Bodies: {model.nbody} | DOFs: {model.nv} | Actuators: {model.nu} | Sensors: {model.nsensor}"
)

# Joint and Actuator lookup
JOINT_IDS = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(i))
    for i in range(1, 9)
]
joint_qpos_ids = {
    j: model.jnt_qposadr[j_id] for j, j_id in zip(range(1, 9), JOINT_IDS)
}
actuator_ids = {
    j: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{j}")
    for j in range(1, 9)
}

# Sensor lookup (IMU & 4x Base ToFs)
gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
accel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
tof_sensor_names = ["tof_fr", "tof_rr", "tof_rl", "tof_fl"]
tof_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    for name in tof_sensor_names
]

foot_sites = {
    "FR": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fr_foot"),
    "RR": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rr_foot"),
    "RL": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rl_foot"),
    "FL": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fl_foot"),
}

# Targets & History Buffers
servo_targets = dict(NEUTRAL_SERVO)
a_prev1 = np.zeros(8, dtype=np.float32)
a_prev2 = np.zeros(8, dtype=np.float32)

# Set initial joint coordinates to neutral pose before first forward pass
for joint, servo in NEUTRAL_SERVO.items():
    data.qpos[joint_qpos_ids[joint]] = servo_to_mujoco(joint, servo)
    data.ctrl[actuator_ids[joint]] = servo_to_mujoco(joint, servo)

mujoco.mj_forward(model, data)

# ============================================================
# Observation & State Functions (POMDP o_t in R^26)
# ============================================================


def get_pomdp_observation():
    """Extracts the exact 26-dim observation vector defined in Section IV-B of the paper."""
    # 1. Torso Gyro (rad/s) [3-dim]
    gyro = np.zeros(3, dtype=np.float32)
    if gyro_id != -1:
        adr = model.sensor_adr[gyro_id]
        gyro = data.sensordata[adr : adr + 3].copy()

    # 2. Projected Gravity Vector R^T * [0, 0, -1] [3-dim]
    torso_quat = data.qpos[3:7]
    rot_mat = np.zeros(9)
    mujoco.mju_quat2Mat(rot_mat, torso_quat)
    rot_mat = rot_mat.reshape(3, 3)
    proj_gravity = rot_mat[2, :].copy()

    # 3. 4x Base ToF Distances (m) [4-dim]
    tof_dists = np.zeros(4, dtype=np.float32)
    for i, s_id in enumerate(tof_ids):
        if s_id != -1:
            adr = model.sensor_adr[s_id]
            tof_dists[i] = data.sensordata[adr]

    # 4. Action History buffers (8-dim each)
    obs = np.concatenate([gyro, proj_gravity, tof_dists, a_prev1, a_prev2])
    return obs


def print_contact_status():
    """Prints foot contact status with the ground."""
    foot_body_ids = {
        name: model.site_bodyid[site_id]
        for name, site_id in foot_sites.items()
        if site_id != -1
    }
    contacts = {name: False for name in foot_body_ids}

    for c_idx in range(data.ncon):
        c = data.contact[c_idx]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        for name, b_id in foot_body_ids.items():
            if b1 == b_id or b2 == b_id:
                contacts[name] = True

    print("\nFoot Contacts:")
    for name, in_contact in contacts.items():
        print(f"  {name}: {'GROUNDED' if in_contact else 'AIRBORNE'}")


# ============================================================
# Interactive Controls & Callback
# ============================================================

selected_joint = [0]
physics_enabled = [True]
data_lock = RLock()


def set_selected_servo(deg):
    with data_lock:
        j = selected_joint[0] + 1
        servo_targets[j] = float(np.clip(deg, 0.0, 180.0))
        data.ctrl[actuator_ids[j]] = servo_to_mujoco(j, servo_targets[j])


def reset_all_to_neutral():
    with data_lock:
        servo_targets.update(NEUTRAL_SERVO)
        for j, deg in servo_targets.items():
            data.qpos[joint_qpos_ids[j]] = servo_to_mujoco(j, deg)
            data.ctrl[actuator_ids[j]] = servo_to_mujoco(j, deg)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)


def key_callback(key):
    with data_lock:
        # Number keys 1-8 to pick joint
        if glfw.KEY_1 <= key <= glfw.KEY_8:
            selected_joint[0] = key - glfw.KEY_1
            print(f"Selected Joint J{selected_joint[0] + 1}")
            return

        # Arrow Up/Down to adjust angle
        if key in (glfw.KEY_LEFT, glfw.KEY_DOWN):
            j = selected_joint[0] + 1
            set_selected_servo(servo_targets[j] - SERVO_STEP)
            print(f"J{j} Target: {servo_targets[j]:.1f}°")
        elif key in (glfw.KEY_RIGHT, glfw.KEY_UP):
            j = selected_joint[0] + 1
            set_selected_servo(servo_targets[j] + SERVO_STEP)
            print(f"J{j} Target: {servo_targets[j]:.1f}°")

        # N: Reset to neutral, Space: Pause physics
        elif key == glfw.KEY_N:
            reset_all_to_neutral()
            print("All joints reset to NEUTRAL stance")
        elif key == glfw.KEY_SPACE:
            physics_enabled[0] = not physics_enabled[0]
            print(
                f"Physics {'ENABLED' if physics_enabled[0] else 'PAUSED (Frozen)'}"
            )

        # O: Inspect POMDP 26-dim Observation vector
        elif key == glfw.KEY_O:
            obs = get_pomdp_observation()
            print("\n================ POMDP OBSERVATION (26-dim) ================")
            print(f"Torso Gyro (rad/s)   : {np.round(obs[0:3], 3)}")
            print(f"Projected Gravity    : {np.round(obs[3:6], 3)}")
            print(
                f"4x Base ToFs (m)     : {np.round(obs[6:10], 4)} [FR, RR, RL, FL]"
            )
            print(f"Mean Ground Clearance: {np.mean(obs[6:10]):.4f} m")
            print(f"Action History a_t-1 : {np.round(obs[10:18], 2)}")
            print(f"Action History a_t-2 : {np.round(obs[18:26], 2)}")
            print("============================================================")

        # F: Foot contact check, M: Mass visualizers toggle
        elif key == glfw.KEY_F:
            print_contact_status()
        elif key == glfw.KEY_M:
            mass_debug_enabled[0] = not mass_debug_enabled[0]
            print(
                f"Mass visualization {'ENABLED' if mass_debug_enabled[0] else 'DISABLED'}"
            )


# ============================================================
# Main Simulation Loop
# ============================================================

print("\n==============================")
print("Sleipnir Simulation Started")
print("==============================")
print(
    "Keys:\n"
    "  1-8       : Select joint 1 to 8\n"
    "  Up/Down   : Nudge selected joint +/- 5 deg\n"
    "  N         : Reset to neutral stance\n"
    "  O         : Print 26-dim POMDP observation vector\n"
    "  F         : Print foot contacts\n"
    "  M         : Toggle point-mass markers\n"
    "  Space     : Pause/Resume physics\n"
)

with mujoco.viewer.launch_passive(
    model, data, key_callback=key_callback
) as viewer:
    step_count = 0
    while viewer.is_running():
        with data_lock:
            if physics_enabled[0]:
                # Send neutral target angles continuously to position actuators
                for j in range(1, 9):
                    data.ctrl[actuator_ids[j]] = servo_to_mujoco(
                        j, servo_targets[j]
                    )

                mujoco.mj_step(model, data)
                step_count += 1

                # Update history buffers at 50 Hz control rate (every 10 physics steps)
                if step_count % 10 == 0:
                    current_actions = np.array(
                        [
                            (servo_targets[j] - NEUTRAL_SERVO[j]) / 90.0
                            for j in range(1, 9)
                        ],
                        dtype=np.float32,
                    )
                    a_prev2[:] = a_prev1[:]
                    a_prev1[:] = current_actions[:]

        viewer.opt.sitegroup[MASS_DEBUG_SITE_GROUP] = int(mass_debug_enabled[0])
        viewer.sync()
        time.sleep(0.002)