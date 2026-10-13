from pathlib import Path
from threading import RLock

import glfw
import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "model"
    / "quadrapod.xml"
)

# Actuator validation settings
TEST_JOINT = 1
TEST_SERVO = 90.0
SERVO_STEP = 5.0
SERVO_TORQUE_LIMIT = 0.98

# ============================================================
# Real-world mass model (kilograms)
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
MASS_DEBUG_SITE_GROUP = 3
MASS_DEBUG_ENABLED = [False]

MASS_COMPONENT_BODIES = {
    "Base printed part": "base_link",
    "FR arm printed link": "fr_arm_1",
    "FR end printed link": "fr_end_1",
    "RR arm printed link": "rr_arm_1",
    "RR end printed link": "rr_end_1",
    "RL arm printed link": "rl_arm_1",
    "RL end printed link": "rl_end_1",
    "FL arm printed link": "fl_arm_1",
    "FL end printed link": "fl_end_1",
    "FR upper servo": "fr_upper_servo_mass",
    "FR lower servo": "fr_lower_servo_mass",
    "RR upper servo": "rr_upper_servo_mass",
    "RR lower servo": "rr_lower_servo_mass",
    "RL upper servo": "rl_upper_servo_mass",
    "RL lower servo": "rl_lower_servo_mass",
    "FL upper servo": "fl_upper_servo_mass",
    "FL lower servo": "fl_lower_servo_mass",
    "Battery": "battery_mass",
    "Electronics": "electronics_mass",
}

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
    """Convert a 0-180 degree servo command to a MuJoCo joint angle."""

    qmin, qmax = JOINT_LIMITS[joint]
    return qmin + (servo_deg / 180.0) * (qmax - qmin)


def mujoco_to_servo(joint, angle):
    """Convert a MuJoCo joint angle to a 0-180 degree servo command."""

    qmin, qmax = JOINT_LIMITS[joint]
    return (angle - qmin) / (qmax - qmin) * 180.0


# ============================================================
# Load MuJoCo model
# ============================================================

print(f"Loading model: {MODEL_PATH}")

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

print("\n==============================")
print("MuJoCo model loaded")
print("==============================")

print(f"Number of bodies    : {model.nbody}")
print(f"Number of joints    : {model.njnt}")
print(f"Number of DOFs      : {model.nv}")
print(f"Number of actuators : {model.nu}")
print(f"Number of sensors   : {model.nsensor}")


def print_mass_properties():
    """Print compiled body masses, inertias, and the composite system COM."""

    print("\n==============================")
    print("MASS PROPERTIES")
    print("==============================")

    total_mass = 0.0
    for body_id in range(1, model.nbody):
        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )
        mass = model.body_mass[body_id]
        total_mass += mass
        local_com = model.body_ipos[body_id]
        inertia = model.body_inertia[body_id]
        print(
            f"{body_name:30s} "
            f"mass={mass:.6f} kg "
            f"COM=({local_com[0]:+.6f}, {local_com[1]:+.6f}, {local_com[2]:+.6f}) "
            f"inertia=({inertia[0]:.6g}, {inertia[1]:.6g}, {inertia[2]:.6g})"
        )

    composite_com = data.subtree_com[1]
    print("--------------------------------------")
    print(f"Expected total mass: {EXPECTED_TOTAL_MASS:.6f} kg")
    print(f"Total compiled mass:  {total_mass:.6f} kg")
    print(
        "Composite COM (world): "
        f"({composite_com[0]:+.6f}, "
        f"{composite_com[1]:+.6f}, {composite_com[2]:+.6f})"
    )
    if not np.isclose(total_mass, EXPECTED_TOTAL_MASS):
        print("WARNING: compiled mass does not match EXPECTED_TOTAL_MASS")


def print_actuator_limits():
    """Print whether each actuator has explicit control or force limits."""

    print("\n==============================")
    print("ACTUATOR LIMITS")
    print("==============================")

    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )
        control_limited = bool(model.actuator_ctrllimited[actuator_id])
        force_limited = bool(model.actuator_forcelimited[actuator_id])
        print(
            f"{actuator_name:12s} "
            f"ctrl={'limited' if control_limited else 'unlimited'} "
            f"{model.actuator_ctrlrange[actuator_id]} "
            f"force={'limited' if force_limited else 'unlimited'} "
            f"{model.actuator_forcerange[actuator_id]}"
        )


# ============================================================
# Joint information
# ============================================================

print("\n==============================")
print("JOINTS")
print("==============================")

for i in range(model.njnt):
    joint_name = model.joint(i).name
    joint_type = model.jnt_type[i]

    print(
        f"Joint {i}: "
        f"name={joint_name}, "
        f"type={joint_type}, "
        f"qpos_addr={model.jnt_qposadr[i]}, "
        f"dof_addr={model.jnt_dofadr[i]}"
    )

JOINT_IDS = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(i))
    for i in range(1, 9)
]

joint_qpos_ids = {
    joint: model.jnt_qposadr[joint_id]
    for joint, joint_id in zip(range(1, 9), JOINT_IDS)
}

actuator_ids = {
    joint: mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        f"motor_{joint}",
    )
    for joint in range(1, 9)
}

servo_targets = dict(NEUTRAL_SERVO)

for joint, servo in NEUTRAL_SERVO.items():
    qpos_idx = joint_qpos_ids[joint]
    data.qpos[qpos_idx] = servo_to_mujoco(joint, servo)

mujoco.mj_forward(model, data)


# ============================================================
# Actuator information
# ============================================================

print("\n==============================")
print("ACTUATORS")
print("==============================")

for i in range(model.nu):
    actuator_name = model.actuator(i).name
    joint_id = model.actuator_trnid[i, 0]

    print(
        f"Actuator {i}: "
        f"name={actuator_name}, "
        f"joint={model.joint(joint_id).name}"
    )

print_actuator_limits()


# ============================================================
# Sensor information
# ============================================================

print("\n==============================")
print("SENSORS")
print("==============================")

for i in range(model.nsensor):
    sensor_name = model.sensor(i).name

    print(
        f"Sensor {i}: "
        f"name={sensor_name}, "
        f"dimension={model.sensor_dim[i]}"
    )


# ============================================================
# Find important sites
# ============================================================

imu_site = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "imu"
)

foot_sites = {
    "FR": mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "fr_foot"
    ),
    "RR": mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "rr_foot"
    ),
    "RL": mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "rl_foot"
    ),
    "FL": mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "fl_foot"
    ),
}

FOOT_CONTACT_WORLD = {
    "FR": np.array([0.0715, -0.1180, 0.0]),
    "RR": np.array([-0.0985, -0.1005, 0.0]),
    "RL": np.array([-0.1030, 0.0995, 0.0]),
    "FL": np.array([0.0670, 0.1200, 0.0]),
}


# ============================================================
# Helper functions
# ============================================================

def get_joint_positions():
    """Return the 8 joint positions in joint-number order."""

    positions = []

    for joint_id in JOINT_IDS:
        qpos_addr = model.jnt_qposadr[joint_id]

        positions.append(data.qpos[qpos_addr])

    return positions


def get_joint_velocities():
    """Return the 8 joint velocities in joint-number order."""

    velocities = []

    for joint_id in JOINT_IDS:
        dof_addr = model.jnt_dofadr[joint_id]

        velocities.append(data.qvel[dof_addr])

    return velocities


def get_imu():
    """Return accelerometer and gyroscope readings."""

    accel = None
    gyro = None

    for i in range(model.nsensor):

        sensor_name = model.sensor(i).name
        start = model.sensor_adr[i]
        dimension = model.sensor_dim[i]

        values = data.sensordata[start:start + dimension]

        if sensor_name == "imu_accel":
            accel = values.copy()

        elif sensor_name == "imu_gyro":
            gyro = values.copy()

    return accel, gyro


def get_foot_contacts():
    """
    Check whether each foot site is close to a contact point.

    This uses MuJoCo's contact pairs and identifies whether one
    of the contact geometries belongs to the foot body.
    """

    contacts = {
        "FR": False,
        "RR": False,
        "RL": False,
        "FL": False,
    }

    foot_body_ids = {}

    for name, site_id in foot_sites.items():
        foot_body_ids[name] = model.site_bodyid[site_id]

    for contact_id in range(data.ncon):

        contact = data.contact[contact_id]

        geom1 = contact.geom1
        geom2 = contact.geom2

        body1 = model.geom_bodyid[geom1]
        body2 = model.geom_bodyid[geom2]

        # Ignore contacts that aren't involving a foot body
        for name, foot_body in foot_body_ids.items():

            if body1 == foot_body or body2 == foot_body:
                contacts[name] = True

    return contacts


def print_contact_pairs():
    """Print the exact geom pairs currently reported by MuJoCo."""

    if data.ncon == 0:
        print("Contacts: none")
        return

    print("Contacts:")
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1 = model.geom(contact.geom1).name
        geom2 = model.geom(contact.geom2).name
        print(f"  {geom1} <-> {geom2}, dist={contact.dist:+.5f}")


def print_contact_points():
    """Print active contact positions in world and end-link-local frames."""

    if data.ncon == 0:
        print("Contact points: none")
        return

    foot_body_ids = {
        foot_name: model.site_bodyid[site_id]
        for foot_name, site_id in foot_sites.items()
    }

    print("Contact points:")
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1 = model.geom(contact.geom1).name
        geom2 = model.geom(contact.geom2).name
        world_point = np.array(contact.pos)
        contact_body_ids = {
            model.geom_bodyid[contact.geom1],
            model.geom_bodyid[contact.geom2],
        }

        foot_name = next(
            (
                name
                for name, body_id in foot_body_ids.items()
                if body_id in contact_body_ids
            ),
            None,
        )

        if foot_name is None:
            print(
                f"  {contact_id}: world="
                f"({world_point[0]:+.6f}, {world_point[1]:+.6f}, {world_point[2]:+.6f}) "
                f"{geom1} <-> {geom2}"
            )
            continue

        body_id = foot_body_ids[foot_name]
        body_position = data.xpos[body_id]
        body_rotation = data.xmat[body_id].reshape(3, 3)
        local_point = body_rotation.T @ (world_point - body_position)
        print(
            f"  {contact_id} {foot_name}: "
            f"world=({world_point[0]:+.6f}, {world_point[1]:+.6f}, {world_point[2]:+.6f}), "
            f"end_link=({local_point[0]:+.6f}, {local_point[1]:+.6f}, {local_point[2]:+.6f}) "
            f"{geom1} <-> {geom2}"
        )


selected_joint = [TEST_JOINT - 1]
physics_enabled = [True]
data_lock = RLock()


def set_selected_servo(servo):
    """Set the selected joint target in servo degrees."""

    with data_lock:
        joint_number = selected_joint[0] + 1
        servo_targets[joint_number] = float(np.clip(servo, 0.0, 180.0))

        if not physics_enabled[0]:
            data.qpos[joint_qpos_ids[joint_number]] = servo_to_mujoco(
                joint_number,
                servo_targets[joint_number],
            )
            mujoco.mj_forward(model, data)


def reset_all_to_neutral():
    """Reset all servo targets and pose coordinates to the neutral pose."""

    with data_lock:
        servo_targets.update(NEUTRAL_SERVO)
        for joint, servo in servo_targets.items():
            data.qpos[joint_qpos_ids[joint]] = servo_to_mujoco(joint, servo)
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)


def shift_servo_group(joints, delta):
    """Shift a group of servo targets by a small amount."""

    with data_lock:
        for joint in joints:
            servo_targets[joint] = float(
                np.clip(servo_targets[joint] + delta, 0.0, 180.0)
            )

        if not physics_enabled[0]:
            for joint in joints:
                data.qpos[joint_qpos_ids[joint]] = servo_to_mujoco(
                    joint,
                    servo_targets[joint],
                )
            mujoco.mj_forward(model, data)


def shift_end_group(delta):
    """Move corresponding end joints in the same physical direction."""

    shift_servo_group(
        (2, 4, 6, 8),
        delta,
    )


def print_servo_pose():
    """Print current MuJoCo joint angles and their servo equivalents."""

    print("\nCurrent servo pose:")
    for joint in range(1, 9):
        q = data.qpos[joint_qpos_ids[joint]]
        servo = mujoco_to_servo(joint, q)
        print(
            f"J{joint}: MuJoCo={np.rad2deg(q):7.2f} deg "
            f"Servo={servo:7.2f} deg "
            f"Target={servo_targets[joint]:7.2f} deg"
        )


def print_mass_model():
    """Print configured component masses, local inertials, and system COM."""

    print("\nMass model:")
    simulated_total = 0.0

    for component_name, body_name in MASS_COMPONENT_BODIES.items():
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        mass = model.body_mass[body_id]
        simulated_total += mass
        local_position = model.body_ipos[body_id]
        print(
            f"{component_name}: {mass * 1000:.1f} g, "
            f"body={body_name}, "
            f"local_pos=({local_position[0]:+.6f}, "
            f"{local_position[1]:+.6f}, {local_position[2]:+.6f})"
        )

    print(f"Expected total mass: {EXPECTED_TOTAL_MASS * 1000:.1f} g")
    print(f"Total simulated mass: {simulated_total * 1000:.1f} g")
    print(
        "System COM (world): "
        f"({data.subtree_com[0][0]:+.6f}, "
        f"{data.subtree_com[0][1]:+.6f}, "
        f"{data.subtree_com[0][2]:+.6f})"
    )


def apply_position_control():
    """Send desired joint positions to the MuJoCo servo actuators."""

    for joint in range(1, 9):
        data.ctrl[actuator_ids[joint]] = servo_to_mujoco(
            joint,
            servo_targets[joint],
        )


def key_callback(key):
    """Edit the selected joint from the MuJoCo viewer keyboard."""

    with data_lock:
        if glfw.KEY_1 <= key <= glfw.KEY_8:
            selected_joint[0] = key - glfw.KEY_1
            print(f"Selected joint {selected_joint[0] + 1}")
            return

        if key in (glfw.KEY_LEFT, glfw.KEY_DOWN):
            joint = selected_joint[0] + 1
            set_selected_servo(servo_targets[joint] - SERVO_STEP)
            print(f"J{joint} target: {servo_targets[joint]:.1f} deg")
        elif key in (glfw.KEY_RIGHT, glfw.KEY_UP):
            joint = selected_joint[0] + 1
            set_selected_servo(servo_targets[joint] + SERVO_STEP)
            print(f"J{joint} target: {servo_targets[joint]:.1f} deg")
        elif key == glfw.KEY_R:
            joint = selected_joint[0] + 1
            set_selected_servo(NEUTRAL_SERVO[joint])
            print(f"J{joint} reset to {NEUTRAL_SERVO[joint]:.1f} deg")
        elif key == glfw.KEY_N:
            reset_all_to_neutral()
            print("All joints reset to neutral")
        elif key == glfw.KEY_A:
            shift_servo_group((1, 3, 5, 7), SERVO_STEP)
            print("Arm joints increased by 5 deg")
        elif key == glfw.KEY_Z:
            shift_servo_group((1, 3, 5, 7), -SERVO_STEP)
            print("Arm joints decreased by 5 deg")
        elif key == glfw.KEY_E:
            shift_end_group(SERVO_STEP)
            print("End joints moved by 5 deg in servo space")
        elif key == glfw.KEY_D:
            shift_end_group(-SERVO_STEP)
            print("End joints moved by -5 deg in servo space")
        elif key == glfw.KEY_SPACE:
            physics_enabled[0] = not physics_enabled[0]
            state = "enabled" if physics_enabled[0] else "paused"
            print(f"Physics {state}")
        elif key == glfw.KEY_P:
            print_servo_pose()
            print_mass_model()
        elif key == glfw.KEY_F:
            print_contact_points()
        elif key == glfw.KEY_M:
            MASS_DEBUG_ENABLED[0] = not MASS_DEBUG_ENABLED[0]
            state = "enabled" if MASS_DEBUG_ENABLED[0] else "disabled"
            print(f"Mass visualization {state}")


# ============================================================
# Main simulation
# ============================================================

print("\n==============================")
print("Starting simulation")
print("==============================")

servo_targets[TEST_JOINT] = TEST_SERVO
print("All-actuator standing hold: all joints start at the configured neutral pose.")
print("Keys: 1-8 select, arrows +/-5 deg, R=joint neutral, N=all neutral, P=pose+mass, F=contacts, M=mass markers, Space=physics.")
print("A/Z=all arm joints +/-5 deg, E/D=end joints +/-5 deg in servo space.")
print(
    "Position servos: "
    f"kp=2.0, dampratio=0.2, torque limit={SERVO_TORQUE_LIMIT:.2f} N*m"
)
print(f"Expected mass model total: {EXPECTED_TOTAL_MASS:.3f} kg")
print_mass_properties()

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback,
) as viewer:

    while viewer.is_running():

        # ----------------------------------------------------
        # Physics
        # ----------------------------------------------------

        with data_lock:
            if physics_enabled[0]:
                apply_position_control()
                mujoco.mj_step(model, data)

        # ----------------------------------------------------
        # Viewer
        # ----------------------------------------------------

        viewer.opt.sitegroup[MASS_DEBUG_SITE_GROUP] = int(MASS_DEBUG_ENABLED[0])
        viewer.sync()
