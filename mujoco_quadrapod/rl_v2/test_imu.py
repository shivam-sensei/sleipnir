import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "../model/quadrapod.xml"
)


# ============================================================
# JOINT LIMITS
# ============================================================

JOINT_LIMITS = {
    1: (-0.733038, 1.605703),
    2: (0.0, 3.141593),
    3: (-1.780236, 1.082104),
    4: (0.0, 3.141593),
    5: (-0.977384, 1.815142),
    6: (-3.141593, 0.0),
    7: (-1.605703, 0.907571),
    8: (-3.141593, 0.0),
}


# ============================================================
# NEUTRAL SERVO POSE
# ============================================================

NEUTRAL_SERVO = {
    1: 90,
    2: 0,
    3: 90,
    4: 0,
    5: 90,
    6: 180,
    7: 90,
    8: 180,
}


# ============================================================
# TOF MOUNTS
# ============================================================

TOF_NAMES = (
    "FRONT",
    "REAR",
    "LEFT",
    "RIGHT",
)

TOF_OFFSETS = (
    np.array([0.060, 0.000, 0.000]),
    np.array([-0.060, 0.000, 0.000]),
    np.array([0.000, 0.060, 0.000]),
    np.array([0.000, -0.060, 0.000]),
)

TOF_DIRECTION = np.array(
    [0.0, 0.0, -1.0]
)


# ============================================================
# SERVO → MUJOCO
# ============================================================

def servo_to_mujoco(joint_id, servo_deg):

    qmin, qmax = JOINT_LIMITS[joint_id]

    return qmin + (
        servo_deg / 180.0
    ) * (qmax - qmin)


# ============================================================
# LOAD MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)


# ============================================================
# FIND BASE
# ============================================================

base_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base_link",
)


# ============================================================
# RESET
# ============================================================

mujoco.mj_resetData(
    model,
    data,
)


# ============================================================
# SET NEUTRAL POSE + ACTUATOR TARGETS
# ============================================================

for joint_id in range(1, 9):

    q = servo_to_mujoco(
        joint_id,
        NEUTRAL_SERVO[joint_id],
    )

    qpos_index = model.jnt_qposadr[joint_id]

    data.qpos[qpos_index] = q

    data.ctrl[joint_id - 1] = q


mujoco.mj_forward(
    model,
    data,
)


# ============================================================
# TOF RAY VISUALIZATION
# ============================================================

def draw_tof_rays(viewer):

    # Clear previous dynamic geoms
    viewer.user_scn.ngeom = 0

    base_pos = data.xpos[base_body_id]

    base_rot = data.xmat[
        base_body_id
    ].reshape(3, 3)

    for i in range(4):

        # ----------------------------------------------------
        # Sensor origin
        # ----------------------------------------------------

        sensor_pos = (
            base_pos
            + base_rot @ TOF_OFFSETS[i]
        )

        # ----------------------------------------------------
        # Ray direction
        # ----------------------------------------------------

        direction = (
            base_rot @ TOF_DIRECTION
        )

        direction /= np.linalg.norm(direction)

        # ----------------------------------------------------
        # Raycast
        # ----------------------------------------------------

        geomid = np.array(
            [-1],
            dtype=np.int32,
        )

        distance = mujoco.mj_ray(
            model,
            data,
            sensor_pos,
            direction,
            None,
            1,
            -1,
            geomid,
        )

        # ----------------------------------------------------
        # Choose ray length
        # ----------------------------------------------------

        if distance >= 0:

            ray_length = distance

        else:

            ray_length = 0.08

        end_pos = (
            sensor_pos
            + direction * ray_length
        )

        # ----------------------------------------------------
        # Add line to viewer
        # ----------------------------------------------------

        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break

        geom = viewer.user_scn.geoms[
            viewer.user_scn.ngeom
        ]

        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([
                0.002,
                ray_length / 2,
                0.0,
            ]),
            np.zeros(3),
            np.eye(3).flatten(),
            np.array([
                1.0,
                0.0,
                0.0,
                1.0,
            ]),
        )

        mujoco.mjv_connector(
    geom,
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    0.002,
    np.asarray(sensor_pos, dtype=np.float64),
    np.asarray(end_pos, dtype=np.float64),
)

        viewer.user_scn.ngeom += 1


# ============================================================
# VIEWER
# ============================================================

print("=" * 60)
print("SLEIPNIR — ToF VISUALIZATION")
print("=" * 60)

print()
print("Four ToF rays are visualized:")
print("  FRONT")
print("  REAR")
print("  LEFT")
print("  RIGHT")
print()
print("Close the viewer to exit.")
print()


with mujoco.viewer.launch_passive(
    model,
    data,
) as viewer:

    viewer.cam.distance = 0.45
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20

    while viewer.is_running():

        # ----------------------------------------------------
        # Physics
        # ----------------------------------------------------

        mujoco.mj_step(
            model,
            data,
        )

        # ----------------------------------------------------
        # Draw ToF rays
        # ----------------------------------------------------

        draw_tof_rays(viewer)

        # ----------------------------------------------------
        # Print distances periodically
        # ----------------------------------------------------

        if int(data.time * 1000) % 100 == 0:

            base_pos = data.xpos[base_body_id]

            base_rot = data.xmat[
                base_body_id
            ].reshape(3, 3)

            distances = []

            for offset in TOF_OFFSETS:

                origin = (
                    base_pos
                    + base_rot @ offset
                )

                direction = (
                    base_rot @ TOF_DIRECTION
                )

                direction /= np.linalg.norm(
                    direction
                )

                geomid = np.array(
                    [-1],
                    dtype=np.int32,
                )

                distance = mujoco.mj_ray(
                    model,
                    data,
                    origin,
                    direction,
                    None,
                    1,
                    -1,
                    geomid,
                )

                distances.append(distance)

            print(
                " | ".join(
                    f"{name}: {distance:.4f} m"
                    for name, distance
                    in zip(TOF_NAMES, distances)
                )
            )

        viewer.sync()

        time.sleep(0.002)