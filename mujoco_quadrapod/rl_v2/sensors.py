import mujoco
import numpy as np
from dataclasses import dataclass

TOF_NAMES = ("front", "rear", "left", "right")
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
@dataclass(frozen=True)
class ToFMount:
    offset: tuple[float, float, float]
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)


TOF_MOUNTS = (ToFMount(offset=(0.060,0.0,0.0)),
                ToFMount(offset=(-0.060,0.0,0.0)),
                ToFMount(offset=(0.0,0.060,0.0)),
                ToFMount(offset=(0.0,-0.060,0.0)))

@dataclass
class SensorSample:
    tof_distance: np.ndarray
    tof_valid: np.ndarray
    accelerometer: np.ndarray
    gyro: np.ndarray

class HardwareSensorSimulator:
    def __init__(self,model_path):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.imu_site_id = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,"imu")
        self.accel_sensor_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            "imu_accel",
        )
        self.gyro_sensor_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            "imu_gyro",
        )
        self.base_body_id = mujoco.mj_name2id(
    self.model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base_link",
)
        if self.imu_site_id < 0:
            raise ValueError("IMU site 'imu' was not found")

        if self.accel_sensor_id < 0:
            raise ValueError("Accelerometer sensor 'imu_accel' was not found")

        if self.gyro_sensor_id < 0:
            raise ValueError("Gyro sensor 'imu_gyro' was not found")

    def reset(self):

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        # Set all eight joints to the real neutral servo pose
        for joint_id in range(1, 9):
            servo_deg = NEUTRAL_SERVO[joint_id]

            qpos_index = self.model.jnt_qposadr[
                joint_id
            ]

            self.data.qpos[qpos_index] = self.servo_to_mujoco(
                joint_id,
                servo_deg,
            )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

    def read_imu(self):
        accel = self.data.sensordata[
            self.model.sensor_adr[self.accel_sensor_id]:
            self.model.sensor_adr[self.accel_sensor_id] + 3
        ]

        gyro = self.data.sensordata[
            self.model.sensor_adr[self.gyro_sensor_id]:
            self.model.sensor_adr[self.gyro_sensor_id] + 3
        ]

        return accel.copy(), gyro.copy()

    def servo_to_mujoco(self, joint_id, servo_deg):
        qmin, qmax = JOINT_LIMITS[joint_id]

        return qmin + (servo_deg / 180.0) * (qmax - qmin)

    def read_tof(self):

        distances = np.zeros(4, dtype=np.float64)
        valid = np.zeros(4, dtype=bool)

        base_pos = self.data.xpos[self.base_body_id]

        base_rot = self.data.xmat[
            self.base_body_id
        ].reshape(3, 3)

        for i, mount in enumerate(TOF_MOUNTS):

            local_offset = np.asarray(
                mount.offset,
                dtype=np.float64,
            )

            sensor_pos = (
                base_pos
                + base_rot @ local_offset
            )

            local_direction = np.asarray(
                mount.direction,
                dtype=np.float64,
            )

            direction = base_rot @ local_direction

            direction /= np.linalg.norm(direction)

            geomid = np.array([-1], dtype=np.int32)

            distance = mujoco.mj_ray(
                self.model,
                self.data,
                sensor_pos,
                direction,
                None,
                1,
                -1,
                geomid,
            )

            if distance >= 0.0:
                distances[i] = distance
                valid[i] = True

        return distances, valid