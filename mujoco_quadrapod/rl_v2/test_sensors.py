import unittest

import mujoco
import numpy as np

from sensors import HardwareSensorSimulator, default_sensor_config


class SensorSimulatorTests(unittest.TestCase):
    def test_sensor_sample_has_hardware_fields_only(self):
        simulator = HardwareSensorSimulator(default_sensor_config())
        simulator.reset()
        sample = simulator.step(np.zeros(simulator.model.nu))

        self.assertEqual(sample.tof_distance.shape, (4,))
        self.assertEqual(sample.tof_valid.shape, (4,))
        self.assertEqual(sample.gravity_body.shape, (3,))
        self.assertEqual(sample.gyro_body.shape, (3,))
        self.assertEqual(sample.accelerometer_body.shape, (3,))
        self.assertTrue(np.all(sample.tof_valid))
        self.assertTrue(np.all(np.isfinite(sample.tof_distance)))

    def test_tof_distances_change_with_body_height(self):
        simulator = HardwareSensorSimulator(default_sensor_config())
        simulator.reset()
        baseline = simulator.step(np.zeros(simulator.model.nu)).tof_distance
        simulator.data.qpos[2] += 0.01
        mujoco.mj_forward(simulator.model, simulator.data)
        raised = simulator.read().tof_distance
        self.assertTrue(np.all(raised > baseline))


if __name__ == "__main__":
    unittest.main()
