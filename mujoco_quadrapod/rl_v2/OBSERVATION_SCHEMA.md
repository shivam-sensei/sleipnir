# Frozen Hardware Observation Schema (V2)

**Version**: 1.0  
**Status**: Locked for Stage 1  
**Validated**: 2024  

---

## Overview

This schema defines the exact observation vector that all Stage 1+ environments must use. It is derived from Stage 0 sensor validation and is hardware-compatible (only includes signals that the physical Sleipnir can measure).

**Key constraint**: If a signal cannot be measured on the real robot, it cannot appear in the observation vector.

---

## Observation Vector (29 features)

### Structure

```python
observation = [
    # ToF distances (4 features) — meters
    tof_front,       # index 0
    tof_rear,        # index 1
    tof_left,        # index 2
    tof_right,       # index 3
    
    # Gravity vector (3 features) — normalized, body frame
    gravity_x,       # index 4  [-1.0, +1.0]
    gravity_y,       # index 5  [-1.0, +1.0]
    gravity_z,       # index 6  [-1.0, +1.0]
    
    # Angular velocity (3 features) — rad/s, body frame
    gyro_x,          # index 7   roll rate
    gyro_y,          # index 8   pitch rate
    gyro_z,          # index 9   yaw rate
    
    # Acceleration (3 features) — m/s², body frame (specific force)
    accel_x,         # index 10
    accel_y,         # index 11
    accel_z,         # index 12
    
    # Current servo commands (8 features) — radians or positions
    cmd_1,           # index 13  (coxa 1)
    cmd_2,           # index 14  (femur 1)
    cmd_3,           # index 15  (tibia 1)
    cmd_4,           # index 16  (coxa 2)
    cmd_5,           # index 17  (femur 2)
    cmd_6,           # index 18  (tibia 2)
    cmd_7,           # index 19  (coxa 3)
    cmd_8,           # index 20  (femur 3)  [note: only 8 joints for quadruped 4 legs]
    
    # Previous servo commands (8 features) — radians or positions (t - 1)
    prev_cmd_1,      # index 21
    prev_cmd_2,      # index 22
    prev_cmd_3,      # index 23
    prev_cmd_4,      # index 24
    prev_cmd_5,      # index 25
    prev_cmd_6,      # index 26
    prev_cmd_7,      # index 27
    prev_cmd_8,      # index 28
]

# Total: 29 features
```

---

## Sensor Specifications

### 1. ToF Distances

**Physical Mounting**:
```
              Front (tof_front)
                    ↓
    Left ←   Body IMU center   → Right
   (tof_left)                   (tof_right)
                    ↑
                Rear (tof_rear)
```

**Typical Range**: 0.030 m to 0.200 m (from ground when upright to nose/rear tips)  
**Mount Height Above Ground** (nominal upright pose): ~0.040 m  
**Update Rate**: 200 Hz (determined by MuJoCo simulation step)  
**Noise**: None in Stage 0; may be added in Stage 1+ to match hardware.

**Interpretation**:
- Used to estimate body height (e.g., median or mean of four distances).
- Asymmetric values (front ≠ rear, left ≠ right) indicate body tilt.
- All four are necessary; not reducible to a single "height" signal (validated in Stage 0).

### 2. Gravity Vector

**Definition**: Downward acceleration in body-frame coordinates, normalized to unit magnitude.  
**Computation**: `gravity_body = R_world_to_body @ [0, 0, -9.81]`, then normalize.

**Typical Values** (static hover, upright):
```
gravity_x ≈ 0.0
gravity_y ≈ 0.0
gravity_z ≈ -1.0
```

**During roll tilt** (nose up):
```
gravity_x ≈ +0.5  (for 30° roll)
gravity_y ≈ 0.0
gravity_z ≈ -0.87
```

**Interpretation**:
- `(gravity_x, gravity_y)` encodes body roll and pitch.
- `gravity_z` remains close to -1 (unless robot is tipping severely).
- Provides orientation estimate independent of ToF geometry.
- Expected stability: std < 0.005 during normal leg motion (validated in Stage 0).

### 3. Angular Velocity (Gyroscope)

**Frame**: Body-frame (rotates with robot; not world-frame).  
**Units**: rad/s.  
**Range**: [−1.0, +1.0] rad/s typical; servo limits allow faster rotation.

**Mapping** (body frame, right-hand rule):
- `gyro_x` = roll rate (rotation about forward axis)
- `gyro_y` = pitch rate (rotation about rightward axis)
- `gyro_z` = yaw rate (rotation about upward axis)

**Interpretation**:
- Immediate angular velocity; integrating gyro gives orientation drift, which is why we use gravity vector.
- Essential for detecting disturbances and dynamic maneuvers.
- Less ambiguous than relying on absolute yaw from magnetometer (which we don't have).

### 4. Accelerometer (Specific Force)

**Definition**: Measured acceleration minus gravity in body frame.  
**Units**: m/s².  
**Range**: [−5, +5] m/s² typical for servo motions; can exceed during impacts.

**Interpretation**:
- Shows body-frame linear acceleration.
- Transient peaks during leg footfalls, sudden weight shifts, or disturbances.
- Helps distinguish standing oscillation from unstable tipping.
- Expected stability: std ~0.5 m/s² during normal leg motion (deterministic transients).

### 5. Servo Commands (Current and Previous)

**Definition**:
- `cmd_i` = command sent to servo `i` in the current step.
- `prev_cmd_i` = command sent to servo `i` in the previous step.

**Values**: Servo angle/position in radians (or native units if servo accepts position; typically normalized [−π, +π] or [0, 2π]).

**Interpretation**:
- The policy always knows what commands it issued.
- Servo tracking error is not modeled in deterministic Stage 0; actual servo position may lag command.
- Current + previous gives 1-step history without requiring true joint encoders.
- Allows policy to learn servo dynamics (e.g., "if I command extension, expect 1-step lag").

---

## Normalization and Scaling

For PPO training, observations should be normalized to approximately zero mean and unit variance.

### Recommended Scaling

```python
# ToF: assume 0.030 m to 0.200 m range
tof_scaled = (tof - 0.115) / 0.085  # Approximate mean ≈ 0.115, std ≈ 0.085

# Gravity: already unit magnitude; can scale by 1.0 (already normalized)
gravity_scaled = gravity  # No scaling needed; magnitude ≈ 1.0

# Gyro: assume ±1.0 rad/s typical
gyro_scaled = gyro / 1.0  # Or use observed std from data

# Accel: assume ±5 m/s² typical
accel_scaled = accel / 5.0

# Servo commands: typically [−π, +π]
cmd_scaled = cmd / 3.14  # Normalized to ≈ [-1, +1]
```

Exact scaling should be determined from training data (running the environment and computing empirical statistics on observation vectors).

---

## Excluded Signals

### Explicitly NOT Included

- **True joint positions** (`q_1`, ..., `q_8`): Not available on physical robot without encoders.
- **True joint velocities** (`qdot_1`, ..., `qdot_8`): Not available on physical robot.
- **Foot contact state**: Not available without pressure sensors or switches.
- **Body linear velocity** (`vx`, `vy`, `vz`): Not directly available; would need integration of accelerometer over time.
- **Absolute yaw orientation**: IMU yaw drifts without external reference; excluded intentionally.
- **MuJoCo internal state** (collision flags, constraint forces, etc.): Not available on physical robot.

### Why Excluded

1. **Causality**: RL policies trained on signals unavailable on hardware cannot transfer to real robot.
2. **Realistic Observation**: Encourages learning control strategies that depend only on hardware-compatible measurements.
3. **Sim-to-Real Transfer**: Policies trained on this observation can run on physical Sleipnir without modification.

---

## Stage 1 Environment Contract

### HardwareObservationEnv Must Provide

```python
class HardwareObservationEnv(gym.Env):
    def step(self, action):
        # ... update simulation ...
        obs = self._compute_hardware_observation()  # 29D vector
        info = self._compute_debug_info()            # True state, only for logging
        return obs, reward, terminated, truncated, info
    
    def _compute_hardware_observation(self):
        # Return exactly 29-element array with no true joint state, no contacts
        # Read from HardwareSensorSimulator
        return np.array([...])
    
    def _compute_debug_info(self):
        # Return dict with MuJoCo ground truth for diagnostics
        # NOT used by PPO; only for visualization/debugging
        return {'body_height': ..., 'roll': ..., 'q': [...], ...}
```

### Policy Input / Output

```
Input:  obs (29D) from HardwareObservationEnv
        ↓
        PPO Policy
        ↓
Output: action (8D servo commands)
```

The policy cannot accidentally receive true joint state or foot contacts because they do not exist in `obs`.

---

## Versioning and Future Changes

**Version 1.0 (Frozen for Stage 1)**:
- 29D observation, all Stage 0 tests passed.
- Lock this version until Stage 1 training is complete.

**Version 2.0 (Future)**:
- If Stage 1 fails to learn standing due to insufficient information, consider adding:
  - Motor current (indirect contact signal) — if servo hardware exposes it.
  - Additional ToF/IMU frames for filtering — if latency becomes critical.
  - Body height estimate (derived ToF feature) — if learned height filtering is insufficient.
- Changes require Stage 0 validation of new signals before deployment.

---

## References

- **SPEC.md**: Comprehensive hardware and design contract; includes stages 1–5 curriculum.
- **STAGE0_COMPLETE.md**: Validation results and rationale for schema choice.
- **sensors.py**: Implementation of HardwareSensorSimulator used to generate this observation.
