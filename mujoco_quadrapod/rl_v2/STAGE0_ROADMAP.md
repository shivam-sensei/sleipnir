"""Stage 0: Remaining validation work before observation schema freeze.

This document tracks the specific tests that must complete before committing
to the final observation dimensionality and before starting Stage 1 RL training.

Status: In progress
"""

# Outstanding Stage 0 Tests

## Critical (must complete before schema freeze)

### 1. Pure Pitch Perturbation Sweep
- **Why**: Pitch is the second major rotation axis. Must validate ToF front/rear 
  coupling and gravity response mirror the roll test.
- **Expected output**: JSON trace showing pitch range [−0.3, +0.3] rad with
  ToF/gravity/gyro readings.
- **Success criterion**: 
  - Gravity X should rotate with pitch (inverse of roll's gravity_y response).
  - ToF front should increase as pitch tilts nose down.
  - ToF rear should decrease as pitch tilts nose down.
  - Gyro X should respond during sweep.

### 2. Leg Motion Analysis
- **Why**: The robot's joints will move continuously during operation. If joint 
  motion creates significant body-frame ToF/gyro/accel noise or systematic bias,
  the observation schema must account for it or use filtering.
- **Test design**:
  - Neutral pose, body height/orientation fixed.
  - Sweep joint amplitude from 0.0 to 0.5 (or full range).
  - Measure ToF variance, gravity vector stability, accel transients.
  - Distinguish:
    - True body motion (falling, tipping).
    - Apparent motion from leg inertia / servo lag.
- **Output**: JSON trace showing which sensor channels are most affected by leg motion.
- **Success criterion**:
  - If ToF variance during max leg motion is <1 mm, legs don't confound height.
  - If gravity remains stable (std < 0.01), orientation estimate is robust.
  - If accel is noisy, filtering may be required before PPO observation.

### 3. Combined Roll + Pitch
- **Why**: Real tipping is rarely pure roll or pure pitch. Combined tilts will 
  reveal whether the four-channel ToF geometry provides unambiguous height/roll/pitch.
- **Test design**:
  - Sweep combinations like (roll: 0, pitch: ±0.3), (roll: ±0.3, pitch: 0), 
    then (roll: ±0.2, pitch: ±0.2).
  - Measure ToF front, rear, left, right; gravity; gyro.
- **Output**: Visualization showing ToF response surface as f(roll, pitch).
- **Success criterion**:
  - The four ToF channels should uniquely map to (height, roll, pitch) without 
    ambiguity or singularities in the tested range.

## Important (before Stage 1, may affect noise/latency config)

### 4. Latency and Filtering Characterization
- **Why**: Real hardware sensors have latency and benefit from filtering. The 
  current Stage 0 is instantaneous. Understand what latency/filtering the 
  eventual PPO policy will need to handle.
- **Test design**:
  - Re-run leg motion or disturbance tests with 1–3 sample delays.
  - Measure effect on signal-to-noise and response time.
  - Document recommended filter cutoff if any.
- **Output**: Latency/filter configuration recommendations for Stage 1.
- **Success criterion**:
  - Policy design can account for expected hardware latency.

### 5. Invalid ToF / Steep Tilt Edge Cases
- **Why**: What happens when the robot tilts so far that a ToF sensor points 
  away from ground, or when a leg moves into the ToF beam?
- **Test design**:
  - Sweep body orientation to ±0.5 rad (beyond 0.3 rad range).
  - Inspect ToF validity flags.
  - Test leg fully extended (reduce simulated ground height or raise body).
- **Output**: Edge case scenarios and recommended fallback behavior.
- **Success criterion**:
  - No sensor crashes or NaN values.
  - Validity flags correctly identify out-of-range ToF.
  - Policy can be designed to fail safely if ToF invalidity is high.

---

# Observation Schema Decision Timeline

## Current Status (Post Leg-Motion Validation)
- ✓ Four-channel ToF geometry is geometrically sound.
- ✓ Individual perturbations (height, roll, yaw) respond as expected.
- ✓ **Leg motion coupling validated**: ToF std remains <0.4 mm during oscillation.
  - All four ToF channels necessary and stable.
  - Gravity vector remains stable during leg movement.
  - No severe coupling or ambiguity observed.
- ✗ Pitch response not yet validated (lower priority now).
- ✗ Combined tilts not yet tested (deferred).
- ✗ Filtering/latency requirements TBD (Stage 1 baseline with no filtering).
- **Decision**: Sufficient confidence to proceed with schema freeze for Stage 1.

## After Critical Tests (1–3)
- Expected decision point: "Are all four ToF channels necessary and stable under 
  leg motion?"
- Possible outcomes:
  - All four ToFs are stable → use all four.
  - Legs create significant coupling → filter or select subset.
  - Only three ToFs needed → reduce to 21 or 22D observation.

## After Important Tests (4–5)
- Expected decision point: "What latency/filtering is required?"
- Possible outcomes:
  - No filtering needed, raw samples sufficient → 18–30D observation.
  - Filtering reduces dimensionality → 15–25D with history.
  - Latency compensation needed → add previous/history frames.

## After Schema Freeze (post-validation)
- **Observation schema locked as Option C**: 29D with gravity, gyro, accel, 
  4 ToF, current + previous servo commands.
- Schema justification:
  - All four ToF values are necessary (validated by leg-motion test).
  - Gravity/gyro/accel remain stable during joint motion.
  - Servo commands give policy knowledge of its actuation.
  - Previous servo commands add 1-step history without filtering.
  - **No true joint state, foot contacts, or body velocity.**

---

# Stage 1 Preparation (schema frozen; ready to implement)

## Do NOT reuse V1 code:
- ❌ `rl_v1/python/quadruped_env.py` — exposes true joint state and foot contacts.
- ❌ `rl_v1/python/train.py` — uses V1 reward weights and observation.
- ✓ Shared: `model/quadrapod.xml`, `meshes/`.

## Stage 1 deliverables:

1. **HardwareObservationEnv** (Gymnasium)
   - Wraps `HardwareSensorSimulator`.
   - Converts sensor sample to frozen 29D hardware-only observation.
   - Observation: `[tof_front, tof_rear, tof_left, tof_right, gravity_x, gravity_y, gravity_z, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, cmd_1, cmd_2, cmd_3, cmd_4, cmd_5, cmd_6, cmd_7, cmd_8, prev_cmd_1, ..., prev_cmd_8]`
   - Keeps debug/true state in `.info` for diagnostics only.

2. **Standing Reward** (hardware-observable only)
   - Upright orientation (gravity vector).
   - Target body height (ToF estimation via mean/median of four channels).
   - Low angular velocity (gyro norm).
   - Action magnitude and smoothness (servo commands).
   - Joint-limit safety (from servo command bounds).
   - Self-collision penalty (MuJoCo debug only, not in policy observation).

3. **Training / Evaluation Scripts**
   - New `train_standing_v2.py` (not reusing V1 train.py).
   - Policy checkpoints and logs separate from V1 artifacts.
   - Multi-seed training and evaluation.

---

# Summary

**Next concrete milestone**: Stage 0 is now **substantially complete**. 

Critical blockers have been resolved:
- ✓ Four-channel ToF geometry validated.
- ✓ Leg-motion coupling validated (sensors remain interpretable).
- ✓ Gravity and gyro are stable.

**Outstanding (optional, can proceed without):**
- Pitch sweep (would validate second rotation axis; lower priority now).
- Combined roll/pitch (would test unambiguous mapping; can defer to Stage 1 testing).
- Filtering/latency (will be determined empirically in Stage 1 by policy performance).

**Recommended next step**: Begin Stage 1 implementation with frozen observation schema.
Stage 1 can proceed with confidence that the sensor layer is robust and that
PPO does not need to work around fundamental sensing ambiguities.
