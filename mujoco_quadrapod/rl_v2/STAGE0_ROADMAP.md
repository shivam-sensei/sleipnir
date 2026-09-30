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

## Current Status (Post Stage 0 Baseline)
- ✓ Four-channel ToF geometry is geometrically sound.
- ✓ Individual perturbations (height, roll, yaw) respond as expected.
- ✗ Leg motion coupling unknown.
- ✗ Pitch response not yet validated.
- ✗ Combined tilts not yet tested.
- ✗ Filtering/latency requirements unknown.
- **Decision**: Do not freeze observation schema yet.

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

## Final Observation Schema (post-tests)
- Candidate **Option C remains primary**: 29D with gravity, gyro, accel, 
  4 ToF, current + previous servo commands.
- But schema will be **locked only after** the above tests confirm:
  - All four ToF values contribute unique information under leg motion.
  - Gravity/gyro/accel remain stable.
  - Latency/filter strategy is clear.

---

# Stage 1 Preparation (after schema is frozen)

## Do NOT reuse V1 code:
- ❌ `rl_v1/python/quadruped_env.py` — exposes true joint state and foot contacts.
- ❌ `rl_v1/python/train.py` — uses V1 reward weights and observation.
- ✓ Shared: `model/quadrapod.xml`, `meshes/`.

## Stage 1 will build:
1. **HardwareObservationEnv** (Gymnasium)
   - Wraps `HardwareSensorSimulator`.
   - Converts sensor sample to hardware-only observation.
   - Enforces the frozen schema.
   - Keeps debug/true state in `.info` for diagnostics.

2. **Standing Reward** (hardware-observable only)
   - Upright orientation (gravity vector).
   - Target height (ToF estimation + IMU roll/pitch).
   - Low angular velocity (gyro).
   - Action magnitude and smoothness.
   - Joint limits (from servo commands).
   - Self-collision (still needs MuJoCo contact data for debug).

3. **Training / Evaluation Scripts**
   - New `train_standing.py` (not reusing V1 train.py).
   - Logging/checkpoint structure for hardware-observed policies.

---

# Summary

**Next concrete milestone**: Complete tests 1, 2, 3 above.

**Blocking issue**: Leg motion coupling is unknown. Once it's characterized, 
the observation schema can be frozen with confidence.

**Then**: Stage 1 implementation with deterministic sensor observation, 
standing reward, and PPO training.
