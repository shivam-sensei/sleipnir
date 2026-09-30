# Sleipnir RL Environment V2 Specification

Status: design baseline  
Scope: standing first; locomotion is deferred  
Shared model: `model/quadrapod.xml` and its meshes remain unchanged

## 1. Hardware contract

### Sensors

- Four downward-facing ToF sensors mounted around the base: front, rear, left,
  and right.
- One body-mounted IMU.
- No joint encoders are assumed.
- No dedicated foot-contact switches are assumed.
- Motor-current/load feedback is optional future hardware, not part of V2.

### Actuation

- Eight servo position targets, one per joint.
- The policy receives the command history it generated, not the true simulated
  joint positions.
- The simulator retains true joint state for diagnostics only.

### Mechanical interface

- Feet use a high-friction rubber pad.
- Rubber friction is represented in the MJCF ground/foot contact model, but
  foot contact is not exposed as an observation.

## 2. Simulated sensor model

The sensor layer is separate from the task/reward layer and produces the only
observations available to PPO.

### ToF

- Each ToF returns the distance from its mounting point along its mounting
  ray to the first valid ground intersection.
- Invalid/no-return behavior is represented explicitly rather than replaced
  with a perfect value.
- The initial model uses one sample per environment step; update-rate
  decimation is configurable.
- Noise, clipping, latency, and dropout are configurable and disabled only
  for deterministic sensor validation.

### IMU

- Orientation signal: gravity vector in the IMU/body frame (3 values).
- Gyroscope: body-frame angular velocity (3 values).
- Accelerometer: specific force in the IMU/body frame (3 values), including
  configurable bias and noise.
- Absolute yaw is not exposed.
- The IMU origin and axes are defined by the existing `imu` site and a named
  frame convention; no implicit world-yaw value is used.

## 3. Observation design

The observation dimensionality is not fixed in advance. Candidate features are:

- four ToF distances;
- gravity vector;
- gyroscope;
- accelerometer;
- current servo command;
- previous servo command;
- optional short sensor history or filtered/finite-difference features.

The first implementation must expose a named observation schema and per-field
normalization/clipping. The schema is selected after Stage 0 sensor
validation, not copied from the V1 29-value state observation.

PPO must never receive:

- true joint position or velocity;
- true foot contact state;
- true body linear velocity;
- absolute world yaw;
- MuJoCo collision/contact flags.

## 4. Action contract

- Action shape: eight normalized values in `[-1, 1]`.
- Values are converted to servo targets using the existing joint limits and
  neutral servo convention.
- A configurable target slew-rate limit models servo command-rate limits.
- The environment records requested action, rate-limited command, and true
  joint state separately for diagnostics.

## 5. Timing and latency

- Physics timestep remains the value defined by the frozen MJCF.
- Control frequency, sensor update frequencies, and action hold duration are
  explicit configuration values.
- Each sensor stream may have a configurable integer-sample delay.
- V2 evaluation must include both zero-latency and hardware-estimate latency
  profiles.

## 6. Noise and domain randomization

Stage 0 starts deterministic for sensor verification. Training profiles then
add:

- ToF Gaussian noise, range clipping, dropout, and bounded bias;
- IMU gyro/accelerometer noise and slowly varying bias;
- servo target tracking error and bounded response delay;
- small mass, friction, and initial-pose variation.

Randomization ranges are configuration data and are logged with every
evaluation run.

## 7. Episode initialization and termination

Initial standing episodes start near the neutral pose with small randomized
body roll, pitch, height, and joint perturbations.

Terminate when any of the following occurs:

- body height is below the configured safe threshold;
- gravity-vector uprightness falls below the configured threshold;
- simulation becomes invalid/non-finite;
- an optional safety-limit condition is triggered.

The episode horizon is a truncation, not a fall termination.

## 8. Stage 1 standing reward

The reward is based on hardware-observable task outcomes and safe actuation:

- upright body orientation;
- target body height;
- low roll/pitch angular rate;
- stable/consistent ToF geometry;
- low action magnitude and action rate;
- distance from joint limits, used as a safety regularizer;
- self-collision penalty;
- fall penalty.

True contacts, true joint state, and true velocities may be included in debug
metrics but must not directly shape the V2 policy reward unless they are
replaced by a hardware-observable proxy.

Reward components and episode metrics are emitted by name on every step.

## 9. Instrumentation

Every episode can log:

- ToF readings, validity, age, and filtered values;
- gravity vector, gyro, and accelerometer;
- requested and applied servo targets;
- true joint positions/velocities for debug only;
- body height, roll, pitch, and yaw rate;
- joint-limit proximity;
- self-collision count;
- action magnitude and action rate;
- reward components;
- termination/truncation reason.

Training and evaluation records include the full V2 configuration and random
seed so policies are reproducible and comparable.

## 10. Development curriculum

1. **Stage 0: sensor simulation** — validate ToF and IMU behavior with logged
   traces and visualization; no PPO.
2. **Stage 1: standing** — maintain a stable body configuration.
3. **Stage 2: disturbance recovery** — front, rear, left, right, and yaw
   disturbances.
4. **Stage 3: deliberate weight shifting** — command four body-shift
   directions.
5. **Stage 4: leg lifting** — unload, lift, and replace one leg.
6. **Stage 5: locomotion** — introduce commanded body velocity and gait
   learning only after the preceding stages pass.

Each stage has a separate policy checkpoint and benchmark; later stages do not
replace earlier regression tests.

## 11. Evaluation criteria

Standing and recovery are evaluated over multiple seeds and disturbance
directions. Required metrics include:

- survival rate and time to failure;
- body-height error;
- roll/pitch RMS and peak angle;
- roll/pitch/yaw-rate RMS;
- ToF validity and stability;
- action magnitude and action-rate RMS;
- joint-limit and self-collision events;
- episode return with reward components.

The best policy is selected by the stage benchmark, not by nominal episode
return alone.

## 12. Implementation boundary

V1 files and policies are retained as historical prototypes. V2 code lives
under `rl_v2/` and may import shared model assets, but it must not silently
reuse V1 observation or reward helpers.

The first V2 implementation deliverables are:

1. sensor configuration/schema types;
2. deterministic ToF and IMU simulation;
3. sensor logging/visualization;
4. hardware-observation Gymnasium environment;
5. instrumented Stage 1 standing reward;
6. evaluation and regression benchmarks.

The following remain explicit decisions after Stage 0, not hidden assumptions:

- final observation field set and dimensionality;
- whether acceleration improves standing/recovery;
- number of history frames;
- ToF normalization and filtering;
- sensor rates and measured hardware latency;
- noise/randomization ranges from real sensor measurements.

## Stage 0 commands

From the project root, using the project virtual environment:

```bash
.quadruped/bin/python rl_v2/test_sensors.py
.quadruped/bin/python rl_v2/stage0_sensor_log.py \
  --output /tmp/sleipnir_stage0_sensor_log.csv
.quadruped/bin/python rl_v2/plot_sensor_log.py \
  /tmp/sleipnir_stage0_sensor_log.csv \
  --output /tmp/sleipnir_stage0_sensor_plot.png
```

The CSV contains the hardware-path ToF/IMU signals and a separate `debug_`
section containing exact MuJoCo state. The debug fields are for diagnosis and
are not consumed by a policy.
