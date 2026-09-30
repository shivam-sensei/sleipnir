# Sleipnir RL Environment V1 Archive

This directory contains the original simulation-centric standing-policy
prototype. It is retained for comparison and reproducibility, but it is not
the hardware-compatible V2 controller.

## Contents

- [python/](./python/) — V1 Gymnasium environment, training, evaluation, and
  visualization scripts.
- [artifacts/](./artifacts/) — saved PPO policies and TensorBoard training
  logs.

The V1 policy observation contains simulated joint state and foot-contact
information. Those values are useful for development, but they are not
available in the planned physical sensor suite and must not be reused as the
V2 policy observation.

The MuJoCo model and meshes remain shared at the project root:

- [model/](../model/)
- [meshes/](../meshes/)

The hardware-first V2 design is documented in [rl_v2/SPEC.md](../rl_v2/SPEC.md).
