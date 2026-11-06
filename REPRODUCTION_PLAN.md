# Reproduction plan: Extreme-RGMT, then SuperMimic

## Scope and order

This repository is the target project. The requested order is strict:

1. Reproduce the Extreme-RGMT baseline from `2607.20110v1.pdf`.
2. Validate its base-policy and continual-learning stages.
3. Only after that baseline is documented, migrate the SuperMimic method from `2610.28480v1.pdf` and the local `supermimic_vibed/` reference into the RGMT training pipeline.

The local `supermimic_vibed/` tree and both PDFs are source material. They must remain untouched. The adjacent `sdog_rl_lab` checkout is a separate Go2 velocity-control project, not an Extreme-RGMT baseline.

## Initial audit (2026-11-05)

- This checkout has only a 12-byte README and no tracked training code, assets, configs, or tests.
- The host has two RTX 4090 GPUs, NVIDIA driver 580.178.04, and about 294 GB free on the workspace filesystem. The initial audit found `/home/zero/miniconda3/envs/edu3_isaaclab` (Isaac Lab 2.2), but this is not the selected runtime. A separate read-only neighboring checkout provides `/home/zero/Zeroth-Origo/origo-supermimic/.venv` (Python 3.12.16, PyTorch 2.11+cu130, Isaac Lab 3.0.0 at commit `46d6cc3d7`, Newton 1.8 dev, MuJoCo 3.15). The selected runtime has not yet been proven compatible with pinned PMT. Docker is not installed, so use the existing native Isaac Lab environment unless a pinned alternative is required.
- A neighboring SuperMimic project contains 40 retargeted G1 LAFAN1 clips and G1 description/USD assets; they are not present in the supplied `supermimic_vibed/` reference tree and remain read-only. It is a separate checkout with local modifications, so treat it as read-only reference material. The paper-specific AMASS challenging clips, OmniXtreme/XtremeMotion set, and in-house Xsens data are not present in the target checkout and have not been verified elsewhere.
- The Extreme-RGMT paper points to an author-hosted project page. No training implementation or checkpoints are present in this workspace. Paper §4.1 does not specify the training physics backend; MuJoCo appears in the evaluation protocol. The supplied `2610.28480v1.pdf` SuperMimic paper does specify MuJoCo Warp for its experiments. Keep simulator claims separate by paper. The PMT PhysX task is a candidate RGMT integration backend, but compatibility and fidelity still require runtime verification.
- The two supplied Bilibili links have not yet been independently inspected; treat them as user-provided reference material until their contents can be checked.
- The paper identifies a Unitree G1, PPO-based Stage I, and Stage II PACE + STAR. Stage I uses a 10-step proprioception/action history and a 21-token reference window. Full faithful reproduction also requires the paper's policy, reward, randomization, motion preprocessing, optimizer and Stage-II schedule details, plus compatible robot assets and motion datasets.
- No evidence currently establishes a completed Extreme-RGMT reproduction in this workspace. We must not describe the later SuperMimic port as complete until the Stage-I/Stage-II checkpoints and paper-aligned evaluations exist.

## Implementation gates

### Gate A — runtime and data inventory

Runtime inventory is recorded above. Before long training runs, pin the selected native Isaac Lab runtime and verify or obtain compatible G1 assets plus the paper-aligned motion splits. The existing 40-clip LAFAN1 set is enough to smoke-test an environment, but it is not the paper training corpus. The local reference checkout lacks its ignored `data/` directory, while a separate adjacent checkout has G1 USD and LAFAN1 assets; those assets are read-only and may be referenced for local validation. Do not start a full run while required data or assets are absent.

### Gate B — Extreme-RGMT Stage I

Implement the paper-aligned G1 tracking environment, policy architecture, rewards, randomization, adaptive motion sampling, and PPO configuration. Validate tensor shapes, reset/termination semantics, a short rollout, and a short training smoke run before a convergence run. Save a reproducible config and checkpoint.

### Gate C — stratification and PACE/STAR

Use Stage-I per-motion completion statistics to construct mastered/challenging partitions. Implement separate acquisition and consolidation environment roles, frozen Stage-I action regularization, progress-adaptive consolidation, difficulty-conditioned advantage normalization, and segment-aware trajectory resampling. Validate sampling distributions and the loss on synthetic rollouts before training. Evaluate retention and acquisition separately.

### Gate D — SuperMimic migration

After Gates B/C have results, port the differentiable learned dynamics, replay buffer, imagined-rollout policy objective and critic into the established RGMT observation/action/reward interface. Preserve Extreme-RGMT sampling and consolidation only where compatible, and document deliberate differences. Compare simulator-only and learned-model runs under the same motion split and compute budget.

## Completion evidence

For each gate, record exact command/config, environment and data versions, seed, hardware, wall time, checkpoint path, training curves, evaluation split and metrics. Smoke tests establish execution only; they do not establish reproduction or convergence.
