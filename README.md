# Extreme-RGMT reproduction

This repository is an in-progress reproduction of **Extreme-RGMT: Continual Learning of Highly Dynamic Skills for Robust Generalist Humanoid Control** (arXiv:2607.20110v1). It is not result-equivalent: local data is a 40-clip LAFAN1 smoke corpus, Stage-I training and deterministic diagnostic evaluations have run, but the learned policy underperforms the zero-residual baseline even after a 1,024-environment PPO diagnostic, and no paper-aligned Stage-II result or full-corpus evaluation exists.

## Current implementation

- Stage I: policy/model code, paper-derived PPO and reward settings, observation noise, domain randomization, LAFAN1-to-PMT motion conversion, and an Isaac Lab/PMT training launcher.
- Stage II: PACE/STAR sampling and objective helpers, plus a simulator runner. A one-iteration Isaac Sim/PPO software smoke completed with two clips assigned one-per-role; this was not a paper-derived motion split. The valid mastered/challenging split from five randomized rollouts per clip and a successful Stage-II training/evaluation run are still outstanding.
- Data: 40 G1-retargeted LAFAN1 clips under `data/pmt_npz/`. The AMASS and in-house Xsens subsets named in the paper are not available locally. The local corpus is for integration checks and cannot establish paper-level performance.
- Runtime: the launcher uses pinned [PMT](https://github.com/Mondo-Robotics/PMT) (`RGMT-G1-v0`) with PhysX. PMT was selected as an existing G1 tracking/environment integration backend; it is not named by the paper and is not evidence of simulator fidelity. The paper does not identify its Stage-I training physics backend. A minimal PPO run has entered simulation and completed one iteration; longer training still has not produced a useful tracking policy.

See [the detailed Stage-I audit](docs/extreme-rgmt-stage1.md) and [the gated reproduction plan](REPRODUCTION_PLAN.md) for assumptions, version pins, evidence, and unresolved requirements.

## Checks

Run the isolated unit suite with plugin autoload disabled in the task venv:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-extreme-rgmt/bin/python -m pytest tests -q
```

Current recorded result: **49 passed** (including the reference-residual action equation tests). These tests validate local components, not simulator behavior or convergence.

Regenerate the local motion manifest after changing the ignored data directory:

```bash
.venv-extreme-rgmt/bin/python scripts/audit_motion_dataset.py data/pmt_npz \
  --present-source LAFAN1 --output data/pmt_npz_manifest.json
```

Check launcher arguments without starting Isaac Sim:

```bash
.venv-extreme-rgmt/bin/python scripts/train_stage1.py --help
```

The Isaac Sim runtime has already completed a minimal PPO smoke iteration. The Stage-I launcher accepts `--motion-dir`, `--num-envs`, `--iterations`, and `--seed`; see `docs/extreme-rgmt-stage1.md` before invoking it. Do not interpret a smoke run as a reproduction result.

## Repository layout

- `extreme_rgmt/`: policy, MDP, randomization, and PACE/STAR components.
- `configs/`: paper-derived Stage-I settings plus explicitly marked assumptions.
- `scripts/`: dataset conversion/audit, simulator probes, and Stage-I launcher.
- `tests/`: simulator-independent component tests.
- `docs/`: implementation audit and evidence log.
