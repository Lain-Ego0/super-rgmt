# Extreme-RGMT Stage-I: current reproduction record

## Data preparation

The local smoke corpus uses 40 G1-retargeted LAFAN1 clips from the Hugging Face dataset `lvhaidong/LAFAN1_Retargeting_Dataset`, under the dataset's CC BY-NC-ND 4.0 terms. Keep the source attribution and license with any research use; do not redistribute transformed clips without checking the license terms. The original CSVs and generated NPZs are local ignored files under `data/`.

CSV layout is 30 Hz `[root_xyz(3), root_quat_xyzw(4), G1 joint positions(29)]`. Convert with the G1 MJCF matching this retarget, preserving its joint order:

```bash
python scripts/convert_lafan1_to_pmt.py \
  --input data/lafan1_g1 --output data/pmt_npz \
  --mjcf /path/to/matching/g1_29dof.xml --src-fps 30 --fps 50
```

The converter resamples root/joints, runs MuJoCo FK for the 14 PMT tracked bodies, and writes `fps`, `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w` (wxyz), `body_lin_vel_w`, and `body_ang_vel_w`. This is a smoke corpus, not the paper's AMASS/XtremeMotion/Xsens training mix. The selected MJCF came from a neighboring local checkout and was read-only; it is not vendored here.

## Stage-I mapping and gaps

The policy implementation and values live in `extreme_rgmt/models.py`, `configs/extreme_rgmt_stage1.yaml`, and `scripts/train_stage1.py`. The runner maps PMT's RGMT observation groups to the paper actor/critic sets and applies the paper's PPO rollout/update values. The FSQ level count (5 per scalar) and attention head count (4) are explicit assumptions because the paper does not specify them.

The current PMT task is an integration backend, not a fully faithful reproduction. The actor/action scale and exact joint ordering require runtime verification against the active robot articulation before claiming policy fidelity.

## Verification state (updated 2026-11-06)

- `conda run -n edu3_isaaclab python -m pytest -q tests/test_models.py`: 4 passed (unit-level model contract only).
- Selected Isaac Lab 3 venv can import this launcher's `--help` path with PMT on `PYTHONPATH`.
- Target Python 3.12 venv: latest unit suite (`tests/test_configuration.py`, `tests/test_models.py`, `tests/test_mdp.py`, `tests/test_domain_randomization.py`, `tests/test_dataset_audit.py`) → 12 passed; `compileall` passes. These tests cover paper reward weights and command-noise ranges as pure code, not simulator execution. Attempting to instantiate PMT RGMT config outside an app fails while importing `omni.usd`, so real config construction remains tied to simulator startup. `pip check` reports Isaac Sim 6.2/Isaac Lab lockfile version conflicts noted below.
- The earlier 2-environment PMT launch stopped at the Omniverse EULA gate before environment creation. Extreme-RGMT does not identify its training physics backend in §4.1; it mentions MuJoCo in its evaluation protocol. Thus neither PhysX nor Newton fidelity can be asserted from that passage. No RGMT tracking rollout or PPO update has yet been verified.
- No convergence checkpoint or paper-level evaluation exists. Stage-I reproduction is incomplete; Stage-II PACE/STAR and SuperMimic migration remain gated on this baseline.

## Runtime and setup

The initial PMT `RGMT-G1-v0` adapter uses PhysX. Extreme-RGMT does not specify the training physics backend in the paper text audited so far, so this is a candidate integration backend pending environment validation. Newton and MuJoCo-Warp modules are also available. The supplied SuperMimic paper and `supermimic_vibed` reference use Newton/MuJoCo-Warp; their settings must not be silently attributed to Extreme-RGMT. The task-local environment was prepared with:

```bash
python3.12 -m venv .venv-extreme-rgmt
.venv-extreme-rgmt/bin/python -m pip install --extra-index-url https://pypi.nvidia.com \
  'isaacsim[all,extscache]==6.2.0.0'
```

The neighboring read-only checkout provides Isaac Lab at commit `46d6cc3d7b115906fad12efd372fbf69f4e3d3ed`; PMT at commit `1a92390077d329d22cbcdbf89c4a5c982003fbc8` was installed for the initial adapter. Their upstream license notices are retained. Isaac Sim 6.2.0.0 and the core Isaac Lab packages are installed in the target venv. Core imports succeed without launching Kit. The PMT source at commit `1a92390077d329d22cbcdbf89c4a5c982003fbc8` is staged under ignored `third_party/PMT-pinned/` and installed editable in the target venv. Its G1 URDF and meshes were downloaded with PMT’s checksum-verifying installer into that ignored checkout. The Isaac Lab 6.2/RSL-RL extras from the neighboring Isaac Lab lockfile are installed in the target venv, and PMT registry plus G1 config imports succeed with `PYTHONPATH` set. A dependency audit reports conflicts between Isaac Sim wheel pins and Isaac Lab overrides. Newton and MuJoCo-Warp imports succeed, but runtime compatibility and a short rollout remain unverified. `scripts/train_stage1.py --help` and `scripts/smoke_newton_backend.py --help` exit without importing the simulator launcher. Attempting the Newton smoke reaches Isaac Lab's Omniverse EULA check during `isaaclab.app` initialization, before Kit or a simulation scene starts. No Kit process remains.

## Simulator validation status

The target-owned PMT probe `scripts/smoke_newton_backend.py` targets Isaac Lab/Newton using the adjacent read-only G1 USD; it has not run because `isaaclab.app` reaches the Omniverse EULA check. The separate `scripts/smoke_mujoco_warp.py` loads a G1 MJCF from the adjacent SuperMimic checkout into MuJoCo Warp on an RTX 4090, verifies 29 joint names, and advances two batched worlds for four 5 ms physics steps. That is a low-level engine/asset check for the SuperMimic asset, not an Extreme-RGMT tracking or PPO result.

The current launcher targets pinned PMT `RGMT-G1-v0` with a PhysX configuration. Extreme-RGMT’s paper text audited here does not name its training physics backend, so backend fidelity remains unresolved. PMT is being used as an integration candidate, not as evidence of exact simulator reproduction. The supplied SuperMimic reference’s Newton/MuJoCo-Warp setup belongs to the later SuperMimic migration stage.

## Reward alignment update

Extreme-RGMT Table I specifies global-anchor orientation 0.5; relative body position/orientation 1.0 each; global body linear/angular velocity 1.0 each; action rate -0.1; joint-position limits -10; undesired contacts -0.1; and feet slip -0.1. The launcher now disables PMT-only global-anchor position, retains the two listed regularizers, and adds feet slip at -0.1. Reward formulas/scales and contact selection still require source-level and simulator-side validation.

## Paper DR audit (Table II, 2026-11-06)

The paper's Table II lists ground friction `[0.10, 1.75]`, added base mass `[-3, 6] kg`, torso CoM offset `x=[-0.025,0.025] m` and `y,z=[-0.05,0.05] m`, motor strength and PD gain scales `[0.8,1.2]`, motor zero offset `[-0.01,0.01] rad`, armature scale `[1.0,1.05]`, and external pushes every `[1,3] s`. Observation noise magnitudes are gravity `0.05`, angular velocity `0.2 rad/s`, joint position `0.01 rad`, and joint velocity `0.5 rad/s`. Command perturbations are base linear velocity `±0.5 m/s`, base angular velocity `±0.52 rad/s`, gravity direction `0.05`, and joint pose `±0.1 rad`.

Against pinned PMT `RGMT-G1-v0`, static source inspection confirms matching torso CoM ranges, joint default-position offset, push interval, angular-velocity/joint-position/joint-velocity noise magnitudes, and paper actor-noisy/critic-clean separation. The target adapter now overrides static/dynamic friction ranges to `[0.10,1.75]`, adds a `[-3,6] kg` mass event on the pelvis (the explicit implementation interpretation of “base mass”), scales motor effort limits by `[0.8,1.2]` using one independent factor per environment shared across joints, scales PD stiffness and damping by `[0.8,1.2]`, scales armature by `[1.0,1.05]`, and injects the paper noise bounds into actor proprioception and reference-window velocity, gravity, and joint-pose channels. Existing PMT terms retain the paper CoM offset, motor zero offset, and 1–3 s push interval. Remaining differences: PD gain factors are sampled independently per joint and for stiffness/damping, which may differ from a single shared scale if that was intended; the friction table gives one range, so applying it to both static and dynamic friction is an explicit assumption; PMT push velocity magnitudes remain its defaults because the paper specifies only push interval, not push magnitude. Event wiring and noise bounds have unit coverage, but actual event execution, effort-limit write behavior, and observed noise distribution still need simulator validation.

The converter's 29-joint CSV order exactly matches the movable-joint order in PMT's checksum-verified `unitree_description/urdf/g1/main.urdf` (static XML check, all 29 names and indices). PMT's multi-motion command config does not declare an independent joint-name remapping; it consumes motion arrays directly against articulation tensors. The final articulation/action-manager order still requires a runtime check after the simulator starts. The launcher default now points at the actual pinned `third_party/PMT-pinned/` path.

## Policy implementation audit (2026-11-06)

A source-level comparison with Sec. III-C and Table III confirms the implementation uses separate state/action/reference MLP branches with widths `[64,128,64]`, `[29,64,64]`, and `[38,128,64]`; per-branch LayerNorm; alternating action/state history tokens with a causal mask; a 21-token command cross-attention path; two 32-scalar quantized outputs; a 29D residual action head; asymmetric privileged critic; and the listed actor/critic MLP widths. The PMT action term sets unit residual scale and tracks the reference pose as its offset, matching `q_target = q_ref + a`.

Implementation assumptions and unresolved details remain: the paper omits attention-head count and FSQ scalar level counts, so this code uses four heads and five odd levels per scalar; exact actor-critic tensor wiring still requires a live RSL-RL environment initialization and rollout. Unit tests cover shapes, gradients, causal attention, and FSQ straight-through gradients, but do not validate paper performance.

## Motion-data availability audit (2026-11-06)

`data/pmt_npz_manifest.json` is generated locally (ignored with `data/`) by `scripts/audit_motion_dataset.py` and contains per-file SHA-256, shapes, frame counts, FPS, and duration. The current 40-clip corpus has 441,140 frames at 50 Hz and 8,822 seconds (147.033 minutes / 2.45056 hours) using endpoint duration `(frames-1)/fps`. This is 22.94 seconds above the paper's 146.651-minute LAFAN1 entry, but matching aggregate duration does not establish matching clip selection; the exact paper clip list is not known. AMASS (30.677 minutes) and in-house Xsens (8.446 minutes) clips are not present in the target data directory, so the existing corpus is not the paper's complete training mixture.

An online check of the author-owned Extreme-RGMT GitHub project page repository found only static site files (`assets`, `README.md`, `index.html`) and no code repository linked from that repo; its GitHub Releases page has no releases. This means there is no public author training implementation available through that project repository as of this audit; it does not rule out unpublished or separately hosted material.
