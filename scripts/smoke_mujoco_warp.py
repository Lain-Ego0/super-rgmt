"""Step a batched G1 MJCF directly in MuJoCo Warp, without launching Kit.

This is a backend/asset probe only: the MJCF contains no actuators, so it does
not validate the Isaac Lab articulation, controller, observations, or PPO runner.
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MJCF = Path("/home/zero/Zeroth-Origo/origo-supermimic/data/g1_description/g1_29dof.xml")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
parser.add_argument("--worlds", type=int, default=2)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
if not args.mjcf.is_file():
    parser.error(f"MJCF not found: {args.mjcf}")
if args.worlds < 1 or args.steps < 1:
    parser.error("--worlds and --steps must be positive")

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

mj_model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
# Paper training uses 50 Hz control and four 5 ms physics substeps. This MJCF
# has a 2 ms standalone FK timestep, so set the paper substep for this probe.
mj_model.opt.timestep = 0.005
if mj_model.nu != 0:
    raise RuntimeError(f"expected kinematic/dynamics MJCF without actuators, found nu={mj_model.nu}")
joint_names = [
    mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(mj_model.njnt)
    if mj_model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
]
if len(joint_names) != 29:
    raise RuntimeError(f"expected 29 actuated G1 joints, found {len(joint_names)}: {joint_names}")

with wp.ScopedDevice(args.device):
    model = mjw.put_model(mj_model)
    data = mjw.put_data(
        mj_model,
        mujoco.MjData(mj_model),
        nworld=args.worlds,
        nconmax=128,
        njmax=500,
    )
    for _ in range(args.steps):
        mjw.step(model, data)
    wp.synchronize()
    times = data.time.numpy()
    qpos = data.qpos.numpy()

if not np.isfinite(qpos).all():
    raise RuntimeError("MuJoCo Warp produced non-finite qpos")
expected_time = args.steps * mj_model.opt.timestep
if not (abs(times - expected_time) < 1e-6).all():
    raise RuntimeError(f"unexpected simulation times {times}; expected {expected_time}")
print(
    f"MUJOCO_WARP_SMOKE_OK worlds={args.worlds} steps={args.steps} "
    f"device={args.device} control_steps={args.steps / 4:g} nq={mj_model.nq} nv={mj_model.nv} nu={mj_model.nu}"
)
print(f"JOINTS({len(joint_names)})={joint_names}")
print(f"TIME={times.tolist()}")
