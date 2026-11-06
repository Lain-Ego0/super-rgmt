"""Verify the local Isaac Lab/Newton G1 runtime using read-only reference assets.

This creates a tiny simulator scene only. It does not train, edit the reference
checkout, or download anything. Supply --g1-usd pointing to a compatible G1 USD.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REFERENCE = Path("/home/zero/Documents/super-rgmt/supermimic_vibed")
DEFAULT_USD = Path("/home/zero/Zeroth-Origo/origo-supermimic/data/g1_usd/g1_29dof/g1_29dof.usda")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=2)
parser.add_argument("--g1-usd", type=Path, default=DEFAULT_USD)
parser.add_argument("--mj-contacts", action=argparse.BooleanOptionalAction, default=True)
if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    parser.print_help()
    raise SystemExit(0)
from isaaclab.app import add_launcher_args, launch_simulation

add_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1:
    parser.error("--num-envs must be positive")
if not args.g1_usd.is_file():
    parser.error(f"G1 USD not found: {args.g1_usd}")
if not (REFERENCE / "supermimic" / "robot.py").is_file():
    parser.error(f"reference robot config not found under {REFERENCE}")
sys.path.insert(0, str(REFERENCE))

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate, replace
from supermimic.robot import G1_CFG, G1_JOINT_NAMES


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = replace(
        G1_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=replace(G1_CFG.spawn, usd_path=str(args.g1_usd.resolve())),
    )


physics = NewtonCfg(
    solver_cfg=MJWarpSolverCfg(
        njmax=500,
        nconmax=128,
        cone="pyramidal",
        integrator="implicitfast",
        use_mujoco_contacts=args.mj_contacts,
    ),
    num_substeps=4,
)
sim_cfg = sim_utils.SimulationCfg(dt=0.02, device=args.device or "cuda:0", physics=physics)
with launch_simulation(sim_cfg, args):
    sim = SimulationContext(sim_cfg)
    scene = instantiate(SceneCfg(num_envs=args.num_envs, env_spacing=0.0, replicate_physics=True))
    sim.reset()
    robot = scene["robot"]
    joints = list(robot.joint_names)
    if joints != G1_JOINT_NAMES:
        raise RuntimeError(f"articulation joint order differs from retarget order:\n{joints}")
    for _ in range(8):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(0.02)
    torch.cuda.synchronize() if str(sim_cfg.device).startswith("cuda") else None
    print(f"NEWTON_SMOKE_OK envs={args.num_envs} device={sim_cfg.device}")
    print(f"JOINTS({len(joints)})={joints}")
    print(f"BODIES({len(robot.body_names)})={list(robot.body_names)}")
    print(f"ROOT_Z={robot.data.root_link_pos_w.torch[:, 2].detach().cpu().tolist()}")
    scene.close()
    sim.stop()
