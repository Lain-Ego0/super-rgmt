#!/usr/bin/env python3
"""Convert G1-retargeted LAFAN1 CSV clips to PMT tracking-motion NPZ files.

CSV source layout is [root_xyz, root_quat_xyzw, 29 SDK-order joint angles] at 30 Hz.
Body poses are recomputed with the matching G1 MuJoCo model; output is resampled to 50 Hz.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link",
    "right_knee_link", "right_ankle_roll_link", "torso_link", "left_shoulder_roll_link", "left_elbow_link",
    "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]


def slerp_xyzw(q0: np.ndarray, q1: np.ndarray, a: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot).clip(0, 1)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    aa = a[:, None]
    near = sin_theta < 1e-6
    out = np.where(near, (1 - aa) * q0 + aa * q1,
                   np.sin((1 - aa) * theta) / np.maximum(sin_theta, 1e-8) * q0
                   + np.sin(aa * theta) / np.maximum(sin_theta, 1e-8) * q1)
    return out / np.linalg.norm(out, axis=-1, keepdims=True).clip(1e-8)


def gradient(x: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(x, dt, axis=0, edge_order=2 if len(x) > 2 else 1).astype(np.float32)


def convert(csv_path: Path, out_path: Path, model: mujoco.MjModel, src_fps: float, fps: float) -> None:
    raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 36 or len(raw) < 2:
        raise ValueError(f"{csv_path}: expected at least 2 rows and 36 columns; got {raw.shape}")
    duration = (len(raw) - 1) / src_fps
    n = int(np.floor(duration * fps)) + 1
    t = np.arange(n, dtype=np.float64) * src_fps / fps
    i0 = np.minimum(np.floor(t).astype(int), len(raw) - 2)
    alpha = (t - i0).clip(0, 1)
    root_pos = raw[i0, :3] * (1 - alpha[:, None]) + raw[i0 + 1, :3] * alpha[:, None]
    root_quat = slerp_xyzw(raw[i0, 3:7], raw[i0 + 1, 3:7], alpha)
    joint_pos = raw[i0, 7:36] * (1 - alpha[:, None]) + raw[i0 + 1, 7:36] * alpha[:, None]
    joint_vel = gradient(joint_pos, 1 / fps)

    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINTS]
    if min(joint_ids) < 0:
        raise ValueError("MJCF is missing one or more expected G1 joints")
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in BODIES]
    if min(body_ids) < 0:
        raise ValueError("MJCF is missing one or more PMT tracked bodies")
    qpos_addresses = [int(model.jnt_qposadr[j]) for j in joint_ids]
    data = mujoco.MjData(model)
    body_pos = np.empty((n, len(BODIES), 3), np.float32)
    body_quat = np.empty((n, len(BODIES), 4), np.float32)  # MuJoCo/Isaac wxyz
    for k in range(n):
        data.qpos[:3] = root_pos[k]
        x, y, z, w = root_quat[k]
        data.qpos[3:7] = (w, x, y, z)
        data.qpos[qpos_addresses] = joint_pos[k]
        mujoco.mj_forward(model, data)
        body_pos[k] = data.xpos[body_ids]
        body_quat[k] = data.xquat[body_ids]
    body_lin_vel = gradient(body_pos, 1 / fps)
    body_ang_vel = np.zeros_like(body_pos)
    dt = 1 / fps
    for k in range(1, n):
        for b in range(len(BODIES)):
            q0 = body_quat[k - 1, b]
            q1 = body_quat[k, b]
            if np.dot(q0, q1) < 0:
                q1 = -q1
            dq = np.empty(4, np.float64)
            mujoco.mju_mulQuat(dq, q1.astype(np.float64), np.array([q0[0], -q0[1], -q0[2], -q0[3]], np.float64))
            angle = 2 * np.arctan2(np.linalg.norm(dq[1:]), max(dq[0], 1e-12))
            if angle > np.pi:
                angle -= 2 * np.pi
            norm = np.linalg.norm(dq[1:])
            if norm > 1e-10:
                body_ang_vel[k, b] = dq[1:] / norm * angle / dt
    body_ang_vel[0] = body_ang_vel[1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, fps=np.float32(fps), joint_pos=joint_pos.astype(np.float32),
                        joint_vel=joint_vel, body_pos_w=body_pos, body_quat_w=body_quat,
                        body_lin_vel_w=body_lin_vel, body_ang_vel_w=body_ang_vel,
                        body_names=np.asarray(BODIES))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--src-fps", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=50.0)
    args = parser.parse_args()
    files = sorted(args.input.glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSV motions found in {args.input}")
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    for source in files:
        target = args.output / f"{source.stem}.npz"
        convert(source, target, model, args.src_fps, args.fps)
        print(f"{source.name} -> {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
