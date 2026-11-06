#!/usr/bin/env python3
"""Audit PMT-compatible motion NPZ files and write a reproducible manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from scripts.convert_lafan1_to_pmt import ISAAC_JOINTS
except ModuleNotFoundError:  # direct `python scripts/audit_motion_dataset.py` invocation
    from convert_lafan1_to_pmt import ISAAC_JOINTS

PAPER_MINUTES = {"LAFAN1": 146.651, "AMASS": 30.677, "in_house_Xsens": 8.446}
REQUIRED = {
    "joint_pos": (29,),
    "joint_vel": (29,),
    "body_pos_w": (14, 3),
    "body_quat_w": (14, 4),
    "body_lin_vel_w": (14, 3),
    "body_ang_vel_w": (14, 3),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_clip(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as clip:
        missing = sorted(set(REQUIRED) - set(clip.files))
        if missing or "fps" not in clip.files:
            raise ValueError(f"{path}: missing arrays {missing + ([] if 'fps' in clip.files else ['fps'])}")
        if "quat_order" not in clip.files or str(np.asarray(clip["quat_order"]).item()) != "xyzw":
            raise ValueError(f"{path}: body_quat_w must declare quat_order=xyzw")
        if "joint_names" not in clip.files or clip["joint_names"].astype(str).tolist() != ISAAC_JOINTS:
            raise ValueError(f"{path}: joint_names do not match the pinned Isaac G1 articulation order")
        fps = float(np.asarray(clip["fps"]).item())
        arrays: dict[str, list[int]] = {}
        frame_counts: set[int] = set()
        for name, trailing_shape in REQUIRED.items():
            array = clip[name]
            expected_ndim = len(trailing_shape) + 1
            if array.ndim != expected_ndim or array.shape[1:] != trailing_shape:
                raise ValueError(f"{path}: {name} shape {array.shape}; expected [T,{','.join(map(str, trailing_shape))}]")
            if not np.isfinite(array).all():
                raise ValueError(f"{path}: {name} contains non-finite values")
            arrays[name] = list(array.shape)
            frame_counts.add(int(array.shape[0]))
        if len(frame_counts) != 1:
            raise ValueError(f"{path}: frame counts differ across arrays: {sorted(frame_counts)}")
        frames = frame_counts.pop()
        if fps <= 0 or frames < 2:
            raise ValueError(f"{path}: invalid fps={fps} or frame count={frames}")
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "frames": frames,
        "fps": fps,
        "quat_order": "xyzw",
        "joint_names": ISAAC_JOINTS,
        "duration_seconds": (frames - 1) / fps,
        "arrays": arrays,
    }


def build_manifest(root: Path, present_sources: set[str] | None = None) -> dict[str, object]:
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise ValueError(f"no .npz clips found under {root}")
    clips = [inspect_clip(path) for path in files]
    total_seconds = sum(float(clip["duration_seconds"]) for clip in clips)
    return {
        "source_directory": str(root.resolve()),
        "clip_count": len(clips),
        "total_frames": sum(int(clip["frames"]) for clip in clips),
        "duration_seconds": total_seconds,
        "duration_hours": total_seconds / 3600,
        "paper_duration_minutes": PAPER_MINUTES,
        "present_sources": sorted(present_sources) if present_sources is not None else None,
        "missing_paper_sources": sorted(set(PAPER_MINUTES) - present_sources) if present_sources is not None else None,
        "clips": clips,
        "note": "File hashes identify this local corpus; they do not establish equivalence to the paper's clip selection.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--present-source", action="append", choices=sorted(PAPER_MINUTES), default=None)
    args = parser.parse_args()
    present_sources = set(args.present_source) if args.present_source is not None else None
    manifest = build_manifest(args.motion_dir, present_sources=present_sources)
    rendered = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
