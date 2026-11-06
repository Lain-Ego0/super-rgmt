#!/usr/bin/env python3
"""Split PMT motion NPZ sequences into deterministic, at-most-N-second clips.

Adjacent clips share one boundary sample so finite-difference velocity and pose
continuity are preserved. A balanced partition avoids tiny trailing clips.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def split_motion(path: Path, output_dir: Path, max_seconds: float) -> list[Path]:
    with np.load(path, allow_pickle=False) as source:
        if "fps" not in source.files or "joint_pos" not in source.files:
            raise ValueError(f"{path}: expected fps and joint_pos arrays")
        fps = float(np.asarray(source["fps"]).item())
        frame_count = int(source["joint_pos"].shape[0])
        if fps <= 0 or frame_count < 2 or max_seconds <= 0:
            raise ValueError(f"{path}: invalid fps, frame count, or max_seconds")
        max_intervals = max(1, int(np.floor(max_seconds * fps + 1e-9)))
        interval_count = max(1, int(np.ceil((frame_count - 1) / max_intervals)))
        # Balanced integer boundaries include both endpoints and share adjacent
        # endpoints. No interval exceeds the requested duration.
        boundaries = np.rint(np.linspace(0, frame_count - 1, interval_count + 1)).astype(int)
        if np.any(np.diff(boundaries) < 1) or np.any(np.diff(boundaries) > max_intervals):
            raise RuntimeError(f"{path}: failed to construct valid clip boundaries")
        payload = {name: source[name] for name in source.files}

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for clip_index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        target = output_dir / f"{path.stem}__clip_{clip_index:04d}.npz"
        clip_payload = {}
        for name, value in payload.items():
            if value.ndim > 0 and value.shape[0] == frame_count:
                clip_payload[name] = value[start : stop + 1]
            else:
                clip_payload[name] = value
        clip_payload["source_file"] = np.asarray(path.name)
        clip_payload["clip_index"] = np.asarray(clip_index, dtype=np.int32)
        clip_payload["source_frame_start"] = np.asarray(start, dtype=np.int64)
        clip_payload["source_frame_stop"] = np.asarray(stop, dtype=np.int64)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing clip: {target}")
        np.savez_compressed(target, **clip_payload)
        written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    args = parser.parse_args()
    sources = sorted(args.input_dir.rglob("*.npz"))
    if not sources:
        raise SystemExit(f"no NPZ motions found under {args.input_dir}")
    if args.input_dir.resolve() == args.output_dir.resolve() or args.input_dir.resolve() in args.output_dir.resolve().parents:
        raise SystemExit("output directory must not be inside the input directory")
    total = 0
    for source in sources:
        clips = split_motion(source, args.output_dir, args.max_seconds)
        total += len(clips)
        print(f"{source.name}: {len(clips)} clip(s)")
    print(f"Wrote {total} clip(s) to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
