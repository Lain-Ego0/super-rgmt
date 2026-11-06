import numpy as np

from scripts.split_motion_clips import split_motion


def make_motion(path, frames=1001, fps=50.0):
    np.savez_compressed(
        path,
        fps=np.float32(fps),
        joint_pos=np.arange(frames * 29, dtype=np.float32).reshape(frames, 29),
        joint_vel=np.zeros((frames, 29), dtype=np.float32),
        body_names=np.asarray([f"body_{i}" for i in range(14)]),
        constant=np.asarray(17, dtype=np.int32),
    )


def test_split_motion_keeps_endpoints_and_bounds_duration(tmp_path):
    source = tmp_path / "source.npz"
    output = tmp_path / "clips"
    make_motion(source, frames=1001)

    clips = split_motion(source, output, max_seconds=10.0)

    assert len(clips) == 2
    with np.load(clips[0], allow_pickle=False) as first, np.load(clips[1], allow_pickle=False) as second:
        assert first["joint_pos"].shape == (501, 29)
        assert second["joint_pos"].shape == (501, 29)
        np.testing.assert_array_equal(first["joint_pos"][-1], second["joint_pos"][0])
        assert (first["joint_pos"].shape[0] - 1) / float(first["fps"]) <= 10
        assert (second["joint_pos"].shape[0] - 1) / float(second["fps"]) <= 10
        assert first["constant"].item() == 17
        assert first["source_file"].item() == source.name
        assert second["source_frame_start"].item() == first["source_frame_stop"].item()


def test_split_short_motion_is_retained_as_one_clip(tmp_path):
    source = tmp_path / "short.npz"
    make_motion(source, frames=120)

    clips = split_motion(source, tmp_path / "short-clips", max_seconds=10)

    assert len(clips) == 1
    with np.load(clips[0], allow_pickle=False) as clip:
        assert clip["joint_pos"].shape == (120, 29)
