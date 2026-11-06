import numpy as np

from scripts.audit_motion_dataset import build_manifest
from scripts.convert_lafan1_to_pmt import ISAAC_JOINTS


def test_dataset_audit_records_shapes_hash_and_duration(tmp_path):
    arrays = {
        "fps": np.float32(50),
        "quat_order": np.asarray("xyzw"),
        "joint_names": np.asarray(ISAAC_JOINTS),
        "joint_pos": np.zeros((3, 29), np.float32),
        "joint_vel": np.zeros((3, 29), np.float32),
        "body_pos_w": np.zeros((3, 14, 3), np.float32),
        "body_quat_w": np.tile(np.array([0, 0, 0, 1], np.float32), (3, 14, 1)),
        "body_lin_vel_w": np.zeros((3, 14, 3), np.float32),
        "body_ang_vel_w": np.zeros((3, 14, 3), np.float32),
    }
    np.savez(tmp_path / "clip.npz", **arrays)

    manifest = build_manifest(tmp_path, present_sources={"LAFAN1"})

    assert manifest["clip_count"] == 1
    assert manifest["total_frames"] == 3
    assert manifest["duration_seconds"] == 2 / 50
    assert len(manifest["clips"][0]["sha256"]) == 64
    assert manifest["missing_paper_sources"] == ["AMASS", "in_house_Xsens"]
