from __future__ import annotations

import sys
from pathlib import Path

PMT_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "PMT-pinned"
sys.path.insert(0, str(PMT_ROOT))

from pmt_tasks.utils.motion_body_mapping import resolve_motion_body_indices


def test_named_motion_channels_are_mapped_by_name_not_articulation_index():
    indices = resolve_motion_body_indices(
        body_count=3,
        requested_body_names=["torso", "pelvis"],
        stored_body_names=["pelvis", "left_foot", "torso"],
        fallback_indices=[17, 0],
    )
    assert indices == [2, 0]


def test_legacy_compact_motion_channels_follow_requested_order():
    indices = resolve_motion_body_indices(
        body_count=2,
        requested_body_names=["pelvis", "torso"],
        stored_body_names=None,
        fallback_indices=[9, 2],
    )
    assert indices == [0, 1]


def test_legacy_motion_with_ambiguous_body_channels_fails_clearly():
    try:
        resolve_motion_body_indices(
            body_count=3,
            requested_body_names=["pelvis", "torso"],
            stored_body_names=None,
            fallback_indices=[9, 2],
        )
    except ValueError as exc:
        assert "cannot safely map motion channels" in str(exc)
    else:
        raise AssertionError("ambiguous name-less body channels must be rejected")
