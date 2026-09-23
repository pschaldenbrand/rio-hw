"""Clutch retargeting math for the Vive tracker. Pure numpy, no hardware."""

import math

import numpy as np
import pytest

from rio_hw.interfaces.vive_retarget import (
    AXIS_MAP_STEAMVR_TO_ROBOT,
    ClutchRetargeter,
    WorkspaceLimits,
    axis_angle_to_rotation,
    matrix_to_pose,
    matrix_to_rotvec,
    orthonormalize,
    pose_to_matrix,
    rotation_to_axis_angle,
    rotvec_to_matrix,
)


def tracker_pose(position, rotation=None):
    pose = np.eye(4)
    pose[:3, 3] = position
    if rotation is not None:
        pose[:3, :3] = rotation
    return pose


@pytest.mark.parametrize(
    "rotvec",
    [
        [0.0, 0.0, 0.0],
        [0.1, -0.2, 0.3],
        [1e-10, 0.0, 0.0],
        [0.0, 0.0, math.pi - 1e-9],
        [0.0, math.pi, 0.0],
        [1.2, -0.4, 2.0],
    ],
)
def test_rotvec_matrix_roundtrip(rotvec):
    rotation = rotvec_to_matrix(rotvec)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    # Rotations are 2-pi periodic, so compare matrices rather than vectors.
    np.testing.assert_allclose(rotvec_to_matrix(matrix_to_rotvec(rotation)), rotation, atol=1e-9)


def test_small_rotation_is_finite():
    """atan2 keeps tiny rotations from producing a NaN axis (0/0)."""
    axis, angle = rotation_to_axis_angle(rotvec_to_matrix([1e-12, 0.0, 0.0]))
    assert np.isfinite(axis).all()
    assert math.isfinite(angle)
    assert angle == pytest.approx(0.0, abs=1e-9)


def test_pose_vector_roundtrip():
    pose = np.array([0.4, -0.1, 0.35, 0.2, -1.3, 0.7])
    np.testing.assert_allclose(matrix_to_pose(pose_to_matrix(pose)), pose, atol=1e-9)


def test_orthonormalize_projects_onto_so3():
    noisy = rotvec_to_matrix([0.3, 0.2, -0.1]) + 1e-3 * np.ones((3, 3))
    fixed = orthonormalize(noisy)
    np.testing.assert_allclose(fixed @ fixed.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(fixed) == pytest.approx(1.0)


def test_workspace_clamps_radius_and_height():
    limits = WorkspaceLimits(min_radius=0.2, max_radius=0.75, min_z=0.05)

    far = limits.clamp(np.array([2.0, 0.0, 0.5]))
    assert np.linalg.norm(far) <= 0.75 + 1e-9

    near = limits.clamp(np.array([0.01, 0.0, 0.1]))
    assert np.linalg.norm(near) >= 0.2 - 1e-9

    low = limits.clamp(np.array([0.4, 0.0, -1.0]))
    assert low[2] >= 0.05 - 1e-9


def test_engage_produces_no_jump():
    """The first target after engaging must equal the TCP it snapshotted."""
    retargeter = ClutchRetargeter()
    tcp_pos = np.array([0.4, 0.1, 0.3])
    tcp_rot = rotvec_to_matrix([0.0, math.pi, 0.0])

    retargeter.engage(tracker_pose([0.0, 1.0, 0.0]), tcp_pos, tcp_rot)
    position, rotation = retargeter.target(tracker_pose([0.0, 1.0, 0.0]), dt=0.02)

    np.testing.assert_allclose(position, tcp_pos, atol=1e-9)
    np.testing.assert_allclose(rotation, tcp_rot, atol=1e-9)


def test_tracker_translation_maps_through_axis_map():
    """Tracker motion is expressed in the robot frame via the axis map."""
    retargeter = ClutchRetargeter(max_pos_speed=10.0)
    tcp_pos = np.zeros(3)
    retargeter.engage(tracker_pose([0.0, 1.0, 0.0]), tcp_pos, np.eye(3))

    delta = np.array([0.1, 0.05, -0.02])
    position, _ = retargeter.target(tracker_pose(np.array([0.0, 1.0, 0.0]) + delta), dt=1.0)

    expected = retargeter.workspace.clamp(AXIS_MAP_STEAMVR_TO_ROBOT @ delta)
    np.testing.assert_allclose(position, expected, atol=1e-9)


def test_position_step_is_speed_limited():
    retargeter = ClutchRetargeter(max_pos_speed=0.1)
    start = np.array([0.4, 0.0, 0.3])
    retargeter.engage(tracker_pose([0.0, 1.0, 0.0]), start, np.eye(3))

    # A one meter tracker jump must not move the TCP more than max_pos_speed * dt.
    position, _ = retargeter.target(tracker_pose([1.0, 1.0, 0.0]), dt=0.02)
    assert np.linalg.norm(position - start) <= 0.1 * 0.02 + 1e-9


def test_rotation_step_is_speed_limited():
    retargeter = ClutchRetargeter(max_rot_speed=0.5)
    retargeter.engage(tracker_pose([0.0, 1.0, 0.0]), np.array([0.4, 0.0, 0.3]), np.eye(3))

    spun = tracker_pose([0.0, 1.0, 0.0], axis_angle_to_rotation(np.array([0.0, 1.0, 0.0]), 3.0))
    _, rotation = retargeter.target(spun, dt=0.02)
    _, angle = rotation_to_axis_angle(rotation)
    assert angle <= 0.5 * 0.02 + 1e-9


def test_disengaged_target_holds_last_pose():
    retargeter = ClutchRetargeter()
    hold_pos = np.array([0.5, 0.0, 0.2])
    retargeter.hold(hold_pos, np.eye(3))

    position, rotation = retargeter.target(tracker_pose([5.0, 5.0, 5.0]), dt=0.02)
    np.testing.assert_allclose(position, hold_pos)
    np.testing.assert_allclose(rotation, np.eye(3))


def test_orientation_disabled_holds_tool_rotation():
    retargeter = ClutchRetargeter(orientation_enabled=False, max_rot_speed=10.0)
    tcp_rot = rotvec_to_matrix([0.0, math.pi, 0.0])
    retargeter.engage(tracker_pose([0.0, 1.0, 0.0]), np.array([0.4, 0.0, 0.3]), tcp_rot)

    spun = tracker_pose([0.0, 1.0, 0.0], axis_angle_to_rotation(np.array([1.0, 0.0, 0.0]), 1.0))
    _, rotation = retargeter.target(spun, dt=0.02)
    np.testing.assert_allclose(rotation, tcp_rot, atol=1e-9)


def test_pose_vector_api_matches_matrix_api():
    retargeter = ClutchRetargeter()
    pose = np.array([0.4, 0.1, 0.3, 0.0, math.pi, 0.0])
    retargeter.hold_pose(pose)
    np.testing.assert_allclose(matrix_to_pose(pose_to_matrix(pose)), pose, atol=1e-9)

    retargeter.engage(tracker_pose([0.0, 1.0, 0.0]), pose[:3], pose_to_matrix(pose)[:3, :3])
    np.testing.assert_allclose(
        retargeter.target_pose(tracker_pose([0.0, 1.0, 0.0]), dt=0.02), pose, atol=1e-9
    )


def test_long_trajectory_stays_finite():
    """Guards the 0/0 axis-normalization that used to surface as NaN."""
    rng = np.random.default_rng(0)
    retargeter = ClutchRetargeter()
    retargeter.engage(tracker_pose([0.0, 1.0, 0.0]), np.array([0.4, 0.0, 0.3]), np.eye(3))

    for step in range(2000):
        # Includes long stretches of near-identity motion.
        wobble = 1e-9 if step % 2 else 1e-3
        pose = tracker_pose(
            np.array([0.0, 1.0, 0.0]) + wobble * rng.standard_normal(3),
            axis_angle_to_rotation(rng.standard_normal(3), wobble),
        )
        position, rotation = retargeter.target(pose, dt=0.008)
        assert np.isfinite(position).all()
        assert np.isfinite(rotation).all()
