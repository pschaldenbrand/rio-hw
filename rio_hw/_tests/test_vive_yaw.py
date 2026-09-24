"""Yaw alignment between the SteamVR world frame and the robot base."""

import math

import numpy as np
import pytest

from rio_hw.interfaces.vive_retarget import (
    AXIS_MAP_STEAMVR_TO_ROBOT,
    ClutchRetargeter,
    dominant_direction,
    horizontal_angle_between,
    yaw_from_motions,
    yaw_rotation,
)


def sweep(direction, travel=0.4, n=60, noise=0.0, seed=0):
    """Points along a straight hand sweep, optionally with tremor."""
    rng = np.random.default_rng(seed)
    ts = np.linspace(0.0, travel, n)
    points = np.outer(ts, np.asarray(direction, dtype=float))
    if noise:
        points = points + noise * rng.standard_normal(points.shape)
    return points


def test_yaw_rotation_is_a_rotation_about_vertical():
    rotation = yaw_rotation(0.7)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    # The vertical axis is the one gravity already fixed, so it must not move.
    np.testing.assert_allclose(rotation @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-12)


def test_dominant_direction_recovers_a_clean_sweep():
    direction = np.array([0.6, -0.8, 0.0])
    np.testing.assert_allclose(dominant_direction(sweep(direction)), direction, atol=1e-9)


def test_dominant_direction_is_signed_by_net_travel():
    """A principal component is sign-free; net travel decides which way it points."""
    forward = sweep([1.0, 0.0, 0.0])
    np.testing.assert_allclose(dominant_direction(forward[::-1]), [-1.0, 0.0, 0.0], atol=1e-9)


def test_dominant_direction_survives_tremor():
    direction = np.array([1.0, 0.0, 0.0])
    estimated = dominant_direction(sweep(direction, travel=0.4, noise=0.005, seed=3))
    assert math.degrees(horizontal_angle_between(estimated, direction)) < 3.0


def test_dominant_direction_needs_two_points():
    with pytest.raises(ValueError, match="at least two samples"):
        dominant_direction(np.zeros((1, 3)))


@pytest.mark.parametrize("truth_deg", [0.0, 12.0, -35.0, 90.0, 179.0, -179.0])
def test_yaw_is_recovered_from_two_sweeps(truth_deg):
    """Rotate the intended axes by a known yaw and check we solve for it."""
    truth = math.radians(truth_deg)
    intended = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    # What the operator's sweeps look like once mismeasured by `truth`.
    measured = [yaw_rotation(-truth) @ axis for axis in intended]

    solved = yaw_from_motions(measured, intended)
    # Compare on the circle so +179 and -181 are not treated as far apart.
    assert math.degrees(abs(math.atan2(math.sin(solved - truth), math.cos(solved - truth)))) < 1e-6


def test_yaw_averages_disagreeing_sweeps():
    """Two sweeps that disagree should land between them, not on either one."""
    intended = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    measured = [
        yaw_rotation(math.radians(-20.0)) @ intended[0],
        yaw_rotation(math.radians(-30.0)) @ intended[1],
    ]
    assert 20.0 < math.degrees(yaw_from_motions(measured, intended)) < 30.0


def test_yaw_ignores_the_vertical_component():
    """A sweep that drifts up or down must not change the answer."""
    intended = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    flat = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    tilted = [np.array([1.0, 0.0, 0.5]), np.array([0.0, 1.0, -0.3])]
    assert yaw_from_motions(tilted, intended) == pytest.approx(yaw_from_motions(flat, intended))


def test_yaw_of_degenerate_input_is_zero():
    intended = [np.array([1.0, 0.0, 0.0])]
    assert yaw_from_motions([np.array([0.0, 0.0, 1.0])], intended) == 0.0


def test_horizontal_angle_between_perpendicular_axes():
    angle = horizontal_angle_between(np.array([1.0, 0.0, 9.0]), np.array([0.0, 1.0, -4.0]))
    assert math.degrees(angle) == pytest.approx(90.0)


def test_calibrated_axis_map_sends_hand_motion_the_intended_way():
    """End to end: a mis-yawed setup, calibrated, moves the TCP where asked."""
    truth = math.radians(25.0)

    # The robot frame this operator's SteamVR install actually produces.
    true_map = yaw_rotation(truth) @ AXIS_MAP_STEAMVR_TO_ROBOT

    def hand_motion_for(robot_direction):
        """Tracker-frame motion that should drive the TCP along robot_direction."""
        return np.linalg.inv(true_map) @ robot_direction

    intended = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    measured = [
        AXIS_MAP_STEAMVR_TO_ROBOT @ dominant_direction(sweep(hand_motion_for(axis)))
        for axis in intended
    ]
    solved = yaw_from_motions(measured, intended)
    assert math.degrees(abs(solved - truth)) < 1e-6

    # Drive the retargeter with the calibrated map and confirm the TCP tracks +X.
    retargeter = ClutchRetargeter(
        axis_map=yaw_rotation(solved) @ AXIS_MAP_STEAMVR_TO_ROBOT,
        max_pos_speed=10.0,
        pos_scale=1.0,
    )
    start = np.eye(4)
    start[:3, 3] = [0.0, 1.0, 0.0]
    retargeter.engage(start, np.array([0.4, 0.0, 0.3]), np.eye(3))

    moved = np.eye(4)
    moved[:3, 3] = start[:3, 3] + 0.1 * hand_motion_for(np.array([1.0, 0.0, 0.0]))
    position, _ = retargeter.target(moved, dt=1.0)

    step = position - np.array([0.4, 0.0, 0.3])
    assert step[0] == pytest.approx(0.1, abs=1e-6)
    assert abs(step[1]) < 1e-6


def test_uncalibrated_setup_is_visibly_off():
    """Guards the test above: without the fix, the same motion goes astray."""
    truth = math.radians(25.0)
    true_map = yaw_rotation(truth) @ AXIS_MAP_STEAMVR_TO_ROBOT
    hand = np.linalg.inv(true_map) @ np.array([1.0, 0.0, 0.0])

    retargeter = ClutchRetargeter(max_pos_speed=10.0, pos_scale=1.0)
    start = np.eye(4)
    start[:3, 3] = [0.0, 1.0, 0.0]
    retargeter.engage(start, np.array([0.4, 0.0, 0.3]), np.eye(3))

    moved = np.eye(4)
    moved[:3, 3] = start[:3, 3] + 0.1 * hand
    position, _ = retargeter.target(moved, dt=1.0)

    step = position - np.array([0.4, 0.0, 0.3])
    assert math.degrees(horizontal_angle_between(step, np.array([1.0, 0.0, 0.0]))) == pytest.approx(
        25.0, abs=0.5
    )
