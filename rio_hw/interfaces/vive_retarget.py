"""Clutch-based retargeting from a SteamVR tracker pose to a robot TCP target.

Pure numpy, so the mapping can be exercised without SteamVR or hardware.

Alignment comes from clutching rather than calibration: engaging snapshots the
tracker pose and the current TCP pose together, and tracker motion is then
applied relative to that pair.
"""

import math
from dataclasses import dataclass, field

import numpy as np

# SteamVR standing space is Y-up with -Z pointing away from the user; robot base
# frames are Z-up with +X forward. Rows map robot axes onto SteamVR axes.
AXIS_MAP_STEAMVR_TO_ROBOT = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)


def rotation_to_axis_angle(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the (unit axis, angle) of a 3x3 rotation matrix.

    The angle comes from atan2 of the skew-symmetric part rather than from the
    trace. Near zero rotation the trace only carries the angle as a second
    order term, which underflows to noise for angles below ~1e-8 rad and would
    otherwise pair a bogus angle with a zero-length axis.
    """
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    sin_term = float(np.linalg.norm(skew)) / 2.0
    cos_term = (float(np.trace(rotation)) - 1.0) / 2.0

    if sin_term > 1e-8:
        return skew / (2.0 * sin_term), math.atan2(sin_term, cos_term)
    if cos_term > 0.0:
        return np.array([0.0, 0.0, 1.0]), 0.0

    # Near 180 deg the skew-symmetric part vanishes, so recover the axis from
    # the dominant column of R + I.
    shifted = rotation + np.eye(3)
    axis = shifted[:, int(np.argmax(np.diag(shifted)))]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    return axis / norm, math.pi


def axis_angle_to_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' formula for a 3x3 rotation matrix."""
    norm = float(np.linalg.norm(axis))
    if abs(angle) < 1e-9 or norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def scale_rotation(rotation: np.ndarray, scale: float) -> np.ndarray:
    """Scale a rotation by shrinking or growing its angle about the same axis."""
    axis, angle = rotation_to_axis_angle(rotation)
    return axis_angle_to_rotation(axis, angle * scale)


def clamp_rotation(rotation: np.ndarray, max_angle: float) -> np.ndarray:
    """Limit a rotation to at most max_angle radians about its own axis."""
    axis, angle = rotation_to_axis_angle(rotation)
    if angle <= max_angle:
        return rotation
    return axis_angle_to_rotation(axis, max_angle)


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Convert a rotation vector (axis scaled by angle) to a 3x3 rotation."""
    rotvec = np.asarray(rotvec, dtype=float)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    return axis_angle_to_rotation(rotvec / angle, angle)


def matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation to a rotation vector, as eef_pose uses."""
    axis, angle = rotation_to_axis_angle(rotation)
    return axis * angle


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert a 6-vector [x, y, z, rx, ry, rz] to a 4x4 homogeneous matrix."""
    pose = np.asarray(pose, dtype=float)
    matrix = np.eye(4)
    matrix[:3, 3] = pose[:3]
    matrix[:3, :3] = rotvec_to_matrix(pose[3:])
    return matrix


def matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    """Convert a 4x4 homogeneous matrix to a 6-vector [x, y, z, rx, ry, rz]."""
    matrix = np.asarray(matrix, dtype=float)
    return np.concatenate([matrix[:3, 3], matrix_to_rotvec(matrix[:3, :3])])


def yaw_rotation(yaw: float) -> np.ndarray:
    """Rotation of `yaw` radians about the robot's vertical axis."""
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cos_y, -sin_y, 0.0],
            [sin_y, cos_y, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def dominant_direction(points: np.ndarray) -> np.ndarray:
    """Unit direction of a recorded path, as its first principal component.

    Differencing only the endpoints would put all of the operator's start and
    stop jitter into the answer, so the whole path votes instead. The sign is
    resolved by the net travel, which a principal component cannot supply.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise ValueError("need at least two samples to estimate a direction")
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    if float(np.dot(points[-1] - points[0], direction)) < 0.0:
        direction = -direction
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("path has no dominant direction")
    return direction / norm


def horizontal_angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two vectors after dropping their vertical components."""
    a = np.asarray(a, dtype=float)[:2]
    b = np.asarray(b, dtype=float)[:2]
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        return 0.0
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.acos(max(-1.0, min(1.0, cosine)))


def yaw_from_motions(measured: list[np.ndarray], intended: list[np.ndarray]) -> float:
    """Least-squares yaw about vertical taking `measured` directions onto `intended`.

    Both are sequences of vectors already in the robot frame; vertical
    components are dropped because gravity has pinned that axis. Summing the
    sine and cosine terms before the atan2 is the circular mean, which averages
    headings correctly where averaging raw angles would not.
    """
    sin_sum = 0.0
    cos_sum = 0.0
    for measured_dir, intended_dir in zip(measured, intended, strict=True):
        m = np.asarray(measured_dir, dtype=float)
        t = np.asarray(intended_dir, dtype=float)
        m_norm = float(np.linalg.norm(m[:2]))
        t_norm = float(np.linalg.norm(t[:2]))
        if m_norm < 1e-9 or t_norm < 1e-9:
            continue
        mx, my = m[:2] / m_norm
        tx, ty = t[:2] / t_norm
        sin_sum += mx * ty - my * tx
        cos_sum += mx * tx + my * ty
    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        return 0.0
    return math.atan2(sin_sum, cos_sum)


def orthonormalize(rotation: np.ndarray) -> np.ndarray:
    """Project a near-rotation onto SO(3) so drift cannot accumulate."""
    u, _, vt = np.linalg.svd(rotation)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


@dataclass
class WorkspaceLimits:
    """Keeps TCP targets inside a reachable shell around the robot base."""

    min_radius: float = 0.20
    max_radius: float = 0.75
    min_z: float = 0.05

    def clamp(self, position: np.ndarray) -> np.ndarray:
        clamped = position.copy()
        clamped[2] = max(clamped[2], self.min_z)
        radius = float(np.linalg.norm(clamped))
        if radius < 1e-6:
            return np.array([self.min_radius, 0.0, max(self.min_z, clamped[2])])
        if radius > self.max_radius:
            clamped = clamped * (self.max_radius / radius)
            clamped[2] = max(clamped[2], self.min_z)
        elif radius < self.min_radius:
            clamped = clamped * (self.min_radius / radius)
        return clamped


@dataclass
class ClutchRetargeter:
    """Maps tracker motion since engage onto a TCP target pose.

    Position and orientation are mapped independently: composing a single
    homogeneous delta would rotate the TCP position about the robot base and
    make wrist twists swing the whole arm.
    """

    pos_scale: float = 1.0
    rot_scale: float = 1.0
    max_pos_speed: float = 0.25
    max_rot_speed: float = 0.6
    axis_map: np.ndarray = field(default_factory=AXIS_MAP_STEAMVR_TO_ROBOT.copy)
    workspace: WorkspaceLimits = field(default_factory=WorkspaceLimits)
    orientation_enabled: bool = True

    engaged: bool = field(default=False, init=False)
    _tracker_pos_0: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
    _tracker_rot_0: np.ndarray = field(default_factory=lambda: np.eye(3), init=False)
    _tcp_pos_0: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
    _tcp_rot_0: np.ndarray = field(default_factory=lambda: np.eye(3), init=False)
    _last_pos: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
    _last_rot: np.ndarray = field(default_factory=lambda: np.eye(3), init=False)

    def engage(
        self,
        tracker_pose: np.ndarray,
        tcp_position: np.ndarray,
        tcp_rotation: np.ndarray,
    ) -> None:
        """Snapshot the tracker and TCP poses that define the correspondence."""
        self._tracker_pos_0 = np.asarray(tracker_pose)[:3, 3].copy()
        self._tracker_rot_0 = orthonormalize(np.asarray(tracker_pose)[:3, :3])
        self._tcp_pos_0 = np.asarray(tcp_position, dtype=float).copy()
        self._tcp_rot_0 = orthonormalize(np.asarray(tcp_rotation, dtype=float))
        self._last_pos = self._tcp_pos_0.copy()
        self._last_rot = self._tcp_rot_0.copy()
        self.engaged = True

    def disengage(self) -> None:
        self.engaged = False

    def target(self, tracker_pose: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the (position, 3x3 rotation) TCP target for the current tracker pose."""
        if not self.engaged:
            return self._last_pos.copy(), self._last_rot.copy()

        tracker_pose = np.asarray(tracker_pose, dtype=float)
        tracker_pos = tracker_pose[:3, 3]
        tracker_rot = orthonormalize(tracker_pose[:3, :3])

        delta_pos = self.axis_map @ (tracker_pos - self._tracker_pos_0)
        goal_pos = self._tcp_pos_0 + self.pos_scale * delta_pos

        if self.orientation_enabled:
            delta_rot = tracker_rot @ self._tracker_rot_0.T
            delta_rot = self.axis_map @ delta_rot @ self.axis_map.T
            goal_rot = scale_rotation(delta_rot, self.rot_scale) @ self._tcp_rot_0
        else:
            goal_rot = self._tcp_rot_0.copy()

        goal_pos = self.workspace.clamp(goal_pos)

        step = goal_pos - self._last_pos
        max_step = self.max_pos_speed * dt
        step_norm = float(np.linalg.norm(step))
        if step_norm > max_step > 0.0:
            step = step * (max_step / step_norm)
        position = self._last_pos + step

        rot_step = clamp_rotation(goal_rot @ self._last_rot.T, self.max_rot_speed * dt)
        rotation = orthonormalize(rot_step @ self._last_rot)

        self._last_pos = position
        self._last_rot = rotation
        return position.copy(), rotation.copy()

    def target_pose(self, tracker_pose: np.ndarray, dt: float) -> np.ndarray:
        """Return the TCP target as a 6-vector [x, y, z, rx, ry, rz]."""
        position, rotation = self.target(tracker_pose, dt)
        return np.concatenate([position, matrix_to_rotvec(rotation)])

    def hold(self, position: np.ndarray, rotation: np.ndarray) -> None:
        """Set the pose the retargeter holds while disengaged."""
        self._last_pos = np.asarray(position, dtype=float).copy()
        self._last_rot = orthonormalize(np.asarray(rotation, dtype=float))

    def hold_pose(self, pose: np.ndarray) -> None:
        """Hold a TCP pose given as a 6-vector [x, y, z, rx, ry, rz]."""
        pose = np.asarray(pose, dtype=float)
        self.hold(pose[:3], rotvec_to_matrix(pose[3:]))
