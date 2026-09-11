"""Pinocchio FK/IK for Kassow KR-series arms (KR1018 URDF).

RIO / ``KassowArm`` pose6 is ``[x, y, z, rx, ry, rz]`` with **axis-angle / rotvec**
rotation (same as the rest of RIO teleop). KORD ``getTCP()`` / ``moveL`` use
**XYZ Euler** for the rotation part — convert at the KORD boundary with
``kord_tcp_to_pose6`` / ``pose6_to_kord_tcp``.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R

try:
    import pinocchio as pin
except ImportError as e:  # pragma: no cover - optional until robots extra installed
    pin = None
    _PIN_IMPORT_ERROR = e
else:
    _PIN_IMPORT_ERROR = None

NUM_JOINTS = 7
DEFAULT_EE_FRAME = "end_effector"
DEFAULT_URDF_PATH = str(
    files("rio_hw").joinpath("assets/kassow/kr2_robot_S00V0000M1018.urdf")
)

# FK vs measured TCP: warn above this, refuse to enable IK above that.
_FK_WARN_POS_M = 0.02
_FK_FAIL_POS_M = 0.08
_FK_WARN_ROT_RAD = 0.15
_FK_FAIL_ROT_RAD = 0.5


def kord_tcp_to_pose6(tcp: np.ndarray) -> np.ndarray:
    """KORD ``getTCP()`` XYZ-Euler → RIO pose6 (rotvec)."""
    tcp = np.asarray(tcp, dtype=np.float64).reshape(6)
    return np.concatenate([tcp[:3], R.from_euler("xyz", tcp[3:]).as_rotvec()])


def pose6_to_kord_tcp(pose6: np.ndarray) -> np.ndarray:
    """RIO pose6 (rotvec) → KORD ``moveL`` XYZ-Euler."""
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(6)
    return np.concatenate([pose6[:3], R.from_rotvec(pose6[3:]).as_euler("xyz")])


def pose6_to_se3(pose6: np.ndarray) -> "pin.SE3":
    """Convert RIO ``[xyz, rotvec]`` to a Pinocchio ``SE3``."""
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(6)
    return pin.SE3(R.from_rotvec(pose6[3:]).as_matrix(), pose6[:3])


def se3_to_pose6(oMf: "pin.SE3") -> np.ndarray:
    """Convert a Pinocchio ``SE3`` to RIO ``[xyz, rotvec]``."""
    return np.concatenate([oMf.translation, R.from_matrix(oMf.rotation).as_rotvec()])


class KassowKinematics:
    """7-DOF Kassow FK/IK backed by Pinocchio."""

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        ee_frame: str = DEFAULT_EE_FRAME,
    ):
        if pin is None:
            raise ImportError(
                "pinocchio (package 'pin') is required for Kassow local IK. "
                "Install with: uv sync --extra robots  # or pip install pin"
            ) from _PIN_IMPORT_ERROR

        self.urdf_path = str(urdf_path or DEFAULT_URDF_PATH)
        self.ee_frame = ee_frame
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        if self.model.nq != NUM_JOINTS or self.model.nv != NUM_JOINTS:
            raise ValueError(
                f"Expected {NUM_JOINTS} actuated DOF in {self.urdf_path}, "
                f"got nq={self.model.nq}, nv={self.model.nv}"
            )
        if not self.model.existFrame(ee_frame):
            raise ValueError(
                f"EE frame {ee_frame!r} not in URDF {self.urdf_path}. "
                f"Frames: {[self.model.frames[i].name for i in range(self.model.nframes)]}"
            )
        self.ee_id = self.model.getFrameId(ee_frame)
        logger.info(
            f"KassowKinematics: loaded {self.urdf_path} "
            f"(nq={self.model.nq}, ee={ee_frame})"
        )

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Forward kinematics: joint positions → TCP pose6."""
        q = np.asarray(q, dtype=np.float64).reshape(NUM_JOINTS)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return se3_to_pose6(self.data.oMf[self.ee_id])

    def twist_to_qd(
        self,
        q: np.ndarray,
        twist: np.ndarray,
        *,
        damping: float = 1e-2,
    ) -> np.ndarray:
        """Map a world-aligned spatial twist ``[v, ω]`` to joint velocities.

        Uses a damped least-squares inverse of the LOCAL_WORLD_ALIGNED frame
        Jacobian so linear/angular parts match base-frame teleop deltas.
        """
        q = np.asarray(q, dtype=np.float64).reshape(NUM_JOINTS)
        twist = np.asarray(twist, dtype=np.float64).reshape(6)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.computeFrameJacobian(
            self.model, self.data, q, self.ee_id, pin.LOCAL_WORLD_ALIGNED
        )
        JJt = J @ J.T
        return J.T @ np.linalg.solve(JJt + (damping * damping) * np.eye(6), twist)

    def ik(
        self,
        q0: np.ndarray,
        target_pose6: np.ndarray,
        *,
        max_iters: int = 60,
        damping: float = 1e-2,
        step: float = 0.5,
        tol_pos: float = 1e-4,
        tol_rot: float = 1e-3,
    ) -> np.ndarray:
        """Damped least-squares IK toward ``target_pose6``, seeded from ``q0``."""
        q = np.asarray(q0, dtype=np.float64).reshape(NUM_JOINTS).copy()
        oMdes = pose6_to_se3(target_pose6)
        for _ in range(max_iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            iMd = self.data.oMf[self.ee_id].actInv(oMdes)
            err = pin.log6(iMd).vector
            if float(np.linalg.norm(err[:3])) < tol_pos and float(np.linalg.norm(err[3:])) < tol_rot:
                break
            J = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_id, pin.LOCAL
            )
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + (damping * damping) * np.eye(6), err)
            q = pin.integrate(self.model, q, step * dq)
        return q

    def check_fk_alignment(self, q_meas: np.ndarray, tcp_meas: np.ndarray) -> tuple[float, float]:
        """Compare URDF FK to the controller TCP; warn / raise on large mismatch.

        Returns:
            ``(pos_err_m, rot_err_rad)``.
        """
        fk_pose = self.fk(q_meas)
        pose_meas = kord_tcp_to_pose6(tcp_meas)
        pos_err = float(np.linalg.norm(fk_pose[:3] - pose_meas[:3]))
        rot_err = float(
            (R.from_rotvec(fk_pose[3:]).inv() * R.from_rotvec(pose_meas[3:])).magnitude()
        )
        msg = (
            f"KassowKinematics: FK vs measured TCP "
            f"pos_err={pos_err * 1e3:.1f} mm, rot_err={rot_err * 1e3:.1f} mrad"
        )
        if pos_err >= _FK_FAIL_POS_M or rot_err >= _FK_FAIL_ROT_RAD:
            raise RuntimeError(
                f"{msg}. URDF/TCP frames look wrong — fix the model or tool offset "
                "before enabling task_pos_ik."
            )
        if pos_err >= _FK_WARN_POS_M or rot_err >= _FK_WARN_ROT_RAD:
            logger.warning(msg)
        else:
            logger.info(msg)
        return pos_err, rot_err
