import queue
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R

from .. import time
from ..filters import LowPassFilter
from ..middleware import ClientFactory, ServerFactory
from ..node import Node
from ..request import Request
from .kassow_kinematics import (
    DEFAULT_EE_FRAME,
    DEFAULT_URDF_PATH,
    KassowKinematics,
    kord_tcp_to_pose6,
    pose6_to_kord_tcp,
)

try:
    import _kord_bridge
except ImportError as e:
    if TYPE_CHECKING:
        raise e
    else:
        _kord_bridge = None

BUILD_HINT = (
    "_kord_bridge extension not found. Build it with:\n"
    "  cmake -S kord_bridge -B kord_bridge/build -DCMAKE_BUILD_TYPE=Release\n"
    "  cmake --build kord_bridge/build --target _kord_bridge -j$(nproc)\n"
    "  cmake --install kord_bridge/build --component kord_bridge"
)

NUM_JOINTS = 7

# Waypoints this close to the previous one mean teleop is idle. Re-sending an
# unchanged via-point only adds KORD traffic, which has dropped sessions before.
IDLE_EPS = 1e-6

DIAGNOSTICS_INTERVAL = 2.0  # seconds between sync/send-rate reports

# systemAlarmState() packs category in bits 0-3, context in 4-7, condition ID in
# 8-19 and severity in 20-23. Condition IDs are listed per category in kord-api
# docs/guides/safety/handle_alarms.rst.
_ALARM_CATEGORY = {1: "SafetyEvent", 2: "SoftStopEvent", 3: "HwStat", 4: "CBunEvent"}
_ALARM_CONTEXT = {0x01: "ESTOP", 0x02: "PSTOP", 0x04: "SSTOP", 0x08: "SYSERR"}
_ALARM_SEVERITY = {0: "recoverable", 1: "latched", 2: "critical"}
_ALARM_CONDITION = {
    1: {
        1001: "EXTERNAL_ESTOP_ACTIVATED",
        1002: "IOB_NOT_RESPONDING",
        1003: "JBS_NOT_RESPONDING",
        1004: "JREF_X_SENSOR_POSITION_SPAN",
        1005: "JREF_POSITION_DELTA_SPAN",
        1006: "JRATED_SPEED_EXCEEDED",
        1007: "JRATED_TORQUE_EXCEEDED",
        1008: "JHOLD_TORQUE_EXCEEDED",
        1009: "JBRATED_TEMP_EXCEEDED",
        1010: "JTORQUE_DEVIATION_EXCEEDED",
        1011: "MODEL_X_TRJ_REFJ_SSPAN_EXC",
        1012: "MODEL_X_TRJ_REFW_SSPAN_EXC",
        1013: "FRAME_SPEED_LIMIT_EXC",
        2001: "EXTERNAL_PSTOP_ACTIVATED",
    },
    2: {
        2001: "MODEL_INVALID_STATE",
        2002: "MODEL_JVELOCITY_LIMITS_EXC",
        2003: "MODEL_JTORQUE_LIMITS_EXC",
        2004: "MODEL_JSDTORQUE_LIMITS_EXC",
        2005: "MODEL_JPOS_LIMITS_VIOLATION_EST",
        3001: "CBUN_KORD_BAD_CONN_QUALITY",
        3002: "INFEASIBLE_MOVE_COMMAND",
        3003: "CBUN_KORD_COMM_ERROR",
        # CBun ≥3.0.4; may also report under CBunEvent (category 4).
        3004: "MOVE_IN_INVALID_MOTION_STATE",
    },
    4: {
        3001: "CBUN_KORD_BAD_CONN_QUALITY",
        3002: "INFEASIBLE_MOVE_COMMAND",
        3003: "CBUN_KORD_COMM_ERROR",
        3004: "MOVE_IN_INVALID_MOTION_STATE",
    },
}


def describe_alarm(code: int) -> str:
    """Render a systemAlarmState() word as a readable condition.

    Args:
        code: Raw alarm word as published in the arm state.

    Returns:
        A string naming the category, condition, active contexts and severity.
    """
    category, context = code & 0xF, (code >> 4) & 0xF
    condition, severity = (code >> 8) & 0xFFF, (code >> 20) & 0xF
    flags = [name for bit, name in _ALARM_CONTEXT.items() if context & bit]
    return (
        f"0x{code:08x} "
        f"{_ALARM_CATEGORY.get(category, f'category {category}')}/"
        f"{_ALARM_CONDITION.get(category, {}).get(condition, f'condition {condition}')} "
        f"[{'|'.join(flags) or 'no context'}, "
        f"{_ALARM_SEVERITY.get(severity, f'severity {severity}')}]"
    )


class RobotController(Enum):
    TASK_POS = auto()
    TASK_POS_IK = auto()  # Cartesian goals → Pinocchio IK → VelCmd / directJControl
    JOINT_POS = auto()
    JOINT_VEL = auto()  # VelCmd → directJControl every waitSync


class RequestType(Enum):
    MOVEL = auto()
    MOVEJ = auto()
    SPEEDJ = auto()


# Joint tracking deadband for task_pos_ik (rad); below this, send zero qd.
_IK_HOLD_EPS = 1e-3
# Cap ||q_ik - q|| before applying ik_kp so a bad IK step cannot command a lurch.
_IK_MAX_Q_ERR = 0.06  # rad
# Hands-off: only force zero VelCmd after teleop stops (not between 100 Hz packets).
_IK_IDLE_HOLD_S = 0.05
# Below this angular error (rad), treat orientation as reached. Keep small so
# translation-only teleop corrects Jacobian coupling before it accumulates.
_IK_ORIENT_HOLD = 0.005


class KassowArm(Node):
    """Kassow KR-series 7-DOF arm, driven over KORD by a C++ real-time bridge.

    The `_kord_bridge` extension runs kord-api's `waitSync()` loop in a dedicated
    C++ thread at 250 Hz.

    Controllers:
      - `task_pos` — Python Cartesian goals → streamed `moveL` micro-steps
      - `task_pos_ik` — Cartesian goals → local Pinocchio IK → joint-vel
        (`VelCmd` / `directJControl`); preferred for EEF teleop
      - `joint_pos` — Python joint goals → streamed `moveJ`
      - `joint_vel` — Python joint velocities → C++ integrates `qd` and calls
        `directJControl` every tick (usually more tolerant of full-rate sends
        than streamed `moveL`)

    Raise `max_pos_speed` / `max_rot_speed` for Cartesian teleop, or
    `max_motor_speed` for joint-velocity teleop.
    """

    __api__ = [
        "get_state",
        "get_all_state",
        "moveL",
        "moveJ",
        "speedJ",
    ]
    __pub__ = True
    __req__ = True

    def __init__(
        self,
        robot_ip: str = "192.168.1.44",
        port: int = 7582,
        session_id: int = 1,
        rt_priority: int = 80,
        robot_controller: str = "task_pos_ik",
        max_pos_speed: float = 0.15,
        max_rot_speed: float = 0.25,
        cmd_freq: int = 10,
        stream_l_mode: str = "time",
        stream_l_tt: float = 0.016,
        stream_l_bt: float = 0.008,
        stream_l_speed: float = 0.0,
        stream_l_min_track_time: float = 0.04,
        stream_l_throttle: int = 2,
        stream_j_speed: float = 0.3,
        stream_j_throttle: int = 2,
        max_motor_speed: float = 1.0,
        joint_limits: tuple[list[float], list[float]] | None = None,
        ws_orientation_speed: float = 1.0,
        max_eef_delta_pos: float = 0.0,
        max_eef_delta_rot: float = 0.0,
        max_cmd_step_pos: float = 0.0,
        max_cmd_step_rot: float = 0.0,
        lowpass_alpha: float | None = 0.2,
        max_joint_accel: float | None = 1.5,
        urdf_path: str | None = None,
        ee_frame: str = DEFAULT_EE_FRAME,
        ik_kp: float = 8.0,
        log_diagnostics: bool = False,
        dtype=np.float64,
        *,
        freq: int = 250,
        max_buffer_size: int | None = None,
        **kwargs,
    ):
        """
        Args:
            robot_ip: Address of the robot controller.
            port: KORD UDP port.
            session_id: KORD session id, must match the CBun configuration.
            rt_priority: SCHED_FIFO priority for the C++ loop; needs CAP_SYS_NICE.
            robot_controller: "task_pos_ik" (local IK → VelCmd), "task_pos"
                (streamL), "joint_pos" (streamJ), or "joint_vel" (VelCmd).
            max_pos_speed: Translation speed ceiling in m/s.
            max_rot_speed: Rotation speed ceiling in rad/s.
            cmd_freq: Rate the caller sends targets at, used to size the guards.
            stream_l_mode: "time" for TT_TIME, "speed" for TT_WS_TARGET_SPEED.
                Speed tracking lets KORD derive the duration, so the resulting
                motion no longer depends on how fast the caller's loop runs.
            stream_l_tt: TT_TIME deadline per C++ micro-step (seconds). Default
                0.016 is 2× the send period at throttle=2 (KORD real_time_patterns).
            stream_l_bt: BT_TIME blend window per micro-step, in seconds.
                Default 0.008 (~50% of TT).
            stream_l_speed: TCP speed in m/s for "speed" mode. 0 derives
                1.5 * max_pos_speed, giving the TCP headroom to converge on the
                target rather than trail it.
            stream_l_min_track_time: Shortest move duration speed tracking may
                imply, in seconds. Targets are held back until the accumulated
                step is long enough that KORD will accept it.
            stream_l_throttle: Send moveL every n sync ticks while micro-stepping.
                Keep waitSync at 250 Hz always; only decimate moveL. Default 2 ≈
                125 Hz (documented when full-rate moveL causes waitSync timeouts).
                Idle still sends nothing once the micro-step lands on the goal.
            stream_j_speed: Max joint speed in rad/s for streamJ waypoints.
            stream_j_throttle: Send moveJ every n sync ticks.
            max_motor_speed: Joint speed ceiling in rad/s, used to size the
                per-command joint step guard.
            joint_limits: Optional (q_min, q_max) to clip joint targets to. The
                controller enforces its own position limits and does not report
                them over KORD, so nothing is clipped by default; a wrong guess
                here would clip a valid pose and command a lurch.
            ws_orientation_speed: Must match max_ws_orientation_speed in the
                controller's KORD.ini; sizes the rotation half of the hold.
            max_eef_delta_pos: Envelope around the measured position in m.
                0 derives it from max_pos_speed.
            max_eef_delta_rot: Envelope around the measured rotation in rad.
                0 derives it from max_rot_speed.
            max_cmd_step_pos: Per-command position jump guard in m.
                0 derives it from max_pos_speed.
            max_cmd_step_rot: Per-command rotation jump guard in rad.
                0 derives it from max_rot_speed.
            lowpass_alpha: EMA smoothing on streamL / streamJ / IK Cartesian
                targets and ``qd`` after the guards. None disables it. Smaller
                is smoother/slower to respond (0.15–0.25 is a good teleop band
                at 100 Hz). Hands-off resets the filter so it does not coast.
            max_joint_accel: Per-joint slew limit on ``qd`` in rad/s² for VelCmd
                paths (`task_pos_ik`, `joint_vel`). Caps how fast velocity may
                change so Spacemouse spikes and idle→hold do not jerk the
                reference. None disables it.
            urdf_path: URDF for `task_pos_ik`. Defaults to the packaged KR1018
                kinematics model.
            ee_frame: Tip frame name in the URDF (default ``end_effector``).
            ik_kp: Task-space P gain (1/s) mapping Cartesian lead to twist
                before the Jacobian map. ~5–10 is responsive at 100 Hz teleop.
            log_diagnostics: Report sync and command rates every two seconds.
            dtype: Published state dtype. KORD is double precision.
        """
        assert 0 < freq <= 500
        assert 0 < cmd_freq
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        assert 0 < max_motor_speed
        assert 0 < ik_kp
        if lowpass_alpha is not None and not (0.0 < lowpass_alpha < 1.0):
            raise ValueError(f"lowpass_alpha must be in (0, 1) or None, got {lowpass_alpha}")
        if max_joint_accel is not None and max_joint_accel <= 0.0:
            raise ValueError(f"max_joint_accel must be > 0 or None, got {max_joint_accel}")
        if stream_l_mode not in ("time", "speed"):
            raise ValueError(f"stream_l_mode must be 'time' or 'speed', got {stream_l_mode!r}")
        if max_buffer_size is None:
            max_buffer_size = int(freq * 5)

        self.robot_ip = robot_ip
        self.port = port
        self.session_id = session_id
        self.rt_priority = rt_priority
        self.robot_controller = RobotController[robot_controller.upper()]
        if self.robot_controller not in (
            RobotController.TASK_POS,
            RobotController.TASK_POS_IK,
            RobotController.JOINT_POS,
            RobotController.JOINT_VEL,
        ):
            raise ValueError(
                "robot_controller must be task_pos_ik, task_pos, joint_pos, or "
                f"joint_vel; got {robot_controller!r}"
            )
        self.num_joints = NUM_JOINTS
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.max_motor_speed = max_motor_speed
        self.cmd_freq = cmd_freq
        self.stream_l_mode = stream_l_mode
        self.stream_l_tt = stream_l_tt
        self.stream_l_bt = stream_l_bt
        self.stream_l_throttle = stream_l_throttle
        self.stream_j_speed = stream_j_speed
        self.stream_j_throttle = stream_j_throttle
        self.lowpass_alpha = lowpass_alpha
        self.max_joint_accel = max_joint_accel
        self.urdf_path = urdf_path or DEFAULT_URDF_PATH
        self.ee_frame = ee_frame
        self.ik_kp = ik_kp
        self._kin: KassowKinematics | None = None
        if joint_limits is None:
            self.joint_limits = None
        else:
            q_bounds = np.asarray(joint_limits, dtype=dtype)
            assert q_bounds.shape == (2, NUM_JOINTS)
            self.joint_limits = (q_bounds[0], q_bounds[1])
        self.log_diagnostics = log_diagnostics
        self.dtype = dtype

        # The guards below are all derived from the speed ceilings so that a
        # fixed value can never silently override max_pos_speed / max_rot_speed.
        dt = 1.0 / cmd_freq
        # A step guard below the per-cycle advance would become the speed limit.
        step_pos = max_cmd_step_pos or 1.5 * max_pos_speed * dt
        step_rot = max_cmd_step_rot or 1.5 * max_rot_speed * dt
        self._max_joint_step = 1.5 * max_motor_speed * dt
        # How far ahead of the measured pose the Python goal may sit. Local IK
        # tracks with joint-vel, so allow a longer Cartesian lead than streamL
        # (which needed a tight envelope for via-point feasibility).
        if self.robot_controller == RobotController.TASK_POS_IK:
            lead = 4.0 * dt
        elif stream_l_mode == "speed":
            lead = 6.0 * dt
        else:
            lead = 1.5 * dt
        env_pos = max_eef_delta_pos or max_pos_speed * lead
        env_rot = max_eef_delta_rot or max_rot_speed * lead

        self._min_step_pos = 0.0
        self._min_step_rot = 0.0
        if stream_l_mode == "speed":
            self._tracking_type = "TT_WS_TARGET_SPEED"
            self._tracking_val = stream_l_speed or 1.5 * max_pos_speed
            self._min_step_pos = self._tracking_val * stream_l_min_track_time
            self._min_step_rot = ws_orientation_speed * stream_l_min_track_time
            # The step guard has to leave room for the target to accumulate past
            # that minimum, or nothing is ever sent.
            step_pos = max(step_pos, 2.0 * self._min_step_pos)
            step_rot = max(step_rot, 2.0 * self._min_step_rot)
        else:
            self._tracking_type = "TT_TIME"
            self._tracking_val = stream_l_tt

        self._env_delta = np.array([env_pos] * 3 + [env_rot] * 3, dtype=dtype)
        self._max_step = np.array([step_pos] * 3 + [step_rot] * 3, dtype=dtype)

        super().__init__(freq=freq, max_buffer_size=max_buffer_size, **kwargs)

    def __post_init__(self):
        if _kord_bridge is None:
            raise ImportError(BUILD_HINT)

        example_request_params = {
            "target_eef_pose": np.zeros((6,), dtype=self.dtype),
            "target_joint_q": np.zeros((NUM_JOINTS,), dtype=self.dtype),
            "target_joint_qd": np.zeros((NUM_JOINTS,), dtype=self.dtype),
        }
        request_params_keys = {
            RobotController.TASK_POS: (RequestType.MOVEL, ("target_eef_pose",)),
            RobotController.TASK_POS_IK: (RequestType.MOVEL, ("target_eef_pose",)),
            RobotController.JOINT_POS: (RequestType.MOVEJ, ("target_joint_q",)),
            RobotController.JOINT_VEL: (RequestType.SPEEDJ, ("target_joint_qd",)),
        }[self.robot_controller][1]
        example_request_params = {k: example_request_params[k] for k in request_params_keys}
        example_request_params["target_time"] = time.now()

        self._bridge = _kord_bridge.KordBridge(self.robot_ip, self.port, self.session_id, self.rt_priority)
        if not self._bridge.connect():
            raise RuntimeError(f"KassowArm: failed to connect to {self.robot_ip}:{self.port}")
        self._bridge.set_stream_l_throttle(self.stream_l_throttle)
        self._bridge.set_stream_j_throttle(self.stream_j_throttle)
        # After SafetyEvent ESTOP (e.g. JREF_X_SENSOR_POSITION_SPAN) a CBun-only
        # clear is not enough — CLEAR_HALT must run too. kord-clean-alarm keeps
        # only the last dedicated flag unless --all is passed.
        if self._bridge.clear_recoverable_alarms():
            logger.info("KassowArm: cleared recoverable halt/CBun/unsuspend latches")

        if self.robot_controller == RobotController.TASK_POS_IK:
            self._kin = KassowKinematics(urdf_path=self.urdf_path, ee_frame=self.ee_frame)
            # FK check waits for the first synced state in req() — get_state() is
            # still zeroed until the RT loop has called fetchData().
            self._fk_checked = False
        else:
            self._fk_checked = True
        # Don't send hold VelCmd zeros until a real teleop command has entered
        # Vel mode — premature directJControl triggers MOVE_IN_INVALID_MOTION_STATE.
        self._vel_mode_active = False
        self._qd_filter = None
        self._last_qd = None
        self._last_qd_t = None
        self._diag_vel_n = 0
        self._diag_qd_sum = 0.0
        self._diag_dp_sum = 0.0

        self.example_request = {
            "type": next(iter(RequestType)).value,
            **example_request_params,
        }
        self.example_data = {
            "eef_pose": np.zeros((6,), dtype=self.dtype),
            "joint_q": np.zeros((NUM_JOINTS,), dtype=self.dtype),
            "joint_qd": np.zeros((NUM_JOINTS,), dtype=self.dtype),
            "joint_tau": np.zeros((NUM_JOINTS,), dtype=self.dtype),
            "alarm_code": np.uint32(0),
            "timestamp": time.now(),
        }
        self.worker = self.pub
        self.run = self.req
        super().__post_init__()

    def pub(self):
        self._bridge.start()
        logger.info(f"KassowArm: KORD real-time bridge started ({self.robot_ip})")

        try:
            rate = time.Rate(self.freq)
            not_pub_ready = True
            alarmed = False
            t_diag = time.now()
            state = self._bridge.get_state()
            tick_0, sends_j_0, sends_l_0 = state.tick, state.stream_j_sends, state.stream_l_sends
            while not self.exit_event.is_set():
                state = self._bridge.get_state()

                # Store current state in ring buffer
                data = {
                    "eef_pose": kord_tcp_to_pose6(state.tcp_pose).astype(self.dtype),
                    "joint_q": np.array(state.joint_q, dtype=self.dtype),
                    "joint_qd": np.array(state.joint_qd, dtype=self.dtype),
                    "joint_tau": np.array(state.joint_tau, dtype=self.dtype),
                    "alarm_code": np.uint32(state.alarm_code),
                    "timestamp": time.now(),
                }
                self.ring_buffer.put(data)
                if not_pub_ready:
                    self.pub_ready_event.set()
                    not_pub_ready = False

                # A rejected or infeasible move only surfaces as a system alarm.
                # Without this the robot just looks like it stopped responding.
                if bool(state.alarm_code) != alarmed:
                    alarmed = not alarmed
                    if alarmed:
                        logger.warning(
                            f"KassowArm: KORD alarm, motion blocked: {describe_alarm(state.alarm_code)} "
                            "(often invisible on the teach pendant; CBun software latch)"
                        )
                    else:
                        logger.info("KassowArm: KORD alarm cleared")

                if self.log_diagnostics:
                    elapsed = data["timestamp"] - t_diag
                    if elapsed >= DIAGNOSTICS_INTERVAL:
                        msg = (
                            f"KassowArm: sync {(state.tick - tick_0) / elapsed:.1f} Hz | "
                            f"moveJ {(state.stream_j_sends - sends_j_0) / elapsed:.1f} Hz | "
                            f"moveL {(state.stream_l_sends - sends_l_0) / elapsed:.1f} Hz"
                        )
                        n = self._diag_vel_n
                        if n > 0:
                            msg += (
                                f" | vel {n / elapsed:.0f} Hz "
                                f"||qd||={self._diag_qd_sum / n:.3f} "
                                f"lead={self._diag_dp_sum / n * 1e3:.1f} mm"
                            )
                            self._diag_vel_n = 0
                            self._diag_qd_sum = 0.0
                            self._diag_dp_sum = 0.0
                        logger.info(msg)
                        t_diag, tick_0 = data["timestamp"], state.tick
                        sends_j_0, sends_l_0 = state.stream_j_sends, state.stream_l_sends
                rate.precise_sleep()
        except KeyboardInterrupt:
            pass
        finally:
            self._bridge.stop()
            logger.info("KassowArm: KORD real-time bridge stopped")

    def req(self):
        tracking_type = getattr(_kord_bridge.TrackingType, self._tracking_type)
        blend_type = _kord_bridge.BlendType.BT_TIME
        self._log_motion_envelope()
        if self.lowpass_alpha is not None:
            logger.info(f"KassowArm: command low-pass alpha={self.lowpass_alpha:.2f}")
        if self.robot_controller == RobotController.TASK_POS_IK:
            accel = (
                f", max_joint_accel={self.max_joint_accel:.1f} rad/s²"
                if self.max_joint_accel is not None
                else ""
            )
            logger.info(
                f"KassowArm: task_pos_ik → Jacobian VelCmd (ik_kp={self.ik_kp:.1f} 1/s, "
                f"max_motor_speed={self.max_motor_speed:.2f} rad/s{accel})"
            )
        elif self.max_joint_accel is not None and self.robot_controller == RobotController.JOINT_VEL:
            logger.info(f"KassowArm: qd slew limit {self.max_joint_accel:.1f} rad/s²")

        # Anchors for the rate limiters, adopted from the measured state on the
        # first command so a stale target can never be streamed at startup.
        last_pose = None
        last_joint_q = None
        pose_filter = None
        joint_filter = None
        idle_cycles = 0
        zeros7 = np.zeros((NUM_JOINTS,), dtype=self.dtype)

        try:
            rate = time.Rate(self.freq)
            self.req_ready_event.set()
            while not self.exit_event.is_set():
                if not self._fk_checked and self._kin is not None:
                    state = self._bridge.get_state()
                    # tick advances only after the RT loop has fetched real status.
                    if state.tick > 0:
                        self._kin.check_fk_alignment(
                            np.array(state.joint_q, dtype=np.float64),
                            np.array(state.tcp_pose, dtype=np.float64),
                        )
                        self._fk_checked = True
                    else:
                        rate.precise_sleep()
                        continue

                # Fetch requests from queue
                try:
                    reqs = self.request_queue.get_all()
                    if isinstance(reqs, dict):
                        reqs = [{k: reqs[k][i] for k in reqs.keys()} for i in range(len(reqs["type"]))]
                except queue.Empty:
                    reqs = []
                if not reqs:
                    idle_cycles += 1
                    # Only hold with zero qd after Vel mode was entered by a real
                    # command. Sending directJControl at startup re-latches
                    # MOVE_IN_INVALID_MOTION_STATE on CBun ≥3.0.4.
                    if (
                        self.robot_controller == RobotController.TASK_POS_IK
                        and self._vel_mode_active
                        and idle_cycles >= max(2, int(self.freq * _IK_IDLE_HOLD_S))
                    ):
                        state = self._bridge.get_state()
                        if not state.alarm_code:
                            self._send_vel_cmd(zeros7)
                elif idle_cycles > 0:
                    # Teleop stopped sending (hands off). Re-anchor so the next
                    # move starts from the measured pose with a fresh filter.
                    state = self._bridge.get_state()
                    last_pose = kord_tcp_to_pose6(state.tcp_pose).astype(self.dtype)
                    last_joint_q = np.array(state.joint_q, dtype=self.dtype)
                    if pose_filter is not None:
                        pose_filter.s = last_pose.copy()
                    if joint_filter is not None:
                        joint_filter.s = last_joint_q.copy()
                    self._qd_filter = None
                    idle_cycles = 0
                for r in reqs:
                    req = Request(RequestType(r.pop("type")), r)
                    state = self._bridge.get_state()

                    if req.type == RequestType.MOVEL and self.robot_controller == RobotController.TASK_POS_IK:
                        pose_meas = kord_tcp_to_pose6(state.tcp_pose).astype(self.dtype)
                        joint_q_now = np.array(state.joint_q, dtype=self.dtype)
                        if last_pose is None or state.alarm_code:
                            last_pose = pose_meas
                            self._qd_filter = None
                            self._last_qd = None
                            self._last_qd_t = None
                        if state.alarm_code:
                            continue
                        target = np.array(req.params["target_eef_pose"], dtype=self.dtype)
                        # Relative Cartesian lead (measured TCP frame). Do NOT
                        # low-pass the pose here — alpha=0.15 left ||qd|| near
                        # zero. Smooth only the resulting joint velocity.
                        dp = target[:3] - pose_meas[:3]
                        dist = float(np.linalg.norm(dp))
                        # Cap lead so a stuck teleop integrator cannot demand a
                        # huge corrective velocity in one tick.
                        lead_pos = max(float(self._max_step[0]) * 4.0, 0.01)
                        if dist > lead_pos:
                            dp = dp * (lead_pos / dist)
                            dist = lead_pos

                        r_meas = R.from_rotvec(pose_meas[3:])
                        r_tgt = R.from_rotvec(target[3:])
                        # World-frame error: LOCAL_WORLD_ALIGNED Jacobian expects ω_world.
                        # (Body-frame r_meas.inv()*r_tgt was wrong and let translation
                        # couple into orientation without a usable correction.)
                        dR = r_tgt * r_meas.inv()
                        ang = float(dR.magnitude())
                        if ang < _IK_ORIENT_HOLD:
                            omega = np.zeros(3, dtype=self.dtype)
                        else:
                            lead_rot = max(float(self._max_step[3]) * 4.0, 0.05)
                            if ang > lead_rot:
                                dR = R.from_rotvec(dR.as_rotvec() * (lead_rot / ang))
                                ang = lead_rot
                            omega = dR.as_rotvec() * self.ik_kp

                        if dist < IDLE_EPS and ang < _IK_ORIENT_HOLD:
                            qd = zeros7
                        else:
                            # Task-space P → twist, then weighted Jacobian DLS → qd.
                            # Tracks Spacemouse lead at up to max_*_speed instead
                            # of the near-zero rates from position-IK * small kp.
                            twist = np.zeros(6, dtype=np.float64)
                            twist[:3] = self.ik_kp * dp
                            sp = float(np.linalg.norm(twist[:3]))
                            if sp > self.max_pos_speed > 0.0:
                                twist[:3] *= self.max_pos_speed / sp
                            twist[3:] = omega
                            sr = float(np.linalg.norm(twist[3:]))
                            if sr > self.max_rot_speed > 0.0:
                                twist[3:] *= self.max_rot_speed / sr
                            qd = self._kin.twist_to_qd(joint_q_now, twist)
                            qd = np.clip(qd, -self.max_motor_speed, self.max_motor_speed).astype(
                                self.dtype
                            )
                        if self.lowpass_alpha is not None:
                            if self._qd_filter is None:
                                self._qd_filter = LowPassFilter(alpha=self.lowpass_alpha, initial=zeros7)
                            qd = np.asarray(self._qd_filter(qd), dtype=self.dtype)
                        self._send_vel_cmd(qd)
                        self._vel_mode_active = True
                        last_pose = target
                        if self.log_diagnostics:
                            self._diag_vel_n += 1
                            self._diag_qd_sum += float(np.linalg.norm(qd))
                            self._diag_dp_sum += dist

                    elif req.type == RequestType.MOVEL:
                        pose_now = kord_tcp_to_pose6(state.tcp_pose).astype(self.dtype)
                        if last_pose is None or state.alarm_code:
                            # First command, or motion is blocked: re-anchor on
                            # the measured pose. Otherwise the caller's target
                            # keeps integrating and the arm lurches toward a
                            # stale pose the moment the alarm clears.
                            last_pose = pose_now
                            if pose_filter is not None:
                                pose_filter.s = pose_now.copy()
                        if state.alarm_code:
                            continue
                        target = np.array(req.params["target_eef_pose"], dtype=self.dtype)
                        target = np.clip(target, pose_now - self._env_delta, pose_now + self._env_delta)
                        # Rate-limit command jumps so via-points stay continuous.
                        target = np.clip(target, last_pose - self._max_step, last_pose + self._max_step)
                        if self.lowpass_alpha is not None and pose_filter is None:
                            pose_filter = LowPassFilter(alpha=self.lowpass_alpha, initial=pose_now)
                        if pose_filter is not None:
                            target = np.asarray(pose_filter(target), dtype=self.dtype)
                        if not self._is_new_waypoint(target, last_pose):
                            continue
                        cmd = _kord_bridge.StreamLCmd()
                        cmd.tcp = pose6_to_kord_tcp(target).tolist()
                        cmd.tt = tracking_type
                        cmd.tt_val = self._tracking_val
                        cmd.bt = blend_type
                        cmd.bt_val = self.stream_l_bt
                        cmd.max_pos_speed = self.max_pos_speed
                        cmd.max_rot_speed = self.max_rot_speed
                        self._bridge.set_stream_l_command(cmd)
                        last_pose = target

                    elif req.type == RequestType.MOVEJ:
                        joint_q_now = np.array(state.joint_q, dtype=self.dtype)
                        if last_joint_q is None or state.alarm_code:
                            last_joint_q = joint_q_now
                            if joint_filter is not None:
                                joint_filter.s = joint_q_now.copy()
                        if state.alarm_code:
                            continue
                        target = np.array(req.params["target_joint_q"], dtype=self.dtype)
                        if self.joint_limits is not None:
                            target = np.clip(target, *self.joint_limits)
                        # Anchoring on the measured position means the first
                        # command can only advance one step from where the arm
                        # actually is, whatever the caller asked for.
                        target = np.clip(target, last_joint_q - self._max_joint_step, last_joint_q + self._max_joint_step)
                        if self.lowpass_alpha is not None and joint_filter is None:
                            joint_filter = LowPassFilter(alpha=self.lowpass_alpha, initial=joint_q_now)
                        if joint_filter is not None:
                            target = np.asarray(joint_filter(target), dtype=self.dtype)
                        if np.allclose(target, last_joint_q, rtol=0, atol=IDLE_EPS):
                            continue
                        cmd = _kord_bridge.StreamJCmd()
                        cmd.q = target.tolist()
                        cmd.tt_val = self.stream_j_speed
                        self._bridge.set_stream_j_command(cmd)
                        last_joint_q = target

                    elif req.type == RequestType.SPEEDJ:
                        # VelCmd: C++ integrates qd and calls directJControl every
                        # waitSync. Keep sending (including zeros) so the RT loop
                        # stays in Vel mode and holds when hands-off.
                        if state.alarm_code:
                            self._last_qd = None
                            self._last_qd_t = None
                            continue
                        qd = np.array(req.params["target_joint_qd"], dtype=self.dtype)
                        qd = np.clip(qd, -self.max_motor_speed, self.max_motor_speed)
                        self._send_vel_cmd(qd)
                        self._vel_mode_active = True

                    else:
                        raise ValueError(req.type)
                rate.precise_sleep()
        except KeyboardInterrupt:
            pass

    def _send_vel_cmd(self, qd: np.ndarray) -> None:
        """Publish a VelCmd (directJControl path) with optional joint limits.

        Applies ``max_joint_accel`` slew limiting so commanded ``qd`` cannot jump
        between consecutive sends (including idle ramp-to-zero).
        """
        qd = np.asarray(qd, dtype=self.dtype).reshape(NUM_JOINTS).copy()
        now = time.now()
        # Always slew from the previous command (treat missing history as zero) so
        # the first Spacemouse touch cannot step qd to the full IK output.
        if self.max_joint_accel is not None:
            prev = self._last_qd if self._last_qd is not None else np.zeros((NUM_JOINTS,), dtype=self.dtype)
            if self._last_qd_t is None:
                dt = 1.0 / max(self.cmd_freq, 1)
            else:
                dt = float(now - self._last_qd_t)
            dt = max(1e-4, min(0.05, dt))
            max_dq = self.max_joint_accel * dt
            qd = prev + np.clip(qd - prev, -max_dq, max_dq)
        self._last_qd = qd.copy()
        self._last_qd_t = now
        cmd = _kord_bridge.VelCmd()
        cmd.qd = qd.tolist()
        if self.joint_limits is not None:
            cmd.q_min = self.joint_limits[0].tolist()
            cmd.q_max = self.joint_limits[1].tolist()
        self._bridge.set_vel_command(cmd)

    def _is_new_waypoint(self, target, last_pose) -> bool:
        """Decide whether a clamped target is worth streaming to the robot."""
        d_pos = float(np.linalg.norm(target[:3] - last_pose[:3]))
        d_rot = float(np.linalg.norm(target[3:] - last_pose[3:]))
        if self._min_step_pos > 0.0:
            # Speed tracking derives the duration from the step size, so a
            # sub-millimetre step implies a near-zero duration and the joint
            # speed check rejects the move. Let the target accumulate first.
            return d_pos >= self._min_step_pos or d_rot >= self._min_step_rot
        return d_pos > IDLE_EPS or d_rot > IDLE_EPS

    def _log_motion_envelope(self):
        """Report the speeds the guards actually permit, once per node start."""
        if self.robot_controller == RobotController.TASK_POS_IK:
            logger.info(
                f"KassowArm: local IK envelope "
                f"{float(self._env_delta[0]) * 1e3:.1f} mm / "
                f"{float(self._env_delta[3]) * 1e3:.1f} mrad, "
                f"step guard {float(self._max_step[0]) * 1e3:.1f} mm "
                f"(cmd_freq={self.cmd_freq} Hz)"
            )
            return
        env_pos, env_rot = float(self._env_delta[0]), float(self._env_delta[3])
        micro_pos = self.max_pos_speed * (self.stream_l_throttle / 250.0)
        micro_rot = self.max_rot_speed * (self.stream_l_throttle / 250.0)
        if self.stream_l_mode == "time":
            logger.info(
                f"KassowArm: streamL micro-steps TT_TIME {self.stream_l_tt:.3f}s, "
                f"BT_TIME {self.stream_l_bt:.3f}s, throttle={self.stream_l_throttle} "
                f"(~{250 / max(1, self.stream_l_throttle):.0f} Hz while moving)"
            )
            logger.info(
                f"KassowArm: micro-step ceiling {micro_pos * 1e3:.2f} mm / "
                f"{micro_rot * 1e3:.2f} mrad per send at "
                f"{self.max_pos_speed:.3f} m/s and {self.max_rot_speed:.3f} rad/s"
            )
        else:
            logger.info(
                f"KassowArm: streamL TT_WS_TARGET_SPEED at {self._tracking_val:.3f} m/s, "
                f"BT_TIME {self.stream_l_bt:.3f}s, holding steps below {self._min_step_pos * 1e3:.1f} mm"
            )
        logger.info(
            f"KassowArm: Python goal envelope {env_pos * 1e3:.1f} mm / "
            f"{env_rot * 1e3:.1f} mrad, step guard {float(self._max_step[0]) * 1e3:.1f} mm "
            f"(cmd_freq={self.cmd_freq} Hz)"
        )

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def moveL(self, target_eef_pose, target_time):
        """Cartesian waypoint as RIO pose6 ``[xyz, rotvec]``.

        Converted to KORD XYZ-Euler at the bridge for ``task_pos`` streamL, or
        solved with local IK for ``task_pos_ik``. `target_time` is unused.
        """
        target_eef_pose = np.array(target_eef_pose, dtype=self.dtype)
        assert target_eef_pose.shape == (6,)
        req = {
            "type": RequestType.MOVEL.value,
            "target_eef_pose": target_eef_pose,
            "target_time": target_time,
        }
        self.request_queue.put(req)

    def moveJ(self, target_joint_q, target_time):
        """Stream a joint position waypoint, clipped to the joint limits.

        KORD derives the motion timing from `stream_j_speed`, so `target_time` is
        accepted for interface compatibility but unused.
        """
        target_joint_q = np.array(target_joint_q, dtype=self.dtype)
        assert target_joint_q.shape == (self.num_joints,)
        req = {
            "type": RequestType.MOVEJ.value,
            "target_joint_q": target_joint_q,
            "target_time": target_time,
        }
        self.request_queue.put(req)

    def speedJ(self, target_joint_qd, target_time):
        """Joint-velocity teleop via C++ VelCmd → directJControl every tick.

        The bridge integrates `qd` into a position reference inside the RT loop,
        so Python scheduling jitter does not create torque spikes the way a
        naive directJ position stream would. `target_time` is unused.
        """
        target_joint_qd = np.array(target_joint_qd, dtype=self.dtype)
        assert target_joint_qd.shape == (self.num_joints,)
        req = {
            "type": RequestType.SPEEDJ.value,
            "target_joint_qd": target_joint_qd,
            "target_time": target_time,
        }
        self.request_queue.put(req)


def KassowArmServer(mw, *args, **kwargs):
    return ServerFactory(mw, KassowArm, *args, **kwargs)


def KassowArmClient(mw, *args, **kwargs):
    return ClientFactory(mw, KassowArm, *args, **kwargs)
