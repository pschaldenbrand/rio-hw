import queue
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from .. import time
from ..filters import LowPassFilter
from ..middleware import ClientFactory, ServerFactory
from ..node import Node
from ..request import Request

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
    JOINT_POS = auto()


class RequestType(Enum):
    MOVEL = auto()
    MOVEJ = auto()


class KassowArm(Node):
    """Kassow KR-series 7-DOF arm, driven over KORD by a C++ real-time bridge.

    The `_kord_bridge` extension runs kord-api's `waitSync()` loop in a dedicated
    C++ thread at 250 Hz, independent of Python's GC. `moveL` and `moveJ` publish
    a target that the bridge streams as an `OT_VIAPOINT` motion, so the robot
    blends through waypoints instead of stopping at each one.

    Targets are conditioned before they reach KORD: clamped to an envelope around
    the measured pose, rate-limited against the previous waypoint, and withheld
    while a system alarm blocks motion. Those guards are what stop KORD from
    rejecting a command as INFEASIBLE_MOVE_COMMAND or faulting on torque
    deviation, so they live here rather than in the caller.

    Raise `max_pos_speed` and `max_rot_speed` to teleoperate faster. Do not
    shorten `stream_l_tt` instead: under TT_TIME that compresses the whole
    trajectory and inflates the acceleration KORD estimates.
    """

    __api__ = [
        "get_state",
        "get_all_state",
        "moveL",
        "moveJ",
    ]
    __pub__ = True
    __req__ = True

    def __init__(
        self,
        robot_ip: str = "192.168.1.44",
        port: int = 7582,
        session_id: int = 1,
        rt_priority: int = 80,
        robot_controller: str = "task_pos",
        max_pos_speed: float = 0.15,
        max_rot_speed: float = 0.25,
        cmd_freq: int = 10,
        stream_l_mode: str = "time",
        stream_l_tt: float = 0.10,
        stream_l_bt: float = 0.07,
        stream_l_speed: float = 0.0,
        stream_l_min_track_time: float = 0.04,
        stream_l_throttle: int = 5,
        stream_j_speed: float = 0.3,
        stream_j_throttle: int = 2,
        max_motor_speed: float = 1.0,
        joint_limits: tuple[list[float], list[float]] | None = None,
        ws_orientation_speed: float = 1.0,
        max_eef_delta_pos: float = 0.0,
        max_eef_delta_rot: float = 0.0,
        max_cmd_step_pos: float = 0.0,
        max_cmd_step_rot: float = 0.0,
        lowpass_alpha: float | None = 0.35,
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
            robot_controller: "task_pos" (streamL) or "joint_pos" (streamJ).
            max_pos_speed: Translation speed ceiling in m/s.
            max_rot_speed: Rotation speed ceiling in rad/s.
            cmd_freq: Rate the caller sends targets at, used to size the guards.
            stream_l_mode: "time" for TT_TIME, "speed" for TT_WS_TARGET_SPEED.
                Speed tracking lets KORD derive the duration, so the resulting
                motion no longer depends on how fast the caller's loop runs.
            stream_l_tt: TT_TIME deadline in seconds; keep it near 1 / cmd_freq.
            stream_l_bt: BT_TIME blend window in seconds.
            stream_l_speed: TCP speed in m/s for "speed" mode. 0 derives
                1.5 * max_pos_speed, giving the TCP headroom to converge on the
                target rather than trail it.
            stream_l_min_track_time: Shortest move duration speed tracking may
                imply, in seconds. Targets are held back until the accumulated
                step is long enough that KORD will accept it.
            stream_l_throttle: Send moveL every n sync ticks.
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
            lowpass_alpha: EMA smoothing on streamL / streamJ targets after the
                guards. None disables it. Updates at the teleop command rate, so
                values near 0.3-0.5 smooth Spacemouse motion without the heavy
                lag that 0.1 would add at 10 Hz. Hands-off snaps reset the
                filter so it does not keep coasting.
            log_diagnostics: Report sync and command rates every two seconds.
            dtype: Published state dtype. KORD is double precision.
        """
        assert 0 < freq <= 500
        assert 0 < cmd_freq
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        assert 0 < max_motor_speed
        if lowpass_alpha is not None and not (0.0 < lowpass_alpha < 1.0):
            raise ValueError(f"lowpass_alpha must be in (0, 1) or None, got {lowpass_alpha}")
        if stream_l_mode not in ("time", "speed"):
            raise ValueError(f"stream_l_mode must be 'time' or 'speed', got {stream_l_mode!r}")
        if max_buffer_size is None:
            max_buffer_size = int(freq * 5)

        self.robot_ip = robot_ip
        self.port = port
        self.session_id = session_id
        self.rt_priority = rt_priority
        self.robot_controller = RobotController[robot_controller.upper()]
        self.num_joints = NUM_JOINTS
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.cmd_freq = cmd_freq
        self.stream_l_mode = stream_l_mode
        self.stream_l_tt = stream_l_tt
        self.stream_l_bt = stream_l_bt
        self.stream_l_throttle = stream_l_throttle
        self.stream_j_speed = stream_j_speed
        self.stream_j_throttle = stream_j_throttle
        self.lowpass_alpha = lowpass_alpha
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
        # How far ahead of the measured pose the target is allowed to sit. Under
        # TT_TIME the robot closes the whole remaining gap within stream_l_tt, so
        # the envelope *is* the peak commanded speed and has to stay near
        # max_pos_speed * stream_l_tt. Under speed tracking the gap does not set
        # the speed, and the target needs to lead by several cycles to keep the
        # arm moving.
        lead = 6.0 * dt if stream_l_mode == "speed" else 1.5 * stream_l_tt
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
        }
        request_params_keys = {
            RobotController.TASK_POS: (RequestType.MOVEL, ("target_eef_pose",)),
            RobotController.JOINT_POS: (RequestType.MOVEJ, ("target_joint_q",)),
        }[self.robot_controller][1]
        example_request_params = {k: example_request_params[k] for k in request_params_keys}
        example_request_params["target_time"] = time.now()

        self._bridge = _kord_bridge.KordBridge(self.robot_ip, self.port, self.session_id, self.rt_priority)
        if not self._bridge.connect():
            raise RuntimeError(f"KassowArm: failed to connect to {self.robot_ip}:{self.port}")
        self._bridge.set_stream_l_throttle(self.stream_l_throttle)
        self._bridge.set_stream_j_throttle(self.stream_j_throttle)

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
                    "eef_pose": np.array(state.tcp_pose, dtype=self.dtype),
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
                        logger.warning(f"KassowArm: KORD alarm, motion blocked: {describe_alarm(state.alarm_code)}")
                    else:
                        logger.info("KassowArm: KORD alarm cleared")

                if self.log_diagnostics:
                    elapsed = data["timestamp"] - t_diag
                    if elapsed >= DIAGNOSTICS_INTERVAL:
                        logger.info(
                            f"KassowArm: sync {(state.tick - tick_0) / elapsed:.1f} Hz | "
                            f"moveJ {(state.stream_j_sends - sends_j_0) / elapsed:.1f} Hz | "
                            f"moveL {(state.stream_l_sends - sends_l_0) / elapsed:.1f} Hz"
                        )
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

        # Anchors for the rate limiters, adopted from the measured state on the
        # first command so a stale target can never be streamed at startup.
        last_pose = None
        last_joint_q = None
        pose_filter = None
        joint_filter = None
        idle_cycles = 0

        try:
            rate = time.Rate(self.freq)
            self.req_ready_event.set()
            while not self.exit_event.is_set():
                # Fetch requests from queue
                try:
                    reqs = self.request_queue.get_all()
                    if isinstance(reqs, dict):
                        reqs = [{k: reqs[k][i] for k in reqs.keys()} for i in range(len(reqs["type"]))]
                except queue.Empty:
                    reqs = []
                if not reqs:
                    idle_cycles += 1
                elif idle_cycles > 0:
                    # Teleop stopped sending (hands off). Re-anchor so the next
                    # move starts from the measured pose with a fresh filter.
                    state = self._bridge.get_state()
                    last_pose = np.array(state.tcp_pose, dtype=self.dtype)
                    last_joint_q = np.array(state.joint_q, dtype=self.dtype)
                    if pose_filter is not None:
                        pose_filter.s = last_pose.copy()
                    if joint_filter is not None:
                        joint_filter.s = last_joint_q.copy()
                    idle_cycles = 0
                for r in reqs:
                    req = Request(RequestType(r.pop("type")), r)
                    state = self._bridge.get_state()

                    if req.type == RequestType.MOVEL:
                        pose_now = np.array(state.tcp_pose, dtype=self.dtype)
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
                        cmd.tcp = target.tolist()
                        cmd.tt = tracking_type
                        cmd.tt_val = self._tracking_val
                        cmd.bt = blend_type
                        cmd.bt_val = self.stream_l_bt
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

                    else:
                        raise ValueError(req.type)
                rate.precise_sleep()
        except KeyboardInterrupt:
            pass

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
        env_pos, env_rot = float(self._env_delta[0]), float(self._env_delta[3])
        if self.stream_l_mode == "time":
            dt = 1.0 / self.cmd_freq
            if not 0.5 * dt <= self.stream_l_tt <= 2.0 * dt:
                logger.warning(
                    f"KassowArm: stream_l_tt={self.stream_l_tt:.3f}s is far from the {dt:.3f}s "
                    f"command period, so TT_TIME motion will stutter. Set it near 1 / cmd_freq "
                    f"or use stream_l_mode='speed'."
                )
            # Under TT_TIME the envelope sets the worst case the controller is
            # ever asked for, and that is the number that trips
            # MOTION_CONSTRAINTS, not max_pos_speed.
            logger.info(
                f"KassowArm: streamL TT_TIME {self.stream_l_tt:.3f}s, BT_TIME {self.stream_l_bt:.3f}s, "
                f"peak commanded {env_pos / self.stream_l_tt:.3f} m/s and "
                f"{env_rot / self.stream_l_tt:.3f} rad/s"
            )
        else:
            logger.info(
                f"KassowArm: streamL TT_WS_TARGET_SPEED at {self._tracking_val:.3f} m/s, "
                f"BT_TIME {self.stream_l_bt:.3f}s, holding steps below {self._min_step_pos * 1e3:.1f} mm"
            )
        logger.info(
            f"KassowArm: speed ceiling {self.max_pos_speed:.3f} m/s and {self.max_rot_speed:.3f} rad/s "
            f"(envelope {env_pos * 1e3:.1f} mm, step guard {float(self._max_step[0]) * 1e3:.1f} mm)"
        )

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def moveL(self, target_eef_pose, target_time):
        """Stream a Cartesian waypoint (position and axis-angle rotation).

        KORD derives the motion timing from the configured tracking type, so
        `target_time` is accepted for interface compatibility but unused.
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


def KassowArmServer(mw, *args, **kwargs):
    return ServerFactory(mw, KassowArm, *args, **kwargs)


def KassowArmClient(mw, *args, **kwargs):
    return ClientFactory(mw, KassowArm, *args, **kwargs)
