import queue
import time as _stdlib_time
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from .. import time
from ..middleware import ClientFactory, ServerFactory
from ..node import Node
from ..request import Request

try:
    from inspire_demos import InspireHandSerial
except ImportError as e:
    if TYPE_CHECKING:
        raise e
    else:
        InspireHandSerial = None  # type: ignore

# Inspire hand joint angle range (integer counts).
_ANGLE_MIN = 0
_ANGLE_MAX = 1000


class RequestType(Enum):
    MOVEH = auto()


class InspireHand(Node):
    """Robot interface for the Inspire RH56 dexterous hand over serial.

    Joint ordering (6 DOF):
        [pinky, ring, middle, index, thumb_flex, thumb_rot]

    All public angles are normalized float in [0, 1]:
        0 = closed / adducted
        1 = open   / abducted
    """

    __api__ = [
        "get_state",
        "get_all_state",
        "moveH",
        "moveJ",
    ]
    __pub__ = True
    __req__ = True

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        hand_id: int = 1,
        generation: int = 3,
        speed: int = 1000,
        force: int = 200,
        home_to_open: bool = True,
        read_state: bool = False,
        dtype=np.float32,
        *,
        freq: int = 30,
        max_buffer_size: int | None = None,
        max_queue_size: int = 128,
        **kwargs,
    ):
        """
        Args:
            port: Serial port for the hand. Defaults to "/dev/ttyUSB0".
            baudrate: Serial baud rate. Defaults to 115200.
            hand_id: Modbus hand ID. Defaults to 1.
            generation: Hardware generation (3 or 4). Defaults to 3.
            speed: Default joint speed [0, 1000]. Defaults to 1000.
            force: Default joint force limit [0, 1000]. Defaults to 200.
            home_to_open: Open the hand fully on startup. Defaults to True.
            read_state: If True, read actual joint angles from the hand each cycle
                (slower, ~10 ms per read). If False (default), the commanded angles
                are reported as state — avoids serial contention with set_angle.
            dtype: NumPy dtype for published joint angles. Defaults to np.float32.
            freq: Control loop frequency in Hz. Defaults to 30.
            max_buffer_size: Ring buffer capacity. Defaults to freq * 10.
            max_queue_size: Command queue depth. Defaults to 128.
        """
        if max_buffer_size is None:
            max_buffer_size = int(freq * 10)
        self.port = port
        self.baudrate = baudrate
        self.hand_id = hand_id
        self.generation = generation
        self.speed = int(np.clip(speed, _ANGLE_MIN, _ANGLE_MAX))
        self.force = int(np.clip(force, _ANGLE_MIN, _ANGLE_MAX))
        self.home_to_open = home_to_open
        self.read_state = read_state
        self.dtype = dtype
        super().__init__(freq=freq, max_buffer_size=max_buffer_size, max_queue_size=max_queue_size, **kwargs)

    def __post_init__(self):
        self.example_request = {
            "type": RequestType.MOVEH.value,
            "target_angles": np.zeros(6, dtype=self.dtype),
            "target_time": time.now(),
        }
        self.example_data = {
            "joint_angles": np.zeros(6, dtype=self.dtype),
            # joint_q mirrors joint_angles so this robot is compatible with the
            # rio SingleArm / teleop_leader_follower framework.
            "joint_q": np.zeros(6, dtype=self.dtype),
            "timestamp": time.now(),
        }
        self.worker = None
        self.run = self.pubreq
        super().__post_init__()

    def pubreq(self):
        print(
            f"[InspireHand] Connecting on {self.port} at {self.baudrate} baud "
            f"(gen{self.generation}, read_state={self.read_state})...",
            flush=True,
        )
        hand = InspireHandSerial(port=self.port, baudrate=self.baudrate, generation=self.generation)
        if not hand.connect():
            raise RuntimeError(f"InspireHand: failed to connect on {self.port}")
        print(f"[InspireHand] Connected. hand_id={self.hand_id} speed={self.speed} force={self.force}", flush=True)

        def _flush():
            """Drain any leftover bytes from the serial receive buffer."""
            _stdlib_time.sleep(0.015)
            if hand._ser and hand._ser.in_waiting:
                hand._ser.reset_input_buffer()

        try:
            hand.reset_error()
            _flush()

            speed_arr = np.full(6, self.speed, dtype=np.int32)
            force_arr = np.full(6, self.force, dtype=np.int32)
            hand.set_speed(speed_arr, hand_id=self.hand_id)
            _flush()
            hand.set_force(force_arr, hand_id=self.hand_id)
            _flush()

            if self.home_to_open:
                print("[InspireHand] Homing to open position...", flush=True)
                hand.perform_open()
                _flush()

            # Initialise target from current state if readable, otherwise from home.
            if self.read_state:
                _flush()
                raw_actual = hand.get_angle_actual(hand_id=self.hand_id)
                if len(raw_actual) == 6:
                    target_angles = raw_actual.astype(np.float64) / _ANGLE_MAX
                    print(f"[InspireHand] Initial angles (raw): {raw_actual.tolist()} → norm: {target_angles.round(3).tolist()}", flush=True)
                else:
                    target_angles = np.ones(6, dtype=np.float64) if self.home_to_open else np.zeros(6, dtype=np.float64)
                    print(f"[InspireHand] WARNING: get_angle_actual returned {len(raw_actual)} values (expected 6). Using default.", flush=True)
            else:
                target_angles = np.ones(6, dtype=np.float64) if self.home_to_open else np.zeros(6, dtype=np.float64)
                print(f"[InspireHand] read_state=False — reporting commanded angles as state. Initial target: {target_angles.tolist()}", flush=True)

            rate = time.Rate(self.freq)
            self.req_ready_event.set()
            not_pub_ready = True
            _iter = 0
            _cmd_count = 0

            while not self.exit_event.is_set():
                # Compute integer command and send to hand.
                cmd = np.round(np.clip(target_angles, 0.0, 1.0) * _ANGLE_MAX).astype(np.int32)
                hand.set_angle(cmd, hand_id=self.hand_id)

                # Optionally read back actual angles; otherwise report commanded.
                if self.read_state:
                    # Flush the set_angle ACK before issuing a read request.
                    _flush()
                    raw_actual = hand.get_angle_actual(hand_id=self.hand_id)
                    if len(raw_actual) == 6:
                        actual_norm = (raw_actual.astype(np.float64) / _ANGLE_MAX).astype(self.dtype)
                    else:
                        actual_norm = cmd.astype(np.float64) / _ANGLE_MAX
                        actual_norm = actual_norm.astype(self.dtype)
                else:
                    actual_norm = cmd.astype(np.float64) / _ANGLE_MAX
                    actual_norm = actual_norm.astype(self.dtype)

                self.ring_buffer.put(
                    {
                        "joint_angles": actual_norm,
                        "joint_q": actual_norm,
                        "timestamp": time.now(),
                    }
                )
                if not_pub_ready:
                    self.pub_ready_event.set()
                    not_pub_ready = False

                # Process incoming movement requests.
                try:
                    reqs = self.request_queue.get_all()
                    if isinstance(reqs, dict):
                        reqs = [{k: reqs[k][i] for k in reqs.keys()} for i in range(len(reqs["type"]))]
                except queue.Empty:
                    reqs = []

                for r in reqs:
                    req = Request(RequestType(r.pop("type")), r)
                    if req.type == RequestType.MOVEH:
                        new_target = np.clip(
                            np.array(req.params["target_angles"], dtype=np.float64),
                            0.0,
                            1.0,
                        )
                        _cmd_count += 1
                        if _cmd_count <= 5 or _cmd_count % 90 == 0:
                            new_cmd = np.round(new_target * _ANGLE_MAX).astype(np.int32)
                            # print(f"[InspireHand] MOVEH #{_cmd_count}: target={new_target.round(3).tolist()} → cmd={new_cmd.tolist()}", flush=True)
                        target_angles = new_target
                    else:
                        raise ValueError(req.type)

                # # Print a status line every 90 iterations (~3 s at 30 Hz).
                # if _iter % 90 == 0:
                #     np.set_printoptions(precision=3, suppress=True)
                #     print(
                #         f"[InspireHand] iter={_iter} cmd={cmd.tolist()} "
                #         f"actual={'(=cmd)' if not self.read_state else actual_norm.round(3).tolist()} "
                #         f"total_cmds_received={_cmd_count}",
                #         flush=True,
                #     )

                _iter += 1
                rate.precise_sleep()

        except KeyboardInterrupt:
            pass
        finally:
            print("[InspireHand] Shutting down: opening hand...", flush=True)
            try:
                hand.perform_open()
            except Exception:
                pass
            hand.disconnect()

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def moveH(self, target_angles, target_time):
        """Command the hand to move to the given joint angles.

        Args:
            target_angles: Array-like of shape (6,) with values in [0, 1].
                Index order: [pinky, ring, middle, index, thumb_flex, thumb_rot].
            target_time: Absolute time (seconds) by which the motion should complete.
                Must be in the future. Currently used only for queue ordering.
        """
        target_angles = np.asarray(target_angles, dtype=self.dtype)
        assert target_angles.shape == (6,), f"Expected shape (6,), got {target_angles.shape}"
        assert target_time > time.now(), "target_time must be in the future"
        req = {
            "type": RequestType.MOVEH.value,
            "target_angles": target_angles,
            "target_time": target_time,
        }
        self.request_queue.put(req)

    def moveJ(self, target_angles, target_time):
        """Alias for moveH for compatibility with the rio SingleArm embodiment.

        Accepts the same arguments as moveH. The joint ordering and [0, 1] range
        from the ManusGlove finger openness map directly to this interface.
        """
        self.moveH(target_angles, target_time)


def InspireHandServer(mw, *args, **kwargs):
    return ServerFactory(mw, InspireHand, *args, **kwargs)


def InspireHandClient(mw, *args, **kwargs):
    return ClientFactory(mw, InspireHand, *args, **kwargs)
