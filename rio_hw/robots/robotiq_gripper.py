import queue
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from .. import time
from ..filters import LowPassFilter
from ..interpolators import PoseTrajectoryInterpolator
from ..middleware import ClientFactory, ServerFactory
from ..node import Node
from ..request import Request
from .utils.robotiq_tcp_driver import RobotiqTcpDriver

try:
    import pyrobotiqgripper as rq
except ImportError as e:
    if TYPE_CHECKING:
        raise e
    else:
        rq = None


class RobotModel(Enum):
    ROBOTIQ_2F85 = auto()
    ROBOTIQ_2F140 = auto()


RobotInfo = {
    RobotModel.ROBOTIQ_2F85: {"range": (0, 85)},
    RobotModel.ROBOTIQ_2F140: {"range": (0, 140)},
}


class RobotController(Enum):
    TASK_POS = auto()


class RequestType(Enum):
    MOVEG = auto()


class RobotiqGripper(Node):
    __api__ = [
        "get_state",
        "get_all_state",
        "moveG",
    ]
    __pub__ = True
    __req__ = True

    def __init__(
        self,
        robot_port: str = "/dev/ttyUSB0",
        robot_model: str = "robotiq_2f85",
        robot_controller: str = "task_pos",
        device_id: int = 9,
        connection_type: str = "RTU",
        port: str | None = None,
        modbus_mode: str | None = None,
        calibrate_speed: bool = False,
        max_gripper_speed: float | None = 10.0,
        home_to_open: bool = True,
        gripper_lowpass_alpha: float = 0.6,
        modbus_timeout: float = 10.0,
        dtype=np.float64,
        *,
        freq: int = 50,
        max_buffer_size: int | None = None,
        max_queue_size: int = 128,
        **kwargs,
    ):
        """
        Args:
            robot_port: serial path for RTU (e.g. "/dev/ttyUSB0"), or
                "host:port" for RTU_VIA_TCP (e.g. "192.168.1.100:54321").
            device_id: Modbus device ID, usually 9.
            connection_type: "RTU" for serial, "RTU_VIA_TCP" for TCP via
                pyrobotiqgripper, or "TCPIP" for the raw Modbus driver.
            robot_model: gripper model, e.g. "robotiq_2f85" or "robotiq_2f140".
            gripper_range: (close_mm, open_mm) override, or None to use
                RobotInfo defaults for the model.
            home_to_open: if True, open the gripper on startup.
            robot_controller: controller type, currently only "task_pos".
            max_gripper_speed: max speed for trajectory interpolation, or
                None to disable interpolation.
            modbus_timeout: pymodbus/socket timeout in seconds for RTU_VIA_TCP and
                RobotiqTcpDriver (TCPIP). pyrobotiqgripper defaults to 1s.
            dtype: numpy dtype for position values.
            freq: control loop frequency in Hz.
            max_buffer_size: ring buffer size, defaults to freq * 10.
            max_queue_size: request queue size.
        """
        # Legacy station configs use port/modbus_mode instead of robot_port/connection_type.
        legacy_model = kwargs.pop("model", None)
        if legacy_model is not None:
            robot_model = legacy_model
        port = port or kwargs.pop("port", None)
        if port is not None:
            robot_port = port
        modbus_mode = modbus_mode or kwargs.pop("modbus_mode", None)
        if modbus_mode is not None:
            modbus_mode = modbus_mode.upper()
            if modbus_mode == "TCPIP":
                connection_type = "TCPIP"
            elif modbus_mode == "SERIAL":
                connection_type = "RTU"

        assert connection_type in ("RTU", "RTU_VIA_TCP", "TCPIP")
        assert robot_port != "auto", "AUTO_DETECTION not supported"
        robot_model = RobotModel[robot_model.upper()]
        robot_controller = RobotController[robot_controller.upper()]
        gripper_range = RobotInfo[robot_model]["range"]
        if max_buffer_size is None:
            max_buffer_size = int(freq * 10)
        self.robot_port = robot_port
        self.robot_model = robot_model
        self.robot_controller = robot_controller
        self.device_id = device_id
        self.connection_type = connection_type
        self.calibrate_speed = calibrate_speed
        self.gripper_range = gripper_range
        self.home_to_open = home_to_open
        self.gripper_lowpass_alpha = gripper_lowpass_alpha
        self.max_gripper_speed = max_gripper_speed
        self.modbus_timeout = modbus_timeout
        self.dtype = dtype
        super().__init__(freq=freq, max_buffer_size=max_buffer_size, max_queue_size=max_queue_size, **kwargs)

    def __post_init__(self):
        example_request_params = {
            "target_pos": np.zeros((1,), dtype=self.dtype),
        }
        request_params_keys = {
            RobotController.TASK_POS: (RequestType.MOVEG, ("target_pos",)),
        }[self.robot_controller][1]
        example_request_params = {k: example_request_params[k] for k in request_params_keys}
        example_request_params["target_time"] = time.now()

        example_robot_state = {
            "gripper_position": 0.0,
        }

        self.example_request = {
            "type": next(iter(RequestType)).value,
            **example_request_params,
        }
        self.example_data = {
            **example_robot_state,
            "timestamp": time.now(),
        }
        self.worker = None
        self.run = self.pubreq
        super().__post_init__()

    def pubreq(self):
        gripper = None
        if self.connection_type == "RTU":
            gripper = rq.RobotiqGripper(
                com_port=self.robot_port,
                device_id=self.device_id,
                connection_type="RTU",
            )
        elif self.connection_type == "RTU_VIA_TCP":
            # The UR socket bridge at port 54321 speaks Modbus RTU-over-TCP (raw
            # RTU frames with CRC, no MBAP header).  pymodbus.ModbusTcpClient
            # defaults to Modbus-TCP framing (MBAP, no CRC), which the bridge
            # rejects by closing the connection immediately.  We therefore:
            #   1. Patch ModbusTcpClient.__init__ to force RTU framing.
            #   2. Patch ModbusTcpClient.execute with a resilient wrapper that
            #      retries on ConnectionException and mocks the non-existent
            #      FC16-write response that the bridge never sends.
            # Both class-level patches are restored in the finally block; the
            # execute patch is then transferred to the instance only.
            from functools import partial
            from types import SimpleNamespace

            import pymodbus.client.tcp as _pymodbus_tcp
            from pymodbus.exceptions import ConnectionException as _ModbusConnErr
            from pymodbus.exceptions import ModbusIOException as _ModbusIOErr
            from pymodbus.framer import FramerType

            _orig_cls_init = _pymodbus_tcp.ModbusTcpClient.__init__
            _orig_cls_execute = _pymodbus_tcp.ModbusTcpClient.execute
            modbus_timeout = self.modbus_timeout

            def _rtu_init(self_client, host, port=502, **kwargs):
                kwargs.setdefault("framer", FramerType.RTU)
                # pyrobotiqgripper passes timeout=1; UR socket bridges need longer.
                kwargs["timeout"] = modbus_timeout
                _orig_cls_init(self_client, host, port=port, **kwargs)

            def _resilient_execute(client, no_response_expected, request):
                fc = getattr(request, "function_code", None)
                # FC16 write: UR bridge processes the write then closes without
                # sending a Modbus response — skip waiting for one.
                if fc == 0x10:
                    no_response_expected = True
                for attempt in range(3):
                    try:
                        if not client.connected:
                            client.connect()
                        result = _orig_cls_execute(client, no_response_expected, request)
                        # Close after FC16 so stale response bytes don't pollute
                        # the next FC03/FC04 read on the same socket.
                        if fc == 0x10:
                            try:
                                client.close()
                            except Exception:
                                pass
                            # pymodbus returns None when no_response_expected=True;
                            # pyrobotiqgripper checks res.count and res.isError() so
                            # return a minimal mock that looks like a successful write.
                            if result is None:
                                result = SimpleNamespace(count=1, registers=[], isError=lambda: False)
                        return result
                    except (_ModbusConnErr, _ModbusIOErr):
                        try:
                            client.close()
                        except Exception:
                            pass
                        if attempt == 2:
                            raise

            _pymodbus_tcp.ModbusTcpClient.__init__ = _rtu_init
            _pymodbus_tcp.ModbusTcpClient.execute = _resilient_execute
            try:
                host, port = self.robot_port.rsplit(":", 1)
                gripper = rq.RobotiqGripper(
                    device_id=self.device_id,
                    connection_type="RTU_VIA_TCP",
                    tcp_host=host,
                    tcp_port=int(port),
                )
            finally:
                # Always restore the class-level patches.
                _pymodbus_tcp.ModbusTcpClient.__init__ = _orig_cls_init
                _pymodbus_tcp.ModbusTcpClient.execute = _orig_cls_execute
                # Transfer the execute patch to the instance only if construction
                # succeeded (gripper is not None).
                if gripper is not None:
                    gripper._client.execute = partial(_resilient_execute, gripper._client)

        elif self.connection_type == "TCPIP":
            tcp_host = self.robot_port
            tcp_port = 54321
            if ":" in self.robot_port:
                tcp_host, tcp_port = self.robot_port.rsplit(":", 1)
                tcp_port = int(tcp_port)
            gripper = RobotiqTcpDriver(
                robot_ip=tcp_host,
                tcp_port=tcp_port,
                freq=self.freq,
                timeout=self.modbus_timeout,
            )
            gripper.start()
        else:
            raise ValueError(f"Unknown connection_type: {self.connection_type}")

        if self.connection_type != "TCPIP":
            gripper.connect()
            gripper.activate()
            if self.calibrate_speed:
                # gripper.calibrate_bit()  # not needed since calibrate_speed() will also perform bit calibration
                gripper.calibrate_speed()
            else:
                gripper.calibrate_bit()
            gripper.calibrate_mm(closemm=self.gripper_range[0], openmm=self.gripper_range[1])
            # gripper.start()  # not needed since activate(start=True)
            if self.home_to_open:
                gripper.open(wait=True)

        try:
            if self.robot_controller == RobotController.TASK_POS:
                curr_pos = gripper.state()["gripper_position"] if self.connection_type == "TCPIP" else 1 - gripper.position() / 255
                if self.max_gripper_speed is not None:
                    # joint interpolation
                    curr_time = time.now()
                    last_waypoint_time = curr_time
                    pose_interp = PoseTrajectoryInterpolator(times=[curr_time], poses=[[curr_pos, 0, 0, 0, 0, 0]])
                    # joint filtering/smoothing
                    lowpass_filter = LowPassFilter(alpha=self.gripper_lowpass_alpha, initial=curr_pos)
                else:
                    target_pos = np.copy(curr_pos)
                    pose_interp = None
                    lowpass_filter = None
            else:
                raise ValueError(self.robot_controller)

            dt = 1.0 / self.freq
            rate = time.Rate(self.freq)
            self.req_ready_event.set()
            not_pub_ready = True
            while not self.exit_event.is_set():
                t_now = time.now()
                if self.robot_controller == RobotController.TASK_POS:
                    if pose_interp is not None:
                        pos_command = pose_interp(t_now)[0]
                        pos_command = lowpass_filter(pos_command)
                    else:
                        pos_command = np.copy(target_pos)
                    pos_command = max(0.0, min(1.0, float(pos_command)))
                    if self.connection_type == "TCPIP":
                        gripper.moveG(pos_command)
                    elif self.calibrate_speed:
                        gripper.realTimeMove(
                            int(255 - pos_command * 255),
                            minimalMotion=0,
                            continuousGrip=False,
                            autoLock=False,
                            objectDetectionDuration=0.0,
                        )
                    else:
                        gripper.move(int(255 - pos_command * 255), speed=255, force=255, wait=False)
                else:
                    raise ValueError(self.robot_controller)

                if self.connection_type == "TCPIP":
                    robot_state = gripper.state()
                else:
                    pos = 1 - gripper.position(refreshStatus=False) / 255
                    robot_state = {"gripper_position": pos}

                data = {
                    **robot_state,
                    "timestamp": time.now(),
                }
                self.ring_buffer.put(data)
                if not_pub_ready:
                    self.pub_ready_event.set()
                    not_pub_ready = False

                try:
                    reqs = self.request_queue.get_all()
                    if isinstance(reqs, dict):
                        reqs = [{k: reqs[k][i] for k in reqs.keys()} for i in range(len(reqs["type"]))]
                except queue.Empty:
                    reqs = []
                for r in reqs:
                    req = Request(RequestType(r.pop("type")), r)
                    if req.type == RequestType.MOVEG:
                        target_pos = np.array(req.params["target_pos"], dtype=self.dtype)[0]
                        target_time = float(req.params["target_time"])
                        if pose_interp is not None:
                            curr_time = t_now + dt
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=[target_pos, 0, 0, 0, 0, 0],
                                time=target_time,
                                max_pos_speed=self.max_gripper_speed,
                                max_rot_speed=self.max_gripper_speed,
                                curr_time=curr_time,
                                last_waypoint_time=last_waypoint_time,
                            )
                            last_waypoint_time = target_time
                    else:
                        raise ValueError(req.type)
                rate.precise_sleep()
        except KeyboardInterrupt:
            pass
        finally:
            if gripper is not None:
                try:
                    gripper.stop()
                except (OSError, ConnectionError, Exception):
                    pass
                if self.connection_type != "TCPIP":
                    try:
                        gripper.disconnect()
                    except (OSError, ConnectionError, Exception):
                        pass

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def moveG(self, target_pos, target_time):
        target_pos = np.array(target_pos, dtype=self.dtype)
        assert target_pos.shape == (1,)
        assert target_time > time.now()
        req = {
            "type": RequestType.MOVEG.value,
            "target_pos": target_pos,
            "target_time": target_time,
        }
        self.request_queue.put(req)


def RobotiqGripperServer(mw, *args, **kwargs):
    return ServerFactory(mw, RobotiqGripper, *args, **kwargs)


def RobotiqGripperClient(mw, *args, **kwargs):
    return ClientFactory(mw, RobotiqGripper, *args, **kwargs)
