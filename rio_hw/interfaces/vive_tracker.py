"""Publisher interface for an HTC Vive Tracker read through SteamVR / OpenVR.

SteamVR handles base-station calibration and multi-lighthouse fusion, so this
node only enumerates devices and republishes the tracker pose. No headset is
required; SteamVR can run with the null driver.
"""

import time as _stdlib_time
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import time
from ..middleware import ClientFactory, ServerFactory
from ..node import Node
from .vive_basestations import ensure_distinct_base_channels
from .vive_retarget import matrix_to_rotvec

try:
    import openvr
except ImportError as e:
    if TYPE_CHECKING:
        raise e
    else:
        openvr = None  # type: ignore


def device_serial(vr_system: Any, index: int) -> str:
    try:
        return vr_system.getStringTrackedDeviceProperty(index, openvr.Prop_SerialNumber_String)
    except openvr.OpenVRError:
        return f"device-{index}"


def find_devices(vr_system: Any, device_class: int) -> list[int]:
    return [
        index
        for index in range(openvr.k_unMaxTrackedDeviceCount)
        if vr_system.isTrackedDeviceConnected(index)
        and vr_system.getTrackedDeviceClass(index) == device_class
    ]


def find_trackers(vr_system: Any) -> list[int]:
    return find_devices(vr_system, openvr.TrackedDeviceClass_GenericTracker)


def find_base_stations(vr_system: Any) -> list[int]:
    return find_devices(vr_system, openvr.TrackedDeviceClass_TrackingReference)


def pose_matrix(matrix: Any) -> np.ndarray:
    """Convert an OpenVR 3x4 pose matrix to a 4x4 homogeneous numpy matrix."""
    result = np.eye(4)
    for row in range(3):
        for col in range(4):
            result[row, col] = matrix[row][col]
    return result


class ViveTracker(Node):
    """Publisher interface for a Vive Tracker pose from a running SteamVR runtime.

    Publishes the tracker pose in the SteamVR standing frame (Y up, -Z away
    from the user) as a 6-vector, matching the position + axis-angle layout
    that rio uses for eef_pose:

        tracker_pose: [x, y, z, rx, ry, rz]
        pose_valid:   1.0 while SteamVR reports a valid pose, else 0.0

    Consumers should ignore the pose whenever pose_valid is 0.0; the last good
    pose is republished so downstream filters see a continuous signal.
    """

    __api__ = [
        "get_state",
        "get_all_state",
        "get_tracker_pose",
        "is_pose_valid",
    ]
    __pub__ = True
    __req__ = False

    def __init__(
        self,
        serial: str | None = None,
        fix_base_channels: bool = True,
        base_scan_timeout: float = 8.0,
        dtype=np.float32,
        *,
        freq: int = 250,
        max_buffer_size: int = 30,
        **kwargs,
    ):
        """
        Args:
            serial: SteamVR serial (e.g. "LHR-XXXXXXXX") of the tracker to follow.
                Defaults to the first tracker SteamVR reports.
            fix_base_channels: If OpenVR sees fewer than two base stations, scan
                Bluetooth and retune Base Station 2.0 units that share an RF
                channel. Free when both bases are already visible; otherwise it
                costs one Bluetooth scan at startup, so raise ``timeout``
                accordingly.
            base_scan_timeout: Bluetooth scan duration for the channel check.
        """
        self.serial = serial
        self.fix_base_channels = fix_base_channels
        self.base_scan_timeout = base_scan_timeout
        self.dtype = dtype
        super().__init__(freq=freq, max_buffer_size=max_buffer_size, **kwargs)

    def __post_init__(self):
        self.example_request = None
        self.example_data = {
            # 3 translation, 3 axis-angle rotation, in the SteamVR standing frame
            "tracker_pose": np.zeros((6,), dtype=self.dtype),
            "pose_valid": 0.0,
            "timestamp": time.now(),
        }
        self.worker = None
        self.run = self.pub
        super().__post_init__()

    def _connect(self):
        """Initialize OpenVR and resolve which tracker to follow."""
        if openvr is None:
            raise RuntimeError(
                "openvr is not installed. Install the vive extra: pip install 'rio_hw[vive]'"
            )

        vr_system = openvr.init(openvr.VRApplication_Background)

        bases = find_base_stations(vr_system)
        if bases:
            print(
                "[ViveTracker] Base stations: "
                + ", ".join(device_serial(vr_system, i) for i in bases),
                flush=True,
            )
        else:
            print("[ViveTracker] No base stations visible to OpenVR yet.", flush=True)

        if self.fix_base_channels and len(bases) < 2:
            try:
                changed = ensure_distinct_base_channels(
                    openvr_base_count=len(bases),
                    scan_timeout_s=self.base_scan_timeout,
                )
            except Exception as error:
                print(f"[ViveTracker] Base station check failed: {error}", flush=True)
            else:
                if changed:
                    # Give SteamVR a moment to enumerate the second base.
                    _stdlib_time.sleep(5.0)

        trackers = find_trackers(vr_system)
        if not trackers:
            openvr.shutdown()
            raise RuntimeError(
                "No Vive trackers in OpenVR. Start SteamVR, power the tracker, and pair it."
            )

        available = [device_serial(vr_system, index) for index in trackers]
        print(f"[ViveTracker] Trackers: {', '.join(available)}", flush=True)
        if self.serial and self.serial not in available:
            openvr.shutdown()
            raise RuntimeError(
                f"Tracker {self.serial!r} not found. Available: {', '.join(available)}"
            )

        label = self.serial or available[0]
        print(f"[ViveTracker] Following {label}", flush=True)
        return vr_system, label

    def pub(self):
        vr_system, label = self._connect()

        try:
            tracker_pose = np.zeros((6,), dtype=self.dtype)

            rate = time.Rate(self.freq)
            not_pub_ready = True
            while not self.exit_event.is_set():
                poses = vr_system.getDeviceToAbsoluteTrackingPose(
                    openvr.TrackingUniverseStanding,
                    0,
                    openvr.k_unMaxTrackedDeviceCount,
                )

                pose_valid = 0.0
                for index in find_trackers(vr_system):
                    if device_serial(vr_system, index) != label:
                        continue
                    pose = poses[index]
                    if not pose.bPoseIsValid:
                        break
                    matrix = pose_matrix(pose.mDeviceToAbsoluteTracking)
                    tracker_pose = np.concatenate(
                        [matrix[:3, 3], matrix_to_rotvec(matrix[:3, :3])]
                    ).astype(self.dtype)
                    pose_valid = 1.0
                    break

                # The last good pose is republished while tracking is lost so
                # downstream filters stay continuous; pose_valid says to ignore it.
                data = {
                    "tracker_pose": tracker_pose,
                    "pose_valid": pose_valid,
                    "timestamp": time.now(),
                }
                self.ring_buffer.put(data)
                if not_pub_ready:
                    self.pub_ready_event.set()
                    not_pub_ready = False
                rate.precise_sleep()
        except KeyboardInterrupt:
            pass
        finally:
            openvr.shutdown()

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def get_tracker_pose(self):
        """Latest tracker pose as [x, y, z, rx, ry, rz]."""
        return self.ring_buffer.get()["tracker_pose"]

    def is_pose_valid(self):
        """Whether SteamVR currently reports a valid pose for this tracker."""
        return bool(self.ring_buffer.get()["pose_valid"] > 0.5)


def ViveTrackerServer(mw, *args, **kwargs):
    return ServerFactory(mw, ViveTracker, *args, **kwargs)


def ViveTrackerClient(mw, *args, **kwargs):
    return ClientFactory(mw, ViveTracker, *args, **kwargs)
