import json
import time as _stdlib_time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import time
from ..middleware import ClientFactory, ServerFactory
from ..node import Node

try:
    from manus_glove import HandMotion, ManusDataPublisher
except ImportError as e:
    if TYPE_CHECKING:
        raise e
    else:
        HandMotion = None  # type: ignore
        ManusDataPublisher = None  # type: ignore


# ---------------------------------------------------------------------------
# Skeleton feature extraction
# Adapted from glove_to_inspire_clean.py
# ---------------------------------------------------------------------------

_JOINT_ORDER = {"MCP": 0, "PIP": 1, "IP": 2, "DIP": 3, "TIP": 4}


def _chain(raw: list[dict[str, Any]], chain: str) -> list[dict[str, Any]]:
    ns = [n for n in raw if n.get("chainType") == chain]
    ns.sort(key=lambda n: _JOINT_ORDER.get(str(n.get("jointType", "")), 99))
    return ns


def _pick(raw: list[dict[str, Any]], chain: str, joints: tuple[str, ...]) -> np.ndarray | None:
    for j in joints:
        for n in raw:
            if n.get("chainType") == chain and str(n.get("jointType")) == j:
                return np.asarray(n["position"], float)
    return None


def _chord_open(ns: list[dict[str, Any]]) -> float:
    """Ratio of end-to-end chord to total chain arc length; 1 = fully extended."""
    if len(ns) < 2:
        return 0.5
    p = [np.asarray(n["position"], float) for n in ns]
    arc = sum(float(np.linalg.norm(p[i + 1] - p[i])) for i in range(len(p) - 1))
    if arc < 1e-9:
        return 0.5
    return float(np.clip(np.linalg.norm(p[-1] - p[0]) / arc, 0.0, 1.0))


def _thumb_chain(ns: list[dict[str, Any]], chord_w: float) -> float:
    if len(ns) < 3:
        return _chord_open(ns)
    p = [np.asarray(n["position"], float) for n in ns]
    sc: list[float] = []
    for i in range(len(p) - 2):
        u, v = p[i + 1] - p[i], p[i + 2] - p[i + 1]
        lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        if lu < 1e-9 or lv < 1e-9:
            continue
        c = float(np.clip(np.dot(u, v) / (lu * lv), -1.0, 1.0))
        sc.append((c + 1.0) * 0.5)
    if not sc:
        return _chord_open(ns)
    bend = 0.5 * min(sc) + 0.5 * float(np.mean(sc))
    w = float(np.clip(chord_w, 0.0, 1.0))
    return (1.0 - w) * bend + w * _chord_open(ns)


def _thumb_flex_01(raw: list[dict[str, Any]], chord_w: float) -> float:
    """Thumb flex openness in [0, 1]; 1 = fully extended away from palm."""
    tip = _pick(raw, "Thumb", ("TIP", "DIP", "IP"))
    mcp = _pick(raw, "Thumb", ("MCP",))
    if tip is None or mcp is None:
        return _thumb_chain(_chain(raw, "Thumb"), chord_w)
    mcps = [_pick(raw, ch, ("MCP",)) for ch in ("Index", "Middle", "Ring", "Pinky")]
    mcps = [x for x in mcps if x is not None]
    if len(mcps) < 2:
        return _thumb_chain(_chain(raw, "Thumb"), chord_w)
    palm = np.mean(np.stack(mcps, axis=0), axis=0)
    u, v = tip - mcp, palm - mcp
    lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if lu < 1e-9 or lv < 1e-9:
        return _thumb_chain(_chain(raw, "Thumb"), chord_w)
    c = float(np.clip(np.dot(u / lu, v / lv), -1.0, 1.0))
    palm_o = float(np.clip(1.0 - (c + 1.0) * 0.5, 0.0, 1.0))
    chain_o = _thumb_chain(_chain(raw, "Thumb"), chord_w)
    return float(np.clip(0.7 * palm_o + 0.3 * chain_o, 0.0, 1.0))


def _thumb_rot_01(raw: list[dict[str, Any]]) -> float:
    """Thumb rotation (abduction/adduction) in [0, 1]; 1 = fully abducted."""
    tip = _pick(raw, "Thumb", ("TIP", "DIP", "IP"))
    tm = _pick(raw, "Thumb", ("MCP",))
    im = _pick(raw, "Index", ("MCP",))
    mm = _pick(raw, "Middle", ("MCP",))
    if tip is None or tm is None or im is None or mm is None:
        return 0.5
    ex = im - mm
    n = float(np.linalg.norm(ex))
    if n < 1e-9:
        return 0.5
    ex /= n
    ez = np.cross(ex, tm - mm)
    nz = float(np.linalg.norm(ez))
    if nz < 1e-9:
        return 0.5
    ez /= nz
    ey = np.cross(ez, ex)
    v = tip - tm
    nv = float(np.linalg.norm(v))
    if nv < 1e-9:
        return 0.5
    v /= nv
    a = float(np.arctan2(float(np.dot(v, ey)), float(np.dot(v, ex))))
    return float(np.clip((a / np.pi + 1.0) * 0.5, 0.0, 1.0))


def _features(raw: list[dict[str, Any]], thumb_chord_w: float) -> np.ndarray:
    """Return 6-element openness vector: [pinky, ring, middle, index, thumb_flex, thumb_rot] in [0, 1]."""
    return np.array(
        [
            _chord_open(_chain(raw, "Pinky")),
            _chord_open(_chain(raw, "Ring")),
            _chord_open(_chain(raw, "Middle")),
            _chord_open(_chain(raw, "Index")),
            _thumb_flex_01(raw, thumb_chord_w),
            _thumb_rot_01(raw),
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Calibration
# Compatible with the JSON format used by glove_to_inspire_calibrated_clean.py
# ---------------------------------------------------------------------------

@dataclass
class GloveCalibration:
    """Per-user calibration mapping raw skeleton features to [0, 1]."""

    open_v: np.ndarray
    closed_v: np.ndarray
    thumb_chord_w: float
    thumb_flex_open: float | None = None
    thumb_flex_closed: float | None = None
    thumb_rot_open: float | None = None
    thumb_rot_closed: float | None = None

    @property
    def thumb_extra(self) -> bool:
        return all(
            v is not None
            for v in (self.thumb_flex_open, self.thumb_flex_closed, self.thumb_rot_open, self.thumb_rot_closed)
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "version": 2 if self.thumb_extra else 1,
            "thumb_chord_weight": self.thumb_chord_w,
            "open": self.open_v.astype(float).tolist(),
            "closed": self.closed_v.astype(float).tolist(),
        }
        if self.thumb_extra:
            out["thumb_flex"] = {"open": float(self.thumb_flex_open), "closed": float(self.thumb_flex_closed)}
            out["thumb_rot"] = {"open": float(self.thumb_rot_open), "closed": float(self.thumb_rot_closed)}
        return out

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "GloveCalibration":
        if int(d.get("version", 1)) not in (1, 2):
            raise ValueError(f"Unsupported calibration version {d.get('version')}")
        o = np.asarray(d["open"], dtype=np.float64).reshape(6)
        c = np.asarray(d["closed"], dtype=np.float64).reshape(6)
        w = float(d.get("thumb_chord_weight", 0.25))
        tfo = tfc = tro = trc = None
        tf, tr = d.get("thumb_flex"), d.get("thumb_rot")
        if isinstance(tf, dict) and isinstance(tr, dict):
            tfo, tfc = float(tf["open"]), float(tf["closed"])
            tro, trc = float(tr["open"]), float(tr["closed"])
        return cls(o, c, w, tfo, tfc, tro, trc)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GloveCalibration":
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))


def _apply_calibration(
    feat: np.ndarray,
    cal: GloveCalibration,
    curl: float = 1.2,
    thumb_curl: float = 1.45,
    invert: bool = False,
    thumb_swap: bool = False,
    thumb_flex_inv: bool = False,
    min_span: float = 0.08,
) -> np.ndarray:
    """Map raw skeleton features through calibration ranges to [0, 1]."""
    lo = cal.closed_v.astype(np.float64, copy=True)
    hi = cal.open_v.astype(np.float64, copy=True)
    if cal.thumb_extra:
        lo[4], hi[4] = cal.thumb_flex_closed, cal.thumb_flex_open
        lo[5], hi[5] = cal.thumb_rot_closed, cal.thumb_rot_open

    den = hi - lo
    span = np.sign(den) * np.maximum(np.abs(den), min_span)
    span = np.where(span == 0.0, min_span, span)
    x = np.clip((feat - lo) / span, 0.0, 1.0)

    if invert:
        x = 1.0 - x

    e = float(np.clip(curl, 1.0, 3.0))
    te = float(np.clip(thumb_curl, 1.0, 3.0))
    if e > 1.0:
        x[:4] = np.clip(x[:4] ** e, 0.0, 1.0)
    if te > 1.0:
        x[4] = float(np.clip(x[4] ** te, 0.0, 1.0))

    if thumb_swap:
        x[4], x[5] = float(x[5]), float(x[4])
    if thumb_flex_inv:
        x[4] = float(np.clip(1.0 - x[4], 0.0, 1.0))

    return x.astype(np.float64)


# ---------------------------------------------------------------------------
# ManusGlove Node
# ---------------------------------------------------------------------------


class ManusGlove(Node):
    """Publisher interface for a Manus haptic glove.

    Publishes per-frame finger openness features in [0, 1]:
        [pinky, ring, middle, index, thumb_flex, thumb_rot]

    If a calibration file is available (or auto_calibrate=True triggers interactive
    calibration), features are remapped through the recorded open/closed ranges before
    publishing.  The calibration JSON format is identical to glove_to_inspire_calibrated_clean.py,
    so existing .json files from that script work directly.
    """

    __api__ = [
        "get_state",
        "get_all_state",
    ]
    __pub__ = True
    __req__ = False

    def __init__(
        self,
        glove_id: int | None = None,
        hand_motion: str = "NoMotion",
        thumb_chord_weight: float = 0.25,
        # --- calibration ---
        calibration_file: str | None = None,
        auto_calibrate: bool = True,
        cal_samples: int = 45,
        cal_sample_dt: float = 0.02,
        no_thumb_endpoints: bool = False,
        # --- retarget parameters (same defaults as glove_to_inspire_calibrated_clean.py) ---
        curl: float = 1.2,
        thumb_curl: float = 1.45,
        invert: bool = False,
        thumb_swap_dofs: bool = False,
        thumb_flex_invert: bool = False,
        min_cal_span: float = 0.08,
        dtype=np.float32,
        *,
        freq: int = 100,
        max_buffer_size: int = 30,
        **kwargs,
    ):
        """
        Args:
            glove_id: Specific glove ID to use. If None, uses the lowest available ID.
            hand_motion: Manus SDK hand motion mode. One of "NoMotion" or "IMU".
            thumb_chord_weight: Blend weight for thumb flex (0 = pure bend-angle,
                1 = pure chord-ratio). Overridden by value stored in calibration file.
            calibration_file: Path to a JSON calibration file. If None, defaults to
                "manus_glove_cal.json" in the working directory when auto_calibrate=True.
            auto_calibrate: If True (default), run interactive calibration when no file
                is found. If False and no file is found, publishes raw features.
            cal_samples: Number of frames to average per calibration pose. Default 45.
            cal_sample_dt: Seconds between samples during calibration. Default 0.02.
            no_thumb_endpoints: If True, only record open + closed (2 poses). If False
                (default), also record 4 thumb-specific poses for better thumb accuracy.
            curl: Power-law exponent applied to finger openness (1 = linear). Default 1.2.
            thumb_curl: Power-law exponent for thumb flex. Default 1.45.
            invert: Invert all openness values (swap open ↔ closed direction). Default False.
            thumb_swap_dofs: Swap thumb_flex and thumb_rot channels. Default False.
            thumb_flex_invert: Invert only the thumb flex channel. Default False.
            min_cal_span: Minimum allowed calibration range per channel. Default 0.08.
            dtype: NumPy dtype for published arrays. Default np.float32.
            freq: Polling frequency in Hz. Default 100.
            max_buffer_size: Ring buffer capacity. Default 30.
        """
        self.glove_id = glove_id
        self.hand_motion = hand_motion
        self.thumb_chord_weight = thumb_chord_weight
        self.calibration_file = calibration_file
        self.auto_calibrate = auto_calibrate
        self.cal_samples = cal_samples
        self.cal_sample_dt = cal_sample_dt
        self.no_thumb_endpoints = no_thumb_endpoints
        self.curl = curl
        self.thumb_curl = thumb_curl
        self.invert = invert
        self.thumb_swap_dofs = thumb_swap_dofs
        self.thumb_flex_invert = thumb_flex_invert
        self.min_cal_span = min_cal_span
        self.dtype = dtype
        super().__init__(freq=freq, max_buffer_size=max_buffer_size, **kwargs)

    def __post_init__(self):
        self.example_request = None
        self.example_data = {
            "finger_openness": np.zeros(6, dtype=self.dtype),
            # joint_q and gripper_position mirror finger_openness for compatibility
            # with the rio SingleArm / teleop_leader_follower framework.
            "joint_q": np.zeros(6, dtype=self.dtype),
            "gripper_position": 0.0,
            "timestamp": time.now(),
        }
        self.worker = None
        self.run = self.pub
        super().__post_init__()

    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------

    def _sample_pose(self, publisher, gid: int) -> np.ndarray | None:
        """Collect cal_samples frames and return their mean feature vector."""
        buf: list[np.ndarray] = []
        for _ in range(max(self.cal_samples, 1)):
            data = publisher.GetGloveData(gid)
            if data is not None:
                raw = data.get("raw_nodes") or []
                if len(raw) >= 8:
                    buf.append(_features(raw, self.thumb_chord_weight))
            _stdlib_time.sleep(self.cal_sample_dt)
        if not buf:
            return None
        return np.mean(np.stack(buf, axis=0), axis=0)

    def _run_calibration(self, publisher, gid: int) -> GloveCalibration:
        """Interactive calibration: prompt user for open/closed (and thumb) poses."""
        ntot = 2 if self.no_thumb_endpoints else 6
        print(f"\n[ManusGlove] === CALIBRATION ({ntot} poses) ===", flush=True)
        print(f"[ManusGlove] thumb_chord_weight={self.thumb_chord_weight}", flush=True)

        input(f"[ManusGlove] 1/{ntot} — Hold hand FLAT and OPEN, then press Enter... ")
        open_v = self._sample_pose(publisher, gid)
        if open_v is None:
            raise RuntimeError("[ManusGlove] Calibration failed: no data for OPEN pose.")
        print(f"[ManusGlove]   open:   {np.round(open_v, 4).tolist()}", flush=True)

        input(f"[ManusGlove] 2/{ntot} — Make a FIST (fully closed), then press Enter... ")
        closed_v = self._sample_pose(publisher, gid)
        if closed_v is None:
            raise RuntimeError("[ManusGlove] Calibration failed: no data for CLOSED pose.")
        print(f"[ManusGlove]   closed: {np.round(closed_v, 4).tolist()}", flush=True)

        tfo = tfc = tro = trc = None
        if not self.no_thumb_endpoints:
            print("[ManusGlove] Thumb-only poses (keep other fingers relaxed).", flush=True)

            input(f"[ManusGlove] 3/{ntot} — Thumb EXTENDED along palm, press Enter... ")
            p = self._sample_pose(publisher, gid)
            if p is None:
                raise RuntimeError("[ManusGlove] Calibration failed: no data for thumb-flex-open pose.")
            tfo = float(p[4])
            print(f"[ManusGlove]   thumb flex open:   {tfo:.4f}", flush=True)

            input(f"[ManusGlove] 4/{ntot} — Thumb MAX CURL, press Enter... ")
            p = self._sample_pose(publisher, gid)
            if p is None:
                raise RuntimeError("[ManusGlove] Calibration failed: no data for thumb-flex-closed pose.")
            tfc = float(p[4])
            print(f"[ManusGlove]   thumb flex closed: {tfc:.4f}", flush=True)

            input(f"[ManusGlove] 5/{ntot} — Thumb MAX ABDUCTION (spread away from palm), press Enter... ")
            p = self._sample_pose(publisher, gid)
            if p is None:
                raise RuntimeError("[ManusGlove] Calibration failed: no data for thumb-rot-open pose.")
            tro = float(p[5])
            print(f"[ManusGlove]   thumb rot open:    {tro:.4f}", flush=True)

            input(f"[ManusGlove] 6/{ntot} — Thumb MAX ADDUCTION (pressed against palm), press Enter... ")
            p = self._sample_pose(publisher, gid)
            if p is None:
                raise RuntimeError("[ManusGlove] Calibration failed: no data for thumb-rot-closed pose.")
            trc = float(p[5])
            print(f"[ManusGlove]   thumb rot closed:  {trc:.4f}", flush=True)

        cal = GloveCalibration(open_v, closed_v, self.thumb_chord_weight, tfo, tfc, tro, trc)
        print("[ManusGlove] === CALIBRATION COMPLETE ===\n", flush=True)
        return cal

    def _resolve_calibration(self, publisher, gid: int) -> GloveCalibration | None:
        """Load calibration from file, or run interactive calibration if needed."""
        # Determine the target path.
        if self.calibration_file is not None:
            cal_path = Path(self.calibration_file).expanduser()
        elif self.auto_calibrate:
            cal_path = Path("manus_glove_cal.json")
        else:
            return None

        if cal_path.is_file():
            print(f"[ManusGlove] Loading calibration from {cal_path.resolve()}", flush=True)
            cal = GloveCalibration.load(cal_path)
            print(
                f"[ManusGlove]   open:   {np.round(cal.open_v, 4).tolist()}\n"
                f"[ManusGlove]   closed: {np.round(cal.closed_v, 4).tolist()}\n"
                f"[ManusGlove]   thumb_chord_w={cal.thumb_chord_w}  thumb_extra={cal.thumb_extra}",
                flush=True,
            )
            return cal

        if self.auto_calibrate:
            print(f"[ManusGlove] No calibration file at {cal_path.resolve()}. Running interactive calibration...", flush=True)
            cal = self._run_calibration(publisher, gid)
            cal.save(cal_path)
            print(f"[ManusGlove] Calibration saved to {cal_path.resolve()}", flush=True)
            return cal

        print(f"[ManusGlove] WARNING: calibration file not found at {cal_path.resolve()} and auto_calibrate=False. Running uncalibrated.", flush=True)
        return None

    # ------------------------------------------------------------------
    # Main publish loop
    # ------------------------------------------------------------------

    def pub(self):
        print("[ManusGlove] Initializing SDK...", flush=True)
        publisher = ManusDataPublisher(hand_motion=HandMotion[self.hand_motion], debug=False)
        publisher.Initialize()
        publisher.Connect()

        try:
            print("[ManusGlove] Waiting for landscape...", flush=True)
            while publisher.GetLandscape() is None:
                _stdlib_time.sleep(0.05)
            print("[ManusGlove] Landscape ready. Loading SDK calibration files...", flush=True)
            publisher.LoadCalibrationFiles()

            print("[ManusGlove] Waiting for glove...", flush=True)
            gid = None
            while gid is None:
                ids = publisher.GetGloveIds()
                if ids:
                    gid = self.glove_id if (self.glove_id is not None and self.glove_id in ids) else min(ids)
                if gid is None:
                    _stdlib_time.sleep(0.05)
            print(f"[ManusGlove] Using glove id={gid}.", flush=True)

            # Load or run calibration (may prompt user interactively).
            cal = self._resolve_calibration(publisher, gid)
            # When calibration is loaded from file, use its thumb_chord_weight.
            thumb_chord_w = cal.thumb_chord_w if cal is not None else self.thumb_chord_weight
            if cal is not None:
                print(
                    f"[ManusGlove] Retarget: curl={self.curl} thumb_curl={self.thumb_curl} "
                    f"invert={self.invert} thumb_swap={self.thumb_swap_dofs} "
                    f"thumb_flex_inv={self.thumb_flex_invert} min_span={self.min_cal_span}",
                    flush=True,
                )
            else:
                print("[ManusGlove] No calibration — publishing raw features.", flush=True)

            print(f"[ManusGlove] Starting publish loop at {self.freq} Hz.", flush=True)
            rate = time.Rate(self.freq)
            not_pub_ready = True
            _iter = 0
            _no_data_warned = False

            while not self.exit_event.is_set():
                # Refresh active glove ID in case landscape changed.
                ids = publisher.GetGloveIds()
                if ids:
                    if self.glove_id is not None and self.glove_id in ids:
                        gid = self.glove_id
                    else:
                        gid = min(ids)

                data = publisher.GetGloveData(gid) if gid is not None else None
                if data is not None:
                    _no_data_warned = False
                    raw = data.get("raw_nodes") or []
                    feat = _features(raw, thumb_chord_w)

                    if cal is not None:
                        openness = _apply_calibration(
                            feat,
                            cal,
                            curl=self.curl,
                            thumb_curl=self.thumb_curl,
                            invert=self.invert,
                            thumb_swap=self.thumb_swap_dofs,
                            thumb_flex_inv=self.thumb_flex_invert,
                            min_span=self.min_cal_span,
                        ).astype(self.dtype)
                    else:
                        openness = feat.astype(self.dtype)

                    self.ring_buffer.put(
                        {
                            "finger_openness": openness,
                            "joint_q": openness,
                            "gripper_position": 0.0,
                            "timestamp": time.now(),
                        }
                    )
                    if not_pub_ready:
                        self.pub_ready_event.set()
                        not_pub_ready = False

                    # Status line every ~2 s.
                    if _iter % 200 == 0:
                        cal_tag = "cal" if cal is not None else "raw"
                        np.set_printoptions(precision=3, suppress=True)
                        # print(
                        #     f"[ManusGlove] glove={gid} nodes={len(raw)} [{cal_tag}] "
                        #     f"openness(P,R,M,I,Tf,Tr)={np.array2string(openness, separator=',')}",
                        #     flush=True,
                        # )
                else:
                    if not _no_data_warned:
                        print(f"[ManusGlove] WARNING: GetGloveData returned None for glove id={gid}", flush=True)
                        _no_data_warned = True

                _iter += 1
                rate.precise_sleep()
        except KeyboardInterrupt:
            pass
        finally:
            print("[ManusGlove] Shutting down SDK.", flush=True)
            publisher.ShutDown()

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()


def ManusGloveServer(mw, *args, **kwargs):
    return ServerFactory(mw, ManusGlove, *args, **kwargs)


def ManusGloveClient(mw, *args, **kwargs):
    return ClientFactory(mw, ManusGlove, *args, **kwargs)
