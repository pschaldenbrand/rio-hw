"""Detect and fix SteamVR Base Station 2.0 channel conflicts over Bluetooth.

Two Base Station 2.0 units sharing an RF channel is the usual reason SteamVR
only tracks with one of them. The channel lives in base station firmware, so
this is a one-time repair; ``ensure_distinct_base_channels`` returns
immediately when OpenVR already sees enough bases and only pays the Bluetooth
scan when something is actually wrong.
"""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

try:
    from bleak import BleakClient, BleakScanner
except ImportError as e:
    if TYPE_CHECKING:
        raise e
    else:
        BleakClient = None  # type: ignore
        BleakScanner = None  # type: ignore

# Valve BS 2.0 control service (ends in ...124 on current firmware).
_CHANNEL_UUID = "00001524-1212-efde-1523-785feabcd124"


@dataclass(frozen=True)
class BaseStationBle:
    name: str
    address: str
    channel: int | None


async def _scan(timeout_s: float) -> list[tuple[str, str]]:
    devices = await BleakScanner.discover(timeout=timeout_s)
    found: list[tuple[str, str]] = []
    for device in devices:
        name = device.name or ""
        if name.startswith("LHB-"):
            found.append((name, device.address))
    found.sort(key=lambda item: item[0])
    return found


async def _read_channel(address: str) -> int:
    async with BleakClient(address, timeout=20.0) as client:
        value = await client.read_gatt_char(_CHANNEL_UUID)
        return int(value[0])


async def _write_channel(address: str, channel: int) -> int:
    async with BleakClient(address, timeout=20.0) as client:
        await client.write_gatt_char(_CHANNEL_UUID, bytes([channel]), response=True)
        await asyncio.sleep(0.5)
        value = await client.read_gatt_char(_CHANNEL_UUID)
        return int(value[0])


async def discover_base_stations(scan_timeout_s: float = 8.0) -> list[BaseStationBle]:
    """Scan for LHB-* base stations and read each RF channel."""
    found = await _scan(scan_timeout_s)
    results: list[BaseStationBle] = []
    for name, address in found:
        try:
            channel = await _read_channel(address)
        except Exception as error:
            print(f"[ViveTracker]   {name}: could not read channel ({error})", flush=True)
            results.append(BaseStationBle(name, address, None))
            continue
        results.append(BaseStationBle(name, address, channel))
    return results


def channels_conflict(stations: list[BaseStationBle]) -> bool:
    known = [s.channel for s in stations if s.channel is not None]
    return len(known) >= 2 and len(set(known)) < len(known)


async def fix_channel_conflicts(
    stations: list[BaseStationBle],
    *,
    preferred_channels: tuple[int, ...] = (1, 2, 3, 4),
) -> list[BaseStationBle]:
    """Assign unique channels to bases that share a channel.

    Leaves already-unique channels alone when possible. Returns the post-fix
    station list (re-read where writes happened).
    """
    if not channels_conflict(stations):
        return stations

    used: set[int] = set()
    updated: list[BaseStationBle] = []
    for station in stations:
        channel = station.channel
        if channel is None:
            updated.append(station)
            continue
        if channel not in used:
            used.add(channel)
            updated.append(station)
            continue

        replacement = next(
            (c for c in preferred_channels if c not in used),
            max(used, default=0) + 1,
        )
        print(
            f"[ViveTracker]   {station.name}: channel {channel} conflict, setting {replacement}",
            flush=True,
        )
        try:
            new_channel = await _write_channel(station.address, replacement)
        except Exception as error:
            print(f"[ViveTracker]   {station.name}: write failed ({error})", flush=True)
            updated.append(station)
            continue
        used.add(new_channel)
        updated.append(BaseStationBle(station.name, station.address, new_channel))
    return updated


async def _ensure_distinct_base_channels_async(
    *,
    openvr_base_count: int,
    scan_timeout_s: float,
    min_bases_for_fix: int,
) -> bool:
    if openvr_base_count >= min_bases_for_fix:
        return False

    print(
        f"[ViveTracker] Only {openvr_base_count} base station(s) in OpenVR; scanning "
        "Bluetooth for Base Station 2.0 channel conflicts...",
        flush=True,
    )
    stations = await discover_base_stations(scan_timeout_s)
    if not stations:
        print("[ViveTracker]   No LHB-* base stations found over Bluetooth.", flush=True)
        return False

    for station in stations:
        ch = "?" if station.channel is None else str(station.channel)
        print(f"[ViveTracker]   {station.name}  channel={ch}", flush=True)

    if len([s for s in stations if s.channel is not None]) < min_bases_for_fix:
        print("[ViveTracker]   Fewer than two readable bases over Bluetooth.", flush=True)
        return False

    if not channels_conflict(stations):
        print("[ViveTracker]   Channels already distinct.", flush=True)
        return False

    before = {(s.name, s.channel) for s in stations}
    fixed = await fix_channel_conflicts(stations)
    changed = before != {(s.name, s.channel) for s in fixed}
    if changed:
        print("[ViveTracker]   Channel fix applied.", flush=True)
    return changed


def ensure_distinct_base_channels(
    *,
    openvr_base_count: int,
    scan_timeout_s: float = 8.0,
    min_bases_for_fix: int = 2,
) -> bool:
    """If BLE sees multiple bases on the same channel, retune them.

    Returns True if any channel was changed. Requires the ``bleak`` extra.
    """
    if BleakScanner is None:
        print("[ViveTracker] bleak is not installed; skipping base station check.", flush=True)
        return False
    return asyncio.run(
        _ensure_distinct_base_channels_async(
            openvr_base_count=openvr_base_count,
            scan_timeout_s=scan_timeout_s,
            min_bases_for_fix=min_bases_for_fix,
        )
    )
