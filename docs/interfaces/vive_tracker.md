# Vive Tracker

Tested: HTC Vive Tracker 3.0 with two Base Station 2.0 units, SteamVR on Linux, no headset.

Reference: [SteamVR](https://store.steampowered.com/app/250820/SteamVR/), [`pyopenvr`](https://github.com/cmbruns/pyopenvr)

SteamVR handles base-station calibration and multi-lighthouse fusion, so the
`ViveTracker` node only enumerates devices and republishes the pose. It
publishes in the SteamVR standing frame (Y up, -Z away from the user) as a
6-vector matching rio's `eef_pose` layout:

| Key | Shape | Meaning |
| --- | --- | --- |
| `tracker_pose` | `(6,)` | `[x, y, z, rx, ry, rz]`, position in meters and axis-angle rotation |
| `pose_valid` | scalar | `1.0` while SteamVR reports a valid pose, else `0.0` |

The last good pose is republished while tracking is lost so downstream filters
stay continuous. Consumers must check `pose_valid` and stop commanding motion
when it drops.

## Install SteamVR

```bash
sudo apt update
sudo apt install steam
steam steam://install/250820
```

Use native Steam, not Flatpak or Snap. Launch SteamVR from Steam
**Library -> Tools**, or `steam steam://rungameid/250820`.

## Run without a headset

SteamVR needs a null HMD when no headset is attached. Fully quit SteamVR, then:

1. Enable the null driver in
   `~/.steam/debian-installation/steamapps/common/SteamVR/drivers/null/resources/settings/default.vrsettings`
   (or `~/.local/share/Steam/...`) by setting `"enable": true`.

2. In `~/.steam/debian-installation/config/steamvr.vrsettings`, make `"steamvr"` include:

```json
"requireHmd": false,
"forcedDriver": "null",
"activateMultipleDrivers": true
```

3. Power both base stations, plug in the tracker dongle, power the tracker.
4. Launch SteamVR and pair via **Devices -> Pair Controller** (tracker LED solid green).
5. Room setup is not required for tracker poses.
6. Leave SteamVR running while nodes are up.

## Base station channels

Two Base Station 2.0 units on the same RF channel is the usual reason only one
of them tracks. With `fix_base_channels=True` (the default), the node retunes
them over Bluetooth when OpenVR sees fewer than two bases. This costs nothing
when both bases are already visible, but a repair adds a Bluetooth scan to node
startup, so keep the node's `timeout` at 60 s or more.

The channel is stored in base station firmware, so this is a one-time repair.

## Retargeting

`rio_hw.interfaces.vive_retarget` holds the clutch retargeting used by
`examples/teleop_vive_hand.py`. It is pure numpy and can be exercised without
SteamVR or hardware.

Alignment comes from clutching rather than calibration. Engaging snapshots the
tracker pose and the current TCP pose together, and tracker motion is applied
relative to that pair. Position and orientation are mapped separately, so a
wrist twist rotates the tool without swinging the arm.

To align: hold your forearm the way you want the tool to point, engage the
clutch, then move. Release and re-engage to re-center, for example after
reaching the edge of your comfortable range.

### Yaw calibration

`AXIS_MAP_STEAMVR_TO_ROBOT` maps SteamVR (Y up) onto a robot base frame (Z up),
but it contains only 0s and ±1s, so it can express 90 degree steps and nothing
in between. That is rarely enough on its own.

SteamVR's standing frame is gravity-aligned, so its vertical axis already agrees
with the robot's and needs no tuning. What SteamVR cannot know is which
direction to call forward — that comes from room setup and base station
placement, which have no relationship to how the robot base is mounted. The
leftover error is therefore a single rotation about vertical, and it shows up as
motion that is consistently off by some intermediate angle.

`examples/teleop_vive_hand.py` solves for that angle by asking the operator to
sweep a hand along two robot axes. The path's dominant direction comes from its
first principal component rather than its endpoints, so start and stop jitter
does not dominate, and the two sweeps are checked against the 90 degrees they
should be apart. The result is written to `vive_yaw_cal.json` and reused, so
calibration happens once. Pass `--calibrate-yaw` to redo it.

Wrist orientation is irrelevant during calibration and during use: position
retargeting reads only tracker translation, and rotation retargeting applies
only the change since the clutch engaged. The tracker does need to be strapped
down rigidly, since a strap that slips reads as real motion.

Because the mapping lives in SteamVR's world frame, a calibrated yaw is correct
for the direction the operator was facing when they measured it. Calibrate
standing where you will actually work, and recalibrate if you relocate.

## FAQ

- SteamVR shows the tracker but the node reports no trackers: confirm a tracker
  icon is present rather than only the null HMD, then power-cycle the tracker
  and replug the dongle.
- `pose_valid` stays at 0: the tracker is paired but no base station can see
  it. Check base station power and line of sight.
- Only one base station appears after a channel fix: give SteamVR a few seconds
  to re-enumerate, or restart it.
