# Kassow

Tested: KR-series 7-DOF arm, Ubuntu 22.04, `kord-api` 3.0.2.

Reference: [KORD API](https://gitlab.com/kassowrobots/kord-api),
[KORD API docs](https://kassowrobots.gitlab.io/kord-api-doc/).

The integration has three layers:

1. The KORD CBun running on the robot controller.
2. Kassow's C++ `kord-api` library, vendored as a submodule.
3. `kord_bridge`, which exposes `kord-api` to Python as `_kord_bridge` and runs
   the `waitSync()` control loop in a dedicated C++ thread at 250 Hz.

`KassowArm` sits on top and streams `moveL` / `moveJ` waypoints as
`OT_VIAPOINT` motions, so the arm blends through targets instead of stopping at
each one.

## 1. Install the KORD CBun on the robot

Install and activate a KORD CBun compatible with both the controller software
and the KORD API version, using Kassow's
[Master CBun compatibility table](https://gitlab.com/kassowrobots/kord-api/-/wikis/Master-CBun):
download the matching `.cbun`, copy it to a USB drive, then install and activate
it through the CBun Manager on the controller.

Do not assume the newest CBun is compatible. A CBun/API protocol mismatch stops
the client connecting at all.

## 2. Install system dependencies

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    libeigen3-dev \
    libncurses-dev
```

The bridge asks for `SCHED_FIFO` priority and runs without it, but scheduling is
noticeably better with a real-time kernel and the matching user permissions.

## 3. Fetch `kord-api`

```bash
git submodule update --init kord_bridge/kord-api
```

To build against an existing checkout instead, pass
`-DKORD_API_DIR=/path/to/kord-api` when configuring CMake below.

## 4. Build the Python bridge

Run this from the `rio` checkout so the extension lands in the environment that
teleoperation actually uses. `nanobind` must be installed in that same
environment, and CMake must be pointed at its interpreter, or it will pick up a
system Python where `nanobind` is missing.

```bash
uv sync
uv pip install nanobind

cmake -S ../rio-hw/kord_bridge -B ../rio-hw/kord_bridge/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build ../rio-hw/kord_bridge/build --target _kord_bridge -j"$(nproc)"
cmake --install ../rio-hw/kord_bridge/build --component kord_bridge
```

Verify it:

```bash
uv run python -c "import _kord_bridge; print(_kord_bridge.__file__)"
```

The extension is built for one specific Python minor version and embeds an
RPATH into its build tree, so rebuild it after changing Python versions or
moving the checkout. If CMake was configured against the wrong interpreter,
delete `../rio-hw/kord_bridge/build` and configure again.

For Spacemouse control, install and start `spacenavd`:

```bash
sudo ./scripts/setup/spacemouse.sh   # from the rio-hw checkout
systemctl status spacenavd
```

## 5. Configure the network

Connect the workstation to the controller directly or over the same Ethernet
network, with different addresses in the same subnet. `KassowStation` defaults
to robot address `192.168.1.44`, KORD port `7582` and session id `1`.

```bash
nmcli device status
sudo ip addr add 192.168.1.43/24 dev <ethernet-interface>
ping 192.168.1.44
```

Do not assign the robot's own address to the workstation.

## 6. Teleoperate

From the `rio` checkout:

```bash
STATION=KassowStation uv run -m examples.teleop_eef
```

Override the robot address, or drive it from the keyboard instead:

```bash
STATION=KassowStation uv run -m examples.teleop_eef --arm-cfg.robot-ip 192.168.1.44
STATION=KassowStation uv run -m examples.teleop_eef --teleop Keyboard
```

Recordings land in `data/pick_and_place/` as `.vla` trajectories. Set
`--instruction` to label them, and add `Camera` entries to `KassowStation` to
capture video.

Joint streaming instead of Cartesian:

```bash
STATION=KassowStation uv run -m examples.teleop_eef \
    --arm-cfg.robot-controller joint_pos --action-space joint_pos
```

## Tuning Cartesian speed

`max_pos_speed` and `max_rot_speed` are the speed knobs. Raise them in steps,
watching the peak the node reports at startup:

```
KassowArm: streamL TT_TIME 0.100s, BT_TIME 0.070s, peak commanded 0.225 m/s and 0.375 rad/s
KassowArm: speed ceiling 0.150 m/s and 0.250 rad/s (envelope 22.5 mm, step guard 22.5 mm)
```

| | `max_pos_speed` | `max_rot_speed` |
| --- | --- | --- |
| Conservative | 0.08 | 0.20 |
| Default | 0.15 | 0.25 |
| Next step | 0.20 | 0.30 |
| Aggressive | 0.30 | 0.40 |

Translation has room: the stock `max_ws_speed` is 2.0 m/s. Rotation is the
tighter constraint, since `max_ws_orientation_speed` is 1.0 rad/s and KORD's
estimate is conservative, so back `max_rot_speed` off first if alarms appear.

Do **not** shorten `stream_l_tt` to go faster. Under `TT_TIME` it is a deadline,
so shortening it compresses the whole trajectory and inflates the acceleration
KORD estimates, which trips `INFEASIBLE_MOVE_COMMAND` or a torque-deviation
fault. Keep it near `1 / freq`.

If motion feels laggy rather than slow, the command period is the cause, not the
speed. Raise `freq` and drop `stream_l_tt` to match, which halves both the
latency and the per-command step at the same speed:

```bash
STATION=KassowStation uv run -m examples.teleop_eef \
    --freq 20 --arm-cfg.stream-l-tt 0.05 --arm-cfg.stream-l-bt 0.035
```

`freq` also sizes the node's guards, so change it there rather than on the arm
config; `KassowStation` keeps `arm_cfg.cmd_freq` in sync automatically.

### Tracking modes

`stream_l_mode` picks the KORD tracking type, and the two fail in mirror-image
ways:

- **`time` (default, `TT_TIME`)** pins the movement duration, which puts a floor
  under every speed and acceleration estimate the controller makes. Recommended.
- **`speed` (`TT_WS_TARGET_SPEED`)** makes `stream_l_speed` the TCP speed in m/s
  and lets KORD derive the duration, so speed no longer depends on `freq` or on
  how far ahead the target sits. That is more robust to a jittery loop, but
  deriving duration from `distance / speed` puts no *lower* bound on it: the
  sub-millimetre steps a Spacemouse produces at low deflection imply a near-zero
  duration and get rejected. The node holds targets back until they accumulate
  past `stream_l_speed × stream_l_min_track_time`, at the cost of moving in
  discrete hops during fine motion. Prefer `time` for precise work.

### Guards

`KassowArm` conditions every target before it reaches KORD. All three guards
derive from the speed ceilings, so raising `max_pos_speed` widens them together
and no fixed value can silently become the real speed limit.

| Guard | Role |
| --- | --- |
| Envelope | How far the commanded pose may lead the measured pose. |
| Step guard | Per-command jump limit, so via-points stay continuous. |
| Minimum step | Speed mode only. Shortest duration a move may imply. |

Joint targets get the same treatment from `max_motor_speed`: the first `moveJ`
after connecting is anchored on the measured position, so it can only advance
one step from where the arm actually is regardless of what the caller sent.
Joint positions are otherwise **not** clipped. The controller enforces its own
position limits and does not report them over KORD, and the roll axes travel
well past the ±170° that would look like a safe guess, so clipping to a guessed
range would reject a perfectly valid pose and command a lurch of a radian or
more. Set `joint_limits` explicitly if you know the limits for your model.

The envelope flips meaning between modes, which is worth understanding before
overriding `max_eef_delta_pos` / `_rot`:

- Under `TT_TIME` the robot closes the whole remaining gap within
  `stream_l_tt`, so the envelope **is** the peak commanded speed
  (`envelope / stream_l_tt`). A deep envelope does not buy smoothness, it
  multiplies the speed. A 0.18 rad envelope at `stream_l_tt 0.10` commands
  1.8 rad/s, already above the stock 1.0 rad/s limit.
- Under `TT_WS_TARGET_SPEED` the gap does not set the speed at all, so the
  target needs to lead by several cycles and too tight an envelope caps the
  speed instead.

## Reading alarms

A rejected or infeasible move surfaces only as a system alarm; without decoding
it the robot just looks like it stopped responding. `KassowArm` logs the
condition on each transition and pins its target to the measured pose until the
alarm clears, so the arm cannot lurch toward a stale target on recovery. The raw
word is also published as `alarm_code` in the arm state.

```
KassowArm: KORD alarm, motion blocked: 0x000bba42 SoftStopEvent/INFEASIBLE_MOVE_COMMAND [SSTOP, recoverable]
```

| Condition | Meaning |
| --- | --- |
| `INFEASIBLE_MOVE_COMMAND` | Commanded speed or acceleration exceeded `[MOTION_CONSTRAINTS]`. Lower the speed or lengthen the implied duration. |
| `JTORQUE_DEVIATION_EXCEEDED` | Expected and measured joint torques diverged, usually too aggressive an acceleration. |
| `JREF_POSITION_DELTA_SPAN` | Consecutive commanded joint positions too far apart, i.e. the target jumped. |
| `MODEL_X_TRJ_REFW_SSPAN_EXC` | Commanded TCP ran too far ahead of the model; shrink the envelope. |
| `EXTERNAL_ESTOP_ACTIVATED` / `EXTERNAL_PSTOP_ACTIVATED` | Physical stop, not a tuning problem. |

Full tables are in `kord_bridge/kord-api/docs/guides/safety/handle_alarms.rst`.
The absolute ceiling is the controller's own `[MOTION_CONSTRAINTS]` section in
`KORD.ini`; commands above it never reach the controller.

Set `arm_cfg.log_diagnostics` to report the achieved sync and command rates
every two seconds, which is the quickest way to tell a throttled command stream
from a controller that is rejecting moves:

```
KassowArm: sync 250.1 Hz | moveJ 0.0 Hz | moveL 9.8 Hz
```

Keep the emergency stop accessible and raise speeds incrementally.

## Troubleshooting

**`kord-api not found`** — the submodule was not initialised. Run
`git submodule update --init kord_bridge/kord-api`, or pass `-DKORD_API_DIR`.

**`No module named nanobind`** — CMake picked an interpreter other than the
project environment. Install `nanobind` there, delete
`kord_bridge/build`, and configure again with `-DPython_EXECUTABLE`.

**`_kord_bridge extension not found`** — the extension was never installed, or
was built for a different Python minor version. Rebuild and reinstall it.

**`KassowArm: failed to connect`** — check that the robot answers `ping`, the
KORD CBun is installed and active, the CBun and API revisions are compatible,
the IP / port / session id match, and no other KORD client holds the session.
