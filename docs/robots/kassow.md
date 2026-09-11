# Kassow

Tested: KR-series 7-DOF arm, Ubuntu 22.04, `kord-api` 3.0.2.

Reference: [KORD API](https://gitlab.com/kassowrobots/kord-api),
[KORD API docs](https://kassowrobots.gitlab.io/kord-api-doc/).

The integration has three layers:

1. The KORD CBun running on the robot controller.
2. Kassow's C++ `kord-api` library, vendored as a submodule.
3. `kord_bridge`, which exposes `kord-api` to Python as `_kord_bridge` and runs
   the `waitSync()` control loop in a dedicated C++ thread at 250 Hz.

`KassowArm` accepts Cartesian goals from Python at the teleop rate (default
100 Hz). The default controller is **`task_pos_ik`**: local Pinocchio IK on the
packaged KR1018 URDF, then joint-velocity tracking via `directJControl`. Legacy
`task_pos` still streams `OT_VIAPOINT` `moveL` (waitSync 250 Hz, send every 2nd
tick ~125 Hz with `TT_TIME=0.016` / `BT_TIME=0.008`).

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

From the `rio` checkout (needs Pinocchio via `rio_hw[robots]` / `pin`):

```bash
STATION=KassowStation uv run -m examples.teleop_eef
```

Default path: Spacemouse EEF deltas → Pinocchio IK (`rio_hw/assets/kassow/kr2_robot_S00V0000M1018.urdf`)
→ `VelCmd` / `directJControl`. You should see `moveL 0.0 Hz` in diagnostics while
the arm still tracks smoothly. Override speeds as needed:

```bash
STATION=KassowStation uv run -m examples.teleop_eef \
    --arm-cfg.max-pos-speed 0.15 --arm-cfg.max-rot-speed 0.45 \
    --arm-cfg.max-motor-speed 0.5 --arm-cfg.log-diagnostics --freq 100
```

Override the robot address, or drive it from the keyboard instead:

```bash
STATION=KassowStation uv run -m examples.teleop_eef --arm-cfg.robot-ip 192.168.1.44
STATION=KassowStation uv run -m examples.teleop_eef --teleop Keyboard
# On Wayland this auto-switches to SshKeyboard (stdin). Same keys: WASD/QE, IJKL/UO.
```

Legacy streamed `moveL` (no local IK):

```bash
STATION=KassowStation uv run -m examples.teleop_eef \
    --arm-cfg.robot-controller task_pos
```

Recordings land in `data/pick_and_place/` as `.vla` trajectories. Set
`--instruction` to label them, and add `Camera` entries to `KassowStation` to
capture video.

Joint streaming instead of Cartesian:

```bash
STATION=KassowStation uv run -m examples.teleop_eef \
    --arm-cfg.robot-controller joint_pos --action-space joint_pos
```

## Local IK (`task_pos_ik`)

Cartesian teleop keeps the `task_pos` action interface (recorded actions stay
EEF poses as RIO ``[xyz, rotvec]``). Inside `KassowArm`, each `moveL` target is
turned into a Cartesian twist ``v = clip(ik_kp * (target - measured))`` and mapped
to joint velocity with a damped Jacobian inverse (`directJControl` / VelCmd).
The measured→commanded lead is used so FK/TCP frame mismatch is not chased.
``qd`` is EMA-smoothed and slew-limited. Hands-off ramps ``qd`` to zero.

Tunables: `--arm-cfg.ik-kp`, `--arm-cfg.max-pos-speed`, `--arm-cfg.max-motor-speed`,
`--arm-cfg.lowpass-alpha`, `--arm-cfg.max-joint-accel`, `--arm-cfg.urdf-path`,
`--arm-cfg.ee-frame`.

Spacemouse axes for Kassow default to the lab cell map (device Z → robot +X,
device X → −Y, device Y → +Z). Override with
`--teleop-cfg.tx-zup-spnav` (length-9 row-major 3×3) if your cell differs.

## directJControl (joint velocity)

`joint_vel` does **not** use `moveL`. Python sends joint velocities; the C++ bridge
integrates them and calls `directJControl` on every `waitSync` (~250 Hz). That
path usually tolerates full-rate sends better than streamed `moveL`.

Spacemouse / Keyboard / Gamepad axes map onto joints 0–5 (joint 7 stays 0):

```bash
STATION=KassowStation uv run -m examples.teleop_joint_vel \
    --arm-cfg.max-motor-speed 0.3 \
    --arm-cfg.log-diagnostics \
    --teleop Spacemouse
# or: --teleop Keyboard
```

Start slow (`max_motor_speed` 0.2–0.3). Hands-off sends zero `qd` so the arm holds.

## Tuning Cartesian speed

Python teleop stays at `freq` (default 100 Hz with `KassowStation`). Speed is set
by `max_pos_speed` and `max_rot_speed`, which also size the C++ micro-steps
(`speed * throttle / 250` per send). Raise them in steps:

```
KassowArm: streamL micro-steps TT_TIME 0.016s, BT_TIME 0.008s, throttle=2 (~125 Hz while moving)
KassowArm: micro-step ceiling 1.20 mm / 2.00 mrad per tick at 0.150 m/s and 0.250 rad/s
KassowArm: Python goal envelope 2.3 mm / 3.8 mrad, step guard 2.3 mm (cmd_freq=100 Hz)
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

Do **not** shrink `stream_l_tt` below ~2× the send period, and do **not** send
large Python steps with a short TT. Without C++ interpolation that compresses
each jump into a few milliseconds and recreates torque spikes. Scale TT/BT with
throttle the way KORD documents (throttle 1 → 0.008/0.004, throttle 2 → 0.016/0.008)
and raise the speed ceilings instead.

If motion still feels stepped on the legacy `task_pos` path, raise teleop `freq`
further (e.g. 125–200) so goals update more often; the bridge still syncs at
250 Hz. `KassowStation` keeps `arm_cfg.cmd_freq` in sync with `freq` automatically.

```bash
STATION=KassowStation uv run -m examples.teleop_eef --freq 125
```

### Tracking modes

`stream_l_mode` picks the KORD tracking type on each micro-step:

- **`time` (default, `TT_TIME`)** — recommended. Each via-point has a fixed
  ~4 ms deadline; C++ keeps the step size small.
- **`speed` (`TT_WS_TARGET_SPEED`)** — `stream_l_speed` is the TCP speed in m/s
  and KORD derives the duration. Prefer `time` for teleop with micro-steps.

### Guards

`KassowArm` conditions every **Python goal** before it reaches the bridge.
Guards derive from the speed ceilings and the teleop period (`cmd_freq`), not
from the micro-step TT:

| Guard | Role |
| --- | --- |
| Envelope | How far the Python goal may lead the measured pose. |
| Step guard | Per-goal jump limit between teleop cycles. |
| Minimum step | Speed mode only. Shortest duration a move may imply. |
| C++ micro-step | Per-tick advance capped at `max_*_speed / 250`. |

Joint targets get the same treatment from `max_motor_speed`: the first `moveJ`
after connecting is anchored on the measured position, so it can only advance
one step from where the arm actually is regardless of what the caller sent.
Joint positions are otherwise **not** clipped. The controller enforces its own
position limits and does not report them over KORD, and the roll axes travel
well past the ±170° that would look like a safe guess, so clipping to a guessed
range would reject a perfectly valid pose and command a lurch of a radian or
more. Set `joint_limits` explicitly if you know the limits for your model.

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
| `MOVE_IN_INVALID_MOTION_STATE` | CBun ≥3.0.4 rejected a move because the controller is not ready (INIT/halt/suspend/brakes). Clear the state, then the alarm. |
| `JTORQUE_DEVIATION_EXCEEDED` | Expected and measured joint torques diverged, usually too aggressive an acceleration. |
| `JREF_POSITION_DELTA_SPAN` | Consecutive commanded joint positions too far apart, i.e. the target jumped. |
| `JREF_X_SENSOR_POSITION_SPAN` | Joint reference vs sensor span — often leftover ESTOP after a jerk, or incomplete recovery. Clear with `sudo kord-clean-alarm -c <ip> --all` (not bare `--halt --cbun`: only the last dedicated flag is applied). Pendant release may still be required. |
| `MODEL_X_TRJ_REFW_SSPAN_EXC` | Commanded TCP ran too far ahead of the model; shrink the envelope. |
| `EXTERNAL_ESTOP_ACTIVATED` / `EXTERNAL_PSTOP_ACTIVATED` | Physical stop, not a tuning problem. |

On connect, `KassowArm` runs `CLEAR_HALT` + `CBUN_EVENT` + `UNSUSPEND` before the
RT loop starts (same set as `kord-clean-alarm --all` minus `CONTINUE_INIT`).
That recovers SafetyEvent ESTOP residue that a CBun-only clear leaves behind.

Full tables are in `kord_bridge/kord-api/docs/guides/safety/handle_alarms.rst`.
The absolute ceiling is the controller's own `[MOTION_CONSTRAINTS]` section in
`KORD.ini`; commands above it never reach the controller.

Set `arm_cfg.log_diagnostics` to report the achieved sync and command rates
every two seconds. While moving you want `sync` near 250 Hz and `moveL` near
the decimated rate (~125 Hz with default throttle=2); hands-off should drop to
~0 Hz (idle stops sending):

```
KassowArm: sync 250.1 Hz | moveJ 0.0 Hz | moveL 124.8 Hz
```

If sessions still drop, try `--arm-cfg.stream-l-throttle 4` with
`--arm-cfg.stream-l-tt 0.032 --arm-cfg.stream-l-bt 0.016`. Full-rate
`--arm-cfg.stream-l-throttle 1 --arm-cfg.stream-l-tt 0.008 --arm-cfg.stream-l-bt 0.004`
matches the linear examples but needs a clean wired link and RT scheduling.

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
