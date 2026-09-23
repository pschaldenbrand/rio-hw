# Inspire Hand

Tested: Inspire RH56 dexterous hand (gen3), USB-serial.

Reference: [`inspire_demos`](https://github.com/TechShare-inc/inspire_demos), [dexsuite `inspire_hand_right.urdf`](https://github.com/dexsuite/dex-urdf/blob/main/robots/hands/inspire_hand/inspire_hand_right.urdf)

The `InspireHand` node ([`rio_hw/robots/inspire_hand.py`](../../rio_hw/robots/inspire_hand.py)) drives the hand over serial via the `inspire_demos` package and exposes a `moveJ`-compatible API so it can be used as the `arm` in a `SingleArm` station (e.g. driven directly by a `ManusGlove` teleop interface).

Joint ordering (6 DOF), normalized to `[0, 1]` (`0` = closed/adducted, `1` = open/abducted):

```
[pinky, ring, middle, index, thumb_flex, thumb_rot]
```

```bash
# Add user to dialout group for serial access
sudo usermod -aG dialout $USER
newgrp dialout

# Verify the hand is enumerated
lsusb
ls /dev/ttyUSB*
```

Minimal station config wiring (see [`examples/cfg/manus_inspire.py`](../../../rio/examples/cfg/manus_inspire.py)):

```python
arm: str = "InspireHand"
arm_cfg: NodeCfg | None = field(
    default_factory=lambda: NodeCfg(
        port="/dev/ttyUSB0",
        baudrate=115200,
        hand_id=1,
        generation=3,
        speed=1000,
        force=200,
        home_to_open=True,
        freq=300,
    )
)
```

## FAQ

- `read_state=False` (default) reports commanded angles as state instead of reading them back over serial each cycle, avoiding serial contention with `set_angle`. Set `read_state=True` if you need measured joint angles.
- If the hand does not respond, check `port`/`baudrate` and that no other process (e.g. a test script) is holding the serial port open.
