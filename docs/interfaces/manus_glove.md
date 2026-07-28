# Manus Glove

Tested: Manus Quantum Metaglove Pro (Manus Core 3.1.1), USB/HID.

Reference: [`manus_glove`](https://github.com/etaoxing/manus_glove)

The `ManusGlove` node ([`rio_hw/interfaces/manus_glove.py`](../../rio_hw/interfaces/manus_glove.py)) publishes per-frame finger openness derived from the glove's hand skeleton, for use as a `teleop` leader (e.g. paired with `InspireHand` — see [`docs/robots/inspire_hand.md`](../robots/inspire_hand.md)).

```bash
# udev rules for HID access
sudo tee /etc/udev/rules.d/70-manus-hid.rules << 'EOF'
# HIDAPI/libusb
SUBSYSTEMS=="usb", ATTRS{idVendor}=="3325", MODE:="0666"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="83fd", MODE:="0666"
# HIDAPI/hidraw
KERNEL=="hidraw*", ATTRS{idVendor}=="3325", MODE:="0666"
EOF

sudo udevadm control --reload-rules && sudo udevadm trigger

# libManusSDK_Integrated.so is downloaded automatically on first use to ~/.cache/manus_glove/lib
```

Published features are a 6-element vector in `[0, 1]` (`0` = closed, `1` = open):

```
[pinky, ring, middle, index, thumb_flex, thumb_rot]
```

## Calibration

On first run, if no calibration file is found and `auto_calibrate=True` (default), the node interactively prompts for open/closed (and optionally thumb-specific) poses, then saves a JSON calibration file (default: `manus_glove_cal.json` in the working directory). Subsequent runs load this file automatically. Pass an existing calibration file directly to skip the prompts:

```python
teleop: str = "ManusGlove"
teleop_cfg: NodeCfg | None = field(
    default_factory=lambda: NodeCfg(
        glove_id=None,           # auto-select lowest available glove ID
        hand_motion="NoMotion",
        calibration_file="my_cal.json",  # or omit + auto_calibrate=True
        auto_calibrate=True,
        curl=1.2,
        thumb_curl=1.45,
        freq=100,
    )
)
```

## FAQ

- Calibration is per-user/per-glove; re-run calibration (delete the saved JSON file) if retargeted motion feels off after a different person wears the glove.
- `no_thumb_endpoints=True` records only the open/closed poses (2-pose calibration) instead of the full 6-pose (including thumb flex/rotation) calibration, at the cost of thumb accuracy.
