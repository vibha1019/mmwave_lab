# Hand Radar Lab

This lab uses the preflashed xWRL6432 mmWave demo and one USB serial port.
The lab cfg files keep `lowPowerCfg 0` so `sensorStop` works when you press
`Ctrl+C`. If the board is currently running an older `lowPowerCfg 1` session,
press the EVM reset button or power-cycle it once before starting this lab.

In the `student-todo` branch, `hand_lab.py` and `hand_motion_lab.py` contain
small `TODO` blocks. The UART setup and plotting code are provided;
students complete the range-profile processing.

## Part 1: Hand Distance

Run:

```bash
python hand_lab.py --port /dev/cu.usbserial-5B1F0090131
```

Keep the area in front of the radar empty while the script captures the
background. Then place a hand in front of the radar. After the TODOs are
implemented, the reported distance should be the strongest
background-subtracted range bin between 15 cm and 2 m. The plot shows
background-subtracted strength in dB by default.

Student TODOs:

1. Collect `N` empty-scene range profiles.
2. Compute the background as the median profile.
3. Subtract the background with `max(profile - background, 0)`.
4. Find the strongest peak inside the valid range window.
5. Convert the peak bin index to meters.
6. Smooth the distance over the last few estimates using both mean and median,
   then decide which is more robust to one bad peak.

## Part 2: Background Subtraction

Have students compare the display with and without clutter in the scene. Static
objects present during the first background frames are removed from the plot,
while a newly introduced hand remains visible.

Useful options:

```bash
python hand_lab.py --port /dev/cu.usbserial-5B1F0090131 --background-frames 40
python hand_lab.py --port /dev/cu.usbserial-5B1F0090131 --min-range 0.1 --max-range 2.0
python hand_lab.py --port /dev/cu.usbserial-5B1F0090131 --no-db
```

## Part 3: Moving Target Detection

Run:

```bash
python hand_motion_lab.py --port /dev/cu.usbserial-5B1F0090131
```

Keep the scene still while the first frames initialize the background. Students
then implement the exponential moving average update so static objects fade out
and moving hands remain visible. The plot shows motion strength in dB for
ranges below 2 m, with a rolling waterfall of recent frames.

Useful options:

```bash
python hand_motion_lab.py --port /dev/cu.usbserial-5B1F0090131 --alpha 0.01
python hand_motion_lab.py --port /dev/cu.usbserial-5B1F0090131 --residual positive
python hand_motion_lab.py --port /dev/cu.usbserial-5B1F0090131 --history-frames 120
python hand_motion_lab.py --port /dev/cu.usbserial-5B1F0090131 --no-db
```

Smaller `--alpha` adapts more slowly and keeps motion visible longer. Larger
`--alpha` adapts faster and suppresses slow or stopped hands sooner.

Student TODO:

```python
background = (1 - alpha) * background + alpha * profile
```

Configs:

- `xwrL64xx-evm/hand_distance.cfg`: range profile only.
- `xwrL64xx-evm/point_cloud.cfg`: point cloud only for `point_cloud_viewer.py`.
- `xwrL64xx-evm/near_field_hand_50cm.cfg`: range profile plus point cloud for
  the near-field viewer. The script filters display and recording to 50 cm.

## Near-Field Gesture Viewer

For fine-grained hand-motion data, use the combined near-field viewer:

```bash
python near_field_gesture_viewer.py --port /dev/cu.usbserial-5B1F0090131
```

It shows the background-subtracted range profile and a short waterfall for
the first 50 cm, plus a top-down 2D point-cloud view restricted to the front
region: `x` left/right +/-15 cm and `y` range less than 50 cm.
The script performs `sensorWarmRst` before configuring the board so repeated
runs start from a clean demo CLI state.

To record a compact dataset file:

```bash
python near_field_gesture_viewer.py --port /dev/cu.usbserial-5B1F0090131 --record data/push_001.npz --label push
```

The `.npz` file stores the clipped range axis, background profile, raw clipped
range profile, background-subtracted range profile, estimated peak range, and
the filtered point cloud for each frame. The raw firmware `z` field is retained
in the saved array, but this lab does not use it for display or features.

For class datasets with repeated labeled trials, use:

```bash
python collect_mmwave_dataset.py --port /dev/cu.usbserial-5B1F0090131 --collector student01 --trials 8
```

This creates a UWB-lab-style dataset folder with `dataset_metadata.json`,
`trials.csv`, and one `trial_data.npz` per accepted gesture trial. See
`MMWAVE_GESTURE_PIPELINE.md` for the saved array format plus the
`train_mmwave.py` and `eval_mmwave_realtime.py` workflow.

The preflashed demo accepted the validated profile used here at about
4.57 cm/bin. More aggressive CLI-only chirps for finer range bins were rejected
by this firmware at `sensorStart`, so this lab gets finer gesture information
from temporal changes, background subtraction, and sub-bin peak tracking rather
than from a denser firmware range FFT.

Optional point-cloud viewer:

```bash
python point_cloud_viewer.py --port /dev/cu.usbserial-5B1F0090131
```

The point-cloud viewer shows a top-down `x/y` view. It colors by
frame-to-frame tracked radial velocity by default because this preflashed demo
may report zero in the raw Doppler field.
To inspect the raw firmware Doppler field, use:

```bash
python point_cloud_viewer.py --port /dev/cu.usbserial-5B1F0090131 --color-by velocity
```

Gray points in raw velocity coloring mean the reported radial velocity is near
zero. To inspect the geometry without velocity coloring, use:

```bash
python point_cloud_viewer.py --port /dev/cu.usbserial-5B1F0090131 --color-by range
```
