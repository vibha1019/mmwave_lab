# mmWave Radar Student Lab

This lab uses the TI xWRL6432/IWR6432 mmWave radar EVM with the preflashed
`mmwDemo` firmware. Students will first visualize simple radar returns, then
collect and classify small datasets.

The code has multiple TODOs. Search for `TODO` in the Python files.

The main activities are:

1. Connect the radar board through a USB-to-UART adapter.
2. Detect the range of a hand in front of the radar.
3. Detect moving targets with a slowly updating background.
4. Visualize point clouds and compare raw returns with background subtraction.
5. "See through the box" by classifying whether a closed box contains a strong
   reflector, such as a phone.
6. Recognize postures from the point cloud.

## 0. Hardware Setup

Use `swru628.pdf` Section 2.3 as the board reference. Handle the EVM carefully
and keep people at least 5 cm from the radar during operation.

Required hardware:

- IWRL6432FSP/xWRL6432 EVM.
- The 1.27 mm to 2.54 mm breakout board from the kit.
- USB-to-UART adapter with 3.3 V logic.
- Female-to-female Dupont jumper wires.
- Micro-USB cable from the USB-to-UART adapter to the computer.

Wire the radar breakout to the USB-to-UART adapter:

| Radar breakout pin | USB-to-UART pin | Notes |
| --- | --- | --- |
| `3V3` | `SYS3V3` | Powers the radar board from USB 3.3 V. |
| `GND` | `GND` | Common ground. |
| `UART RX` | `TXD` | TX/RX are crossed. |
| `UART TX` | `RXD` | TX/RX are crossed. |
| `SOP1` | `GND` | Required for functional mode. |
| `SOP0` | `3V3` | Functional mode for the preflashed demo. |

Do not connect `SOP0` to `GND` for this lab; that is flashing mode. Power-cycle
the board after changing SOP wiring.

Also see the course website for the pictures.

After wiring, plug the USB-to-UART adapter into the computer.

## 1. Find The Serial Port

Use the serial port from the USB-to-UART adapter in all commands below.

### macOS

Run:

```bash
ls /dev/cu.usbserial* /dev/cu.usbmodem* 2>/dev/null
python -m serial.tools.list_ports -v
```

The port usually looks like:

```text
/dev/cu.usbserial-BH00LV2S
/dev/cu.usbserial-5B1F0090131
```

Use the `/dev/cu...` device, not `/dev/tty...`.

### Windows

open Device Manager, then look under `Ports (COM & LPT)` for
`USB Serial Port (COMx)`.

In the commands below, replace `<PORT>` with your actual port, for example
`/dev/cu.usbserial-BH00LV2S` on macOS or `COM5` on Windows.

## 2. Hand Range Experiment

Goal: detect the distance from the radar to your hand using the range FFT profile.

Run:

```bash
python hand_lab.py --port <PORT>
```

Procedure:

1. Keep the scene empty while the first background frames are captured.
2. Put one hand in front of the radar.
3. Move the hand slowly between about 15 cm and 1 m.
4. Watch the printed distance and the background-subtracted range plot.

Useful options:

```bash
python hand_lab.py --port <PORT> --background-frames 40
python hand_lab.py --port <PORT> --min-range 0.2 --max-range 2.0
python hand_lab.py --port <PORT> --no-db
```

Student TODOs in `hand_lab.py`:

1. Implement background collection: collect `N` empty-scene range profiles.
2. Compute the background as the median profile.
3. Implement background subtraction: `max(profile - background, 0)`.
4. Find the strongest peak inside a valid range window, e.g. `0.15 m` to `2 m`.
5. Convert the peak bin index to meters.
6. Add simple smoothing over the last few distance estimates. Implement both
   mean smoothing and median smoothing, then decide which is more reasonable
   when one frame has a bad peak.

## 3. Moving Target Lab

Goal: detect a moving hand or reflector by subtracting a background that slowly
updates over time.

Run:

```bash
python hand_motion_lab.py --port <PORT>
```

Procedure:

1. Keep the scene still while the initial background frames are captured.
2. Move one hand or a reflector in front of the radar.
3. Watch the moving-target range estimate and waterfall plot.
4. Try several `--alpha` values.

Student TODO in `hand_motion_lab.py`:

1. Implement the exponential background update:

```python
background = (1 - alpha) * background + alpha * profile
```

Think about the tradeoff: a small `alpha` adapts slowly and keeps motion visible
longer; a large `alpha` adapts quickly and may absorb slow hand motion into the
background.

## 4. Point Cloud And Background Subtraction

Goal: see how the radar point cloud changes when a strong reflector moves in
front of the board. A phone, metal plate, or corner reflector works better than
a small hand.

Raw point-cloud viewer:

```bash
python point_cloud_viewer.py --port <PORT> --cfg xwrL64xx-evm/point_cloud.cfg
```

Move the reflector left/right and nearer/farther from the radar. The viewer is
a top-down `x/y` display:

- `x`: left/right position.
- `y`: forward range.

The preflashed demo often reports raw `z` near zero, so this lab treats point clouds as 2D.


For near-field hand and reflector motion, run:

```bash
python near_field_gesture_viewer.py --port <PORT>
```

This shows the first 50 cm of background-subtracted range profile plus the front-region point cloud.

For a raw range profile without subtraction:

```bash
python get_range_profile.py --port <PORT> --cfg xwrL64xx-evm/hand_distance.cfg
```

Compare the raw profile with the background-subtracted displays. The raw view
contains the room, table, box, and static clutter. Background subtraction makes
new or moving reflectors easier to see.

## 5. See Through The Box

Goal: decide whether a closed box is empty or contains a strong reflector.

Use a strong reflector first, such as a phone. Keep the radar and box position
as fixed as possible. For each object recording, put the reflector inside the
closed box and slowly move or rotate the box during capture. For each empty
recording, close the same box with nothing inside and move it in a similar way.

Collect data:

```bash
python collect_box_presence_dataset.py --port <PORT> --trails 3
```

The script collects:

- `empty`: closed empty box.
- `object`: closed box with the reflector inside.

The first calibration captures the empty scene with no box in front of the
radar. The same background is saved in the dataset folder as `background.npz`.

Train the classifier. The default classifier is random forest:

```bash
python train_box_presence.py datasets/box_presence_dataset_YYYYMMDD_HHMMSS
```

The model uses a background-subtracted dB range image over a 2-second window.
Normalization is off by default so the absolute reflection-change strength is
kept as part of the feature.

Evaluate in real time:

```bash
python eval_box_presence_realtime.py \
  --model datasets/box_presence_dataset_YYYYMMDD_HHMMSS/models/random_forest_box_presence_YYYYMMDD_HHMMSS.joblib \
  --port <PORT> \
  --background-npz datasets/box_presence_dataset_YYYYMMDD_HHMMSS/background.npz
```

Use the `--background-npz` option when the radar setup has not moved since data
collection. If the setup moved, omit it and let the evaluator capture a fresh
background.

For reliable results:

- Use the same box for both classes.
- Move the empty box and object-filled box in similar ways.
- Interleave collection when possible: empty, object, empty, object.
- Collect more than 3 trails per class for a better model.

More detail: [BOX_PRESENCE_PIPELINE.md](BOX_PRESENCE_PIPELINE.md).

## 6. Posture Recognition

Goal: classify human posture using the radar point cloud.

Default posture labels:

- `empty`: no person in front of the radar.
- `sitting`: person sitting.
- `standing`: person standing naturally.
- `standing_arms_forward`: person standing with both arms extended straight forward.
- `squat`: person squatting or crouching.

Collect data. Each student should use a unique collector ID:

```bash
python collect_posture_dataset.py \
  --port <PORT> \
  --collector student01 \
  --trials 3
```

Short test run:

```bash
python collect_posture_dataset.py \
  --port <PORT> \
  --collector student01 \
  --trials 1 \
  --duration 10
```

Train:

```bash
python train_posture.py datasets/posture_dataset_YYYYMMDD_HHMMSS
```

If multiple students collect datasets, combine them first:

```bash
python combine_posture_datasets.py \
  datasets/posture_dataset_STUDENT_A \
  datasets/posture_dataset_STUDENT_B \
  --output datasets/combined_posture
python train_posture.py datasets/combined_posture --group-by collector
```

Evaluate:

```bash
python eval_posture_realtime.py \
  --model datasets/posture_dataset_YYYYMMDD_HHMMSS/models/random_forest_posture_YYYYMMDD_HHMMSS.joblib \
  --port <PORT>
```

The posture model uses top-down point-cloud features: point count, `x/y`
centroid, `x/y` spread, body width/depth, and `x/y` occupancy over time.

Student TODO after the first posture model works:

1. Train at least three different classifiers, for example:

```bash
python train_posture.py DATASET --classifier random_forest
python train_posture.py DATASET --classifier knn
python train_posture.py DATASET --classifier svm_rbf
python train_posture.py DATASET --classifier decision_tree
```

2. Repeat training with several random seeds.
3. Plot a bar chart showing the average validation accuracy for each classifier.
4. Use the bar chart to justify which model you would use for real-time posture
   recognition.

Start from:

```bash
python plot_posture_classifier_accuracy.py \
  datasets/posture_dataset_YYYYMMDD_HHMMSS/models/*.joblib
```

Downstream demo TODO: count squats with a simple state machine:

```bash
python squat_counter_gui.py \
  --model datasets/posture_dataset_YYYYMMDD_HHMMSS/models/random_forest_posture_YYYYMMDD_HHMMSS.joblib \
  --port <PORT>
```

Student TODOs in `squat_counter_gui.py`:

1. Implement a consecutive-prediction filter so one noisy model output does not
   immediately change the displayed stable posture.
2. Implement the state machine that counts one squat only after the stable
   prediction sequence `standing -> squat -> standing`.
3. Use `--stable-predictions`, `--confidence-threshold`, and
   `--min-count-interval` to tune responsiveness versus false counts.

More detail: [POSTURE_PIPELINE.md](POSTURE_PIPELINE.md).

## Troubleshooting

- If a script cannot configure the radar after a previous run, press the reset
  button or power-cycle the EVM, then rerun the script.
- If no serial port appears, install the FTDI USB-to-UART driver and reconnect
  the adapter.
- If the CLI reports a DPU or configuration error, confirm that the board is in
  functional mode: `SOP0 -> 3V3`, `SOP1 -> GND`.
- If evaluation is stuck on one class, collect more trails and retrain with
  the default random forest before trying SVM models.
- Use `Ctrl+C` to stop live scripts. The scripts send `sensorStop 0` before
  exiting.

## Files

- `hand_lab.py`: hand distance from background-subtracted range profile.
- `hand_motion_lab.py`: moving-target range waterfall with exponential
  background subtraction.
- `point_cloud_viewer.py`: top-down point-cloud viewer.
- `near_field_gesture_viewer.py`: near-field range and point-cloud viewer.
- `collect_box_presence_dataset.py`, `train_box_presence.py`,
  `eval_box_presence_realtime.py`: box empty/object classifier.
- `collect_posture_dataset.py`, `train_posture.py`,
  `eval_posture_realtime.py`: posture classifier.
- `plot_posture_classifier_accuracy.py`: TODO scaffold for comparing posture
  classifiers.
- `squat_counter_gui.py`: TODO scaffold for a downstream squat-counting GUI.

Raw recordings and trained models are ignored by git under `datasets/`,
`sessions/`, and `models/`.
