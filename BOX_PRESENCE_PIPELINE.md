# Box Presence Lab

This lab answers a binary question: is the closed box empty, or is there an
object inside?

## 1. Collect

```bash
python collect_box_presence_dataset.py \
  --port /dev/cu.usbserial-BH00LV2S \
  --trails 3
```

At the start, the script captures an empty-scene background. Remove the box
from the radar view, keep the scene still, and press Enter.

For each occasion, prepare the requested condition:

- `empty`: closed empty box.
- `object`: closed box with any object inside.

During the 60-second recording, slowly move the closed box. Change distance
and direction gently so the radar sees several box orientations.

Useful shorter test run:

```bash
python collect_box_presence_dataset.py \
  --port /dev/cu.usbserial-BH00LV2S \
  --trails 1 \
  --duration 10
```

Each occasion saves raw per-frame data:

- `range_profile`: raw range FFT profile, shape `frames x bins`.
- `range_background`: empty-scene background profile.
- `points_xyz`: NaN-padded firmware point cloud over time. It is saved for
  inspection; the default classifier uses the range image.
- `point_count`: valid point count per frame.
- `time_s` and `range_m`: time and range axes.

## 2. Train

After multiple groups collect data, combine their dataset folders:

```bash
python combine_box_presence_datasets.py \
  datasets/box_presence_dataset_YYYYMMDD_HHMMSS \
  datasets/box_presence_dataset_YYYYMMDD_HHMMSS \
  --output datasets/combined_box_presence_all_students
```

Optional label filtering:

```bash
python combine_box_presence_datasets.py DATASET_A DATASET_B --label object
```

```bash
python train_box_presence.py datasets/box_presence_dataset_YYYYMMDD_HHMMSS
```

Training cuts each 60-second recording into 2-second segments with 50% overlap.
Each segment becomes one training example. The default feature is a
background-subtracted dB range image resampled to `24 x 32`. The image is not
normalized by default, so absolute reflection-change strength remains part of
the feature.

Training uses a recording-level split by default, so the test set contains
different box-moving trails than the training set. This requires at least
two accepted recordings per label. A segment-level split can report 100%
accuracy just because overlapping windows from the same recording appear in
both train and test; use `--allow-segment-split` only for quick code smoke
tests.

Useful options:

```bash
python train_box_presence.py DATASET --window-seconds 2.0 --overlap 0.5
python train_box_presence.py DATASET --classifier svm_rbf
python train_box_presence.py DATASET --max-range 0.80 --resample-frames 24 --resample-bins 32
python train_box_presence.py DATASET --normalize
python train_box_presence.py DATASET --allow-segment-split
```

Use the default random forest first for live demos. RBF SVM probabilities can
be overconfident with only a few recordings per label, even when one held-out
recording split reports high validation accuracy.

The model is saved under:

```text
datasets/box_presence_dataset_YYYYMMDD_HHMMSS/models/
```

## 3. Evaluate

```bash
python eval_box_presence_realtime.py \
  --model datasets/box_presence_dataset_YYYYMMDD_HHMMSS/models/random_forest_box_presence_YYYYMMDD_HHMMSS.joblib \
  --port /dev/cu.usbserial-BH00LV2S
```

Evaluation first captures a new empty-scene background, then predicts on a
rolling 2-second window. A live plot shows the current background-subtracted
range profile and the recent range image used for prediction.

If the radar and box setup has not moved since data collection, use the exact
same collection background:

```bash
python eval_box_presence_realtime.py \
  --model MODEL.joblib \
  --port /dev/cu.usbserial-BH00LV2S \
  --background-npz datasets/box_presence_dataset_YYYYMMDD_HHMMSS/background.npz
```

Use Ctrl+C to stop. Predictions are saved in the session folder under
`sessions/`.
