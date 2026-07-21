# See Through The Box Lab

This task treats the fixed radar-plus-box setup as a static classification
problem. Each sample is a short capture after the box is closed.

## 1. Collect

Default labels are `empty`, `board`, and `marker_pen`:

```bash
python collect_box_dataset.py \
  --port /dev/cu.usbserial-BH00LV2S \
  --trials 8
```

Custom labels:

```bash
python collect_box_dataset.py \
  --port /dev/cu.usbserial-BH00LV2S \
  --contents empty,board,marker_pen,phone \
  --trials 8
```

At the start, the script first asks for an empty-box calibration. Remove all
objects, close the empty box, and press Enter. That calibration is reused for
all trials in the dataset.

Before each trial, the script asks the student to open the box, change the
object, close the box, and press Enter. The default trial capture is only 20
valid radar frames.

The point-cloud slice is taken near a fixed forward distance:

```bash
python collect_box_dataset.py \
  --port /dev/cu.usbserial-BH00LV2S \
  --point-distance 0.20 \
  --point-window 0.05
```

`--point-distance 0.20` means points with forward `y` range within
`0.20 +/- 0.05 m` are saved. Use this to match the physical box location.

## 2. Saved Data

Each dataset has:

```text
datasets/box_contents_dataset_YYYYMMDD_HHMMSS/
  dataset_metadata.json
  trials.csv
  sessions/
    box_empty_trial_001/
      trial_metadata.json
      trial_data.npz
```

Important arrays in `trial_data.npz`:

- `range_m`: range axis.
- `range_background`: mean empty-box calibration profile.
- `range_profile`: raw per-frame range profiles, shape `frames x bins`.
- `mean_range_profile`: mean of the captured range profiles.
- `slice_point_count`: valid point count per frame in the requested distance slice.
- `slice_points_xyz`: NaN-padded point cloud for the requested distance slice.
  The raw `z` column is saved, but point-slice features use only `x/y`.

## 3. Train

```bash
python train_box.py datasets/box_contents_dataset_YYYYMMDD_HHMMSS
```

Default features are:

- 64 resampled values from `mean_range_profile - range_background`.
- 12 point-slice statistics: count, `x/y` centroid, `x/y` spread, and min/max
  lateral/range position.

Useful options:

```bash
python train_box.py datasets/box_contents_dataset_YYYYMMDD_HHMMSS --classifier svm_rbf
python train_box.py datasets/box_contents_dataset_YYYYMMDD_HHMMSS --no-points
python train_box.py datasets/box_contents_dataset_YYYYMMDD_HHMMSS --max-range 0.60 --range-bins 64
```

The model is saved under:

```text
datasets/box_contents_dataset_YYYYMMDD_HHMMSS/models/
```

## 4. Evaluate

```bash
python eval_box_realtime.py \
  --model datasets/box_contents_dataset_YYYYMMDD_HHMMSS/models/random_forest_box_contents_YYYYMMDD_HHMMSS.joblib \
  --port /dev/cu.usbserial-BH00LV2S
```

By default, evaluation first asks for an empty-box calibration. During
interactive evaluation:

- Press Enter to classify the current closed box.
- Type `b` and press Enter to capture a new empty-box calibration.
- Type `q` and press Enter to quit.

Each prediction prints the top class and a descending score list for all
classes when the model supports probabilities.

Existing box-content models trained before the 2D point-slice update should be
retrained.

Use continuous mode when the setup should keep classifying:

```bash
python eval_box_realtime.py --model MODEL.joblib --port /dev/cu.usbserial-BH00LV2S --continuous
```
