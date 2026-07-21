# Posture Recognition Lab

This lab classifies posture from the radar point cloud. The default labels are
`empty`, `sitting`, `standing`, `standing_arms_forward`, and `squat`.

## 1. Collect

Each student should use a different collector ID:

```bash
python collect_posture_dataset.py \
  --port /dev/cu.usbserial-BH00LV2S \
  --collector student01 \
  --trials 3
```

For each posture, the script records a continuous 60-second point-cloud stream.

- `empty`: keep the space in front of the radar empty.
- `sitting`: sit in front of the radar and slowly vary position or angle.
- `standing`: stand in front of the radar and slowly vary position or angle.
- `standing_arms_forward`: stand with both arms extended straight forward.
- `squat`: squat or crouch in front of the radar.

Short test run:

```bash
python collect_posture_dataset.py \
  --port /dev/cu.usbserial-BH00LV2S \
  --collector student01 \
  --trials 1 \
  --duration 10
```

Important arrays saved in each `trial_data.npz`:

- `points_xyz`: NaN-padded firmware point cloud. The raw `z` column is saved
  for compatibility, but posture features use only `x` and `y`.
- `points_velocity`: NaN-padded point velocity from the point-cloud TLV.
- `point_count`: valid point count per frame.
- `time_s`: frame timestamps.

The default cfg is `xwrL64xx-evm/point_cloud.cfg`, which outputs point clouds
without range profiles to reduce UART traffic.

## 2. Train

After each student collects data, combine the dataset folders:

```bash
python combine_posture_datasets.py \
  datasets/posture_dataset_YYYYMMDD_HHMMSS \
  datasets/posture_dataset_YYYYMMDD_HHMMSS \
  --output datasets/combined_posture_all_students
```

Optional filtering:

```bash
python combine_posture_datasets.py DATASET_A DATASET_B --collector student01
python combine_posture_datasets.py DATASET_A DATASET_B --posture sitting --posture standing
```

```bash
python train_posture.py datasets/posture_dataset_YYYYMMDD_HHMMSS
```

Training cuts each 60-second recording into 2-second segments with 50% overlap.
Each segment becomes one training example. Features are computed only from the
top-down point cloud: point count over time, `x/y` centroid and spatial extent
over time, and `x/y` occupancy distributions.

Useful options:

```bash
python train_posture.py DATASET --window-seconds 2.0 --overlap 0.5
python train_posture.py DATASET --classifier svm_rbf
python train_posture.py DATASET --max-range 4.0 --x-limit 1.5
```

Student TODO after the first model works:

1. Train several classifiers, for example `random_forest`, `knn`, `svm_rbf`,
   and `decision_tree`.
2. Repeat each classifier with several random seeds.
3. Complete `plot_posture_classifier_accuracy.py` so it reads saved `.joblib`
   model payloads and plots average validation accuracy for each classifier.
4. Use the plot to justify which classifier should be used for real-time
   evaluation.

Run the completed plotting helper like this:

```bash
python plot_posture_classifier_accuracy.py DATASET/models/*.joblib
```

When you have data from multiple students, use a collector split to estimate
cross-student performance:

```bash
python train_posture.py DATASET_A DATASET_B DATASET_C --group-by collector
```

The model is saved under:

```text
datasets/posture_dataset_YYYYMMDD_HHMMSS/models/
```

## 3. Evaluate

```bash
python eval_posture_realtime.py \
  --model datasets/posture_dataset_YYYYMMDD_HHMMSS/models/random_forest_posture_YYYYMMDD_HHMMSS.joblib \
  --port /dev/cu.usbserial-BH00LV2S
```

Evaluation predicts on a rolling 2-second point-cloud window. The live plot
shows the current top-down `x/y` point cloud and the current predicted posture.

Existing posture models trained before the 2D feature update should be
retrained.

Use Ctrl+C to stop. Predictions and optional live frames are saved in the
session folder under `sessions/`.

## 4. Downstream Demo TODO: Squat Counter

After training a model that includes `standing` and `squat`, complete
`squat_counter_gui.py`.

Run it with:

```bash
python squat_counter_gui.py \
  --model datasets/posture_dataset_YYYYMMDD_HHMMSS/models/random_forest_posture_YYYYMMDD_HHMMSS.joblib \
  --port /dev/cu.usbserial-BH00LV2S
```

The GUI code, model loading, feature extraction, and plotting are provided.
Students complete two pieces:

1. `ConsecutivePredictionFilter.update`: convert noisy rolling model
   predictions into a stable posture label.
2. `SquatCounter.update`: count one squat when the stable labels follow:

```text
standing -> squat -> standing
```

Useful tuning options:

```bash
python squat_counter_gui.py --model MODEL.joblib --port PORT --stable-predictions 3
python squat_counter_gui.py --model MODEL.joblib --port PORT --confidence-threshold 0.60
python squat_counter_gui.py --model MODEL.joblib --port PORT --min-count-interval 1.5
```

The `r` key or Reset button clears the count. The `q` key or Quit button stops
the GUI. Prediction logs and optional point-cloud frames are saved under
`sessions/`.
