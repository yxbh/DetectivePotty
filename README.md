# DetectivePotty

DetectivePotty is a prototype for real-time dog potty detection on UniFi Protect cameras / UNVR Pro. It records candidate potty events, saves clips and high-resolution dog-centered crops, and provides a local review app so a human can label each event as `pee`, `poop`, `not_potty`, or `unknown`.

The v0 pee-vs-poop result is a **weak guess, not ground truth**. Every recorded event is saved as `label_status=unlabeled` until reviewed in the web app. The goal is to build a clean training set for a future custom classifier.

## Architecture

DetectivePotty keeps camera streams warm, uses triggers to mark interesting time windows, detects and tracks dogs, then records reviewable events. The key strategy is **detect small, crop big**: YOLO runs on downscaled frames for speed, but bounding boxes are mapped back to original-resolution frames before saving full frames and dog crops for training.

Latency is handled with a warm `RollingBuffer` pre-roll. A late Protect or YOLO trigger reaches backward into already-decoded frames, and Protect recording download is available as a best-effort high-quality source for the same window.

```mermaid
flowchart LR
    A[UniFi cameras / file clips] --> B[Protect client / VideoSource]
    B --> C[Warm RollingBuffer pre-roll]
    B --> D[Triggers]
    D --> D1[Protect animal WS primary]
    D --> D2[YOLO fallback / corroboration]
    C --> E[DogDetector YOLO]
    E --> F[Tracker]
    F --> G[PottyEventDetector state machine]
    G --> H[HeuristicPottyClassifier weak guess]
    H --> I[EventRecorder]
    I --> J[Dataset on disk]
    J --> K[FastAPI review + labeling app]
```

Main flow: cameras/Protect (or a local file) feed `VideoSource`; a warm buffer preserves pre-roll; `DogDetector` finds dogs; `Tracker` builds tracks; `PottyEventDetector` emits generic potty candidates from stationary+squat posture; `HeuristicPottyClassifier` pre-fills a weak guess; `EventRecorder` writes clips, frames, crops, and metadata; the FastAPI app reads the dataset for review and labeling.

For more detail, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Requirements

- macOS / Apple Silicon is the target prototype environment.
- Python **3.12** (`pyproject.toml` pins `>=3.12,<3.13`). Avoid Python 3.13/3.14 for now because the torch/Ultralytics stack may lag.
- [`uv`](https://docs.astral.sh/uv/) for environment and command execution.
- `ffmpeg` available on the system for video workflows (`brew install ffmpeg`). OpenCV writes the test/demo MP4s.
- Optional for live mode: UniFi Protect NVR / UNVR Pro with RTSP enabled and Animal smart-detect enabled on supported cameras.
- MPS acceleration is used when available (`device: auto` or `mps`); the detector falls back to CPU when needed.

## Setup

```bash
uv sync
```

`uv sync` creates/updates `.venv` from `pyproject.toml` and `uv.lock`. YOLO weights such as `yolo11n.pt` are loaded by Ultralytics on first detector use and may be downloaded into its normal cache if not already present. Model weight files are ignored by git.

## Configuration

Copy the example config and edit it locally:

```bash
cp config.example.yaml config.yaml
```

Do **not** put Protect credentials or RTSP tokens in YAML. The code resolves secrets from environment variables only:

```bash
export DETECTIVEPOTTY_NVR_API_KEY="..."
# or username/password instead of API key:
export DETECTIVEPOTTY_NVR_USERNAME="..."
export DETECTIVEPOTTY_NVR_PASSWORD="..."
```

Important fields:

### `global`

- `dataset_dir`: root directory for recorded events.
- `model_name`: YOLO model name/path, e.g. `yolo11n.pt`.
- `inference_long_edge_px`: long edge used for downscaled inference.
- `device`: `auto`, `mps`, or `cpu`.
- `log_level`: Python logging level.
- `dogs`: optional roster of dog names (e.g. `[Gromit, WALL-E, Apollo]`) offered as manual identity labels in the review portal. Leave empty to allow free-form dog names.

### `protect`

- `nvr_host`: NVR base URL, e.g. `https://unvr.example.lan`; never include credentials.
- `api_key_env`, `username_env`, `password_env`: environment variable names for secrets.
- `verify_tls`: set `false` only for prototype/self-signed-cert troubleshooting.

### `cameras[]`

- `id`, `name`, `enabled`: camera identity and selection.
- `input.kind`: `protect` for UniFi Protect, `file` for offline clips.
- `input.path`: local video path for `file` cameras.
- `input.source_id`: optional sanitized source label.
- `substream_choice`: `low`, `medium`, or `high` RTSPS channel preference.
- `animal_supported`: notes whether Protect Animal smart-detect is expected.
- `detection_conf_threshold`: dog confidence threshold.
- `event_duration_s`: how long a candidate must persist before recording.
- `stationary_threshold_s`: stationary posture window.
- `squat_threshold`: bbox posture threshold for squat-like motion.
- `sample_rate_fps`: detector sampling rate.
- `pre_roll_s`, `post_roll_s`: event window around the candidate.
- `roi`, `ignore_zones`: normalized polygon zones for include/exclude filtering.
- `retention_days`, `retention_max_gb`: per-camera cleanup policy.

`config.example.yaml` includes a disabled `file` sample camera for the Gromit pee clip. Enable it and adjust thresholds if you want a no-NVR end-to-end run against that local file.

## Usage

### Offline single-clip detection demo

```bash
uv run detectivepotty detect-file \
  --input "<clip>" \
  --output outputs/annotated.mp4 \
  --save-crops outputs/crops \
  --every-n 3
```

This runs YOLO on every Nth frame, writes an annotated MP4, and optionally saves high-resolution dog crops.

### Run the pipeline

```bash
uv run detectivepotty run --config config.yaml
uv run detectivepotty run --config config.yaml --camera backyard-grass
uv run detectivepotty run --config config.yaml --max-workers 2
```

The pipeline processes enabled cameras (or selected `--camera` IDs) and writes dataset event directories under `global.dataset_dir`. For offline testing, enable the sample `file` camera in `config.example.yaml` after copying it to `config.yaml`.

**Concurrency:** When more than one camera is selected, each runs on its own
worker thread so multiple live cameras are monitored simultaneously (the first
camera no longer blocks the rest). By default every camera gets a dedicated
thread; each live camera always keeps one because its loop runs until
interrupted. `--max-workers` can cap the pool — if you set it lower than the
number of live cameras the pipeline raises it back up (with a warning) so no
live camera is starved. Pass `--max-workers 1` to force sequential processing
(only safe when every camera is a finite file). GPU inference is serialized with
an internal lock because the MPS/torch backend is not reliably safe for
concurrent model execution — I/O (RTSP reads, buffering, encoding) still runs in
parallel, which is where the live-monitoring win comes from. Live (Protect)
cameras stream until you interrupt with Ctrl-C; file cameras finish when the clip
ends. By default a single camera's failure is logged and isolated so the other
cameras keep running.

### List Protect cameras

```bash
uv run detectivepotty list-cameras --config config.yaml
```

Requires `protect.nvr_host` plus either `DETECTIVEPOTTY_NVR_API_KEY` or `DETECTIVEPOTTY_NVR_USERNAME`/`DETECTIVEPOTTY_NVR_PASSWORD`.

### Review and label events

```bash
uv run detectivepotty serve --config config.yaml
```

Open <http://127.0.0.1:8000>. The app lists events, plays `clip.mp4`, shows crops/frames and metadata, and writes labels back into `metadata.json`. The sidebar shows how many events match the current filter and the total recorded on disk.

> One recorded event = one detected potty behavior, **not** one input file. A clip with no qualifying stationary+squat behavior produces zero events, and a busy clip can produce several. So 4 input files may legitimately yield 3 events. The status filter defaults to **All**; switching it to `Unlabeled` hides events once you label them.

Label workflow:

1. Select an event and watch the clip / inspect crops.
2. Choose `Pee`, `Poop`, `Not potty`, or `Unknown`; keyboard shortcuts `1`, `2`, `3`, `0` select those labels.
3. Optionally pick which dog it was (from the `global.dogs` roster) in the **Dog** selector.
4. Pick status (`labeled`, `rejected`, or `uncertain`), add an optional note, and click **Save label**.

The **Dog** label is a manual human identity tag (the `dog` field in `metadata.json`); automatic per-dog re-identification is intentionally out of scope for v0.

## Dataset layout

```text
<dataset_dir>/<camera>/<YYYY-MM-DD>/events/<YYYYMMDDTHHMMSSZ>_<camera>_<track>_<eventId>/
    clip.mp4
    protect_recording.mp4   # optional
    frames/000.jpg ...
    crops/000.jpg ...
    metadata.json
```

Event directories are UTC-sortable, idempotent, filesystem-safe, and secret-free. `metadata.json` includes camera IDs/names, sanitized source ID, trigger reason, timestamps, config hash, model info, detections/tracks, frame records, crop boxes, `classifier_guess`, `classifier_confidence`, `label`, `label_status`, and `dog` (manual identity label, `null` until assigned).

`classifier_guess` is the weak v0 heuristic. `label` and `label_status` are the human-reviewed truth fields used for training.

## Development

```bash
uv run pytest -q
uv run ruff check .
```

The integration tests are offline: they inject fake detectors and do not require GPU, model downloads, cameras, or network access.

## Roadmap

- Train a custom pee/poop classifier on the high-resolution dog crops and stored original-resolution boxes.
- Replace the heuristic guess with a trained model once enough labeled data exists.
- Add stronger tracking such as ByteTrack and better multi-dog handling.
- Improve posture modeling beyond bbox height/aspect heuristics.
- Expand multi-camera live workflows and hard-negative/background capture.

DetectivePotty is a prototype. Generic potty-event detection is the useful v0 output; pee-vs-poop classification is intentionally treated as unreliable until the labeled dataset is large enough to train and validate a real classifier.
