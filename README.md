# DetectivePotty

DetectivePotty is a prototype for real-time dog potty detection on UniFi Protect cameras / UNVR Pro. It records candidate potty events, saves clips and high-resolution dog-centered crops, and provides a local review app so a human can label each event as `pee`, `poop`, `not_potty`, or `unknown`.

The v0 pee-vs-poop result is a **weak guess, not ground truth**. Every recorded event is saved as `label_status=unlabeled` until reviewed in the web app. The goal is to build a clean training set for a future custom classifier.

## Architecture

DetectivePotty keeps camera streams warm, samples frames for YOLO dog detections, tracks dogs, then records reviewable potty candidates. The key strategy is **detect small, crop big**: YOLO runs on downscaled frames for speed, but bounding boxes are mapped back to original-resolution frames before saving full frames and dog crops for training.

Latency is handled with a warm `RollingBuffer` pre-roll, so recording can reach backward into already-decoded frames once the sustained-dwell state machine emits a candidate.

```mermaid
flowchart LR
    A[UniFi cameras / file clips] --> B[Protect client / VideoSource]
    B --> C[Warm RollingBuffer pre-roll]
    C --> E[DogDetector YOLO]
    E --> F[Tracker]
    F --> G[PottyEventDetector state machine]
    G --> H[HeuristicPottyClassifier weak guess]
    H --> I[EventRecorder]
    I --> J[Dataset on disk]
    J --> K[FastAPI review + labeling app]
```

Main flow: cameras/Protect (or a local file) feed `VideoSource`; a warm buffer preserves pre-roll; `DogDetector` finds dogs; `Tracker` builds tracks; `PottyEventDetector` emits generic potty candidates from a sustained stationary dwell; `HeuristicPottyClassifier` pre-fills a weak guess; `EventRecorder` writes clips, frames, crops, and metadata; the FastAPI app reads the dataset for review and labeling.

For more detail, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Requirements

- macOS / Apple Silicon is the target prototype/dev environment; a Windows + NVIDIA GPU (CUDA) box is the semi-permanent runtime target.
- Python **3.12** (`pyproject.toml` pins `>=3.12,<3.13`). Avoid Python 3.13/3.14 for now because the torch/Ultralytics stack may lag.
- [`uv`](https://docs.astral.sh/uv/) for environment and command execution.
- `ffmpeg` available on the system for video workflows (`brew install ffmpeg`). OpenCV writes the test/demo MP4s.
- Optional for live mode: UniFi Protect NVR / UNVR Pro with RTSP enabled and Animal smart-detect enabled on supported cameras.
- GPU acceleration is selected automatically (`device: auto` resolves CUDA → MPS → CPU); the detector and pose backend fall back to CPU when no accelerator is available.
- The project runs on **numpy 1.x / OpenCV 4.11** (pinned `numpy>=1.23,<2`, `opencv-python>=4.6,<4.12`) so the optional pose backend (DeepLabCut 3.x, which caps `numpy<2`) can share a single environment. Nothing in the core app needs numpy 2.

## Setup

```bash
uv sync
```

`uv sync` creates/updates `.venv` from `pyproject.toml` and `uv.lock`. YOLO weight files live in `models/` (e.g. `models/yolo11m.pt`) and are ignored by git. They are loaded by Ultralytics on first detector use; if a configured weight is missing, Ultralytics downloads it (a bare name like `yolo11m.pt` downloads into the working directory, so prefer a `models/…` path or pre-place the file under `models/`).

### Optional: keypoint-pose backend (DeepLabCut)

The pose backend (SuperAnimal-Quadruped via DeepLabCut 3.x) is an **optional extra** — it is heavy and not required for detection, and the unit tests never need it. Install it only when you want pose:

```bash
uv sync --extra pose
```

This shares the base environment (numpy 1.x / OpenCV 4.11); DeepLabCut runs in-process on the same torch as the detector. On the Windows + NVIDIA runtime, the same command installs the CUDA-enabled wheels and `device: auto` resolves to CUDA.

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

**Auth precedence:** when both are set, the **private** API (username/password) is preferred and the
public API key is only used as a fallback if private auth is unavailable or fails. Private auth gives
the full bootstrap that powers RTSPS, snapshots, and recording downloads, so when it succeeds the
heavy public Integration API bootstrap is skipped entirely (running both per camera otherwise trips
UniFi's rate limit with `429`s). A transient private `401` is retried once with a fresh login before
falling back.

Important fields:

### `global`

- `dataset_dir`: root directory for recorded events.
- `model_name`: YOLO model name/path. Default `models/yolo11m.pt`, chosen after sweeping the yolo11/12/26 families (n→x): it was the only off-the-shelf model consistently top-tier across day and night clips, roughly doubling night dog-detection recall over `yolo11n.pt` at negligible extra latency on MPS. Weight files live in `models/` (gitignored); use `models/yolo11n.pt` for a lighter/faster model if needed.
- `inference_long_edge_px`: YOLO network input long edge, passed to ultralytics as `imgsz` (rounded/padded to a stride multiple internally). Default `640`, the model's native training size, which gives the best accuracy on our footage and the lowest latency. Raising it does **not** improve recall — it is slower and can actually *reduce* detection of small/distant dogs (the network is optimised for ~640). The frame is letterboxed by ultralytics directly; we no longer pre-resize, which previously dominated per-frame cost.
- `device`: `auto`, `cuda`, `mps`, or `cpu`. `auto` resolves CUDA → MPS → CPU; an explicitly requested accelerator that is unavailable warns and falls back to CPU.
- `log_level`: Python logging level.
- `dogs`: optional roster of dog names (e.g. `[Gromit, WALL-E, Apollo]`) offered as manual identity labels in the review portal. Leave empty to allow free-form dog names.
- `dedupe_reruns`: when `true` (default), re-running detection over the same source carries your existing human labels forward onto the matched event and supersedes the duplicate instead of piling up a fresh copy. See [Re-running detection](#re-running-detection-reruns--dedupe). Set `false` to keep the legacy behavior where every run writes brand-new events.
- `rerun_match_tolerance_s`: how close (in seconds) a freshly-detected event's start must be to a prior event to be treated as the same event when intervals can't be compared directly. Default `5.0`. Also used as the clustering window for the `dedupe-events` cleanup command.

### `protect`

- `nvr_host`: NVR base URL, e.g. `https://unvr.example.lan`; never include credentials.
- `api_key_env`, `username_env`, `password_env`: environment variable names for secrets.
- `verify_tls`: set `false` only for prototype/self-signed-cert troubleshooting.

### `cameras[]`

- `id`, `name`, `enabled`: camera identity and selection.
- `input.kind`: `protect` for UniFi Protect, `rtsp` for a direct RTSP stream, `file` for offline clips.
- `input.path`: local video path for `file` cameras.
- `input.url_env`: **required for `rtsp` cameras** — the name of an environment variable holding the
  full `rtsp://user:pass@host:554/path` URL (credentials included). The URL is read from the
  environment so secrets never touch YAML; it is also redacted from logs and stripped from source
  IDs. Use this for third-party/ONVIF cameras (e.g. a Dahua camera adopted into the NVR) whose UniFi
  Protect RTSPS re-stream is broken — streaming directly from the camera bypasses UniFi entirely.
- `input.source_id`: optional sanitized source label.
- `substream_choice`: `low`, `medium`, or `high` RTSPS channel preference (`protect` cameras only).
  Streams are read over TCP (`OPENCV_FFMPEG_CAPTURE_OPTIONS` defaults to `rtsp_transport;tcp`) for
  reliability — UDP drops packets and floods the console with `Empty H.264 RTP packet`. Override the
  env var to change it.
- `animal_supported`: notes whether Protect Animal smart-detect is expected.
- `detection_conf_threshold`: dog confidence threshold.
- `event_duration_s`: how long a candidate must persist before recording.
- `stationary_threshold_s`: stationary posture window.
- `dwell_trigger_s`: continuous stationary hold (seconds) that triggers a potty candidate. This is
  the sole detection trigger — a viewpoint-invariant sustained-dwell cue that works on high/top-down
  cameras where a bbox squat metric is unreliable. Must be `> 0` (default 2). (The old
  `squat_threshold` field was removed; delete it from any existing config or validation will fail.)
- `sample_rate_fps`: detector sampling rate.
- `pre_roll_s`, `post_roll_s`: event window around the candidate.
- `roi`, `ignore_zones`: normalized polygon zones for include/exclude filtering.
- `retention_days`, `retention_max_gb`: per-camera cleanup policy.

`config.example.yaml` includes a disabled `file` sample camera for the Gromit pee clip. Enable it and adjust thresholds if you want a no-NVR end-to-end run against that local file.

### `pose` (optional, additive)

Keypoint pose is **off by default** and additive: when disabled, or when keypoints are
missing/low-confidence/too-sparse for a window, the system falls back to the existing bbox
heuristics, so enabling pose never removes the bbox coverage/recall behavior. Requires the
`pose` extra (`uv sync --extra pose`) for the real `superanimal` backend.

- `enabled`: master switch for the shared pose estimator. Default `false`.
- `backend`: `superanimal` (DeepLabCut, real) or `mock` (deterministic, no model — for tests/wiring).
- `model_name`, `device`, `crop_margin_frac`: pose head, device, and bbox-expansion margin (the under-bound-crop rescue).
- `min_keypoint_conf`, `min_required_frames`, `min_pose_coverage`, `min_torso_keypoints`: pose-quality
  gates separate from feature thresholds — how much pose must be present before it is trusted.
- `box_union_window_s`: temporal box union. Pose crops are built from the union of a dog's detector
  boxes over this trailing window (seconds) to recover full extent when a single IR frame under-segments
  (the "detection bounds pose" night failure). `0.0` (default) disables it — the pose crop is then the raw
  detector box. Guarded against ballooning (center-drift + max-growth caps); only affects pose, never the
  tracking/posture/recorder boxes. Currently wired into the pose classifier crop only (the per-frame gate
  observes before tracking, so it uses raw boxes).
- `candidate_only`: when set, the classifier only runs pose on candidate windows (not every dog every frame).
- `enable_pose_classifier`: let `PosePottyClassifier` use pose for the pee/poop guess (still `needs_label=true`).
- `enable_pose_gate`: **experimental — not yet validated with the real pose backend.** Lets pose augment
  the detection gate's stationarity signal. It is strictly additive (it can relax the motion-jitter check
  via OR for the dwell trigger, but never removes the bbox `covered_long_enough` requirement, and its
  squat signal is no longer consumed). The gate-OFF path is byte-for-byte identical to the bbox baseline.
  Keep it `false` until it has been validated end-to-end
  against the real `superanimal` backend on the night/IR clips; the mock backend only proves wiring.



### Offline single-clip detection demo

```bash
uv run detectivepotty detect-file \
  --input "<clip>" \
  --output outputs/annotated.mp4 \
  --save-crops outputs/crops \
  --every-n 3
```

This runs YOLO on every Nth frame, writes an annotated MP4, and optionally saves high-resolution dog crops.

### Tune `detection_conf_threshold` interactively

```bash
# Local clip:
uv run detectivepotty tune-detect --input "<clip>"

# A camera from a config (file, rtsp, or protect):
uv run detectivepotty tune-detect --config config.yaml --camera backyard-grass
```

Opens an OpenCV window that plays the video with **every** detected dog drawn. A
`conf x100` slider sets the confidence threshold live: boxes at or above it render solid
green (kept), boxes below render dim red (dropped). Detection runs at a low `--floor`
(default `0.05`) so borderline boxes stay visible — the slider only re-colors them, it does
not re-run inference, so dragging it is instant. The chosen threshold is printed on exit so
you can paste it into the camera's `detection_conf_threshold`.

Keys: `space` play/pause · `n`/`→` step forward · `p`/`←` step back (file only) · `r`
restart (file only) · `q`/`Esc` quit. Live RTSP/Protect streams are play-only (no
step-back or restart). Useful options: `--conf` (initial slider value), `--floor`,
`--model`, `--long-edge`, `--every-n` (run detection every N played frames for smoother
playback). The window needs a desktop session (it uses `cv2.imshow`).

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

### Build training data: harvest → label → export

The data engine turns long recordings into a labeled image-classifier dataset (pee/poop and dog identity). Clips are written as **browser-playable H.264** so they scrub in the Label tab.

1. **Harvest dog-present spans** from a recording into reviewable clip dirs (`<out>/<span_id>/clip.mp4` + `metadata.json`):

   ```bash
   # From a local file:
   uv run detectivepotty harvest --input "<clip>.mp4" --out dataset/harvest

   # From historical UNVR footage, a whole day in hourly chunks (skips failed/empty
   # chunks; re-runs are idempotent). --camera takes an id or name (see list-cameras):
   uv run detectivepotty harvest-camera --config config.yaml \
     --camera "Backyard Grass" --date 2026-06-06 --utc-offset 10 --out dataset/harvest
   ```

2. **Label** each clip in the portal's **Label** tab (`serve` below): scrub, mark In/Out, pick behavior + dog, add ranges, save. Writes a `labels.json` sidecar per clip. Keyboard-driven (Space play · ←/→ step · I/O mark · 1–4 behavior · Enter add · S save · j/k clip).

3. **Export** the labeled ranges to a YOLO-cls dataset (densely re-detects + crops the labeled dog per frame):

   ```bash
   uv run detectivepotty export-dataset --clips dataset/harvest --out dataset/export
   ```

### Review and label events

The review portal is a Svelte + Vite + TypeScript app that FastAPI serves from a prebuilt bundle (`src/detectivepotty/web/frontend/dist/`, gitignored). Build it once, and again after any frontend change:

```bash
cd src/detectivepotty/web/frontend
npm install   # first run only
npm run build
```

Then start the server from the repo root:

```bash
uv run detectivepotty serve --config config.yaml
```

Open <http://127.0.0.1:8000>. The portal is a keyboard-first, two-pane "review console": an independently-scrollable event list on the left and the selected event (clip, crops/frames, metadata, label panel) on the right. It writes labels back into `metadata.json`. The command bar shows a status filter, a camera filter, and a progress meter ("X of Y labeled"); the sidebar/header report how many events match the current filter and the total recorded on disk. If the bundle hasn't been built yet, `serve` still starts and the page tells you to run `npm run build`. (For active frontend development with hot reload, see [Development](#development).)

> One recorded event = one detected potty behavior, **not** one input file. A clip with no qualifying stationary-dwell behavior produces zero events, and a busy clip can produce several. So 4 input files may legitimately yield 3 events. The status filter defaults to **All**; switching it to `Unlabeled` hides events once you label them.

**Two timestamps, clearly separated.** Each event shows both:

- **Event time** (primary, `utc_ts`) — when the behavior happened in the footage. For file cameras this is real recording time (see [Deterministic file timeline](#deterministic-file-timeline)); when it can only be approximated, a small basis hint appears (`approx · file time` for `file_mtime`, `approx · run time` for `runtime_now`).
- **Generated** (secondary, `recorded_at`) — when this detection run wrote the record, shown relative ("3h ago"). This is bookkeeping, not when the dog acted.

Label workflow:

1. Select an event and watch the clip / inspect crops.
2. Choose `Pee`, `Poop`, `Not potty`, or `Unknown`.
3. Optionally pick which dog it was (from the `global.dogs` roster) in the **Dog** selector — by click, or with `⇧1`…`⇧N` to assign the Nth roster dog and `⇧0` to unassign.
4. Pick status (`labeled`, `rejected`, or `uncertain`), add an optional note, and **Save**.

The **Dog** label is a manual human identity tag (the `dog` field in `metadata.json`); automatic per-dog re-identification is intentionally out of scope for v0.

**Pose overlay on crops.** When the [pose backend](#pose-optional-additive) is enabled, the crops strip draws the detected keypoint skeleton on each posed crop, and the toggle above the crops is **on by default**. Posed crops carry a small teal dot; un-posed crops are de-emphasized while the overlay is on, and a one-line note states the exact coverage (e.g. "Pose dots on 30 of 121 crops…"). The dots are deliberately a **subset** of the crops, and a low count is expected — not a tracking failure. Two things gate it:

- **Sampling.** Pose is computed top-down and **independently per frame** (there is no frame-to-frame pose tracking to lose). To keep runs fast it executes on an **evenly-spaced subset of at most 30 frames** per event (`max_pose_frames`), only over the candidate window (`pose.candidate_only`). A crop written for a frame that wasn't sampled simply has no skeleton.
- **Quality gating.** Each sampled frame is still best-effort and is skipped when no confident keypoints come back — a small/occluded/blurry/IR-night dog, a tiny crop, or keypoint confidence below `pose.min_keypoint_conf`.

So **posed crops ≈ min(detected frames, 30) − low-quality skips.** Short, clear, close-up events (often the live-feed clips) can be fully posed; long or low-quality events show dots on roughly 30 evenly-spread frames. The detail panel reports the posed-frame count and the pose `coverage` fraction alongside the strip note.

**Keyboard shortcuts.** The portal is built for fast triage; press `?` in the app for the full reference. Shortcuts are suppressed while you're typing in a text field.

| Key | Action |
| --- | --- |
| `j` / `↓`, `k` / `↑` | Next / previous event |
| `n` / `N` | Next / previous **unlabeled** event |
| `g` / `G` | Jump to top / bottom of the list |
| `1` / `2` / `3` / `0` | Set label: Pee / Poop / Not potty / Unknown |
| `⇧1`…`⇧N` / `⇧0` | Assign the Nth roster dog / unassign |
| `r` / `u` | Stage status rejected / uncertain |
| `s` / `Enter` | Save label (only when there are unsaved changes) |
| `/` | Focus the camera filter |
| `Space` | Play / pause the clip |
| `a` | Toggle auto-advance (jump to next unlabeled after a successful save) |
| `v` | Toggle the Live feed (Review ⇄ Live) |
| `?` | Toggle the help overlay |
| `Esc` | Close the overlay, blur the focused input, or leave the Live feed |

### Real-time alerts (Live)

The portal keeps itself current without a manual refresh. The backend exposes a Server-Sent Events stream (`GET /api/stream`) that the page subscribes to; the moment the running pipeline finalizes a new event on disk, it shows up in the portal within a couple of seconds — no polling lag. This works because `serve` watches the **dataset on disk**, decoupled from the separate `run` process that writes it. A slow background poll runs as a safety net and the stream auto-reconnects, so the UI stays correct even if the connection drops (or a proxy buffers it). `npm run dev` proxies the stream through Vite, so live updates work in development too.

- On the **Review** page, new arrivals surface as a non-disruptive **"N new events · click to load"** banner. It never steals your selection or discards an in-progress label — click it when you're ready and the list refreshes.
- A dedicated **Live** page (`v`, or the Review | Live toggle in the command bar) shows a reverse-chronological feed that prepends events as they land. Each arrival can raise a **browser notification** (the `notify` toggle, which requests permission) and/or a **sound** chime (the `sound` toggle); both are off by default. Clicking a live event opens it in the Review console.

> **Latency floor.** Live removes _polling_ delay, not _pipeline_ delay. An event isn't written until its capture's post-roll ends (`cameras[].post_roll_s`), so a behavior is surfaced roughly `post_roll_s` after it happens. Alerting earlier — at the trigger/candidate stage, before post-roll — is future work.

## Re-running detection (reruns & dedupe)

### Deterministic file timeline

For file cameras, each frame is timestamped as `base_wall_ts + frame_idx / fps`. The anchor `base_wall_ts` is derived once per open from the **first source that succeeds** (the chosen source is recorded as `time_basis`):

1. `filename` — the recording time parsed from the clip filename (e.g. UniFi exports like `... 6-6-2026, 19.46.40 GMT+10 - ....mp4`). This is the real wall-clock the dog acted, converted to UTC.
2. `file_mtime` — the file's modification time, when the filename carries no parseable time. Lower confidence (surfaced as `approx · file time`) but stable across reruns of the same untouched file.
3. `runtime_now` — last resort if even `stat()` fails. The only non-deterministic basis (surfaced as `approx · run time`).

This replaces the old behavior, where every run re-anchored the file timeline to a fresh `datetime.now()`, which made `utc_ts` meaningless and made the same footage look like a brand-new event on each run. Now the same clip+frame yields the **same `utc_ts` across runs** whenever the basis is `filename`/`file_mtime`. Live (Protect/RTSP) sources are unchanged.

Crucially, **dedupe does not depend on the wall-clock anchor.** Each event also persists source-relative offsets — `source_start_s` / `source_end_s` (start/end minus the anchor) — which are identical across runs of the same clip regardless of which basis was used. Rerun matching keys on these offsets, so dedupe is correct even under the `runtime_now` fallback. The anchor governs display honesty; the offsets govern dedupe.

### Carrying labels forward

Detection is meant to be re-run as you tune thresholds. With `global.dedupe_reruns: true` (the default), re-running over the same source **does not** pile up duplicate event directories next to the old ones. Instead, each freshly-detected event is matched against the events already on disk for the same camera + source, and on a match it:

- **reuses the prior event's identity** and **carries your human review forward** — `label`, `label_status`, `dog`, the review note, and `labeled_at`;
- **refreshes the media** (frames, crops, clip) so you review the latest detector output; and
- **supersedes the old directory** after the replacement metadata/media has been committed.

Matching is per camera + sanitized source, by (in priority order) an exact Protect `event_id`, overlapping source-relative `[source_start_s, source_end_s]` offsets (the anchor-independent key for file reruns), overlapping wall-clock `[start_ts, end_ts]` intervals, or a start time within `global.rerun_match_tolerance_s` seconds (the fallback for older events recorded before offsets/`end_ts` were stored). Events written by the current run are never matched against each other.

Safety guarantees:

- **Conflicts are never auto-resolved.** If one new event matches several prior events whose human labels disagree, nothing is carried forward and nothing is deleted — the conflict is logged and every prior is left in place.
- **Your labeled work is never silently dropped.** A previously labeled event that the new run no longer reproduces is kept, not deleted.
- **No stale media in the browser.** Each event exposes a `media_version` (derived from the `metadata.json` modification time) that the portal appends to every media URL, so refreshed clips/crops bust the cache while unchanged media stays cached.

Set `dedupe_reruns: false` to restore the legacy append-everything behavior.

### Cleaning up older duplicates

If you already have duplicate events from before this feature (or from runs with it disabled), collapse them in place without re-detecting:

```bash
uv run detectivepotty dedupe-events --config config.yaml --dry-run   # preview
uv run detectivepotty dedupe-events --config config.yaml             # apply
```

It groups events by camera + source and time (`rerun_match_tolerance_s` window), keeps the newest-media copy of each duplicate cluster, carries labels forward with the same conflict guard, and deletes the rest. `--dry-run` prints exactly what it *would* keep, remove, and flag as a conflict without touching disk.

### Cleaning up pre-determinism legacy duplicates

Events recorded before the [deterministic file timeline](#deterministic-file-timeline) have `now()`-anchored timestamps and no source offsets, so a fresh rerun can't dedupe against them — `dedupe-events` (which clusters by time) also can't reliably collapse them. `cleanup-legacy` quarantines just those legacy duplicates so a clean rerun can regenerate honest, deduplicated events:

```bash
uv run detectivepotty cleanup-legacy --config config.yaml            # dry run (default)
uv run detectivepotty cleanup-legacy --config config.yaml --apply    # quarantine
```

It is conservative and reversible. Per camera + source, an event is removed **only** if it is unlabeled with zero human signal (no `dog`, note, or `labeled_at`), lacks deterministic-era markers (`recorded_at`/`end_ts`), **and** its source video still exists on disk (so a rerun can regenerate an equivalent). Removable events are **moved to `<dataset>/.trash/<timestamp>/`**, not deleted, so the operation can be undone. Every reviewed event and every deterministic-era event is preserved; events whose source video is missing are skipped and reported. The dry run prints the kept/removable/skipped breakdown before you `--apply`.

> If essentially all of your events predate this fix, the simplest one-time migration is to remove the old dataset (back it up first — it is gitignored) and re-run detection on your file cameras. The rerun regenerates clean, deduplicated events with real recording-time timestamps; any human labels on the old events would need to be re-applied.

## Dataset layout

```text
<dataset_dir>/<camera>/<YYYY-MM-DD>/events/<YYYYMMDDTHHMMSSZ>_<camera>_<track>_<eventId>/
    clip.mp4
    frames/000.jpg ...
    crops/000.jpg ...
    metadata.json
```

Event directories are UTC-sortable, idempotent, filesystem-safe, and secret-free. `metadata.json` includes camera IDs/names, sanitized source ID, trigger reason, timestamps (the event start `utc_ts` and `end_ts`, the `recorded_at` wall-clock write time used by rerun matching, the source-relative `source_start_s`/`source_end_s` offsets, and `time_basis` describing how the file timeline was anchored — `filename`/`file_mtime`/`runtime_now`), config hash, model info, detections/tracks, frame records, crop boxes, `classifier_guess`, `classifier_confidence`, `label`, `label_status`, and `dog` (manual identity label, `null` until assigned).

`classifier_guess` is the weak v0 heuristic. `label` and `label_status` are the human-reviewed truth fields used for training.

## Development

```bash
uv run pytest -q
uv run ruff check .
```

The integration tests are offline: they inject fake detectors and do not require GPU, model downloads, cameras, or network access.

### Frontend

The review portal lives in `src/detectivepotty/web/frontend/` (Svelte 5 + Vite + TypeScript). From that directory:

```bash
npm install        # first run only
npm run check      # svelte-check type + a11y check
npm run build      # production bundle into dist/ (what `serve` serves)
npm run dev        # hot-reloading dev server on http://localhost:5173
```

`npm run dev` is self-sufficient: it serves the UI **and** boots the FastAPI backend for you (proxying `/api` to it on `:8000`), so a single command gives you a working portal with hot reload:

```bash
cd src/detectivepotty/web/frontend && npm run dev
```

Open the dev URL it prints (`http://localhost:5173`, or the next free port), not the backend's `:8000`, to get hot reload. Notes:

- The backend it launches uses `config.yaml` at the repo root by default; point it elsewhere with `DETECTIVEPOTTY_CONFIG=path/to/config.yaml npm run dev`. Its logs are prefixed with `[backend]` in the same terminal.
- If you already have `detectivepotty serve` running on `:8000`, the dev server detects and reuses it instead of starting a second one. Set `DETECTIVEPOTTY_NO_BACKEND=1` to never auto-start (e.g. when you run the backend yourself).
- The auto-started backend is shut down when you stop `npm run dev`.
- After editing `vite.config.ts`, restart `npm run dev` so the change takes effect.

For just reviewing events (no UI changes), skip the dev server entirely and use the built bundle via `serve` as in [Review and label events](#review-and-label-events).

## Roadmap

- Train a custom pee/poop classifier on the high-resolution dog crops and stored original-resolution boxes.
- Replace the heuristic guess with a trained model once enough labeled data exists.
- Add stronger tracking such as ByteTrack and better multi-dog handling.
- Improve posture modeling beyond bbox height/aspect heuristics.
- Expand multi-camera live workflows and hard-negative/background capture.

DetectivePotty is a prototype. Generic potty-event detection is the useful v0 output; pee-vs-poop classification is intentionally treated as unreliable until the labeled dataset is large enough to train and validate a real classifier.
