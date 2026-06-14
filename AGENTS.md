# AGENTS.md

Agent onboarding + guardrails for DetectivePotty. Read this first, then see
[`README.md`](README.md) (setup/usage) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
(design/contracts) for depth. This file stays short on purpose — link out, don't duplicate.

## What this is

A prototype that detects when a dog goes potty from UniFi Protect / ONVIF cameras, records
reviewable events (clip + frames + dog crops + `metadata.json`), and serves a local web app to
human-label each event `pee` / `poop` / `not_potty` / `unknown`. The v0 pee-vs-poop output is a
**weak guess, not ground truth** — the real goal is to build a clean labeled training set.

## Environment & commands

- **Toolchain:** [`uv`](https://docs.astral.sh/uv/), Python **3.12 only** (`>=3.12,<3.13`).
- **Install:** `uv sync` (detection/core). Pose backend is opt-in: `uv sync --extra pose`.
- **Test:** `uv run pytest -q` — fully offline, no GPU/model/network. Single test:
  `uv run pytest tests/test_pipeline.py -q` or `uv run pytest -k name -q`.
- **Lint:** `uv run ruff check src/ tests/`.
- **Run anything via `uv run`** (e.g. `uv run detectivepotty run --config config.yaml`,
  `uv run detectivepotty serve --config config.yaml`, `uv run detectivepotty detect-file ...`).
  Full CLI usage is in [`README.md`](README.md).
- **Loading secrets:** the usual `set -a; source .env; set +a` before `uv run`. **Quote** `.env`
  values containing shell metacharacters (e.g. `&` in an RTSP URL) — unquoted, `source` treats `&`
  as "run in background" and the var ends up empty. Suspect this first when a credential/stream
  isn't being picked up.

Always run **ruff + the full suite** and confirm both are clean before calling a change done.

## Frontend (Svelte review portal)

The review UI is a Svelte 5 + Vite + TypeScript app in `src/detectivepotty/web/frontend/` (the old
vanilla-JS `web/static/` portal is gone). Two non-obvious gotchas — full dev loop is in
[`README.md`](README.md#frontend):

- `serve` serves the **prebuilt** `web/frontend/dist/` bundle, so run `npm run build` after editing
  frontend source. For iterative UI work use `npm run dev` (hot reload on `:5173`, auto-boots the
  backend on `:8000`; open `:5173`, not `:8000`).
- Svelte CSS is **component-scoped**: a class defined in `App.svelte` won't style markup rendered by
  another component (e.g. `LiveFeed.svelte`). Use a global stylesheet or redefine it in the
  consuming component.
- **Verify UI changes with the Playwright MCP server** (configured globally, so it's already
  available): start `npm run dev` and point it at the `:5173` URL to click through the portal
  (event list, labeling, live feed, toggles) instead of eyeballing screenshots.

## Hard constraints — do not break these

- **numpy pinned `>=1.23,<2`.** Do NOT upgrade to numpy 2. The optional DeepLabCut pose extra caps
  `numpy<2`, and pinning lets pose share the single base env. Nothing in core needs numpy 2.
- **Pose is an opt-in extra.** Core/detection and **every unit test must work without it installed.**
  Tests exercise pose through an injected `infer_fn` / `MockPoseEstimator` — never a real model
  download or GPU. Don't add a hard import of `deeplabcut` to a core/test path.
- **Secrets via env vars only** (`DETECTIVEPOTTY_NVR_API_KEY` or `DETECTIVEPOTTY_NVR_USERNAME` /
  `DETECTIVEPOTTY_NVR_PASSWORD`). Never put credentials/tokens in YAML or commit them.
- **GPU inference is serialized by an internal lock** because MPS/torch isn't reliably safe for
  concurrent model execution. Keep it — parallelism comes from I/O (RTSP/decode/encode), not inference.
- **Python 3.12 only.** Avoid 3.13+ (the torch/Ultralytics stack lags).

## Testing & regression discipline

- Tests are offline and inject fake detectors — no cameras, models, GPU, or network.
- **Pose-OFF baseline on `configs/baseline.yaml` is 9 events** (7 cameras, `dwell_trigger_s`
  default 2 s): sample 1 / 1946 1 / 1949 2 / 2240 2 / 0903 1 / 0908 1 / **1729 1**. The trigger is
  **dwell-only** — a non-suppressed track that reads stationary continuously for at least
  `dwell_trigger_s` becomes a candidate. The old bbox **squat** trigger was removed entirely (it was
  unreliable on high/top-down cameras and missed the 17:29 Apollo potty); at the old 5 s default
  dwell-only dropped two real short pees (the Gromit sample and 0903), so the default was lowered to
  2 s, which recovers them and adds Apollo. A pose-disabled change that shifts these counts is a
  regression; over-capture is acceptable by design (every event is `needs_label`). (To run e2e on
  real clips, copy a config to a temp file and point `global.dataset_dir` at a temp dir —
  `data/`/`dataset/` are gitignored.)
- Add tests with new behavior; keep the suite green.

## Domain invariants (easy to get wrong)

- **Detect small, crop big:** YOLO runs on downscaled frames for speed, but boxes are mapped back to
  original-resolution frames before saving frames/crops. Don't save crops from the downscaled frame.
- **One event = one potty behavior, not one input file.** A quiet clip yields 0 events; a busy one
  yields several.
- **`classifier_guess` / `classifier_confidence` are a weak v0 heuristic.** `label` / `label_status`
  (human-reviewed) are the training-truth fields. Every event starts `unlabeled`. If guesses look
  wrong (e.g. "everything is 65% poop"), first confirm pose is actually enabled in the config being
  run — a config with no `pose:` block is heuristic-only — before suspecting a wiring bug.
- **Pee vs poop from posture is genuinely ambiguous** (squat-pee looks poop-like). Guesses carry
  `needs_label=True`; don't treat a mis-guess as a wiring bug.
- **Native UniFi vs third-party ONVIF cameras.** ONVIF cameras behind the NVR often can't be
  reliably re-streamed through Protect's RTSPS proxy — FFmpeg then floods `Empty H.264 RTP packet` /
  "session invalidated" (a symptom, not a bug). Stream those directly with a per-camera RTSP
  override (`DETECTIVEPOTTY_<NAME>_RTSP_URL`, quoted) instead of through UniFi.

## Pose subsystem status

Fully built and validated against the real SuperAnimal-Quadruped (DeepLabCut 3.x, HRNet-W32) backend
on MPS, **but kept default-OFF / experimental**: `pose.enabled=false`, `enable_pose_gate=false`,
`box_union_window_s=0.0`. The pose gate runs inference on every sampled frame (~2× cost) and only ever
*adds* candidate windows (it never removes the `covered_long_enough` recall guard). Leave the defaults
off unless a task explicitly says otherwise.

## Repo map (`src/detectivepotty/`)

- `pipeline.py` — orchestrates per-camera worker threads; wires sources → detect → track → event → record.
- `config.py` — Pydantic config models (`global`, `protect`, `cameras[]`, `pose`); loads YAML + env secrets.
- `sources/` — `VideoSource` impls: `file.py`, `rtsp.py`, `rolling_buffer.py` (warm pre-roll buffer).
- `detect/yolo.py` — `DogDetector` (Ultralytics YOLO, downscaled inference).
- `tracking.py` — `Tracker` + `temporal_box_union` (multi-frame box recovery for pose crops).
- `potty_event.py` — `PottyEventDetector` state machine (sustained stationary dwell → potty candidate).
- `events.py` — core data models/enums (`Detection`, `Track`, `FrameRecord`, `CropRecord`,
  `EventMetadata`, `Label`/`LabelStatus`/`ClassifierGuess`) + `write_metadata_json`.
- `classify/` — `base.py`, `heuristic.py` (weak pee/poop guess), `pose.py` (pose-based classifier, opt-in).
- `pose/` — keypoint backends: `MockPoseEstimator` (tests) + SuperAnimal/DeepLabCut real backend, gate.
- `recording/` — `recorder.py`, `clip_writer.py`, `dataset.py` (on-disk event layout), `retention.py`.
- `web/` — FastAPI review/labeling app (`app.py`, `dataset_index.py`) + `frontend/` (Svelte 5 + Vite
  portal, built to `frontend/dist/`).
- `protect/` — UniFi Protect client + animal smart-detect trigger. `triggers/` — trigger interfaces + YOLO fallback.
- `geometry.py` — `BBox` (crop/expand/`union`) and coordinate helpers.

## Git / workflow

- `data/`, `dataset/`, `outputs/`, `*.pt`/`*.onnx`/`*.engine`, and local/secret files are **gitignored**.
- **Commit only when explicitly asked.** Sessions run in-place on `main`; don't branch/push/commit on
  your own initiative. There is significant in-flight **uncommitted** work (the whole pose subsystem) —
  check `git status` before assuming the working tree is clean.
- When you do commit (on request), append:
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.

## Where to look next

[`README.md`](README.md) · [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ·
[`config.example.yaml`](config.example.yaml) · the session `plan.md` + checkpoints (prior decisions/history).
