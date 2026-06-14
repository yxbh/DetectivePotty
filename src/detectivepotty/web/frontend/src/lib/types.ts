export interface EventSummary {
  event_id: string;
  camera: string;
  camera_id: string | null;
  camera_name: string | null;
  utc_ts: string | null;
  end_ts: string | null;
  recorded_at: string | null;
  source_start_s: number | null;
  source_end_s: number | null;
  time_basis: string | null;
  trigger_reason: string | null;
  classifier_guess: string | null;
  classifier_confidence: number | null;
  label: string;
  label_status: string;
  multi_dog: boolean;
  ambiguous: boolean;
  dog: string | null;
  thumbnail_url: string | null;
  frames_count: number;
  crops_count: number;
  relative_dir: string;
  media_version: number;
}

export interface MediaItem {
  name: string;
  url: string;
}

export interface EventMedia {
  clip: string | null;
  frames: MediaItem[];
  crops: MediaItem[];
  crops_overlay: MediaItem[];
}

export interface EventDetail {
  summary: EventSummary;
  metadata: Record<string, unknown>;
  media: EventMedia;
}

export interface EventsPage {
  events: EventSummary[];
  unfilteredTotal: number | null;
}

export interface EventFilters {
  labelStatus: string;
  camera: string;
}

export interface LabelPayload {
  label: string;
  label_status: string;
  note: string | null;
  dog: string | null;
}

/** Mutable label-editor draft, owned by App.svelte (single source of truth). */
export interface LabelDraft {
  label: string;
  status: string;
  dog: string;
  note: string;
}

// --- Detection tuner (/tune) ---------------------------------------------

export interface TuneEntry {
  name: string;
  kind: "dir" | "video";
  path: string;
  size?: number;
}

export interface TuneListing {
  /** "" at the synthetic top level (the list of roots). */
  path: string;
  /** "" = go to root list, a dir path = parent dir, null = no parent (top). */
  parent: string | null;
  entries: TuneEntry[];
}

export interface TuneDetection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  class_name: string;
}

export interface TuneKeypoint {
  name: string;
  x: number;
  y: number;
  confidence: number;
}

export interface TunePose {
  bbox: number[];
  keypoints: TuneKeypoint[];
}

/** Allow-list of selectable YOLO weights for the model picker. */
export interface TuneModelList {
  models: string[];
  default: string;
  /** Per-`.mlpackage` baked max batch size (e.g. 16 = batched, 1 = single-frame). */
  coreml_batch?: Record<string, number>;
}

/** Result of a one-off CoreML export: the new model + refreshed allow-list. */
export interface TuneExportResult {
  model: string;
  models: string[];
  default: string;
  /** Per-`.mlpackage` baked max batch size (e.g. 16 = batched, 1 = single-frame). */
  coreml_batch?: Record<string, number>;
}

/** Result of the decoupled pose pass (`POST /api/tune/pose`). */
export interface TunePoseResult {
  index: number;
  pose: TunePose[];
  pose_available: boolean;
}

/** Batched pose results for a contiguous run of frames
 *  (`POST /api/tune/pose-range`). Each entry is shaped like {@link TunePoseResult}. */
export interface TunePoseRangeResult {
  frames: TunePoseResult[];
}

/** Clip properties for index<->time mapping (no inference). */
export interface TuneMeta {
  path: string;
  total_frames: number | null;
  fps: number;
  width: number;
  height: number;
  duration: number;
}

/** Detections (+optional pose) for one frame — the async overlay buffer's payload. */
export interface TuneDetectResult {
  path: string;
  index: number;
  total_frames: number | null;
  fps: number;
  width: number;
  height: number;
  model: string;
  detection_floor: number;
  detections: TuneDetection[];
  pose: TunePose[];
  pose_available: boolean;
}

/** Batched detections for a contiguous frame window — the backfill filler's
 *  payload. Each entry is shaped like {@link TuneDetectResult}. */
export interface TuneDetectRangeResult {
  model: string;
  frames: TuneDetectResult[];
}

/** One top-N "object in scene" row (any class, no dog filter) with its box. */
export interface TuneSceneObject {
  class_name: string;
  confidence: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** Top-N all-class detections for one frame (`GET /api/tune/scene`) — the
 *  diagnostic "objects in scene" list. Each object carries its original-frame box
 *  so the client can optionally overlay it; this tells the reviewer what the
 *  detector sees on a frame, including non-dog classes, and where. */
export interface TuneSceneResult {
  path: string;
  index: number;
  total_frames: number | null;
  fps: number;
  width: number;
  height: number;
  model: string;
  detection_floor: number;
  objects: TuneSceneObject[];
}

/** One detection box carrying a persistent track ID (the tracker overlay). */
export interface TuneTrackedDetection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  class_name: string;
  track_id: string;
}

/** Per-frame tracked boxes for one sampled frame (`GET /api/tune/track-range`). */
export interface TuneTrackedFrame {
  index: number;
  detections: TuneTrackedDetection[];
}

/** De-fragmentation stats for a tracked range — the model x tracker comparison. */
export interface TuneTrackStats {
  /** Which backend produced these: `ours` | `bytetrack` | `botsort` | `botsort_reid`. */
  tracker: string;
  n_tracks: number;
  track_ids: string[];
  n_sampled_frames: number;
  n_detections: number;
  n_spans: number;
  n_presence_windows: number;
  spans_per_window: number;
  sample_every: number;
  /** `ours`-tracker knobs; `null` for the Ultralytics backends (own yaml params). */
  iou_threshold: number | null;
  max_age_frames: number | null;
  center_dist_gate: number | null;
  /** Ultralytics tracker knobs used for this run; `null` for the `ours` backend. */
  ultralytics: TuneUltralyticsTrackerParams | null;
}

/** Terminal record of the NDJSON Track-range stream (everything but the frames,
 *  which arrive incrementally as `frames` records during the forward pass). */
export interface TuneTrackStreamDone {
  model: string;
  start: number;
  count: number;
  fps: number;
  total_frames: number | null;
  detection_floor: number;
  stats: TuneTrackStats;
}

/** The tracker backends the Tune surface offers. `off` = per-frame boxes (no
 *  tracking); `ours` = the harvest IoU `Tracker` replay (every model incl. CoreML);
 *  the `bytetrack`/`botsort*` options are Ultralytics native tracking (`.pt`-only). */
export type TuneTracker = "off" | "ours" | "bytetrack" | "botsort" | "botsort_reid";

/** Per-run Ultralytics tracking knobs. `null` threshold fields keep YAML defaults. */
export interface TuneUltralyticsTrackerParams {
  conf: number;
  track_high_thresh: number | null;
  track_low_thresh: number | null;
  new_track_thresh: number | null;
  track_buffer: number | null;
  match_thresh: number | null;
  proximity_thresh: number | null;
  appearance_thresh: number | null;
  with_reid: boolean;
}

/** Client-side Track range request params, shared by streaming and non-streaming APIs. */
export interface TuneTrackRequestParams {
  sampleEvery: number;
  iouThreshold: number;
  maxAgeFrames: number;
  centerDistGate: number;
  ultralytics: TuneUltralyticsTrackerParams;
}

// --- Range labeling (/label) ---------------------------------------------

/** One harvested clip's listing row (identity + labeling progress). */
/** A detected-object class and how many of this clip's detections carried it. */
export interface ClassCount {
  class_name: string;
  count: number;
}

export interface LabelClipSummary {
  span_id: string;
  clip_path: string;
  source_id: string;
  camera_id: string | null;
  camera_name: string | null;
  date: string;
  span_start_utc: string | null;
  span_end_utc: string | null;
  fps: number;
  frame_count: number;
  frame_times_s: number[] | null;
  width: number;
  height: number;
  duration_s: number;
  detect_conf: number | null;
  model_name: string | null;
  class_distribution: ClassCount[];
  track_id: string | null;
  n_detections: number;
  labeled: boolean;
  n_ranges: number;
  n_trainable_ranges: number;
  behaviors: string[];
  dogs: string[];
  scene_id: string | null;
  scene_size: number;
}

/** A recorded detection box for a track at one clip frame. */
export interface LabelTrackBox {
  clip_frame_idx: number;
  bbox: { x1: number; y1: number; x2: number; y2: number };
  confidence: number;
  class_name?: string;
}

/** A dog track present in a clip's window (own track or a sibling's). */
export interface LabelPresentTrack {
  span_id: string;
  track_id: string;
  is_self: boolean;
  camera_name: string | null;
  boxes: LabelTrackBox[];
}

/** One labeled frame range bound to a dog track (mirrors `LabelRange`). */
export interface LabelRangeItem {
  start_frame: number;
  end_frame: number;
  start_s: number;
  end_s: number;
  behavior: string;
  dog: string;
  track_id: string | null;
  time_basis?: string;
  created_at?: string;
}

/** The `labels.json` body for one clip. */
export interface ClipLabelsBody {
  schema_version?: string;
  clip?: string;
  ranges: LabelRangeItem[];
}

/** Full payload for the labeling screen (summary + tracks + existing labels). */
export interface LabelClipDetail extends LabelClipSummary {
  tracks: Record<string, LabelTrackBox[]>;
  present_tracks: Record<string, LabelPresentTrack>;
  n_tracks: number;
  labels: ClipLabelsBody;
}

/** Fixed enum choices the labeler renders. */
export interface LabelVocabulary {
  behaviors: string[];
  dogs: string[];
}

/** Response of `GET /api/label/clips`. */
export interface LabelClipList {
  clips: LabelClipSummary[];
  vocabulary: LabelVocabulary;
}
