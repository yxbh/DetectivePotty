import type {
  ClipLabelsBody,
  EventDetail,
  EventFilters,
  EventSummary,
  EventsPage,
  LabelClipDetail,
  LabelClipList,
  LabelPayload,
  TuneDetectRangeResult,
  TuneDetectResult,
  TuneExportResult,
  TuneListing,
  TuneMeta,
  TuneModelList,
  TunePoseRangeResult,
  TuneSceneResult,
  TuneTracker,
  TuneTrackRequestParams,
  TuneTrackStreamDone,
  TuneTrackedFrame,
} from "./types";

const EVENTS_LIMIT = 200;

// Surface the server's error detail when present (the dev proxy returns a JSON
// {detail} with a "start the backend" hint on connection failure). Falls back to
// a status-code message, with an extra nudge for the gateway codes that mean the
// API backend is unreachable.
async function errorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.clone().json()) as {
      detail?: unknown;
      error?: { message?: unknown };
    };
    if (typeof body.error?.message === "string" && body.error.message.trim()) {
      return body.error.message;
    }
    if (typeof body.detail === "string" && body.detail.trim()) {
      return body.detail;
    }
  } catch {
    // Non-JSON error body; fall through to a generic message.
  }
  if (response.status >= 502 && response.status <= 504) {
    return `HTTP ${response.status} — the API backend isn't responding yet. It starts automatically with \`npm run dev\`; check the terminal for [backend] errors, then retry.`;
  }
  return `HTTP ${response.status}`;
}

async function jsonOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
  return (await response.json()) as T;
}

export async function fetchDogs(): Promise<string[]> {
  const response = await fetch("/api/dogs");
  const data = await jsonOrThrow<{ dogs?: unknown }>(response);
  if (!Array.isArray(data.dogs) || data.dogs.some((dog) => typeof dog !== "string")) {
    throw new Error("Invalid dog roster response");
  }
  return data.dogs;
}

export async function fetchEvents(filters: EventFilters): Promise<EventsPage> {
  const params = new URLSearchParams({ limit: String(EVENTS_LIMIT) });
  if (filters.labelStatus) {
    params.set("label_status", filters.labelStatus);
  }
  const camera = filters.camera.trim();
  if (camera) {
    params.set("camera", camera);
  }
  const response = await fetch(`/api/events?${params.toString()}`);
  const data = await jsonOrThrow<{
    events: EventSummary[];
    total: number;
    unfiltered_total: number | null;
    limit: number;
    offset: number;
    filters: EventsPage["filters"];
  }>(response);
  return {
    events: data.events,
    total: data.total,
    unfilteredTotal: data.unfiltered_total,
    limit: data.limit,
    offset: data.offset,
    filters: data.filters,
  };
}

export async function fetchEventDetail(eventId: string): Promise<EventDetail> {
  const response = await fetch(`/api/events/${encodeURIComponent(eventId)}`);
  return jsonOrThrow<EventDetail>(response);
}

export async function saveLabel(
  eventId: string,
  payload: LabelPayload,
): Promise<EventSummary> {
  const response = await fetch(
    `/api/events/${encodeURIComponent(eventId)}/label`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
  return jsonOrThrow<EventSummary>(response);
}

// Reruns can replace media under a reused event_id, so a fixed per-event media
// URL would serve stale (browser-cached) clips/crops. Appending the event's
// media_version busts the cache only when the media actually changed.
export function versioned(url: string | null | undefined, version: number): string | null {
  if (!url) {
    return null;
  }
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}v=${version}`;
}

// --- Detection tuner (/tune) ---------------------------------------------

export async function fetchTuneFiles(path: string): Promise<TuneListing> {
  const params = new URLSearchParams();
  if (path) {
    params.set("path", path);
  }
  const response = await fetch(`/api/tune/files?${params.toString()}`);
  return jsonOrThrow<TuneListing>(response);
}

export async function fetchTuneModels(): Promise<TuneModelList> {
  const response = await fetch("/api/tune/models");
  return jsonOrThrow<TuneModelList>(response);
}

/** Export a discovered .pt model to a GPU-safe CoreML .mlpackage (one-off). */
export async function exportCoreml(model: string): Promise<TuneExportResult> {
  const response = await fetch("/api/tune/export-coreml", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
  return jsonOrThrow<TuneExportResult>(response);
}

export async function fetchTuneMeta(
  path: string,
  signal?: AbortSignal,
): Promise<TuneMeta> {
  const params = new URLSearchParams({ path });
  const response = await fetch(`/api/tune/meta?${params.toString()}`, { signal });
  return jsonOrThrow<TuneMeta>(response);
}

export async function fetchTuneDetect(
  path: string,
  index: number,
  model: string,
  pose: boolean,
  signal?: AbortSignal,
): Promise<TuneDetectResult> {
  const params = new URLSearchParams({
    path,
    index: String(index),
    model,
    pose: pose ? "1" : "0",
  });
  const response = await fetch(`/api/tune/detect?${params.toString()}`, { signal });
  return jsonOrThrow<TuneDetectResult>(response);
}

/** Top-N all-class detections for one frame (the "objects in scene" diagnostic).
 *  Unlike `fetchTuneDetect` this skips the dog filter, so a reviewer can see whether
 *  a frame with no dog box is empty, a sub-threshold dog, or an animal classed as
 *  something else (cat/person/...). Read-only — never touches the harvest pipeline. */
export async function fetchTuneScene(
  path: string,
  index: number,
  model: string,
  topN: number,
  signal?: AbortSignal,
): Promise<TuneSceneResult> {
  const params = new URLSearchParams({
    path,
    index: String(index),
    model,
    top_n: String(topN),
  });
  const response = await fetch(`/api/tune/scene?${params.toString()}`, { signal });
  return jsonOrThrow<TuneSceneResult>(response);
}

/** Batched detections for a contiguous `[start, start+count)` frame window. One
 *  sequential decode + one `detect_batch` forward replaces `count` single-frame
 *  round-trips, which is what lifts GPU utilization off the batch-1 floor. The
 *  backend caps `count` at `tune_detection_batch_size`, so the returned
 *  `frames` may be shorter than requested (also at EOF). */
export async function fetchTuneDetectRange(
  path: string,
  start: number,
  count: number,
  model: string,
  signal?: AbortSignal,
): Promise<TuneDetectRangeResult> {
  const params = new URLSearchParams({
    path,
    start: String(start),
    count: String(count),
    model,
  });
  const response = await fetch(`/api/tune/detect-range?${params.toString()}`, { signal });
  return jsonOrThrow<TuneDetectRangeResult>(response);
}

function addTrackParams(query: URLSearchParams, params: TuneTrackRequestParams): void {
  query.set("sample_every", String(params.sampleEvery));
  query.set("iou_threshold", String(params.iouThreshold));
  query.set("max_age_frames", String(params.maxAgeFrames));
  query.set("center_dist_gate", String(params.centerDistGate));
  query.set("ultra_conf", String(params.ultralytics.conf));
  for (const [key, value] of Object.entries(params.ultralytics)) {
    if (key === "conf" || key === "with_reid" || value === null) continue;
    query.set(key, String(value));
  }
}

/** Stream a Track-range pass as newline-delimited JSON so the timeline fills and a
 *  progress bar advances as the in-order forward pass computes (instead of waiting
 *  for the whole pass). `onFrames` fires per decode chunk, `onDone` once at the end,
 *  `onError` on a non-200 or a mid-stream `error` line. Tracking is sequential +
 *  stateful, so this is always a single 0→end pass (never cursor-first). */
export async function streamTuneTrackRange(
  path: string,
  start: number,
  count: number,
  model: string,
  tracker: TuneTracker,
  params: TuneTrackRequestParams,
  handlers: {
    onFrames: (frames: TuneTrackedFrame[]) => void;
    onDone: (done: TuneTrackStreamDone) => void;
    onError: (message: string) => void;
  },
  signal?: AbortSignal,
): Promise<void> {
  const query = new URLSearchParams({
    path,
    start: String(start),
    count: String(count),
    model,
    tracker,
  });
  addTrackParams(query, params);
  const response = await fetch(`/api/tune/track-range-stream?${query.toString()}`, {
    signal,
  });
  if (!response.ok || !response.body) {
    handlers.onError(await errorMessage(response));
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const handleLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    const rec = JSON.parse(trimmed) as
      | { type: "frames"; frames: TuneTrackedFrame[] }
      | ({ type: "done" } & TuneTrackStreamDone)
      | { type: "error"; detail: string };
    if (rec.type === "frames") handlers.onFrames(rec.frames);
    else if (rec.type === "done") handlers.onDone(rec);
    else if (rec.type === "error") handlers.onError(rec.detail);
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline = buffer.indexOf("\n");
    while (newline !== -1) {
      handleLine(buffer.slice(0, newline));
      buffer = buffer.slice(newline + 1);
      newline = buffer.indexOf("\n");
    }
  }
  if (buffer.trim()) handleLine(buffer);
}

/** Pose pass for client-supplied boxes across a contiguous run of frames — no
 *  YOLO re-run. Drives the decoupled, proactive pose lane so flipping the overlay
 *  to pose is instant, and batches the whole run into one server-side forward. */
export async function fetchTunePoseRange(
  path: string,
  frames: { index: number; boxes: number[][] }[],
  signal?: AbortSignal,
): Promise<TunePoseRangeResult> {
  const response = await fetch("/api/tune/pose-range", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, frames }),
    signal,
  });
  return jsonOrThrow<TunePoseRangeResult>(response);
}

/** URL for the raw clip the <video> element streams (Range-seekable). */
export function tuneClipUrl(path: string): string {
  const params = new URLSearchParams({ path });
  return `/api/tune/clip?${params.toString()}`;
}

// --- Range labeling (/label) ---------------------------------------------

export async function fetchLabelClips(): Promise<LabelClipList> {
  const response = await fetch("/api/label/clips");
  return jsonOrThrow<LabelClipList>(response);
}

export async function fetchLabelClipDetail(spanId: string): Promise<LabelClipDetail> {
  const response = await fetch(`/api/label/clips/${encodeURIComponent(spanId)}`);
  return jsonOrThrow<LabelClipDetail>(response);
}

export async function saveLabelClip(
  spanId: string,
  body: ClipLabelsBody,
): Promise<LabelClipDetail> {
  const response = await fetch(
    `/api/label/clips/${encodeURIComponent(spanId)}/labels`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
  return jsonOrThrow<LabelClipDetail>(response);
}

/** URL for a harvested clip's video (Range-seekable, by span id). */
export function labelClipVideoUrl(spanId: string): string {
  return `/api/label/clips/${encodeURIComponent(spanId)}/video`;
}
