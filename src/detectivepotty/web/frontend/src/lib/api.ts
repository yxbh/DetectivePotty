import type {
  EventDetail,
  EventFilters,
  EventSummary,
  EventsPage,
  LabelPayload,
  TuneDetectResult,
  TuneExportResult,
  TuneFrame,
  TuneListing,
  TuneMeta,
  TuneModelList,
} from "./types";

const EVENTS_LIMIT = 200;

// Surface the server's error detail when present (the dev proxy returns a JSON
// {detail} with a "start the backend" hint on connection failure). Falls back to
// a status-code message, with an extra nudge for the gateway codes that mean the
// API backend is unreachable.
async function errorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.clone().json()) as { detail?: unknown };
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
  try {
    const response = await fetch("/api/dogs");
    if (!response.ok) {
      return [];
    }
    const data = (await response.json()) as { dogs?: unknown };
    return Array.isArray(data.dogs) ? (data.dogs as string[]) : [];
  } catch {
    return [];
  }
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
  const events = await jsonOrThrow<EventSummary[]>(response);
  const header = response.headers.get("X-Unfiltered-Count");
  const unfilteredTotal = header == null ? null : Number(header);
  return {
    events,
    unfilteredTotal:
      unfilteredTotal == null || Number.isNaN(unfilteredTotal) ? null : unfilteredTotal,
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

export async function fetchTuneFrame(
  path: string,
  index: number,
  pose: boolean,
  signal?: AbortSignal,
): Promise<TuneFrame> {
  const params = new URLSearchParams({
    path,
    index: String(index),
    pose: pose ? "1" : "0",
  });
  const response = await fetch(`/api/tune/frame?${params.toString()}`, { signal });
  return jsonOrThrow<TuneFrame>(response);
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

/** URL for the raw clip the <video> element streams (Range-seekable). */
export function tuneClipUrl(path: string): string {
  const params = new URLSearchParams({ path });
  return `/api/tune/clip?${params.toString()}`;
}
