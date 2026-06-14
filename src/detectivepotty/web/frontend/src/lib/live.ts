import { writable } from "svelte/store";
import type { EventSummary } from "./types";
import { fetchEvents } from "./api";

// Newest-first running log shown on the Live page (capped so a long-running
// session can't grow it without bound).
export const liveEvents = writable<EventSummary[]>([]);
// Count of genuinely-new events that aren't yet in the Review list — drives the
// non-disruptive "N new events" banner.
export const liveNewCount = writable<number>(0);
// SSE health indicator (the slow poll keeps data correct even when this is off).
export const liveConnected = writable<boolean>(false);

const MAX_FEED = 50;
const POLL_MS = 20000;
const RECONNECT_MS = 15000;

// Ids already present in the Review list (App tells us via acknowledgeEvents),
// so reconnect reconciles and the safety-net poll never re-count/re-notify them.
const acknowledged = new Set<string>();
// Ids already ingested into the feed — dedupes the SSE push vs the poll/reconcile
// paths so an event is only ever surfaced once.
const seen = new Set<string>();
// Seen events that are not part of the currently-acknowledged Review list.
const pendingNew = new Set<string>();
const hooks = new Set<(summary: EventSummary) => void>();

let source: EventSource | null = null;
let pollTimer: ReturnType<typeof setInterval> | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let started = false;

/** Register a callback fired once per genuinely-new event (toast/sound/notify). */
export function onLiveEvent(cb: (summary: EventSummary) => void): () => void {
  hooks.add(cb);
  return () => hooks.delete(cb);
}

/** Mark events as already shown in Review; keeps hidden filtered events pending. */
export function acknowledgeEvents(ids: string[]): void {
  for (const id of ids) {
    acknowledged.add(id);
    pendingNew.delete(id);
  }
  liveNewCount.set(pendingNew.size);
}

function ingest(summary: EventSummary, notify: boolean): void {
  const id = summary?.event_id;
  if (!id || seen.has(id)) {
    return;
  }
  seen.add(id);
  liveEvents.update((list) => [summary, ...list].slice(0, MAX_FEED));
  if (!acknowledged.has(id)) {
    pendingNew.add(id);
    liveNewCount.set(pendingNew.size);
    if (notify) {
      for (const cb of hooks) {
        try {
          cb(summary);
        } catch {
          /* a misbehaving hook must not break the stream */
        }
      }
    }
  }
}

// One-shot catch-up used on (re)connect and by the safety-net poll. Acknowledged
// and already-seen events are suppressed inside ingest, so this is idempotent.
async function reconcile(notify: boolean): Promise<void> {
  try {
    const page = await fetchEvents({ labelStatus: "", camera: "" });
    // Oldest-first so the newest ends up on top of the feed.
    for (const ev of [...page.events].reverse()) {
      ingest(ev, notify);
    }
  } catch {
    /* transient; the next poll tick retries */
  }
}

function teardownSource(): void {
  if (source) {
    source.close();
    source = null;
  }
}

function scheduleReconnect(): void {
  if (reconnectTimer) {
    return;
  }
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, RECONNECT_MS);
}

function connect(): void {
  if (source || typeof EventSource === "undefined") {
    return;
  }
  try {
    source = new EventSource("/api/stream");
  } catch {
    return; // the safety-net poll still keeps the UI current
  }
  source.addEventListener("ready", () => {
    liveConnected.set(true);
    // Catch up on anything missed while disconnected.
    void reconcile(true);
  });
  source.addEventListener("new", (ev) => {
    try {
      ingest(JSON.parse((ev as MessageEvent).data) as EventSummary, true);
    } catch {
      /* ignore malformed frame */
    }
  });
  source.onerror = () => {
    liveConnected.set(false);
    // EventSource silently auto-reconnects while CONNECTING; only when it gives
    // up (CLOSED) do we tear down and schedule a manual retry. The slow poll
    // keeps data correct in the meantime either way.
    if (source && source.readyState === EventSource.CLOSED) {
      teardownSource();
      scheduleReconnect();
    }
  };
}

export function startLive(): void {
  if (started) {
    return;
  }
  started = true;
  connect();
  // Always-on safety net: guarantees correctness even if SSE is silently
  // buffered/dropped by a proxy. Cheap (one /api/events call) and fully deduped.
  pollTimer = setInterval(() => {
    void reconcile(true);
  }, POLL_MS);
}

export function stopLive(): void {
  started = false;
  teardownSource();
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  liveConnected.set(false);
}
