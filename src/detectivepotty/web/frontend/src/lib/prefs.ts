import { writable } from "svelte/store";

function boolStore(key: string, fallback: boolean) {
  function read(): boolean {
    try {
      const raw = localStorage.getItem(key);
      return raw == null ? fallback : raw === "1";
    } catch {
      return fallback;
    }
  }
  const store = writable<boolean>(read());
  store.subscribe((value) => {
    try {
      localStorage.setItem(key, value ? "1" : "0");
    } catch {
      /* ignore storage failures (private mode, etc.) */
    }
  });
  return store;
}

// Versioned key so the new on-by-default actually takes effect for reviewers who
// already have the old "poseOverlay" key persisted to "0" from a prior session.
// Their explicit toggles still persist — under the new key — going forward.
export const poseOverlay = boolStore("poseOverlay.v2", true);

/**
 * When on, a successful label save jumps to the next unlabeled event so the
 * reviewer keeps a hands-on-keyboard rhythm. Off by default — opt-in throughput.
 */
export const autoAdvance = boolStore("autoAdvance", false);

/**
 * Live page: fire a browser Notification for each new event (default off;
 * enabling prompts for OS permission via a user gesture).
 */
export const liveNotifications = boolStore("liveNotifications", false);

/** Live page: play a short chime for each new event (default off). */
export const liveSound = boolStore("liveSound", false);

const TUNE_LAST_DIR_KEY = "tuneLastDir";
const REVIEW_FILTERS_KEY = "reviewFilters.v1";
const REVIEW_STATUS_VALUES = new Set(["", "unlabeled", "labeled", "rejected", "uncertain"]);

/**
 * Last directory the Tune file browser was viewing, so reopening the tab resumes
 * where the reviewer left off instead of dropping back to the root list. Stored
 * as the server-resolved absolute path ("" = the synthetic root list). Read/write
 * are explicit (not a reactive store) so only *successfully loaded* dirs persist —
 * a stale/removed path must never get pinned.
 */
export function loadTuneLastDir(): string {
  try {
    return localStorage.getItem(TUNE_LAST_DIR_KEY) ?? "";
  } catch {
    return "";
  }
}

export function saveTuneLastDir(path: string): void {
  try {
    if (path) {
      localStorage.setItem(TUNE_LAST_DIR_KEY, path);
    } else {
      localStorage.removeItem(TUNE_LAST_DIR_KEY);
    }
  } catch {
    /* ignore storage failures (private mode, etc.) */
  }
}

export interface ReviewFiltersPref {
  status: string;
  camera: string;
}

export function loadReviewFilters(): ReviewFiltersPref {
  try {
    const raw = localStorage.getItem(REVIEW_FILTERS_KEY);
    if (!raw) {
      return { status: "", camera: "" };
    }
    const parsed = JSON.parse(raw) as { status?: unknown; camera?: unknown };
    const status = typeof parsed.status === "string" && REVIEW_STATUS_VALUES.has(parsed.status)
      ? parsed.status
      : "";
    const camera = typeof parsed.camera === "string" ? parsed.camera : "";
    return { status, camera };
  } catch {
    return { status: "", camera: "" };
  }
}

export function saveReviewFilters(filters: ReviewFiltersPref): void {
  try {
    const status = REVIEW_STATUS_VALUES.has(filters.status) ? filters.status : "";
    const camera = filters.camera.trim();
    if (!status && !camera) {
      localStorage.removeItem(REVIEW_FILTERS_KEY);
      return;
    }
    localStorage.setItem(REVIEW_FILTERS_KEY, JSON.stringify({ status, camera }));
  } catch {
    /* ignore storage failures (private mode, etc.) */
  }
}
