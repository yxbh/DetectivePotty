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
