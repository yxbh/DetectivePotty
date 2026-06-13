import type { EventSummary } from "./types";

export function notificationsSupported(): boolean {
  return typeof Notification !== "undefined";
}

/**
 * Request browser-notification permission. Must be called from a user gesture
 * (a toggle click), per browser policy. Returns the resulting permission.
 */
export async function requestNotificationPermission(): Promise<NotificationPermission> {
  if (!notificationsSupported()) {
    return "denied";
  }
  if (Notification.permission !== "default") {
    return Notification.permission;
  }
  try {
    return await Notification.requestPermission();
  } catch {
    return "denied";
  }
}

export function showEventNotification(summary: EventSummary): void {
  if (!notificationsSupported() || Notification.permission !== "granted") {
    return;
  }
  const camera = summary.camera || "camera";
  const guess = summary.classifier_guess ? ` · ${summary.classifier_guess}` : "";
  try {
    const note = new Notification(`Potty event · ${camera}`, {
      body: `New event detected${guess}. Open the portal to review.`,
      tag: summary.event_id,
    });
    note.onclick = () => {
      try {
        window.focus();
      } catch {
        /* ignore */
      }
      note.close();
    };
  } catch {
    /* notifications are best-effort */
  }
}

let audioCtx: AudioContext | null = null;

/** Short synthesized two-note chime — no asset, no autoplay-blocked <audio>. */
export function playChime(): void {
  try {
    const Ctx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctx) {
      return;
    }
    audioCtx = audioCtx ?? new Ctx();
    const ctx = audioCtx;
    if (ctx.state === "suspended") {
      void ctx.resume();
    }
    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(880, now);
    osc.frequency.setValueAtTime(1320, now + 0.09);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.12, now + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.32);
    osc.connect(gain).connect(ctx.destination);
    osc.start(now);
    osc.stop(now + 0.34);
  } catch {
    /* audio is best-effort */
  }
}
