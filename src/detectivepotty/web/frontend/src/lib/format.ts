import type { EventSummary } from "./types";

function parseDate(value: string | null | undefined): Date | null {
  if (!value) {
    return null;
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatTime(value: string | null | undefined): string {
  const date = parseDate(value);
  if (!date) {
    return value ? value : "Unknown time";
  }
  return date.toLocaleString();
}

/** Compact wall-clock for the event timeline: "Jun 6 · 19:46:40". */
export function formatClock(value: string | null | undefined): string {
  const date = parseDate(value);
  if (!date) {
    return "—";
  }
  const day = date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  const time = date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
  return `${day} · ${time}`;
}

/** Coarse relative age, e.g. "just now", "12m ago", "3h ago", "2d ago". */
export function formatRelative(value: string | null | undefined): string {
  const date = parseDate(value);
  if (!date) {
    return "—";
  }
  const seconds = Math.round((Date.now() - date.getTime()) / 1000);
  if (seconds < 45) {
    return "just now";
  }
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ago`;
  }
  const hours = Math.round(minutes / 60);
  if (hours < 24) {
    return `${hours}h ago`;
  }
  const days = Math.round(hours / 24);
  if (days < 7) {
    return `${days}d ago`;
  }
  const weeks = Math.round(days / 7);
  if (weeks < 5) {
    return `${weeks}w ago`;
  }
  return date.toLocaleDateString();
}

/**
 * How trustworthy is utc_ts? Returns a short qualifier the UI shows next to the
 * event time when the wall-clock anchor was inferred rather than authoritative.
 */
export function basisHint(timeBasis: string | null | undefined): string | null {
  switch (timeBasis) {
    case "file_mtime":
      return "approx · file time";
    case "runtime_now":
      return "approx · run time";
    default:
      // config / filename / explicit / unknown(live) are authoritative.
      return null;
  }
}

export function formatConfidence(value: number | null | undefined): string {
  return value == null ? "" : `${Math.round(value * 100)}%`;
}

export function formatGuess(
  guess: string | null | undefined,
  confidence: number | null | undefined,
): string {
  const conf = formatConfidence(confidence);
  return `${guess || "unknown"}${conf ? ` ${conf}` : ""}`.trim();
}

const LABEL_TEXT: Record<string, string> = {
  pee: "Pee",
  poop: "Poop",
  not_potty: "Not potty",
  unknown: "Unknown",
};

export function labelText(label: string | null | undefined): string {
  if (!label) {
    return "Unknown";
  }
  return LABEL_TEXT[label] ?? label;
}

export function flagsText(multiDog: boolean, ambiguous: boolean): string {
  const flags: string[] = [];
  if (multiDog) {
    flags.push("multi-dog");
  }
  if (ambiguous) {
    flags.push("ambiguous");
  }
  return flags.length ? flags.join(", ") : "none";
}

export function summaryFlags(event: EventSummary): string {
  return flagsText(event.multi_dog, event.ambiguous);
}
