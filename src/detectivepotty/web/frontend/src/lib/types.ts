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
  protect_recording_exists: boolean;
  relative_dir: string;
  media_version: number;
}

export interface MediaItem {
  name: string;
  url: string;
}

export interface EventMedia {
  clip: string | null;
  protect_recording: string | null;
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
