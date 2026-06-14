<script lang="ts">
  import type { TuneTracker, TuneTrackStats } from "./types";
  import TuneNumberKnob from "./TuneNumberKnob.svelte";

  interface Props {
    tracker: TuneTracker;
    isMlpackage: boolean;
    selectedPath: string | null;
    tracking: boolean;
    tracked: boolean;
    trackDone: number;
    trackExpected: number;
    trackError: string | null;
    trackStats: TuneTrackStats | null;
    trackSampleEvery: number;
    trackIou: number;
    trackMaxAge: number;
    trackCenterGate: number;
    ultraConf: number;
    ultraTrackHigh: number;
    ultraTrackLow: number;
    ultraNewTrack: number;
    ultraTrackBuffer: number;
    ultraMatch: number;
    ultraProximity: number;
    ultraAppearance: number;
    ontracker: (tracker: TuneTracker) => void;
    oninvalidate: () => void;
    onruntrack: () => void | Promise<void>;
    oncanceltrack: () => void;
  }

  let {
    tracker,
    isMlpackage,
    selectedPath,
    tracking,
    tracked,
    trackDone,
    trackExpected,
    trackError,
    trackStats,
    trackSampleEvery = $bindable(),
    trackIou = $bindable(),
    trackMaxAge = $bindable(),
    trackCenterGate = $bindable(),
    ultraConf = $bindable(),
    ultraTrackHigh = $bindable(),
    ultraTrackLow = $bindable(),
    ultraNewTrack = $bindable(),
    ultraTrackBuffer = $bindable(),
    ultraMatch = $bindable(),
    ultraProximity = $bindable(),
    ultraAppearance = $bindable(),
    ontracker,
    oninvalidate,
    onruntrack,
    oncanceltrack,
  }: Props = $props();

  function isUltralyticsTracker(value: TuneTracker = tracker): boolean {
    return value === "bytetrack" || value === "botsort" || value === "botsort_reid";
  }

  function isBotsortTracker(value: TuneTracker = tracker): boolean {
    return value === "botsort" || value === "botsort_reid";
  }

  function trackStatsTitle(stats: TuneTrackStats): string {
    const base =
      "Distinct track IDs · harvest spans · merged presence windows · spans-per-window (the de-fragmentation metric)";
    if (!stats.ultralytics) return base;
    const u = stats.ultralytics;
    const parts = [
      `det-conf ${u.conf.toFixed(2)}`,
      u.track_high_thresh === null ? null : `high ${u.track_high_thresh.toFixed(2)}`,
      u.track_low_thresh === null ? null : `low ${u.track_low_thresh.toFixed(2)}`,
      u.new_track_thresh === null ? null : `new ${u.new_track_thresh.toFixed(2)}`,
      u.track_buffer === null ? null : `buffer ${u.track_buffer}`,
      u.match_thresh === null ? null : `match ${u.match_thresh.toFixed(2)}`,
      u.proximity_thresh === null ? null : `prox ${u.proximity_thresh.toFixed(2)}`,
      u.appearance_thresh === null ? null : `appear ${u.appearance_thresh.toFixed(2)}`,
    ].filter(Boolean);
    return `${base} · ${parts.join(" · ")}`;
  }
</script>

<div class="track-row">
  <label class="tracker">
    <span class="mono muted small">tracker</span>
    <select
      value={tracker}
      onchange={(event) => ontracker((event.currentTarget as HTMLSelectElement).value as TuneTracker)}
    >
      <option value="off">Off (per-frame)</option>
      <option value="ours">Ours (IoU + gate)</option>
      <option value="bytetrack" disabled={isMlpackage}>ByteTrack</option>
      <option value="botsort" disabled={isMlpackage}>BoT-SORT</option>
      <option value="botsort_reid" disabled={isMlpackage}>BoT-SORT + ReID</option>
    </select>
  </label>

  {#if tracker !== "off"}
    <div class="track-knobs mono small">
      <TuneNumberKnob
        label="stride"
        title="Sample every N source frames before tracking. Affects both Ours and Ultralytics; changing it requires a new track pass."
        min="1"
        max="60"
        step="1"
        bind:value={trackSampleEvery}
        oninput={oninvalidate}
      />
      {#if tracker === "ours"}
        <TuneNumberKnob
          label="iou"
          title="Minimum box overlap to associate a detection with an existing Ours track. Lower joins more jumps; higher splits more tracks."
          min="0"
          max="1"
          step="0.05"
          bind:value={trackIou}
        />
        <TuneNumberKnob
          label="max-age"
          title="Sampled frames an unmatched Ours track survives before it dies. Higher bridges longer gaps."
          min="0"
          max="300"
          step="1"
          bind:value={trackMaxAge}
        />
        <TuneNumberKnob
          label="gate"
          title="Center-distance OR-gate in box diagonals for Ours. 0 means IoU-only; higher reconnects larger jumps."
          min="0"
          max="20"
          step="0.1"
          bind:value={trackCenterGate}
        />
      {:else if isUltralyticsTracker() && !isMlpackage}
        <TuneNumberKnob
          label="det-conf"
          title="YOLO confidence floor passed to Ultralytics model.track(). Lower can expose more detections to the tracker; requires Re-track range."
          min="0"
          max="1"
          step="0.01"
          bind:value={ultraConf}
          oninput={oninvalidate}
        />
        <TuneNumberKnob
          label="track-high"
          title="Ultralytics track_high_thresh: high-confidence detections used for the primary association pass."
          min="0"
          max="1"
          step="0.01"
          bind:value={ultraTrackHigh}
          oninput={oninvalidate}
        />
        <TuneNumberKnob
          label="track-low"
          title="Ultralytics track_low_thresh: lower-confidence detections still eligible for secondary association."
          min="0"
          max="1"
          step="0.01"
          bind:value={ultraTrackLow}
          oninput={oninvalidate}
        />
        <TuneNumberKnob
          label="new-track"
          title="Ultralytics new_track_thresh: minimum confidence required to start a new track ID."
          min="0"
          max="1"
          step="0.01"
          bind:value={ultraNewTrack}
          oninput={oninvalidate}
        />
        <TuneNumberKnob
          label="buffer"
          title="Ultralytics track_buffer: sampled frames an unmatched track stays alive before removal."
          min="0"
          max="10000"
          step="1"
          bind:value={ultraTrackBuffer}
          oninput={oninvalidate}
        />
        <TuneNumberKnob
          label="match"
          title="Ultralytics match_thresh: association matching threshold. Higher is stricter; lower can bridge more uncertain matches."
          min="0"
          max="1"
          step="0.01"
          bind:value={ultraMatch}
          oninput={oninvalidate}
        />
        {#if isBotsortTracker()}
          <TuneNumberKnob
            label="prox"
            title="BoT-SORT proximity_thresh: spatial proximity gate before appearance matching. Lower is more permissive."
            min="0"
            max="1"
            step="0.01"
            bind:value={ultraProximity}
            oninput={oninvalidate}
          />
          <TuneNumberKnob
            label="appear"
            title="BoT-SORT appearance_thresh: appearance/ReID similarity threshold. Higher requires stronger visual match."
            min="0"
            max="1"
            step="0.01"
            bind:value={ultraAppearance}
            oninput={oninvalidate}
          />
        {/if}
      {/if}
    </div>
  {/if}

  {#if isUltralyticsTracker() && !isMlpackage}
    <span class="track-note mono small muted">
      Overrides Ultralytics YAML for this run · press Track range
    </span>
  {/if}

  {#if tracker !== "off"}
    <button
      type="button"
      class="track-btn"
      class:cancel={tracking}
      onclick={tracking ? oncanceltrack : onruntrack}
      disabled={!tracking && (!selectedPath || (isMlpackage && tracker !== "ours"))}
      title={tracking
        ? "Cancel the in-progress track pass"
        : "Decode + detect + track the whole clip in frame order, then scrub to see persistent track IDs"}
    >
      {tracking ? "Cancel" : tracked ? "Re-track range" : "Track range"}
    </button>
  {/if}

  {#if tracking}
    <span
      class="track-progress mono small muted"
      title="Forward 0→end track sweep progress (shown live in the track lane of the timeline strip)"
    >
      tracking… {Math.min(100, Math.round((trackDone / Math.max(1, trackExpected)) * 100))}%
      ({trackDone}/{trackExpected})
    </span>
  {/if}

  {#if trackError}
    <span class="export-error mono small" role="alert">{trackError}</span>
  {/if}

  {#if isMlpackage && tracker !== "off" && tracker !== "ours"}
    <span class="track-note mono small muted">
      Ultralytics trackers need a .pt model — pick a .pt to use this.
    </span>
  {/if}

  {#if trackStats}
    <div class="track-stats mono small" title={trackStatsTitle(trackStats)}>
      <span class="hud-yolo">{trackStats.tracker}</span>
      {#if trackStats.ultralytics}
        <span>conf {trackStats.ultralytics.conf.toFixed(2)}</span>
      {/if}
      <span>tracks {trackStats.n_tracks}</span>
      <span>spans {trackStats.n_spans}</span>
      <span>windows {trackStats.n_presence_windows}</span>
      <span class="frag">spans/win {trackStats.spans_per_window.toFixed(2)}</span>
    </div>
  {/if}
</div>

<style>
  .track-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.8rem;
    margin-top: 0.6rem;
    padding-top: 0.6rem;
    border-top: 1px solid var(--line, #243042);
  }

  .tracker {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }

  .tracker select {
    background: var(--bg-1, #141a24);
    border: 1px solid var(--line-strong, #324056);
    color: var(--text, #d8e0ec);
    border-radius: 6px;
    padding: 0.25rem 0.4rem;
    font-size: 0.78rem;
    font-family: ui-monospace, monospace;
  }

  .track-knobs {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.55rem;
  }

  .track-btn {
    background: var(--bg-1, #141a24);
    border: 1px solid var(--accent, #3f7d5a);
    color: var(--text, #d8e0ec);
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
    font-size: 0.76rem;
    font-family: ui-monospace, monospace;
    cursor: pointer;
    white-space: nowrap;
  }

  .track-btn:hover:not(:disabled) {
    border-color: var(--amber, #f0b35a);
  }

  .track-btn:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }

  .track-btn.cancel {
    border-color: var(--amber, #f0b35a);
    color: var(--amber, #f0b35a);
  }

  .track-progress {
    color: var(--text, #d8e0ec);
  }

  .track-note {
    max-width: 34ch;
  }

  .track-stats {
    display: inline-flex;
    align-items: center;
    gap: 0.7rem;
    color: var(--text, #d8e0ec);
  }

  .track-stats .frag {
    color: var(--amber, #f0b35a);
  }

  .hud-yolo {
    color: #28d17c;
  }

  .export-error {
    color: var(--amber, #f0b35a);
    max-width: 22ch;
  }

  .muted {
    color: var(--muted, #8a97a8);
  }

  .small {
    font-size: 0.74rem;
  }
</style>
