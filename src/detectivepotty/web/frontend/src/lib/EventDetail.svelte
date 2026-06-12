<script lang="ts">
  import type { EventDetail, LabelDraft } from "./types";
  import { versioned } from "./api";
  import {
    basisHint,
    formatClock,
    formatConfidence,
    formatRelative,
    formatTime,
  } from "./format";
  import { poseOverlay } from "./prefs";

  interface Props {
    detail: EventDetail | null;
    dogs: string[];
    loading: boolean;
    error: string | null;
    draft: LabelDraft;
    dirty: boolean;
    saving: boolean;
    saveStatus: string;
    onsave: () => void;
  }

  let { detail, dogs, loading, error, draft, dirty, saving, saveStatus, onsave }: Props =
    $props();

  const LABEL_BUTTONS: Array<[string, string, string]> = [
    ["pee", "Pee", "1"],
    ["poop", "Poop", "2"],
    ["not_potty", "Not potty", "3"],
    ["unknown", "Unknown", "0"],
  ];

  let hero = $state<{ src: string; alt: string } | null>(null);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let meta = $derived((detail?.metadata ?? {}) as Record<string, any>);
  let summary = $derived(detail?.summary ?? null);
  let mediaVersion = $derived(detail?.summary.media_version ?? 0);
  let overlayByName = $derived(
    new Map((detail?.media.crops_overlay ?? []).map((item) => [item.name, item.url])),
  );
  let hasOverlays = $derived(overlayByName.size > 0);
  let showOverlay = $derived(hasOverlays && $poseOverlay);
  let eventHint = $derived(basisHint(summary?.time_basis));

  let duration = $derived.by<string | null>(() => {
    if (!summary) {
      return null;
    }
    if (summary.source_start_s != null && summary.source_end_s != null) {
      return `${(summary.source_end_s - summary.source_start_s).toFixed(1)}s`;
    }
    if (summary.utc_ts && summary.end_ts) {
      const span = (new Date(summary.end_ts).getTime() - new Date(summary.utc_ts).getTime()) / 1000;
      return Number.isFinite(span) && span >= 0 ? `${span.toFixed(1)}s` : null;
    }
    return null;
  });

  let dogChips = $derived.by<Array<{ name: string; hint: string | null }>>(() => {
    const chips = dogs.map((name, i) => ({
      name,
      hint: i < 9 ? `⇧${i + 1}` : null,
    }));
    // Surface an off-roster manual entry as a selected chip (no shortcut).
    if (draft.dog && !dogs.includes(draft.dog)) {
      chips.unshift({ name: draft.dog, hint: null });
    }
    return chips;
  });

  // Reset the in-detail hero whenever a different event loads.
  $effect(() => {
    void detail?.summary.event_id;
    hero = null;
  });

  function posturePhrase(ps: unknown): string {
    if (!ps || typeof ps !== "object") {
      return "n/a";
    }
    const s = ps as Record<string, unknown>;
    const num = (v: unknown): number | null =>
      typeof v === "number" && Number.isFinite(v) ? v : null;
    const parts: string[] = [];
    const dur = num(s.stationary_duration_s);
    const durThr = num(s.stationary_threshold_s);
    if (dur != null) {
      parts.push(`stationary ${dur.toFixed(1)}s${durThr != null ? ` ≥ ${durThr.toFixed(1)}s` : ""}`);
    }
    const dwell = num(s.dwell_duration_s);
    const dwellThr = num(s.dwell_trigger_s);
    if (dwell != null) {
      parts.push(`held ${dwell.toFixed(1)}s${dwellThr != null ? ` ≥ ${dwellThr.toFixed(1)}s` : ""}`);
    }
    return parts.length ? parts.join(" · ") : "n/a";
  }

  interface SummaryRow {
    title: string;
    value: string;
    hint: string;
  }

  let summaryRows = $derived.by<SummaryRow[]>(() => {
    if (!detail) {
      return [];
    }
    const latency =
      meta.trigger_latency_s == null
        ? "n/a"
        : `${Number(meta.trigger_latency_s).toFixed(2)}s`;
    const flags: string[] = [];
    if (meta.multi_dog) {
      flags.push("multi-dog");
    }
    if (meta.ambiguous) {
      flags.push("ambiguous");
    }
    return [
      {
        title: "Guess",
        value: `${meta.classifier_guess || "unknown"} ${formatConfidence(meta.classifier_confidence)}`.trim(),
        hint: "Weak v0 classifier prefill — not ground truth. Your saved label is the real training signal.",
      },
      {
        title: "Saved label",
        value: `${meta.label || "unknown"} / ${meta.label_status || "unlabeled"}`,
        hint: "The human-reviewed label + status currently written to metadata.json.",
      },
      {
        title: "Dog",
        value: meta.dog || "unassigned",
        hint: "Manual identity tag from the configured roster (Shift+1…N). Not auto-detected.",
      },
      {
        title: "Trigger",
        value: meta.trigger_reason || "unknown",
        hint: "What started capture: a Protect Animal smart-detect, or the YOLO fallback trigger.",
      },
      {
        title: "Latency",
        value: latency,
        hint: "Trigger latency: delay between the behavior appearing in the footage and the Protect smart-detect notification. 'n/a' for file/YOLO events with no Protect notification.",
      },
      {
        title: "Flags",
        value: flags.length ? flags.join(", ") : "none",
        hint: "Ambiguity markers — multi-dog (more than one dog in the window) / ambiguous (tracker identity may have swapped). 'none' is ideal.",
      },
      {
        title: "Posture",
        value: posturePhrase(meta.extra?.posture_summary),
        hint: "Why this fired: how long the dog held still (dwell) measured against the camera's trigger threshold.",
      },
    ];
  });

  let poseRows = $derived.by<Array<[string, string]>>(() => {
    const features = meta.extra?.pose?.features;
    if (!features) {
      return [];
    }
    const candidates: Array<[string, unknown, string]> = [
      ["Spine angle", features.spine_angle_deg, "°"],
      ["Hip offset", features.hip_offset_ratio, ""],
      ["Tail angle", features.tail_angle_deg, "°"],
      ["Centroid motion", features.centroid_motion_ratio, ""],
      ["Dwell", features.dwell_duration_s, "s"],
      ["Coverage", features.coverage, ""],
    ];
    return candidates
      .filter(([, value]) => value != null)
      .map(([name, value, unit]) => [name, `${Number(value).toFixed(2)}${unit}`]);
  });

  let poseFrameCount = $derived(meta.extra?.pose?.keypoints?.length ?? 0);

  let posedCropCount = $derived(
    detail ? detail.media.crops.filter((c) => overlayByName.has(c.name)).length : 0,
  );
  let totalCropCount = $derived(detail?.media.crops.length ?? 0);
  let poseCoverageHint = $derived.by<string | null>(() => {
    if (!hasOverlays || totalCropCount === 0) {
      return null;
    }
    const base = `Pose dots on ${posedCropCount} of ${totalCropCount} crops`;
    if (posedCropCount >= totalCropCount) {
      return `${base} — every frame in this event was posed.`;
    }
    return `${base}. Pose runs on an evenly-sampled subset (max 30 frames) and skips small / occluded / low-confidence frames, so bare crops are expected — not a tracking failure.`;
  });

  function cropUrl(name: string, fallback: string): string {
    const base = showOverlay && overlayByName.has(name) ? overlayByName.get(name)! : fallback;
    return versioned(base, mediaVersion) ?? base;
  }

  function showHero(src: string, alt: string): void {
    hero = { src, alt };
  }
</script>

<div class="detail">
  {#if loading}
    <div class="empty-state">Loading event…</div>
  {:else if error}
    <div class="error-state">Failed to load event: {error}</div>
  {:else if !detail}
    <div class="empty-state">Select an event to begin reviewing.</div>
  {:else}
    {@const media = detail.media}
    <header class="detail-head">
      <div class="head-main">
        <h2>{detail.summary.camera}</h2>
        <span class="event-id mono">{detail.summary.event_id}</span>
      </div>
      <div class="stamps">
        <div class="stamp">
          <span class="stamp-key">Event time</span>
          <span class="stamp-val mono" title={formatTime(summary?.utc_ts)}>
            {formatClock(summary?.utc_ts)}
          </span>
          <span class="stamp-sub">
            {#if eventHint}{eventHint}{:else}in footage{/if}{#if duration} · {duration}{/if}
          </span>
        </div>
        <div class="stamp">
          <span class="stamp-key">Generated</span>
          <span class="stamp-val mono" title={formatTime(summary?.recorded_at)}>
            {formatRelative(summary?.recorded_at)}
          </span>
          <span class="stamp-sub">this run</span>
        </div>
      </div>
    </header>

    <div class="video-wrap">
      {#if media.clip}
        {#key detail.summary.event_id + mediaVersion}
          <!-- svelte-ignore a11y_media_has_caption -->
          <video controls preload="metadata" playsinline loop src={versioned(media.clip, mediaVersion)}
          ></video>
        {/key}
      {:else}
        <div class="empty-state">No clip.mp4 found.</div>
      {/if}
    </div>

    {#if hero}
      <div class="hero-slot">
        <img class="hero-image" src={hero.src} alt={hero.alt} />
      </div>
    {/if}

    <section class="label-panel">
      <div class="panel-head">
        <h3>Label</h3>
        {#if dirty}<span class="dirty-dot" title="Unsaved changes">unsaved</span>{/if}
      </div>
      <div class="label-buttons">
        {#each LABEL_BUTTONS as [value, text, key] (value)}
          <button
            type="button"
            class="label-btn"
            class:selected={draft.label === value}
            onclick={() => (draft.label = value)}
          >
            <span>{text}</span>
            <kbd>{key}</kbd>
          </button>
        {/each}
      </div>
      <div class="dog-group">
        <span class="dog-heading">Dog</span>
        <div class="dog-chips">
          <button
            type="button"
            class="dog-chip"
            class:selected={!draft.dog}
            onclick={() => (draft.dog = "")}
          >
            <span>Unassigned</span>
            <kbd>⇧0</kbd>
          </button>
          {#each dogChips as chip (chip.name)}
            <button
              type="button"
              class="dog-chip"
              class:selected={draft.dog === chip.name}
              onclick={() => (draft.dog = chip.name)}
            >
              <span>{chip.name}</span>
              {#if chip.hint}<kbd>{chip.hint}</kbd>{/if}
            </button>
          {/each}
        </div>
      </div>
      <div class="form-row">
        <label>
          Status
          <select bind:value={draft.status}>
            <option value="labeled">labeled</option>
            <option value="rejected">rejected</option>
            <option value="uncertain">uncertain</option>
          </select>
        </label>
        <label class="note-field">
          Note
          <textarea bind:value={draft.note} placeholder="Optional training note"></textarea>
        </label>
        <button
          type="button"
          class="btn-primary save-btn"
          onclick={onsave}
          disabled={saving || !dirty}
        >
          {saving ? "Saving…" : "Save"}
          <kbd>S</kbd>
        </button>
      </div>
      {#if saveStatus}<p class="save-status muted">{saveStatus}</p>{/if}
    </section>

    <section class="summary-panel">
      <div class="summary-grid">
        {#each summaryRows as row (row.title)}
          <div class="summary-card" title={row.hint}>
            <span class="info-label">{row.title}</span><strong>{row.value}</strong>
          </div>
        {/each}
      </div>
    </section>

    {#if media.crops.length > 0}
      <section class="media-strip">
        <div class="strip-head">
          <h3>Crops</h3>
          <label class="overlay-toggle" title="Draw the detected keypoint skeleton on crops. Pose runs on an evenly-sampled subset (max 30 frames) and skips low-quality frames, so only some crops have dots.">
            <input type="checkbox" bind:checked={$poseOverlay} disabled={!hasOverlays} />
            Pose overlay{#if !hasOverlays}<span class="muted"> · none</span>{/if}
          </label>
        </div>
        {#if poseCoverageHint}
          <p class="strip-note muted">{poseCoverageHint}</p>
        {/if}
        <div class="strip-grid">
          {#each media.crops as crop (crop.name)}
            {@const posed = overlayByName.has(crop.name)}
            <button
              type="button"
              class="strip-item"
              class:posed
              class:no-pose={showOverlay && !posed}
              title={posed ? "Pose estimated for this frame" : "No pose for this frame (not sampled, or low-quality crop)"}
              onclick={() => showHero(cropUrl(crop.name, crop.url), crop.name)}
            >
              <img src={cropUrl(crop.name, crop.url)} alt={crop.name} loading="lazy" />
              {#if posed}<span class="pose-badge" aria-hidden="true"></span>{/if}
            </button>
          {/each}
        </div>
      </section>
    {/if}

    {#if media.frames.length > 0}
      <section class="media-strip">
        <h3>Frames</h3>
        <div class="strip-grid">
          {#each media.frames as frame (frame.name)}
            <button
              type="button"
              class="strip-item"
              onclick={() => showHero(versioned(frame.url, mediaVersion) ?? frame.url, frame.name)}
            >
              <img src={versioned(frame.url, mediaVersion)} alt={frame.name} loading="lazy" />
            </button>
          {/each}
        </div>
      </section>
    {/if}

    {#if media.protect_recording}
      <section class="media-strip">
        <h3>Protect recording</h3>
        <a class="protect-link" href={versioned(media.protect_recording, mediaVersion)}>
          Open Protect recording →
        </a>
      </section>
    {/if}

    {#if poseRows.length > 0}
      <section class="pose-panel">
        <h3>Pose features <span class="muted">({poseFrameCount} posed frames)</span></h3>
        <div class="summary-grid">
          {#each poseRows as [title, value] (title)}
            <div class="summary-card"><span>{title}</span><strong>{value}</strong></div>
          {/each}
        </div>
      </section>
    {/if}

    <details class="metadata-panel">
      <summary>Raw metadata</summary>
      <pre>{JSON.stringify(meta, null, 2)}</pre>
    </details>
  {/if}
</div>

<style>
  .detail {
    padding: 1.25rem 1.4rem 3rem;
    max-width: 64rem;
  }

  .detail-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1.5rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
  }

  .head-main {
    display: grid;
    gap: 0.3rem;
    min-width: 0;
  }

  .head-main h2 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 700;
  }

  .event-id {
    font-size: 0.7rem;
    color: var(--text-faint);
    word-break: break-all;
  }

  .stamps {
    display: flex;
    gap: 1.5rem;
    flex: 0 0 auto;
  }

  .stamp {
    display: grid;
    gap: 0.12rem;
  }

  .stamp-key {
    font-size: 0.64rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-faint);
  }

  .stamp-val {
    font-size: 0.98rem;
    color: var(--text);
    font-weight: 500;
  }

  .stamps .stamp:first-child .stamp-val {
    color: var(--amber);
  }

  .stamp-sub {
    font-size: 0.66rem;
    color: var(--text-faint);
  }

  .video-wrap {
    background: var(--bg-inset);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    overflow: hidden;
  }

  video,
  .hero-image {
    display: block;
    width: 100%;
    max-height: 58vh;
    object-fit: contain;
    background: var(--bg-inset);
  }

  .hero-slot {
    margin-top: 0.8rem;
    border: 1px solid var(--line);
    border-radius: var(--radius);
    overflow: hidden;
  }

  .summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
    gap: 0.7rem;
  }

  .summary-card {
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    padding: 0.7rem 0.8rem;
    background: var(--bg-1);
    cursor: help;
  }

  .summary-card span {
    display: block;
    color: var(--text-faint);
    font-size: 0.66rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
  }

  .summary-card .info-label {
    text-decoration: underline dotted var(--line-strong);
    text-underline-offset: 2px;
  }

  .summary-card strong {
    font-weight: 500;
    font-size: 0.9rem;
  }

  .label-panel,
  .summary-panel,
  .media-strip,
  .pose-panel,
  .metadata-panel {
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 1rem;
    background: var(--bg-1);
    margin-top: 1rem;
  }

  .label-panel {
    border-color: var(--line-strong);
    background: var(--bg-2);
  }

  .panel-head {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.7rem;
  }

  .panel-head h3 {
    margin: 0;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-dim);
  }

  .dirty-dot {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--amber);
    border: 1px solid rgba(245, 165, 36, 0.4);
    background: var(--amber-soft);
    border-radius: 999px;
    padding: 0.08rem 0.42rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .label-buttons {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(7.5rem, 1fr));
    gap: 0.5rem;
    margin-bottom: 0.85rem;
  }

  .label-btn {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    background: var(--bg-inset);
    border-color: var(--line-strong);
  }

  .label-btn.selected {
    background: var(--amber);
    border-color: var(--amber);
    color: #1a1204;
    font-weight: 700;
  }

  .label-btn.selected kbd {
    background: rgba(0, 0, 0, 0.18);
    color: #1a1204;
    border-color: rgba(0, 0, 0, 0.25);
  }

  .dog-group {
    margin-bottom: 0.85rem;
  }

  .dog-heading {
    display: block;
    color: var(--text-faint);
    font-size: 0.66rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
  }

  .dog-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
  }

  .dog-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.4rem 0.6rem;
    background: var(--bg-inset);
    border: 1px solid var(--line-strong);
    border-radius: var(--radius-sm);
    color: var(--text);
    cursor: pointer;
    font-size: 0.82rem;
  }

  .dog-chip:hover {
    border-color: var(--teal);
  }

  .dog-chip.selected {
    background: var(--teal);
    border-color: var(--teal);
    color: #04201d;
    font-weight: 700;
  }

  .dog-chip.selected kbd {
    background: rgba(0, 0, 0, 0.18);
    color: #04201d;
    border-color: rgba(0, 0, 0, 0.25);
  }

  .form-row {
    display: grid;
    grid-template-columns: minmax(7rem, 9rem) 1fr auto;
    gap: 0.7rem;
    align-items: end;
  }

  .save-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
  }

  .save-btn kbd {
    background: rgba(0, 0, 0, 0.18);
    color: inherit;
    border-color: rgba(0, 0, 0, 0.22);
  }

  .save-status {
    margin: 0.7rem 0 0;
    font-size: 0.82rem;
  }

  .strip-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
  }

  .media-strip h3,
  .pose-panel h3 {
    margin: 0 0 0.6rem;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-dim);
  }

  .strip-head h3 {
    margin: 0;
  }

  .overlay-toggle {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    margin: 0;
    font-size: 0.78rem;
    text-transform: none;
    letter-spacing: 0;
    color: var(--text-dim);
    cursor: pointer;
  }

  .overlay-toggle input {
    width: auto;
    cursor: pointer;
  }

  .strip-grid {
    display: flex;
    gap: 0.5rem;
    overflow-x: auto;
    padding-bottom: 0.3rem;
  }

  .strip-item {
    position: relative;
    flex: 0 0 auto;
    padding: 0;
    border: 2px solid transparent;
    border-radius: var(--radius-sm);
    background: transparent;
    cursor: pointer;
    line-height: 0;
  }

  .strip-item:hover {
    border-color: var(--teal);
  }

  .strip-item.posed {
    border-color: color-mix(in srgb, var(--teal) 45%, transparent);
  }

  .strip-item.posed:hover {
    border-color: var(--teal);
  }

  /* When the overlay is ON, fade frames that have no pose so the posed ones read first. */
  .strip-item.no-pose img {
    opacity: 0.5;
    filter: grayscale(0.4);
  }

  .pose-badge {
    position: absolute;
    top: 0.28rem;
    right: 0.28rem;
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 0 2px var(--bg-1), 0 0 6px color-mix(in srgb, var(--teal) 70%, transparent);
  }

  .strip-note {
    margin: 0 0 0.6rem;
    font-size: 0.74rem;
    line-height: 1.4;
    max-width: 60ch;
  }

  .strip-grid img {
    width: 7rem;
    height: 5rem;
    object-fit: cover;
    border-radius: calc(var(--radius-sm) - 2px);
    display: block;
  }

  .protect-link {
    color: var(--teal);
    text-decoration: none;
    font-weight: 600;
  }

  .protect-link:hover {
    text-decoration: underline;
  }

  .metadata-panel summary {
    cursor: pointer;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-dim);
  }

  pre {
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text-dim);
    font-family: var(--font-mono);
    font-size: 0.76rem;
    margin: 0.8rem 0 0;
  }

  @media (max-width: 850px) {
    .form-row {
      grid-template-columns: 1fr;
    }

    .stamps {
      gap: 1rem;
    }
  }
</style>
