<script lang="ts">
  import { onMount } from "svelte";
  import {
    fetchLabelClips,
    fetchLabelClipDetail,
    labelClipVideoUrl,
    saveLabelClip,
  } from "./api";
  import type {
    ClipLabelsBody,
    LabelClipDetail,
    LabelClipSummary,
    LabelRangeItem,
    LabelVocabulary,
  } from "./types";

  const BEHAVIOR_KEYS: Record<string, string> = {
    "1": "pee",
    "2": "poop",
    "3": "not_potty",
    "4": "excluded",
  };

  let clips = $state<LabelClipSummary[]>([]);
  let vocabulary = $state<LabelVocabulary>({ behaviors: [], dogs: [] });
  let listLoading = $state(true);
  let listError = $state<string | null>(null);

  let selectedId = $state<string | null>(null);
  let detail = $state<LabelClipDetail | null>(null);
  let detailLoading = $state(false);
  let detailError = $state<string | null>(null);

  // Working copy of the clip's ranges (saved on demand). dirty tracks unsaved edits.
  let ranges = $state<LabelRangeItem[]>([]);
  let dirty = $state(false);
  let saving = $state(false);
  let saveStatus = $state<string | null>(null);

  let videoEl = $state<HTMLVideoElement | null>(null);
  let currentFrame = $state(0);
  let playing = $state(false);
  let markIn = $state<number | null>(null);
  let markOut = $state<number | null>(null);
  let pendingBehavior = $state("pee");
  let pendingDog = $state("unknown");

  const fps = $derived(detail && detail.fps > 0 ? detail.fps : 30);
  const totalFrames = $derived(detail ? Math.max(1, detail.frame_count) : 1);
  const trackId = $derived(detail?.track_id ?? null);
  const hasRvfc =
    typeof window !== "undefined" &&
    "requestVideoFrameCallback" in HTMLVideoElement.prototype;

  // The recorded detection box nearest the current frame for the span's track,
  // so the labeler always sees which dog the range will bind to.
  const activeBox = $derived.by(() => {
    if (!detail || trackId == null) return null;
    const boxes = detail.tracks[trackId];
    if (!boxes || boxes.length === 0) return null;
    let best = boxes[0];
    let bestDist = Math.abs(best.clip_frame_idx - currentFrame);
    for (const box of boxes) {
      const dist = Math.abs(box.clip_frame_idx - currentFrame);
      if (dist < bestDist) {
        best = box;
        bestDist = dist;
      }
    }
    return best;
  });

  onMount(loadClips);

  async function loadClips(): Promise<void> {
    listLoading = true;
    listError = null;
    try {
      const data = await fetchLabelClips();
      clips = data.clips;
      vocabulary = data.vocabulary;
      if (vocabulary.behaviors.length) pendingBehavior = vocabulary.behaviors[0];
      if (!selectedId && clips.length) void selectClip(clips[0].span_id);
    } catch (err) {
      listError = err instanceof Error ? err.message : String(err);
    } finally {
      listLoading = false;
    }
  }

  async function selectClip(spanId: string): Promise<void> {
    if (dirty && spanId !== selectedId) {
      const ok = confirm("Discard unsaved label changes for this clip?");
      if (!ok) return;
    }
    selectedId = spanId;
    detailLoading = true;
    detailError = null;
    detail = null;
    try {
      const data = await fetchLabelClipDetail(spanId);
      detail = data;
      ranges = data.labels.ranges.map((r) => ({ ...r }));
      dirty = false;
      saveStatus = null;
      currentFrame = 0;
      markIn = null;
      markOut = null;
    } catch (err) {
      detailError = err instanceof Error ? err.message : String(err);
    } finally {
      detailLoading = false;
    }
  }

  // --- video / frame sync -------------------------------------------------

  function clampFrame(frame: number): number {
    return Math.max(0, Math.min(totalFrames - 1, frame));
  }

  function syncFrame(): void {
    if (!videoEl || fps <= 0) return;
    currentFrame = clampFrame(Math.floor(videoEl.currentTime * fps + 1e-6));
  }

  function seekToFrame(frame: number): void {
    const target = clampFrame(frame);
    currentFrame = target;
    if (videoEl) videoEl.currentTime = (target + 0.5) / fps;
  }

  function stepFrame(delta: number): void {
    if (videoEl && !videoEl.paused) videoEl.pause();
    seekToFrame(currentFrame + delta);
  }

  function togglePlay(): void {
    if (!videoEl) return;
    if (videoEl.paused) void videoEl.play().catch(() => undefined);
    else videoEl.pause();
  }

  function onLoadedMeta(): void {
    if (videoEl) videoEl.currentTime = 0.5 / fps;
    if (hasRvfc && videoEl) {
      const cb = (): void => {
        syncFrame();
        if (videoEl) (videoEl as HTMLVideoElement).requestVideoFrameCallback(cb);
      };
      (videoEl as HTMLVideoElement).requestVideoFrameCallback(cb);
    }
  }

  // --- range editing ------------------------------------------------------

  function setMarkIn(): void {
    markIn = currentFrame;
    if (markOut != null && markOut < markIn) markOut = null;
  }

  function setMarkOut(): void {
    markOut = currentFrame;
    if (markIn != null && markIn > markOut) markIn = null;
  }

  function addRange(): void {
    if (!detail) return;
    const a = markIn ?? currentFrame;
    const b = markOut ?? currentFrame;
    const start = Math.min(a, b);
    const end = Math.max(a, b);
    ranges = [
      ...ranges,
      {
        start_frame: start,
        end_frame: end,
        start_s: start / fps,
        end_s: end / fps,
        behavior: pendingBehavior,
        dog: pendingDog,
        track_id: trackId,
      },
    ];
    dirty = true;
    saveStatus = null;
    markIn = null;
    markOut = null;
  }

  function deleteRange(idx: number): void {
    ranges = ranges.filter((_, i) => i !== idx);
    dirty = true;
    saveStatus = null;
  }

  function seekToRange(r: LabelRangeItem): void {
    seekToFrame(r.start_frame);
  }

  async function save(): Promise<void> {
    if (!detail || saving) return;
    saving = true;
    saveStatus = null;
    const body: ClipLabelsBody = {
      schema_version: detail.labels.schema_version ?? "labels-1.0",
      clip: "clip.mp4",
      ranges,
    };
    try {
      const updated = await saveLabelClip(detail.span_id, body);
      detail = updated;
      ranges = updated.labels.ranges.map((r) => ({ ...r }));
      dirty = false;
      saveStatus = "saved";
      clips = clips.map((c) => (c.span_id === updated.span_id ? updated : c));
    } catch (err) {
      saveStatus = err instanceof Error ? err.message : String(err);
    } finally {
      saving = false;
    }
  }

  function moveClip(delta: number): void {
    if (!clips.length) return;
    const idx = clips.findIndex((c) => c.span_id === selectedId);
    const next = clampIndex(idx + delta, clips.length);
    void selectClip(clips[next].span_id);
  }

  function clampIndex(i: number, len: number): number {
    return Math.max(0, Math.min(len - 1, i));
  }

  function fmtFrame(frame: number): string {
    return `${frame} · ${(frame / fps).toFixed(2)}s`;
  }

  // --- keyboard -----------------------------------------------------------

  function onKey(event: KeyboardEvent): void {
    const target = event.target as HTMLElement | null;
    if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) {
      if (event.key === "Escape") target.blur();
      return;
    }
    switch (event.key) {
      case " ":
        event.preventDefault();
        togglePlay();
        break;
      case "ArrowRight":
        event.preventDefault();
        stepFrame(event.shiftKey ? 10 : 1);
        break;
      case "ArrowLeft":
        event.preventDefault();
        stepFrame(event.shiftKey ? -10 : -1);
        break;
      case "i":
      case "I":
        event.preventDefault();
        setMarkIn();
        break;
      case "o":
      case "O":
        event.preventDefault();
        setMarkOut();
        break;
      case "Enter":
        event.preventDefault();
        addRange();
        break;
      case "s":
      case "S":
        event.preventDefault();
        void save();
        break;
      case "j":
        event.preventDefault();
        moveClip(1);
        break;
      case "k":
        event.preventDefault();
        moveClip(-1);
        break;
      default:
        if (event.key in BEHAVIOR_KEYS) {
          pendingBehavior = BEHAVIOR_KEYS[event.key];
        }
        break;
    }
  }
</script>

<svelte:window onkeydown={onKey} />

<div class="label-root">
  <aside class="clip-list">
    <div class="list-head">
      <h2>Harvested clips</h2>
      <button type="button" class="ghost" onclick={() => void loadClips()} title="Reload">↻</button>
    </div>
    {#if listLoading}
      <p class="muted pad">Loading clips…</p>
    {:else if listError}
      <p class="error pad">{listError}</p>
    {:else if clips.length === 0}
      <p class="muted pad">
        No harvested clips found. Run <code>detectivepotty harvest</code> to populate the
        harvest dir.
      </p>
    {:else}
      <ul>
        {#each clips as clip (clip.span_id)}
          <li>
            <button
              type="button"
              class:active={clip.span_id === selectedId}
              onclick={() => void selectClip(clip.span_id)}
            >
              <span class="row1">
                <span class="src">{clip.source_id}</span>
                <span class="badge" class:done={clip.labeled}>
                  {clip.labeled ? `${clip.n_trainable_ranges}✓` : "·"}
                </span>
              </span>
              <span class="row2 mono">
                {clip.date} · {clip.duration_s.toFixed(1)}s · t{clip.track_id ?? "?"}
              </span>
            </button>
          </li>
        {/each}
      </ul>
    {/if}
  </aside>

  <section class="stage">
    {#if detailLoading}
      <p class="muted pad">Loading clip…</p>
    {:else if detailError}
      <p class="error pad">{detailError}</p>
    {:else if !detail}
      <p class="muted pad">Select a clip to start labeling.</p>
    {:else}
      <div class="player">
        <div class="video-wrap" style="aspect-ratio: {detail.width || 16} / {detail.height || 9}">
          <!-- svelte-ignore a11y_media_has_caption -->
          <video
            bind:this={videoEl}
            src={labelClipVideoUrl(detail.span_id)}
            preload="auto"
            onloadedmetadata={onLoadedMeta}
            onseeked={syncFrame}
            ontimeupdate={syncFrame}
            onplay={() => (playing = true)}
            onpause={() => (playing = false)}
          ></video>
          {#if activeBox}
            <svg
              class="overlay"
              viewBox="0 0 {detail.width} {detail.height}"
              preserveAspectRatio="xMidYMid meet"
            >
              <rect
                x={activeBox.bbox.x1}
                y={activeBox.bbox.y1}
                width={activeBox.bbox.x2 - activeBox.bbox.x1}
                height={activeBox.bbox.y2 - activeBox.bbox.y1}
                class="box"
              />
            </svg>
          {/if}
        </div>

        <div class="transport">
          <button type="button" onclick={() => stepFrame(-10)} title="Back 10 (Shift+←)">⏪</button>
          <button type="button" onclick={() => stepFrame(-1)} title="Back 1 (←)">◀</button>
          <button type="button" class="play" onclick={togglePlay} title="Play/Pause (Space)">
            {playing ? "❚❚" : "►"}
          </button>
          <button type="button" onclick={() => stepFrame(1)} title="Forward 1 (→)">▶</button>
          <button type="button" onclick={() => stepFrame(10)} title="Forward 10 (Shift+→)">⏩</button>
          <span class="frame-readout mono">f{currentFrame} / {totalFrames - 1} · {(currentFrame / fps).toFixed(2)}s</span>
        </div>

        <input
          class="scrub"
          type="range"
          min="0"
          max={totalFrames - 1}
          value={currentFrame}
          oninput={(e) => seekToFrame(Number((e.target as HTMLInputElement).value))}
        />
      </div>

      <div class="editor">
        <div class="marks">
          <button type="button" onclick={setMarkIn} title="Mark in (I)">
            In <span class="mono">{markIn ?? "—"}</span>
          </button>
          <button type="button" onclick={setMarkOut} title="Mark out (O)">
            Out <span class="mono">{markOut ?? "—"}</span>
          </button>
        </div>

        <div class="pickers">
          <div class="picker">
            <span class="lbl">Behavior</span>
            <div class="seg">
              {#each vocabulary.behaviors as b (b)}
                <button
                  type="button"
                  class:active={pendingBehavior === b}
                  onclick={() => (pendingBehavior = b)}
                >
                  {b}
                </button>
              {/each}
            </div>
          </div>
          <div class="picker">
            <span class="lbl">Dog</span>
            <div class="seg">
              {#each vocabulary.dogs as d (d)}
                <button
                  type="button"
                  class:active={pendingDog === d}
                  onclick={() => (pendingDog = d)}
                >
                  {d}
                </button>
              {/each}
            </div>
          </div>
        </div>

        <div class="actions">
          <button type="button" class="primary" onclick={addRange}>
            + Add range (Enter)
          </button>
          <button
            type="button"
            class="save"
            class:dirty
            disabled={saving || !dirty}
            onclick={() => void save()}
          >
            {saving ? "Saving…" : dirty ? "Save labels (S)" : "Saved"}
          </button>
          {#if saveStatus && saveStatus !== "saved"}
            <span class="error">{saveStatus}</span>
          {:else if saveStatus === "saved"}
            <span class="ok">✓ saved</span>
          {/if}
        </div>

        <div class="ranges">
          <h3>Ranges ({ranges.length})</h3>
          {#if ranges.length === 0}
            <p class="muted">No ranges yet. Mark In/Out, pick behavior + dog, then Add.</p>
          {:else}
            <ul>
              {#each ranges as r, idx (idx)}
                <li>
                  <button type="button" class="seek" onclick={() => seekToRange(r)} title="Seek to start">
                    <span class="r-frames mono">{fmtFrame(r.start_frame)} → {fmtFrame(r.end_frame)}</span>
                    <span class="r-tags">
                      <span class="tag b-{r.behavior}">{r.behavior}</span>
                      <span class="tag dog">{r.dog}</span>
                      <span class="tag mono">t{r.track_id ?? "?"}</span>
                    </span>
                  </button>
                  <button type="button" class="del" onclick={() => deleteRange(idx)} aria-label="Delete range">×</button>
                </li>
              {/each}
            </ul>
          {/if}
        </div>

        <p class="legend mono">
          Space play · ←/→ step (Shift ×10) · I/O mark · 1-4 behavior · Enter add · S save · j/k clip
        </p>
      </div>
    {/if}
  </section>
</div>

<style>
  .label-root {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 0;
    height: 100%;
    min-height: 0;
  }
  .clip-list {
    border-right: 1px solid var(--border, #243042);
    overflow-y: auto;
    min-height: 0;
  }
  .list-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0.75rem;
    position: sticky;
    top: 0;
    background: var(--bg, #0c1018);
    border-bottom: 1px solid var(--border, #243042);
  }
  .list-head h2 {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0;
    color: var(--muted, #8aa);
  }
  .clip-list ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .clip-list li button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    border-bottom: 1px solid var(--border, #1b2433);
    padding: 0.5rem 0.75rem;
    cursor: pointer;
    color: inherit;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }
  .clip-list li button.active {
    background: var(--accent-soft, #16314d);
  }
  .clip-list li button:hover {
    background: var(--hover, #131c28);
  }
  .row1 {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem;
  }
  .src {
    font-weight: 600;
    font-size: 0.85rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .row2 {
    font-size: 0.72rem;
    color: var(--muted, #7e8ea0);
  }
  .badge {
    font-size: 0.7rem;
    padding: 0.05rem 0.4rem;
    border-radius: 999px;
    background: var(--border, #243042);
    color: var(--muted, #9ab);
  }
  .badge.done {
    background: #1f7a3f;
    color: #d6ffe2;
  }

  .stage {
    overflow-y: auto;
    min-height: 0;
    padding: 0.75rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .player {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .video-wrap {
    position: relative;
    width: 100%;
    max-height: 56vh;
    background: #000;
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .video-wrap video {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }
  .overlay {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }
  .overlay .box {
    fill: none;
    stroke: #36d07a;
    stroke-width: 3;
    vector-effect: non-scaling-stroke;
  }
  .transport {
    display: flex;
    align-items: center;
    gap: 0.35rem;
  }
  .transport button {
    background: var(--border, #243042);
    border: none;
    color: inherit;
    border-radius: 6px;
    padding: 0.3rem 0.6rem;
    cursor: pointer;
    font-size: 0.85rem;
  }
  .transport button.play {
    background: var(--accent, #2d6cdf);
    min-width: 3rem;
  }
  .frame-readout {
    margin-left: auto;
    font-size: 0.8rem;
    color: var(--muted, #9ab);
  }
  .scrub {
    width: 100%;
  }

  .editor {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }
  .marks {
    display: flex;
    gap: 0.5rem;
  }
  .marks button {
    flex: 1;
    background: var(--border, #243042);
    border: none;
    color: inherit;
    border-radius: 6px;
    padding: 0.4rem;
    cursor: pointer;
  }
  .pickers {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .picker {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }
  .picker .lbl {
    width: 4.5rem;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted, #9ab);
  }
  .seg {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
  }
  .seg button {
    background: var(--border, #243042);
    border: 1px solid transparent;
    color: inherit;
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
    cursor: pointer;
    font-size: 0.8rem;
  }
  .seg button.active {
    background: var(--accent, #2d6cdf);
    border-color: #5b8cf0;
  }
  .actions {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }
  .actions .primary {
    background: #2f7d4f;
    border: none;
    color: #eafff2;
    border-radius: 6px;
    padding: 0.45rem 0.8rem;
    cursor: pointer;
    font-weight: 600;
  }
  .actions .save {
    background: var(--border, #243042);
    border: none;
    color: inherit;
    border-radius: 6px;
    padding: 0.45rem 0.8rem;
    cursor: pointer;
  }
  .actions .save.dirty {
    background: var(--accent, #2d6cdf);
    color: #fff;
  }
  .actions .save:disabled {
    opacity: 0.6;
    cursor: default;
  }
  .ranges h3 {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted, #9ab);
    margin: 0.4rem 0;
  }
  .ranges ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }
  .ranges li {
    display: flex;
    align-items: stretch;
    gap: 0.3rem;
  }
  .ranges .seek {
    flex: 1;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem;
    background: var(--hover, #131c28);
    border: 1px solid var(--border, #243042);
    border-radius: 6px;
    padding: 0.35rem 0.6rem;
    cursor: pointer;
    color: inherit;
  }
  .r-frames {
    font-size: 0.75rem;
  }
  .r-tags {
    display: flex;
    gap: 0.3rem;
  }
  .tag {
    font-size: 0.7rem;
    padding: 0.05rem 0.4rem;
    border-radius: 4px;
    background: var(--border, #243042);
  }
  .tag.b-pee { background: #2d6cdf; color: #fff; }
  .tag.b-poop { background: #8a5a2b; color: #fff; }
  .tag.b-not_potty { background: #444c5a; color: #cdd; }
  .tag.b-excluded { background: #5a2b3a; color: #fdd; }
  .tag.dog { background: #2f5d4a; color: #dfe; }
  .ranges .del {
    background: transparent;
    border: 1px solid var(--border, #243042);
    color: var(--muted, #c88);
    border-radius: 6px;
    width: 2rem;
    cursor: pointer;
    font-size: 1rem;
  }
  .legend {
    font-size: 0.72rem;
    color: var(--muted, #7e8ea0);
    border-top: 1px solid var(--border, #1b2433);
    padding-top: 0.5rem;
  }

  .pad { padding: 1rem; }
  .muted { color: var(--muted, #7e8ea0); }
  .error { color: #ff6b6b; }
  .ok { color: #36d07a; }
  .ghost {
    background: transparent;
    border: 1px solid var(--border, #243042);
    color: inherit;
    border-radius: 6px;
    cursor: pointer;
    padding: 0.15rem 0.4rem;
  }
  code {
    background: var(--border, #243042);
    padding: 0.05rem 0.3rem;
    border-radius: 4px;
    font-size: 0.85em;
  }
</style>
