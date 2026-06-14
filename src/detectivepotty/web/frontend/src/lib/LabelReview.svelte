<script lang="ts">
  import { onDestroy, onMount } from "svelte";
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
    LabelPresentTrack,
    LabelRangeItem,
    LabelVocabulary,
  } from "./types";
  import { boxAtFrame } from "./labelBox";
  import { errMsg } from "./errors";
  import { isTypingTarget } from "./keys";
  import { formatClock } from "./format";
  import { BOX_DOG, BOX_SIBLING, boxLabelFontPx, formatDetLabel, formatTrackLabel, isAliasClass } from "./overlayStyle";
  import { observeResize } from "./resize";
  import LabelClipList from "./LabelClipList.svelte";
  import LabelFilmstrip from "./LabelFilmstrip.svelte";
  import LabelRangeEditor from "./LabelRangeEditor.svelte";
  import Transport from "./Transport.svelte";

  const BEHAVIOR_KEYS: Record<string, string> = {
    "1": "pee",
    "2": "poop",
    "3": "not_potty",
    "4": "excluded",
  };
  const BEH_COLOR: Record<string, string> = {
    pee: "#f1cf5b",
    poop: "#c08a55",
    not_potty: "#8290a8",
    excluded: "#a35a74",
  };
  const MAX_FILMSTRIP = 24;

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
  let thumbEl = $state<HTMLVideoElement | null>(null);
  let laneEl = $state<HTMLCanvasElement | null>(null);
  let currentFrame = $state(0);
  let playing = $state(false);
  let markIn = $state<number | null>(null);
  let markOut = $state<number | null>(null);
  let pendingBehavior = $state("pee");
  let pendingDog = $state("unknown");

  // Per-detection crop thumbnails for the active (own) track.
  let crops = $state<{ frame: number; url: string | null }[]>([]);
  let filmstripToken = 0;
  let detailToken = 0;
  let rvfcHandle: number | null = null;
  let stopLaneResize: (() => void) | null = null;
  let cleanupThumbWait: (() => void) | null = null;

  const fps = $derived(detail && detail.fps > 0 ? detail.fps : 30);
  const totalFrames = $derived(detail ? Math.max(1, detail.frame_count) : 1);
  const trackId = $derived(detail?.track_id ?? null);
  // Detector provenance + detected-object-class mix surfaced in the clip header.
  const modelLabel = $derived(detail?.model_name || "unknown");
  const classCounts = $derived(detail?.class_distribution ?? []);
  const hasRvfc =
    typeof window !== "undefined" &&
    "requestVideoFrameCallback" in HTMLVideoElement.prototype;

  // The own track's sampled detection boxes (what a new range binds to).
  const ownBoxes = $derived.by(() => {
    if (!detail || trackId == null) return [];
    return detail.tracks[trackId] ?? [];
  });

  // Other track segments overlapping this clip's window (often the same dog
  // re-detected after the tracker lost it), for context + jump.
  const siblingTracks = $derived.by<LabelPresentTrack[]>(() => {
    if (!detail) return [];
    return Object.values(detail.present_tracks).filter((t) => !t.is_self);
  });

  // Interpolated box for the track this clip follows (Workstream D).
  const activeBox = $derived(boxAtFrame(ownBoxes, currentFrame));

  // Sibling boxes at the current frame (dimmed, click to jump).
  const siblingBoxes = $derived.by(() => {
    const out: {
      track: LabelPresentTrack;
      bbox: { x1: number; y1: number; x2: number; y2: number };
      class_name: string;
      confidence: number;
    }[] = [];
    for (const t of siblingTracks) {
      const b = boxAtFrame(t.boxes, currentFrame);
      if (b && !b.extrapolated) out.push({ track: t, bbox: b.bbox, class_name: b.class_name, confidence: b.confidence });
    }
    return out;
  });

  // Box-label font sized off the larger image edge so it reads consistently
  // on screen regardless of source resolution (overlay scales uniformly).
  const labelFont = $derived(detail ? boxLabelFontPx(Math.max(detail.width, detail.height)) : 14);
  const isAlias = isAliasClass;

  onMount(loadClips);

  onDestroy(() => {
    cancelFrameLoop();
    stopLaneResize?.();
    cleanupThumbWait?.();
    filmstripToken++;
    detailToken++;
  });

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
      listError = errMsg(err);
    } finally {
      listLoading = false;
    }
  }

  async function selectClip(spanId: string): Promise<void> {
    if (dirty && spanId !== selectedId) {
      const ok = confirm("Discard unsaved label changes for this clip?");
      if (!ok) return;
    }
    const token = ++detailToken;
    selectedId = spanId;
    detailLoading = true;
    detailError = null;
    cancelFrameLoop();
    cleanupThumbWait?.();
    cleanupThumbWait = null;
    detail = null;
    crops = [];
    try {
      const data = await fetchLabelClipDetail(spanId);
      if (token !== detailToken || spanId !== selectedId) return;
      detail = data;
      ranges = data.labels.ranges.map((r) => ({ ...r }));
      dirty = false;
      saveStatus = null;
      currentFrame = 0;
      markIn = null;
      markOut = null;
    } catch (err) {
      if (token !== detailToken) return;
      detailError = errMsg(err);
    } finally {
      if (token === detailToken) {
        detailLoading = false;
      }
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
    registerFrameLoop();
  }

  function cancelFrameLoop(): void {
    if (hasRvfc && videoEl && rvfcHandle !== null) {
      try {
        videoEl.cancelVideoFrameCallback(rvfcHandle);
      } catch {
        /* element may be detaching */
      }
    }
    rvfcHandle = null;
  }

  function registerFrameLoop(): void {
    if (!hasRvfc || !videoEl) return;
    cancelFrameLoop();
    const cb = (): void => {
      rvfcHandle = null;
      syncFrame();
      if (videoEl) {
        rvfcHandle = videoEl.requestVideoFrameCallback(cb);
      }
    };
    rvfcHandle = videoEl.requestVideoFrameCallback(cb);
  }

  // --- detection filmstrip (per-detection crop thumbnails) ----------------

  function seekThumb(time: number): Promise<void> {
    return new Promise((resolve) => {
      const v = thumbEl;
      if (!v) {
        resolve();
        return;
      }
      const on = (): void => {
        cleanupThumbWait = null;
        v.removeEventListener("seeked", on);
        resolve();
      };
      cleanupThumbWait = () => {
        v.removeEventListener("seeked", on);
        resolve();
      };
      v.addEventListener("seeked", on);
      v.currentTime = time;
    });
  }

  async function buildFilmstrip(): Promise<void> {
    const token = ++filmstripToken;
    const boxes = ownBoxes;
    if (!detail || boxes.length === 0) {
      crops = [];
      return;
    }
    // Stride down to a manageable number of cards.
    const stride = Math.max(1, Math.ceil(boxes.length / MAX_FILMSTRIP));
    const picked = boxes.filter((_, i) => i % stride === 0);
    crops = picked.map((b) => ({ frame: b.clip_frame_idx, url: null }));

    const v = thumbEl;
    if (!v) return;
    if (v.readyState < 1) {
      await new Promise<void>((r) => {
        const on = (): void => {
          cleanupThumbWait = null;
          v.removeEventListener("loadedmetadata", on);
          r();
        };
        cleanupThumbWait = () => {
          v.removeEventListener("loadedmetadata", on);
          r();
        };
        v.addEventListener("loadedmetadata", on);
      });
    }
    if (token !== filmstripToken) return;

    const natW = v.videoWidth || detail.width || 1;
    const natH = v.videoHeight || detail.height || 1;
    const sx = natW / (detail.width || natW);
    const sy = natH / (detail.height || natH);
    const oc = document.createElement("canvas");
    const ctx = oc.getContext("2d");
    if (!ctx) return;

    for (let i = 0; i < picked.length; i += 1) {
      if (token !== filmstripToken) return;
      const b = picked[i];
      await seekThumb((b.clip_frame_idx + 0.5) / fps);
      if (token !== filmstripToken) return;
      const bw = b.bbox.x2 - b.bbox.x1;
      const bh = b.bbox.y2 - b.bbox.y1;
      const mx = bw * 0.15;
      const my = bh * 0.15;
      const cropX = Math.max(0, (b.bbox.x1 - mx)) * sx;
      const cropY = Math.max(0, (b.bbox.y1 - my)) * sy;
      const cropW = Math.min(natW - cropX, (bw + 2 * mx) * sx);
      const cropH = Math.min(natH - cropY, (bh + 2 * my) * sy);
      const outW = 96;
      const outH = Math.max(48, Math.round((cropH / Math.max(1, cropW)) * outW));
      oc.width = outW;
      oc.height = outH;
      try {
        ctx.drawImage(v, cropX, cropY, Math.max(1, cropW), Math.max(1, cropH), 0, 0, outW, outH);
        const url = oc.toDataURL("image/jpeg", 0.6);
        if (token !== filmstripToken) return;
        crops[i] = { frame: b.clip_frame_idx, url };
        crops = [...crops];
      } catch {
        // CORS/decoding hiccup — leave the placeholder.
      }
    }
  }

  $effect(() => {
    const id = detail?.span_id;
    // Re-extract when the clip (and therefore its own track) changes.
    void id;
    void trackId;
    void buildFilmstrip();
  });

  // --- detection / confidence lane ----------------------------------------

  function drawLane(): void {
    const canvas = laneEl;
    if (!canvas || !detail) return;
    const w = canvas.clientWidth || 1;
    const h = canvas.clientHeight || 1;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const total = Math.max(1, totalFrames - 1);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0a0e16";
    ctx.fillRect(0, 0, w, h);

    // Labeled range bands (behind the detection ticks).
    for (const r of ranges) {
      const x0 = (r.start_frame / total) * w;
      const x1 = (r.end_frame / total) * w;
      ctx.fillStyle = (BEH_COLOR[r.behavior] ?? "#444") + "55";
      ctx.fillRect(x0, 0, Math.max(2, x1 - x0), h);
    }

    // Confidence gate line.
    const gate = detail.detect_conf;
    if (gate != null && gate > 0) {
      const gy = h - gate * h;
      ctx.strokeStyle = "#5b6b80";
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(w, gy);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Detection ticks: height + colour by confidence, gated green/amber.
    for (const b of ownBoxes) {
      const x = (b.clip_frame_idx / total) * w;
      const conf = Math.max(0, Math.min(1, b.confidence));
      const bh = Math.max(2, conf * h);
      const passed = gate == null || conf >= gate;
      ctx.fillStyle = passed ? BOX_DOG : BOX_SIBLING;
      ctx.fillRect(x, h - bh, 2, bh);
    }

    // Playhead.
    const px = (currentFrame / total) * w;
    ctx.strokeStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, h);
    ctx.stroke();
  }

  $effect(() => {
    // Redraw on frame/track/range/clip changes.
    void currentFrame;
    void ownBoxes;
    void ranges;
    void detail;
    drawLane();
  });

  $effect(() => {
    stopLaneResize?.();
    stopLaneResize = observeResize(laneEl, drawLane);
  });

  function laneSeek(event: MouseEvent): void {
    const canvas = laneEl;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const t = (event.clientX - rect.left) / Math.max(1, rect.width);
    seekToFrame(Math.round(t * (totalFrames - 1)));
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

  function updateRange(idx: number, patch: Partial<LabelRangeItem>): void {
    ranges = ranges.map((r, i) => (i === idx ? { ...r, ...patch } : r));
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
      saveStatus = errMsg(err);
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

  // --- keyboard -----------------------------------------------------------

  function onKey(event: KeyboardEvent): void {
    if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) {
      return;
    }
    if (isTypingTarget(event.target, { allowRange: true })) {
      if (event.key === "Escape") (event.target as HTMLElement).blur();
      return;
    }
    // Dog hotkeys: Shift+1..9 map to the vocabulary's dogs (use physical key).
    if (event.shiftKey && /^Digit[1-9]$/.test(event.code)) {
      const idx = Number(event.code.slice(5)) - 1;
      if (idx < vocabulary.dogs.length) {
        event.preventDefault();
        pendingDog = vocabulary.dogs[idx];
      }
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
  <LabelClipList
    {clips}
    {selectedId}
    loading={listLoading}
    error={listError}
    onreload={loadClips}
    onselect={(spanId) => void selectClip(spanId)}
  />

  <section class="stage">
    {#if detailLoading}
      <p class="muted pad">Loading clip…</p>
    {:else if detailError}
      <p class="error pad">{detailError}</p>
    {:else if !detail}
      <p class="muted pad">Select a clip to start labeling.</p>
    {:else}
      <!-- hidden video used only to extract per-detection crop thumbnails -->
      <video
        bind:this={thumbEl}
        class="thumb-src"
        src={labelClipVideoUrl(detail.span_id)}
        preload="auto"
        muted
        playsinline
      ></video>

      <div class="player-col">
        <div class="clip-head">
          <div class="ch-main">
            <strong title={detail.source_id}>{detail.camera_name ?? detail.camera_id ?? "Unknown camera"}</strong>
            <span class="ch-when" title="Clip window (local time)">
              {formatClock(detail.span_start_utc)} → {formatClock(detail.span_end_utc)}
            </span>
            <span class="ch-prov" title="Detector model · detected object classes (dog vs accepted aliases like sheep/zebra)">
              <span class="prov-model" class:unknown={!detail.model_name}>{modelLabel}</span>
              {#each classCounts as c}
                <span class="prov-class" class:alias={c.class_name.toLowerCase() !== "dog"}>{c.class_name} ×{c.count}</span>
              {:else}
                <span class="prov-class">—</span>
              {/each}
            </span>
          </div>
          <div class="ch-meta">
            <span class="pill" title="The single track segment this clip follows. Its boxes/labels bind to this track.">Following Track {trackId ?? "?"}</span>
            <span class="pill" class:multi={detail.n_tracks > 1} title="Distinct track segments in this clip's time window (including siblings from overlapping clips). These are often the same dog re-detected after the tracker lost it — not confirmed separate dogs.">
              {detail.n_tracks} segment{detail.n_tracks === 1 ? "" : "s"} in window
            </span>
          </div>
        </div>

        <div class="video-wrap" style="aspect-ratio: {detail.width || 16} / {detail.height || 9}">
          <!-- svelte-ignore a11y_media_has_caption -->
          <video
            bind:this={videoEl}
            src={labelClipVideoUrl(detail.span_id)}
            preload="auto"
            loop
            onloadedmetadata={onLoadedMeta}
            onseeked={syncFrame}
            ontimeupdate={syncFrame}
            onplay={() => (playing = true)}
            onpause={() => (playing = false)}
          ></video>
          <svg
            class="overlay"
            viewBox="0 0 {detail.width} {detail.height}"
            preserveAspectRatio="xMidYMid meet"
          >
            {#each siblingBoxes as sb (sb.track.span_id + ':' + sb.track.track_id)}
              <!-- svelte-ignore a11y_click_events_have_key_events -->
              <rect
                x={sb.bbox.x1}
                y={sb.bbox.y1}
                width={sb.bbox.x2 - sb.bbox.x1}
                height={sb.bbox.y2 - sb.bbox.y1}
                class="box sibling"
                role="button"
                tabindex="-1"
                onclick={() => void selectClip(sb.track.span_id)}
              ><title>Other segment (Track {sb.track.track_id}) — may be the same dog; click to open its clip</title></rect>
              <text
                x={sb.bbox.x1 + labelFont * 0.2}
                y={sb.bbox.y1 - labelFont * 0.3 < labelFont ? sb.bbox.y1 + labelFont : sb.bbox.y1 - labelFont * 0.3}
                class="box-label sibling"
                class:alias={isAlias(sb.class_name)}
                font-size={labelFont}
              >{formatTrackLabel(sb.track.track_id, sb.confidence, sb.class_name)}</text>
            {/each}
            {#if activeBox && !activeBox.extrapolated}
              <rect
                x={activeBox.bbox.x1}
                y={activeBox.bbox.y1}
                width={activeBox.bbox.x2 - activeBox.bbox.x1}
                height={activeBox.bbox.y2 - activeBox.bbox.y1}
                class="box active"
              />
              <text
                x={activeBox.bbox.x1 + labelFont * 0.2}
                y={activeBox.bbox.y1 - labelFont * 0.3 < labelFont ? activeBox.bbox.y1 + labelFont : activeBox.bbox.y1 - labelFont * 0.3}
                class="box-label active"
                class:alias={isAlias(activeBox.class_name)}
                font-size={labelFont}
              >{formatDetLabel(activeBox.class_name, activeBox.confidence)}</text>
            {/if}
          </svg>
          {#if activeBox && activeBox.extrapolated}
            <div class="no-detect" title="This frame is in the clip's padding, before/after this track's first/last detection. No box is drawn rather than freeze a stale one.">no detection at this frame</div>
          {/if}
        </div>

        <div class="box-legend mono" aria-label="Box legend">
          <span class="lg"><span class="sw own"></span>followed track (this clip)</span>
          {#if siblingTracks.length}
            <span class="lg"><span class="sw sib"></span>other segment — click to open</span>
          {/if}
        </div>

        <Transport
          playing={playing}
          frame={currentFrame}
          total={totalFrames}
          fps={fps}
          skipN={10}
          showReadout={true}
          onTogglePlay={togglePlay}
          onStep={stepFrame}
        />

        <input
          class="scrub"
          type="range"
          min="0"
          max={totalFrames - 1}
          value={currentFrame}
          oninput={(e) => seekToFrame(Number((e.target as HTMLInputElement).value))}
        />

        <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
        <canvas
          bind:this={laneEl}
          class="lane"
          title="Detection confidence over the clip. Bars = sampled detections (height/colour = confidence); dashed line = harvest detect gate; tinted bands = your labeled ranges. Click to seek."
          onclick={laneSeek}
          role="slider"
          tabindex="-1"
          aria-label="Detection confidence timeline"
          aria-valuenow={currentFrame}
        ></canvas>

        <LabelFilmstrip {crops} {currentFrame} {fps} onseek={seekToFrame} />
      </div>

      <LabelRangeEditor
        {siblingTracks}
        {markIn}
        {markOut}
        {pendingBehavior}
        {pendingDog}
        {vocabulary}
        {ranges}
        {fps}
        {dirty}
        {saving}
        {saveStatus}
        onselectclip={(spanId) => void selectClip(spanId)}
        onsetmarkin={setMarkIn}
        onsetmarkout={setMarkOut}
        onbehavior={(behavior) => (pendingBehavior = behavior)}
        ondog={(dog) => (pendingDog = dog)}
        onaddrange={addRange}
        onsave={save}
        onseekrange={seekToRange}
        onupdaterange={updateRange}
        ondeleterange={deleteRange}
      />
    {/if}
  </section>
</div>

<style>
  .label-root {
    display: grid;
    grid-template-columns: 256px 1fr;
    gap: 0;
    height: 100%;
    min-height: 0;
  }
  .stage {
    display: grid;
    grid-template-columns: minmax(0, 1.5fr) minmax(300px, 0.85fr);
    gap: 0.75rem;
    min-height: 0;
    padding: 0.6rem 0.8rem;
    overflow: hidden;
  }
  .thumb-src {
    position: absolute;
    width: 1px;
    height: 1px;
    opacity: 0;
    pointer-events: none;
    left: -9999px;
  }
  .player-col {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    min-width: 0;
    min-height: 0;
  }
  .clip-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  .ch-main { display: flex; flex-direction: column; min-width: 0; }
  .ch-main strong { font-size: 0.95rem; }
  .ch-when { font-size: 0.72rem; color: var(--text-dim); }
  .ch-prov {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    align-items: center;
    margin-top: 0.2rem;
    font-size: 0.68rem;
  }
  .prov-model {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    color: var(--text-dim);
  }
  .prov-model.unknown { font-style: italic; opacity: 0.7; }
  .prov-class {
    padding: 0.05rem 0.35rem;
    border-radius: 0.5rem;
    background: rgba(120, 140, 160, 0.18);
    color: var(--text-dim);
  }
  .prov-class.alias {
    background: rgba(34, 211, 238, 0.18);
    color: #22d3ee;
  }
  .ch-meta { display: flex; gap: 0.3rem; flex-wrap: wrap; }
  .pill {
    font-size: 0.68rem;
    padding: 0.1rem 0.45rem;
    border-radius: 999px;
    background: var(--bg-3);
    color: #bcd;
    white-space: nowrap;
  }
  .pill.multi { background: #3a2c12; color: #f0c869; }
  .video-wrap {
    position: relative;
    width: 100%;
    max-height: 52vh;
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
  .overlay .box { fill: none; vector-effect: non-scaling-stroke; }
  .overlay .box.active { stroke: var(--green); stroke-width: 3; }
  .overlay .box-label {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-weight: 600;
    paint-order: stroke;
    stroke: rgba(0, 0, 0, 0.82);
    stroke-width: 4px;
    vector-effect: non-scaling-stroke;
    pointer-events: none;
    dominant-baseline: alphabetic;
  }
  .overlay .box-label.active { fill: var(--green); }
  .overlay .box-label.sibling { fill: var(--amber); opacity: 0.85; }
  .overlay .box-label.alias { fill: var(--teal); }
  .no-detect {
    position: absolute;
    left: 50%;
    bottom: 8px;
    transform: translateX(-50%);
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    background: rgba(20, 26, 36, 0.72);
    color: #9fb0c4;
    font-size: 0.7rem;
    letter-spacing: 0.02em;
    pointer-events: none;
  }
  .box-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    padding: 0.2rem 0.1rem 0;
    font-size: 0.7rem;
    color: var(--text-dim);
  }
  .box-legend .lg { display: inline-flex; align-items: center; gap: 0.3rem; }
  .box-legend .sw { width: 16px; height: 0; border-top-width: 3px; border-top-style: solid; }
  .box-legend .sw.own { border-top-color: var(--green); }
  .box-legend .sw.sib { border-top-color: var(--amber); border-top-style: dashed; }
  .overlay .box.sibling {
    stroke: var(--amber);
    stroke-width: 2;
    stroke-dasharray: 5 4;
    opacity: 0.8;
    pointer-events: auto;
    cursor: pointer;
    fill: rgba(240, 169, 58, 0.06);
  }
  .scrub { width: 100%; }
  .lane {
    width: 100%;
    height: 44px;
    border-radius: 6px;
    border: 1px solid var(--line-strong);
    cursor: pointer;
    display: block;
  }
  .pad { padding: 1rem; }
  .muted { color: var(--text-dim); }
  .error { color: #ff6b6b; }

  @media (max-width: 1100px) {
    .stage { grid-template-columns: 1fr; overflow-y: auto; }
  }
</style>
