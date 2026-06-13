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
    LabelPresentTrack,
    LabelRangeItem,
    LabelVocabulary,
  } from "./types";
  import { boxAtFrame } from "./labelBox";
  import { errMsg } from "./errors";
  import { isTypingTarget } from "./keys";
  import { formatClock } from "./format";
  import { BOX_DOG, BOX_SIBLING, boxLabelFontPx, formatDetLabel, formatTrackLabel, isAliasClass } from "./overlayStyle";
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

  // Group the clip list into scenes (siblings clustered, first-appearance order).
  interface ClipGroup {
    key: string;
    scene: string | null;
    size: number;
    camera: string | null;
    items: LabelClipSummary[];
  }
  const clipGroups = $derived.by<ClipGroup[]>(() => {
    const groups: ClipGroup[] = [];
    const byKey = new Map<string, ClipGroup>();
    for (const c of clips) {
      const key = c.scene_size > 1 && c.scene_id ? `scene:${c.scene_id}` : `solo:${c.span_id}`;
      let g = byKey.get(key);
      if (!g) {
        g = {
          key,
          scene: c.scene_size > 1 ? c.scene_id : null,
          size: c.scene_size,
          camera: c.camera_name,
          items: [],
        };
        byKey.set(key, g);
        groups.push(g);
      }
      g.items.push(c);
    }
    return groups;
  });

  const dogKeyHint = $derived.by<Record<string, string>>(() => {
    const map: Record<string, string> = {};
    vocabulary.dogs.forEach((d, i) => {
      if (i < 9) map[d] = `\u21e7${i + 1}`;
    });
    return map;
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
    if (hasRvfc && videoEl) {
      const cb = (): void => {
        syncFrame();
        if (videoEl) (videoEl as HTMLVideoElement).requestVideoFrameCallback(cb);
      };
      (videoEl as HTMLVideoElement).requestVideoFrameCallback(cb);
    }
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

  function fmtFrame(frame: number): string {
    return `${frame} \u00b7 ${(frame / fps).toFixed(2)}s`;
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
  <aside class="clip-list">
    <div class="list-head">
      <h2>Harvested clips</h2>
      <button type="button" class="ghost" onclick={() => void loadClips()} title="Reload clip list">↻</button>
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
      {#each clipGroups as group (group.key)}
        {#if group.scene}
          <div class="scene-head" title="Same camera + overlapping time window — {group.size} detection segments (often the same dog re-detected after the tracker lost it, not confirmed separate dogs). Label each on its own clip.">
            <span class="scene-cam">{group.camera ?? "camera"}</span>
            <span class="scene-when">{formatClock(group.items[0].span_start_utc)}</span>
            <span class="scene-badge">×{group.size} segments</span>
          </div>
        {/if}
        <ul class:scene-group={group.scene}>
          {#each group.items as clip (clip.span_id)}
            <li>
              <button
                type="button"
                class:active={clip.span_id === selectedId}
                onclick={() => void selectClip(clip.span_id)}
                title={`${clip.camera_name ?? clip.camera_id ?? "unknown camera"}\n${clip.source_id}\n${formatClock(clip.span_start_utc)} → ${formatClock(clip.span_end_utc)}`}
              >
                <span class="row1">
                  <span class="cam">{clip.camera_name ?? clip.camera_id ?? clip.source_id}</span>
                  <span class="badge" class:done={clip.labeled} title={clip.labeled ? `${clip.n_trainable_ranges} trainable / ${clip.n_ranges} ranges` : "Not labeled yet"}>
                    {clip.labeled ? `✓${clip.n_trainable_ranges}` : "·"}
                  </span>
                </span>
                <span class="row2">
                  <span class="when" title="Clip start (local time)">{formatClock(clip.span_start_utc)}</span>
                  <span class="dur" title="Clip duration">{clip.duration_s.toFixed(1)}s</span>
                  <span class="trk" title="Track segment this clip follows — its boxes/labels bind to this track">T{clip.track_id ?? "?"}</span>
                </span>
                {#if clip.labeled && (clip.behaviors.length || clip.dogs.length)}
                  <span class="row3">
                    {#each clip.behaviors as b (b)}
                      <span class="chip b-{b}">{b}</span>
                    {/each}
                    {#each clip.dogs as d (d)}
                      <span class="chip dog">{d}</span>
                    {/each}
                  </span>
                {/if}
              </button>
            </li>
          {/each}
        </ul>
      {/each}
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

        <div class="filmstrip" title="Every sampled detection of the followed track — click a crop to seek there.">
          {#if crops.length === 0}
            <span class="muted small">No detections sampled for this track.</span>
          {:else}
            {#each crops as c, i (i)}
              <button
                type="button"
                class="film-card"
                class:cur={Math.abs(c.frame - currentFrame) < (fps / 2)}
                onclick={() => seekToFrame(c.frame)}
                title={`Frame ${c.frame} · ${(c.frame / fps).toFixed(2)}s`}
              >
                {#if c.url}
                  <img src={c.url} alt={`detection at frame ${c.frame}`} />
                {:else}
                  <span class="film-ph"></span>
                {/if}
                <span class="film-f mono">f{c.frame}</span>
              </button>
            {/each}
          {/if}
        </div>
      </div>

      <div class="editor-col">
        {#if siblingTracks.length}
          <div class="siblings">
            <span class="lbl" title="Other track segments overlapping this clip's window — often the same dog re-detected, not confirmed separate dogs.">Other segments here</span>
            <div class="sib-chips">
              {#each siblingTracks as t (t.span_id + ':' + t.track_id)}
                <button
                  type="button"
                  class="sib-chip"
                  onclick={() => void selectClip(t.span_id)}
                  title="Open the clip that follows this segment (labels bind to a clip's own track)"
                >
                  → Track {t.track_id}
                </button>
              {/each}
            </div>
          </div>
        {/if}

        <div class="marks">
          <button type="button" onclick={setMarkIn} title="Mark range start at current frame (I)">
            In <span class="mono">{markIn ?? "—"}</span>
          </button>
          <button type="button" onclick={setMarkOut} title="Mark range end at current frame (O)">
            Out <span class="mono">{markOut ?? "—"}</span>
          </button>
        </div>

        <div class="pickers">
          <div class="picker">
            <span class="lbl">Behavior</span>
            <div class="seg">
              {#each vocabulary.behaviors as b, i (b)}
                <button
                  type="button"
                  class:active={pendingBehavior === b}
                  onclick={() => (pendingBehavior = b)}
                  title={`Set behavior to ${b}${i < 9 ? ` (${i + 1})` : ""}`}
                >
                  {b}{#if i < 9}<span class="kh">{i + 1}</span>{/if}
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
                  title={`Set dog to ${d}${dogKeyHint[d] ? ` (${dogKeyHint[d]})` : ""}`}
                >
                  {d}{#if dogKeyHint[d]}<span class="kh">{dogKeyHint[d]}</span>{/if}
                </button>
              {/each}
            </div>
          </div>
        </div>

        <div class="actions">
          <button type="button" class="primary" onclick={addRange} title="Add a range from In→Out (Enter)">
            + Add range
          </button>
          <button
            type="button"
            class="save"
            class:dirty
            disabled={saving || !dirty}
            onclick={() => void save()}
            title="Save labels.json (S)"
          >
            {saving ? "Saving…" : dirty ? "Save (S)" : "Saved"}
          </button>
          {#if saveStatus && saveStatus !== "saved"}
            <span class="error small">{saveStatus}</span>
          {:else if saveStatus === "saved"}
            <span class="ok small">✓ saved</span>
          {/if}
        </div>

        <div class="ranges">
          <h3>Ranges ({ranges.length})</h3>
          {#if ranges.length === 0}
            <p class="muted small">No ranges yet. Mark In/Out, pick behavior + dog, then Add.</p>
          {:else}
            <ul>
              {#each ranges as r, idx (idx)}
                <li>
                  <button type="button" class="seek" onclick={() => seekToRange(r)} title="Seek to range start">
                    <span class="r-frames mono">{fmtFrame(r.start_frame)} → {fmtFrame(r.end_frame)}</span>
                  </button>
                  <select
                    class="r-sel"
                    value={r.behavior}
                    onchange={(e) => updateRange(idx, { behavior: (e.target as HTMLSelectElement).value })}
                    title="Behavior for this range"
                  >
                    {#each vocabulary.behaviors as b (b)}
                      <option value={b}>{b}</option>
                    {/each}
                  </select>
                  <select
                    class="r-sel"
                    value={r.dog}
                    onchange={(e) => updateRange(idx, { dog: (e.target as HTMLSelectElement).value })}
                    title="Dog for this range"
                  >
                    {#each vocabulary.dogs as d (d)}
                      <option value={d}>{d}</option>
                    {/each}
                  </select>
                  <button type="button" class="del" onclick={() => deleteRange(idx)} aria-label="Delete range" title="Delete range">×</button>
                </li>
              {/each}
            </ul>
          {/if}
        </div>

        <p class="legend mono">
          Space play · ←/→ step (⇧×10) · I/O mark · 1-4 behavior · ⇧1-4 dog · Enter add · S save · j/k clip
        </p>
      </div>
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
  .clip-list {
    border-right: 1px solid var(--line-strong);
    overflow-y: auto;
    min-height: 0;
  }
  .list-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.4rem 0.6rem;
    position: sticky;
    top: 0;
    background: var(--bg, #0c1018);
    border-bottom: 1px solid var(--line-strong);
    z-index: 1;
  }
  .list-head h2 {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0;
    color: var(--text-dim);
  }
  .clip-list ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .scene-head {
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
    padding: 0.3rem 0.6rem 0.15rem;
    font-size: 0.68rem;
    color: var(--text-dim);
    border-top: 1px solid var(--line-strong);
  }
  .scene-cam { font-weight: 600; color: #b9c6d6; }
  .scene-when { margin-left: auto; }
  .scene-badge {
    background: #3a2c12;
    color: #f0c869;
    border-radius: 999px;
    padding: 0.02rem 0.4rem;
  }
  ul.scene-group {
    border-left: 2px solid #3a2c12;
    margin-left: 0.35rem;
  }
  .clip-list li { margin: 0; }
  .clip-list li button {
    display: flex;
    flex-direction: column;
    gap: 0.12rem;
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    border-bottom: 1px solid var(--line-strong);
    color: inherit;
    padding: 0.32rem 0.6rem;
    cursor: pointer;
  }
  .clip-list li button:hover { background: var(--hover, #131c28); }
  .clip-list li button.active {
    background: var(--hover, #16202e);
    box-shadow: inset 3px 0 0 var(--amber);
  }
  .row1 {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.4rem;
  }
  .cam {
    font-size: 0.82rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .row2 {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.7rem;
    color: var(--text-dim);
  }
  .row2 .trk { margin-left: auto; }
  .row3 {
    display: flex;
    flex-wrap: wrap;
    gap: 0.2rem;
    margin-top: 0.1rem;
  }
  .chip {
    font-size: 0.62rem;
    padding: 0.02rem 0.32rem;
    border-radius: 4px;
    background: var(--bg-3);
    color: var(--text-dim);
  }
  .chip.b-pee { background: #f1cf5b; color: #1a1204; }
  .chip.b-poop { background: #c08a55; color: #1a1204; }
  .chip.b-not_potty { background: #3a4150; color: var(--text); }
  .chip.b-excluded { background: #5a2f42; color: #fdd; }
  .chip.dog { background: #2f5d4a; color: #dfe; }
  .badge {
    font-size: 0.66rem;
    padding: 0.03rem 0.38rem;
    border-radius: 999px;
    background: var(--bg-3);
    color: var(--text-dim);
    flex: none;
  }
  .badge.done { background: #1f7a3f; color: #d6ffe2; }

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
  .filmstrip {
    display: flex;
    gap: 0.3rem;
    overflow-x: auto;
    padding-bottom: 0.2rem;
  }
  .film-card {
    flex: none;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.1rem;
    background: var(--bg-3);
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 0.15rem;
    cursor: pointer;
  }
  .film-card.cur { border-color: var(--green); }
  .film-card img {
    width: 72px;
    height: 54px;
    object-fit: cover;
    border-radius: 3px;
    background: #000;
    display: block;
  }
  .film-ph {
    width: 72px;
    height: 54px;
    border-radius: 3px;
    background: #0a0e16;
    display: block;
  }
  .film-f { font-size: 0.6rem; color: var(--text-dim); }

  .editor-col {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    min-height: 0;
    overflow-y: auto;
  }
  .siblings { display: flex; flex-direction: column; gap: 0.25rem; }
  .sib-chips { display: flex; flex-wrap: wrap; gap: 0.3rem; }
  .sib-chip {
    background: #3a2c12;
    border: 1px solid #5a4520;
    color: #f0c869;
    border-radius: 6px;
    padding: 0.2rem 0.5rem;
    cursor: pointer;
    font-size: 0.74rem;
  }
  .marks {
    display: flex;
    gap: 0.4rem;
  }
  .marks button {
    flex: 1;
    background: var(--bg-3);
    border: none;
    color: inherit;
    border-radius: 6px;
    padding: 0.35rem;
    cursor: pointer;
  }
  .pickers {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .picker {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .picker .lbl {
    width: 4rem;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-dim);
    flex: none;
  }
  .lbl {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-dim);
  }
  .seg {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
  }
  .seg button {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    background: var(--bg-3);
    border: 1px solid transparent;
    color: inherit;
    border-radius: 6px;
    padding: 0.22rem 0.5rem;
    cursor: pointer;
    font-size: 0.78rem;
  }
  .seg button.active {
    background: var(--amber);
    border-color: var(--amber-bright);
    color: #1a1204;
  }
  .kh {
    font-size: 0.6rem;
    opacity: 0.7;
    background: rgba(0, 0, 0, 0.25);
    border-radius: 3px;
    padding: 0 0.2rem;
  }
  .actions {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  .actions .primary {
    background: #2f7d4f;
    border: none;
    color: #eafff2;
    border-radius: 6px;
    padding: 0.4rem 0.7rem;
    cursor: pointer;
    font-weight: 600;
  }
  .actions .save {
    background: var(--bg-3);
    border: none;
    color: inherit;
    border-radius: 6px;
    padding: 0.4rem 0.7rem;
    cursor: pointer;
  }
  .actions .save.dirty { background: var(--amber); color: #1a1204; }
  .actions .save:disabled { opacity: 0.6; cursor: default; }
  .ranges { display: flex; flex-direction: column; min-height: 0; }
  .ranges h3 {
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-dim);
    margin: 0.2rem 0;
  }
  .ranges ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  .ranges li {
    display: flex;
    align-items: stretch;
    gap: 0.25rem;
  }
  .ranges .seek {
    flex: 1;
    display: flex;
    align-items: center;
    background: var(--hover, #131c28);
    border: 1px solid var(--line-strong);
    border-radius: 6px;
    padding: 0.3rem 0.45rem;
    cursor: pointer;
    color: inherit;
    min-width: 0;
  }
  .r-frames { font-size: 0.7rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .r-sel {
    background: var(--bg-3);
    color: inherit;
    border: 1px solid var(--line-strong);
    border-radius: 6px;
    font-size: 0.7rem;
    padding: 0.15rem 0.2rem;
    max-width: 5.5rem;
  }
  .ranges .del {
    background: transparent;
    border: 1px solid var(--line-strong);
    color: var(--red);
    border-radius: 6px;
    width: 1.8rem;
    cursor: pointer;
    font-size: 1rem;
  }
  .legend {
    font-size: 0.68rem;
    color: var(--text-dim);
    border-top: 1px solid var(--line-strong);
    padding-top: 0.4rem;
    margin-top: auto;
  }

  .pad { padding: 1rem; }
  .small { font-size: 0.72rem; }
  .muted { color: var(--text-dim); }
  .error { color: #ff6b6b; }
  .ok { color: var(--green); }
  .ghost {
    background: transparent;
    border: 1px solid var(--line-strong);
    color: inherit;
    border-radius: 6px;
    cursor: pointer;
    padding: 0.12rem 0.38rem;
  }
  code {
    background: var(--bg-3);
    padding: 0.05rem 0.3rem;
    border-radius: 4px;
    font-size: 0.85em;
  }

  @media (max-width: 1100px) {
    .stage { grid-template-columns: 1fr; overflow-y: auto; }
  }
</style>
