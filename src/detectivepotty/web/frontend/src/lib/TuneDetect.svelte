<script lang="ts">
  import { onDestroy } from "svelte";
  import {
    fetchTuneDetect,
    fetchTuneFiles,
    fetchTuneMeta,
    fetchTuneModels,
    tuneClipUrl,
  } from "./api";
  import type {
    TuneDetection,
    TuneEntry,
    TuneListing,
    TuneMeta,
    TunePose,
  } from "./types";

  // How many frames Shift+Arrow skips.
  const SKIP_N = 10;
  // Cap concurrent detection fetches. Server serializes inference under a lock,
  // so a deep queue just delays interactive seeks; 2 keeps the pipe busy while
  // a fresh seek's frame can still jump the queue within one inference slot.
  const MAX_INFLIGHT = 2;
  // The floor the slider can't go below (overwritten by the first detect result).
  const DEFAULT_FLOOR = 0.05;

  // Skeleton edges by raw DeepLabCut keypoint name. Drawn only when both
  // endpoints are present, so the mock (a torso+paws subset) and the full
  // 39-point SuperAnimal backend both render sensibly.
  const POSE_EDGES: Array<[string, string]> = [
    ["nose", "upper_jaw"],
    ["nose", "neck_base"],
    ["neck_base", "back_base"],
    ["back_base", "neck_end"],
    ["back_base", "back_middle"],
    ["back_middle", "back_end"],
    ["back_end", "tail_base"],
    ["tail_base", "tail_end"],
    ["back_base", "front_left_paw"],
    ["back_base", "front_right_paw"],
    ["front_left_thai", "front_left_knee"],
    ["front_left_knee", "front_left_paw"],
    ["front_right_thai", "front_right_knee"],
    ["front_right_knee", "front_right_paw"],
    ["back_end", "back_left_paw"],
    ["back_end", "back_right_paw"],
    ["back_left_thai", "back_left_knee"],
    ["back_left_knee", "back_left_paw"],
    ["back_right_thai", "back_right_knee"],
    ["back_right_knee", "back_right_paw"],
  ];

  type OverlayMode = "boxes" | "pose" | "both";

  interface BufferEntry {
    scopeKey: string;
    detections: TuneDetection[];
    pose: TunePose[];
  }

  const hasRvfc =
    typeof HTMLVideoElement !== "undefined" &&
    "requestVideoFrameCallback" in HTMLVideoElement.prototype;

  let listing = $state<TuneListing | null>(null);
  let listingLoading = $state(false);
  let listingError = $state<string | null>(null);

  let models = $state<string[]>([]);
  let selectedModel = $state<string>("");

  let selectedPath = $state<string | null>(null);
  let selectedName = $state<string>("");
  let meta = $state<TuneMeta | null>(null);
  let metaError = $state<string | null>(null);

  // presentedIndex = the frame the <video> says it actually painted (rVFC truth).
  // intendedIndex = the frame the user asked to seek to (may briefly differ while
  // the browser settles a seek). Overlay always draws presentedIndex.
  let presentedIndex = $state(0);
  let intendedIndex = 0;
  let threshold = $state(0.25);
  let overlayMode = $state<OverlayMode>("boxes");
  let playing = $state(false);
  let poseAvailable = $state(true);
  let floor = $state(DEFAULT_FLOOR);
  let bufferedCount = $state(0);

  let videoEl = $state<HTMLVideoElement | null>(null);
  let canvasEl = $state<HTMLCanvasElement | null>(null);
  let stripEl = $state<HTMLCanvasElement | null>(null);

  // The scope-valid detections/pose for the currently presented frame. Kept as
  // reactive state (the `buffer` Map itself is non-reactive) so the draw effect
  // and HUD counts update when the playhead or buffer changes.
  let frameDetections = $state<TuneDetection[]>([]);
  let framePose = $state<TunePose[]>([]);

  // Async overlay buffer: detections (+pose) per frame index, each tagged with the
  // (path, model, pose) scope it was fetched under so a stale draw is impossible.
  const buffer = new Map<number, BufferEntry>();
  const inFlight = new Map<number, AbortController>();
  // Bumped on every scope change (clip / model / pose). In-flight fetches whose
  // token is stale are dropped; a new clip selection also bumps selectSeq.
  let scopeToken = 0;
  let selectSeq = 0;
  let rvfcHandle: number | null = null;
  // EMA of detect latency (ms) used to lead the playhead while playing.
  let detectMsEma = 250;

  // Derived counts/fps for the HUD. Counts are derived (never written from an
  // effect) so the draw effect can't feed back into its own dependencies.
  const fps = $derived(effectiveFps(meta));
  const totalFrames = $derived(meta?.total_frames ?? 0);
  const aboveCount = $derived(
    frameDetections.filter((d) => d.confidence >= threshold).length,
  );
  const belowCount = $derived(frameDetections.length - aboveCount);

  void loadListing("");
  void loadModels();

  onDestroy(() => {
    teardownClip();
  });

  function clamp(v: number, lo: number, hi: number): number {
    return v < lo ? lo : v > hi ? hi : v;
  }

  function effectiveFps(m: TuneMeta | null): number {
    if (!m) return 30;
    if (m.fps && m.fps > 0) return m.fps;
    if (m.duration > 0 && m.total_frames) return m.total_frames / m.duration;
    return 30;
  }

  function poseWanted(): boolean {
    return overlayMode !== "boxes";
  }

  function currentScopeKey(): string {
    return `${selectedPath}|${selectedModel}|${poseWanted() ? 1 : 0}`;
  }

  // --- listing + models -----------------------------------------------------

  async function loadListing(path: string): Promise<void> {
    listingLoading = true;
    listingError = null;
    try {
      listing = await fetchTuneFiles(path);
    } catch (err) {
      listingError = err instanceof Error ? err.message : String(err);
    } finally {
      listingLoading = false;
    }
  }

  async function loadModels(): Promise<void> {
    try {
      const data = await fetchTuneModels();
      models = data.models;
      selectedModel = data.default || data.models[0] || "";
    } catch {
      // Non-fatal: the picker just stays empty; detection still uses the default.
    }
  }

  function openEntry(entry: TuneEntry): void {
    if (entry.kind === "dir") {
      void loadListing(entry.path);
      return;
    }
    void selectVideo(entry.path, entry.name);
  }

  function goUp(): void {
    if (!listing || listing.parent === null) {
      return;
    }
    void loadListing(listing.parent);
  }

  // --- clip selection / lifecycle ------------------------------------------

  async function selectVideo(path: string, name: string): Promise<void> {
    teardownClip();
    const seq = ++selectSeq;
    selectedName = name;
    metaError = null;
    meta = null;
    let m: TuneMeta;
    try {
      m = await fetchTuneMeta(path);
    } catch (err) {
      if (seq !== selectSeq) return;
      metaError = err instanceof Error ? err.message : String(err);
      selectedPath = path;
      return;
    }
    if (seq !== selectSeq) return; // superseded by a newer selection
    meta = m;
    intendedIndex = 0;
    presentedIndex = 0;
    playing = false;
    scopeToken++;
    buffer.clear();
    bufferedCount = 0;
    // Setting selectedPath renders the <video> with the new src; onLoadedMetadata
    // then seeks to frame 0, registers rVFC, and starts the filler.
    selectedPath = path;
  }

  function teardownClip(): void {
    selectSeq++;
    scopeToken++;
    for (const controller of inFlight.values()) {
      controller.abort();
    }
    inFlight.clear();
    buffer.clear();
    bufferedCount = 0;
    if (videoEl) {
      try {
        videoEl.pause();
      } catch {
        /* element may be detaching */
      }
      if (hasRvfc && rvfcHandle != null) {
        try {
          (videoEl as HTMLVideoElement).cancelVideoFrameCallback(rvfcHandle);
        } catch {
          /* ignore */
        }
      }
    }
    rvfcHandle = null;
    playing = false;
  }

  function onLoadedMetadata(): void {
    if (!videoEl || !meta) {
      return;
    }
    resizeCanvas();
    // Seek to the middle of frame 0 so a frame paints (firing rVFC) and the
    // index math is unambiguous.
    videoEl.currentTime = 0.5 / fps;
    registerRvfc();
    pump();
  }

  function resizeCanvas(): void {
    if (!canvasEl || !meta) return;
    if (canvasEl.width !== meta.width) canvasEl.width = meta.width;
    if (canvasEl.height !== meta.height) canvasEl.height = meta.height;
  }

  function registerRvfc(): void {
    if (!hasRvfc || !videoEl) return;
    if (rvfcHandle != null) {
      try {
        videoEl.cancelVideoFrameCallback(rvfcHandle);
      } catch {
        /* ignore */
      }
    }
    rvfcHandle = videoEl.requestVideoFrameCallback(frameCb);
  }

  function frameCb(_now: number, metadata: { mediaTime: number }): void {
    rvfcHandle = null;
    if (!videoEl || !meta) {
      return;
    }
    presentedIndex = clamp(
      Math.round(metadata.mediaTime * fps),
      0,
      Math.max(0, totalFrames - 1),
    );
    syncView();
    pump();
    rvfcHandle = videoEl.requestVideoFrameCallback(frameCb);
  }

  // Fallback for browsers without rVFC: currentTime sits *inside* a frame
  // interval, so floor() (not round()) gives the displayed frame.
  function syncFromCurrentTime(): void {
    if (hasRvfc || !videoEl || !meta) {
      return;
    }
    presentedIndex = clamp(
      Math.floor(videoEl.currentTime * fps + 1e-6),
      0,
      Math.max(0, totalFrames - 1),
    );
    syncView();
    pump();
  }

  // Pull the presented frame's scope-valid detections/pose out of the buffer into
  // reactive state; the draw effect and HUD counts react to these. A stale or
  // missing entry yields empty overlays.
  function syncView(): void {
    const entry = buffer.get(presentedIndex);
    if (entry && entry.scopeKey === currentScopeKey()) {
      frameDetections = entry.detections;
      framePose = entry.pose;
    } else {
      frameDetections = [];
      framePose = [];
    }
  }

  // --- transport ------------------------------------------------------------

  function seekToIndex(target: number): void {
    if (!videoEl || !meta || totalFrames <= 0) {
      return;
    }
    const next = clamp(target, 0, totalFrames - 1);
    intendedIndex = next;
    if (playing) {
      videoEl.pause();
    }
    videoEl.currentTime = (next + 0.5) / fps;
    pump();
  }

  function step(delta: number): void {
    seekToIndex(presentedIndex + delta);
  }

  function togglePlay(): void {
    if (!videoEl || !meta) {
      return;
    }
    if (playing) {
      videoEl.pause();
    } else {
      void videoEl.play().catch(() => {
        /* autoplay/gesture rejection — ignore on a local tool */
      });
    }
  }

  function onPlay(): void {
    playing = true;
    pump();
  }

  function onPause(): void {
    playing = false;
    pump();
  }

  function onScrub(event: Event): void {
    const value = Number((event.currentTarget as HTMLInputElement).value);
    seekToIndex(value);
  }

  function setModel(value: string): void {
    if (value === selectedModel) return;
    selectedModel = value;
    resetScope();
  }

  function setOverlay(mode: OverlayMode): void {
    if (mode === overlayMode) {
      return;
    }
    const wasPose = poseWanted();
    overlayMode = mode;
    if (poseWanted() !== wasPose) {
      resetScope(); // pose payload differs -> the buffer must be rebuilt
    } else {
      syncView();
    }
  }

  // Invalidate + rebuild the detection buffer after a model/pose change.
  function resetScope(): void {
    scopeToken++;
    for (const controller of inFlight.values()) {
      controller.abort();
    }
    inFlight.clear();
    buffer.clear();
    bufferedCount = 0;
    syncView();
    pump();
  }

  // --- detection buffer / background filler --------------------------------

  function leadFrames(): number {
    if (!playing) return 0;
    // Prefetch roughly one inference-latency worth of frames ahead of the
    // playhead so overlays are ready by the time playback reaches them.
    return clamp(Math.round((detectMsEma / 1000) * fps), 2, 60);
  }

  // Ordered indices the filler should try, highest priority first. Paused: the
  // cursor then outward (snappy stepping/scrubbing). Playing: ahead of the
  // playhead first (so we fill the future, never re-chase the moving current
  // frame), then the small near-future gap, then the past for replay.
  function* candidateOrder(): Generator<number> {
    const total = totalFrames;
    if (total <= 0) return;
    if (playing) {
      const anchor = clamp(presentedIndex + leadFrames(), 0, total - 1);
      for (let i = anchor; i < total; i++) yield i;
      for (let i = presentedIndex; i < anchor; i++) yield i;
      for (let i = presentedIndex - 1; i >= 0; i--) yield i;
    } else {
      const anchor = clamp(intendedIndex, 0, total - 1);
      yield anchor;
      for (let r = 1; r < total; r++) {
        if (anchor + r < total) yield anchor + r;
        if (anchor - r >= 0) yield anchor - r;
      }
    }
  }

  function nextNeeded(): number | null {
    for (const idx of candidateOrder()) {
      if (!buffer.has(idx) && !inFlight.has(idx)) {
        return idx;
      }
    }
    return null;
  }

  function pump(): void {
    if (!selectedPath || !meta) {
      return;
    }
    while (inFlight.size < MAX_INFLIGHT) {
      const idx = nextNeeded();
      if (idx === null) break;
      void fetchInto(idx);
    }
  }

  async function fetchInto(idx: number): Promise<void> {
    if (!selectedPath) return;
    const path = selectedPath;
    const model = selectedModel;
    const pose = poseWanted();
    const scopeKey = currentScopeKey();
    const token = scopeToken;
    const controller = new AbortController();
    inFlight.set(idx, controller);
    const started = performance.now();
    try {
      const res = await fetchTuneDetect(path, idx, model, pose, controller.signal);
      if (token !== scopeToken) {
        return; // scope changed while in flight -> drop
      }
      detectMsEma = detectMsEma * 0.7 + (performance.now() - started) * 0.3;
      if (threshold < res.detection_floor) {
        threshold = res.detection_floor;
      }
      floor = res.detection_floor;
      poseAvailable = res.pose_available;
      buffer.set(res.index, {
        scopeKey,
        detections: res.detections,
        pose: res.pose,
      });
      bufferedCount = buffer.size;
      if (res.index === presentedIndex) {
        syncView();
      }
      updateStrip();
    } catch {
      // Aborted or a transient single-frame error: skip it quietly rather than
      // surfacing per-frame noise. The frame can be retried on the next pass.
    } finally {
      inFlight.delete(idx);
      if (token === scopeToken) {
        pump();
      }
    }
  }

  // --- rendering ------------------------------------------------------------

  // Redraw the box/pose overlay when its inputs change. This effect only paints
  // the canvas — it never writes reactive state, so it can't feed back into its
  // own dependencies (the cause of effect_update_depth loops).
  $effect(() => {
    void frameDetections;
    void framePose;
    void threshold;
    void overlayMode;
    drawOverlay();
  });

  // Redraw the "analyzed" strip when the buffer grows or the playhead moves.
  $effect(() => {
    void bufferedCount;
    void presentedIndex;
    void totalFrames;
    updateStrip();
  });

  function drawOverlay(): void {
    const canvas = canvasEl;
    if (!canvas || !meta) {
      return;
    }
    const w = meta.width;
    const h = meta.height;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.clearRect(0, 0, w, h);

    const lineW = Math.max(2, Math.round(w / 400));
    const fontPx = Math.max(12, Math.round(w / 45));

    if (overlayMode === "boxes" || overlayMode === "both") {
      ctx.lineWidth = lineW;
      ctx.font = `${fontPx}px ui-monospace, monospace`;
      ctx.textBaseline = "bottom";
      for (const det of frameDetections) {
        const keep = det.confidence >= threshold;
        const color = keep ? "#28d17c" : "#e0556b";
        ctx.strokeStyle = color;
        ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
        const label = `${det.class_name} ${det.confidence.toFixed(2)}`;
        ctx.fillStyle = color;
        const ty = det.y1 > fontPx + 4 ? det.y1 - 2 : det.y1 + fontPx + 2;
        ctx.fillText(label, det.x1, ty);
      }
    }

    if (overlayMode === "pose" || overlayMode === "both") {
      drawPose(ctx, w, framePose);
    }
  }

  function drawPose(
    ctx: CanvasRenderingContext2D,
    w: number,
    poses: TunePose[],
  ): void {
    const dot = Math.max(2, Math.round(w / 300));
    const lineW = Math.max(1, Math.round(w / 600));
    for (const pose of poses) {
      const byName = new Map(pose.keypoints.map((kp) => [kp.name, kp]));
      ctx.lineWidth = lineW;
      ctx.strokeStyle = "#5ad1ff";
      for (const [a, b] of POSE_EDGES) {
        const pa = byName.get(a);
        const pb = byName.get(b);
        if (!pa || !pb) {
          continue;
        }
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
      }
      ctx.fillStyle = "#ffd166";
      for (const kp of pose.keypoints) {
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, dot, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // The "analyzed" strip under the scrub bar: which frames have detections yet,
  // plus a playhead marker.
  function updateStrip(): void {
    const c = stripEl;
    if (!c || totalFrames <= 0) {
      return;
    }
    const w = c.clientWidth || 300;
    const h = 8;
    if (c.width !== w) c.width = w;
    if (c.height !== h) c.height = h;
    const ctx = c.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#23303f";
    ctx.fillRect(0, 0, w, h);
    const colW = Math.max(1, Math.ceil(w / totalFrames));
    ctx.fillStyle = "#3f7d5a";
    const key = currentScopeKey();
    for (const [idx, entry] of buffer) {
      if (entry.scopeKey !== key) continue;
      const x = Math.floor((idx / totalFrames) * w);
      ctx.fillRect(x, 0, colW, h);
    }
    ctx.fillStyle = "#f0b35a";
    const px = Math.floor((presentedIndex / totalFrames) * w);
    ctx.fillRect(px, 0, 2, h);
  }

  function onKey(event: KeyboardEvent): void {
    if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) {
      return;
    }
    const target = event.target as HTMLElement | null;
    const typing =
      target &&
      (target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.tagName === "SELECT" ||
        target.isContentEditable) &&
      (target as HTMLInputElement).type !== "range";
    if (typing) {
      return;
    }
    switch (event.key) {
      case " ":
        event.preventDefault();
        togglePlay();
        break;
      case "ArrowRight":
        event.preventDefault();
        step(event.shiftKey ? SKIP_N : 1);
        break;
      case "ArrowLeft":
        event.preventDefault();
        step(event.shiftKey ? -SKIP_N : -1);
        break;
      default:
        break;
    }
  }
</script>

<svelte:window onkeydown={onKey} />

<div class="tune">
  <aside class="browser">
    <div class="browser-head">
      <span class="eyebrow mono">CLIPS</span>
      {#if listing && listing.parent !== null}
        <button type="button" class="up" onclick={goUp} title="Up one level">↑ up</button>
      {/if}
    </div>
    {#if listing}
      <div class="crumb mono" title={listing.path}>
        {listing.path || "roots"}
      </div>
    {/if}
    <div class="entries">
      {#if listingLoading}
        <p class="muted">Loading…</p>
      {:else if listingError}
        <p class="error">{listingError}</p>
      {:else if listing && listing.entries.length === 0}
        <p class="muted">No clips or folders here.</p>
      {:else if listing}
        {#each listing.entries as entry (entry.path)}
          <button
            type="button"
            class="entry"
            class:active={entry.path === selectedPath}
            onclick={() => openEntry(entry)}
          >
            <span class="icon">{entry.kind === "dir" ? "📁" : "🎬"}</span>
            <span class="name">{entry.name}</span>
          </button>
        {/each}
      {/if}
    </div>
  </aside>

  <section class="stage">
    {#if !selectedPath}
      <div class="empty">
        <h2>Detection tuner</h2>
        <p class="muted">
          Pick a clip on the left to play it with live YOLO boxes. Boxes fill in
          as detection runs in the background. Drag the confidence slider to see
          which survive — green is kept, red is dropped — then copy that value
          into <code>detection_conf_threshold</code>.
        </p>
        <p class="muted small">
          Space play/pause · ← / → step one frame · Shift + ← / → skip {SKIP_N}
        </p>
      </div>
    {:else}
      <div class="player">
        <div class="viewport">
          <!-- svelte-ignore a11y_media_has_caption -->
          <video
            bind:this={videoEl}
            src={tuneClipUrl(selectedPath)}
            preload="auto"
            playsinline
            muted
            onloadedmetadata={onLoadedMetadata}
            onplay={onPlay}
            onpause={onPause}
            onended={onPause}
            onseeked={syncFromCurrentTime}
            ontimeupdate={syncFromCurrentTime}
          ></video>
          <canvas bind:this={canvasEl}></canvas>
          {#if metaError}
            <div class="frame-error">{metaError}</div>
          {/if}
        </div>

        <div class="hud mono">
          <span class="clip" title={selectedPath}>{selectedName}</span>
          <span>frame {presentedIndex}{totalFrames ? ` / ${totalFrames - 1}` : ""}</span>
          {#if meta}<span>{fps.toFixed(1)} fps</span>{/if}
          <span class="kept">▣ {aboveCount}</span>
          <span class="dropped">▢ {belowCount}</span>
          {#if totalFrames}<span class="muted">analyzed {bufferedCount}/{totalFrames}</span>{/if}
        </div>

        <div class="scrub">
          <input
            type="range"
            class="seek"
            min="0"
            max={Math.max(0, totalFrames - 1)}
            step="1"
            value={presentedIndex}
            oninput={onScrub}
            aria-label="Timeline"
          />
          <canvas bind:this={stripEl} class="strip"></canvas>
        </div>

        <div class="controls">
          <div class="transport">
            <button type="button" onclick={() => step(-SKIP_N)} title="Back {SKIP_N} (Shift+←)">⏮</button>
            <button type="button" onclick={() => step(-1)} title="Back 1 (←)">◀</button>
            <button type="button" class="play" onclick={togglePlay} title="Play/pause (Space)">
              {playing ? "⏸" : "▶"}
            </button>
            <button type="button" onclick={() => step(1)} title="Forward 1 (→)">▶</button>
            <button type="button" onclick={() => step(SKIP_N)} title="Forward {SKIP_N} (Shift+→)">⏭</button>
          </div>

          {#if models.length > 1}
            <label class="model">
              <span class="mono muted small">model</span>
              <select
                value={selectedModel}
                onchange={(e) => setModel((e.currentTarget as HTMLSelectElement).value)}
              >
                {#each models as model (model)}
                  <option value={model}>{model.split("/").pop()}</option>
                {/each}
              </select>
            </label>
          {/if}

          <label class="slider">
            <span class="mono">conf ≥ {threshold.toFixed(2)}</span>
            <input
              type="range"
              min={floor}
              max="1"
              step="0.01"
              bind:value={threshold}
            />
            <span class="mono muted small">floor {floor.toFixed(2)}</span>
          </label>

          <div class="overlay-toggle" role="group" aria-label="Overlay">
            {#each ["boxes", "pose", "both"] as const as mode (mode)}
              <button
                type="button"
                class:active={overlayMode === mode}
                onclick={() => setOverlay(mode as OverlayMode)}
              >
                {mode}
              </button>
            {/each}
          </div>
        </div>

        {#if overlayMode !== "boxes" && !poseAvailable}
          <p class="pose-hint muted small">
            Pose overlay needs the optional pose backend
            (<code>uv sync --extra pose</code>). Boxes still work.
          </p>
        {/if}
      </div>
    {/if}
  </section>
</div>

<style>
  .tune {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 1rem;
    min-height: 0;
    height: 100%;
    padding: 1rem 1.25rem;
    box-sizing: border-box;
  }

  .browser {
    display: flex;
    flex-direction: column;
    min-height: 0;
    border: 1px solid var(--line, #243042);
    border-radius: 10px;
    background: var(--bg-1, #141a24);
    overflow: hidden;
  }

  .browser-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0.75rem;
    border-bottom: 1px solid var(--line, #243042);
  }

  .eyebrow {
    font-size: 0.6rem;
    letter-spacing: 0.28em;
    color: var(--amber, #f0b35a);
  }

  .up {
    font-size: 0.72rem;
    background: transparent;
    border: 1px solid var(--line-strong, #324056);
    color: var(--text, #d8e0ec);
    border-radius: 6px;
    padding: 0.15rem 0.45rem;
    cursor: pointer;
  }

  .crumb {
    font-size: 0.68rem;
    color: var(--muted, #8a97a8);
    padding: 0.4rem 0.75rem;
    border-bottom: 1px solid var(--line, #243042);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    direction: rtl;
    text-align: left;
  }

  .entries {
    overflow-y: auto;
    min-height: 0;
    padding: 0.35rem;
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .entry {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    color: var(--text, #d8e0ec);
    padding: 0.4rem 0.5rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.82rem;
  }

  .entry:hover {
    background: var(--bg-2, #1b2330);
  }

  .entry.active {
    background: var(--accent-dim, #1d3346);
    color: #fff;
  }

  .entry .name {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .stage {
    min-width: 0;
    min-height: 0;
    display: flex;
  }

  .empty {
    margin: auto;
    max-width: 460px;
    text-align: center;
  }

  .empty h2 {
    margin: 0 0 0.5rem;
  }

  .player {
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
    width: 100%;
    min-height: 0;
  }

  .viewport {
    position: relative;
    width: 100%;
    background: #000;
    border-radius: 10px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
  }

  .viewport video {
    display: block;
    width: 100%;
    height: auto;
    max-height: 70vh;
    object-fit: contain;
    background: #000;
  }

  .viewport canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }

  .scrub {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }

  .scrub .seek {
    width: 100%;
    margin: 0;
  }

  .scrub .strip {
    width: 100%;
    height: 8px;
    border-radius: 3px;
    display: block;
  }

  .model {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }

  .model select {
    background: var(--bg-1, #141a24);
    border: 1px solid var(--line-strong, #324056);
    color: var(--text, #d8e0ec);
    border-radius: 6px;
    padding: 0.25rem 0.4rem;
    font-size: 0.78rem;
    font-family: ui-monospace, monospace;
    max-width: 16ch;
  }

  .frame-error {
    position: absolute;
    bottom: 0.5rem;
    left: 0.5rem;
    background: rgba(140, 30, 40, 0.85);
    color: #fff;
    padding: 0.3rem 0.6rem;
    border-radius: 6px;
    font-size: 0.78rem;
  }

  .hud {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.9rem;
    font-size: 0.74rem;
    color: var(--muted, #8a97a8);
  }

  .hud .clip {
    color: var(--text, #d8e0ec);
    max-width: 36ch;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .hud .kept {
    color: #28d17c;
  }

  .hud .dropped {
    color: #e0556b;
  }

  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 1rem;
  }

  .transport {
    display: flex;
    gap: 0.25rem;
  }

  .transport button {
    background: var(--bg-1, #141a24);
    border: 1px solid var(--line-strong, #324056);
    color: var(--text, #d8e0ec);
    border-radius: 6px;
    padding: 0.3rem 0.55rem;
    cursor: pointer;
    font-size: 0.9rem;
    min-width: 2.1rem;
  }

  .transport button:hover {
    background: var(--bg-2, #1b2330);
  }

  .transport .play {
    background: var(--accent-dim, #1d3346);
  }

  .slider {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex: 1 1 260px;
  }

  .slider input[type="range"] {
    flex: 1;
    min-width: 140px;
  }

  .overlay-toggle {
    display: inline-flex;
    border: 1px solid var(--line-strong, #324056);
    border-radius: 6px;
    overflow: hidden;
  }

  .overlay-toggle button {
    background: var(--bg-1, #141a24);
    border: none;
    color: var(--muted, #8a97a8);
    padding: 0.3rem 0.7rem;
    cursor: pointer;
    font-size: 0.78rem;
    text-transform: capitalize;
  }

  .overlay-toggle button + button {
    border-left: 1px solid var(--line-strong, #324056);
  }

  .overlay-toggle button.active {
    background: var(--accent-dim, #1d3346);
    color: #fff;
  }

  .muted {
    color: var(--muted, #8a97a8);
  }

  .small {
    font-size: 0.74rem;
  }

  .error {
    color: #e0556b;
  }

  .pose-hint code,
  .empty code {
    background: var(--bg-2, #1b2330);
    padding: 0.05rem 0.3rem;
    border-radius: 4px;
  }
</style>
