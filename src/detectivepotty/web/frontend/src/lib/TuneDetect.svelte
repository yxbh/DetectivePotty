<script lang="ts">
  import { onDestroy } from "svelte";
  import {
    exportCoreml,
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
    // Whether this entry was fetched with pose=1. Box detections are valid for
    // the (clip, model) scope regardless of overlay mode; pose is layered on
    // only when wanted, so toggling boxes<->pose never discards boxes.
    posed: boolean;
  }

  interface ZoomCard {
    det: TuneDetection;
    pose: TunePose | null;
    kept: boolean;
  }

  const hasRvfc =
    typeof HTMLVideoElement !== "undefined" &&
    "requestVideoFrameCallback" in HTMLVideoElement.prototype;

  let listing = $state<TuneListing | null>(null);
  let listingLoading = $state(false);
  let listingError = $state<string | null>(null);

  let models = $state<string[]>([]);
  let selectedModel = $state<string>("");
  // One-off CoreML export (the "Export to CoreML (GPU)" button) state.
  let exporting = $state(false);
  let exportError = $state<string | null>(null);

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
  // Per-detection zoom-crop canvases, indexed alongside `zoomCards`.
  let zoomCanvases: HTMLCanvasElement[] = [];
  let showZoom = $state(true);

  // Scrub state. `scrubbing` is true while the pointer drags the native range;
  // `scrubIndex` is the live drag target (drives the thumb instantly, decoupled
  // from the slower video seek). `pendingDisplayIndex` holds the thumb at a
  // committed seek target until the video actually paints it (no snap-back).
  let scrubbing = $state(false);
  let scrubIndex = $state(0);
  let pendingDisplayIndex = $state<number | null>(null);

  // Coalesced seeking: at most one seek in flight, always converging on the
  // latest requested target so a fast drag can't pile up dozens of seeks.
  let seekPendingTime: number | null = null;
  let seekPendingPrecise = false;
  let seekBusy = false;
  let seekWatchdog: ReturnType<typeof setTimeout> | null = null;

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

  // The frame the scrub thumb should show. While dragging it follows the live
  // drag target; after a committed seek it holds the requested frame until the
  // video paints it; otherwise it tracks the actually-presented frame (so
  // playback and stepping move the thumb). The overlay still always draws
  // `presentedIndex` — only the thumb uses this.
  const displayIndex = $derived(
    scrubbing
      ? scrubIndex
      : pendingDisplayIndex !== null
        ? pendingDisplayIndex
        : presentedIndex,
  );

  // Per-detection zoom cards for the presented frame (highest confidence first,
  // capped). Each carries the best-matching pose (by box IoU) for its crop.
  const zoomCards = $derived(buildZoomCards(frameDetections, framePose, threshold));

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

  // Scope is (clip, model) only — NOT overlay mode. Box detections don't change
  // when the user toggles the pose overlay, so the buffer stays valid and pose
  // is filled in additively (see `frameComplete`), never rebuffering on toggle.
  function currentScopeKey(): string {
    return `${selectedPath}|${selectedModel}`;
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
    resetSeekState();
    scopeToken++;
    buffer.clear();
    bufferedCount = 0;
    // Setting selectedPath renders the <video> with the new src; onLoadedMetadata
    // then seeks to frame 0, registers rVFC, and starts the filler.
    selectedPath = path;
  }

  // Drop any in-flight/queued seek bookkeeping so a new clip doesn't inherit a
  // stale "busy" flag (which would otherwise wedge future seeks).
  function resetSeekState(): void {
    seekPendingTime = null;
    seekPendingPrecise = false;
    seekBusy = false;
    if (seekWatchdog !== null) {
      clearTimeout(seekWatchdog);
      seekWatchdog = null;
    }
    scrubbing = false;
    scrubIndex = 0;
    pendingDisplayIndex = null;
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
    resetSeekState();
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
    onPresented();
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
    onPresented();
    syncView();
    pump();
  }

  // Called whenever a new frame is actually presented (rVFC or fallback). Keeps
  // the paused filler anchor (`intendedIndex`) tracking the on-screen frame
  // during playback, and releases the scrub-thumb hold once the video catches
  // up to a committed seek target.
  function onPresented(): void {
    if (playing) {
      intendedIndex = presentedIndex;
    }
    if (pendingDisplayIndex !== null && presentedIndex === pendingDisplayIndex) {
      pendingDisplayIndex = null;
    }
  }

  // Pull the presented frame's scope-valid detections/pose out of the buffer into
  // reactive state; the draw effect and HUD counts react to these. A stale or
  // missing entry yields empty overlays.
  function syncView(): void {
    const entry = buffer.get(presentedIndex);
    if (entry && entry.scopeKey === currentScopeKey()) {
      frameDetections = entry.detections;
      // Show pose (in both the main overlay and the zoom crops) only when the
      // overlay wants it; in boxes mode the zoom still shows enlarged crops.
      framePose = entry.posed && poseWanted() ? entry.pose : [];
    } else {
      frameDetections = [];
      framePose = [];
    }
  }

  // --- transport ------------------------------------------------------------

  // Request a seek to a frame index. `precise` true lands exactly on the frame
  // (stepping, click, the final landing after a drag); `precise` false allows an
  // approximate fast preview (mid-drag) where supported. Seeks are coalesced:
  // only one is ever in flight and we always converge on the latest target, so a
  // fast drag can't queue dozens of seeks.
  function requestSeek(target: number, precise: boolean): void {
    if (!videoEl || !meta || totalFrames <= 0) {
      return;
    }
    const next = clamp(target, 0, totalFrames - 1);
    intendedIndex = next;
    if (precise) {
      // Hold the thumb at the committed target until the video paints it.
      pendingDisplayIndex = next;
    }
    seekPendingTime = (next + 0.5) / fps;
    seekPendingPrecise = precise;
    if (!seekBusy) {
      issueSeek();
    }
  }

  function issueSeek(): void {
    if (seekPendingTime === null || !videoEl) {
      return;
    }
    const t = seekPendingTime;
    const precise = seekPendingPrecise;
    seekPendingTime = null;
    seekBusy = true;
    // Pause synchronously (not just videoEl.pause(), whose onpause is async) so
    // the immediate pump() below uses the paused filler anchor (intendedIndex),
    // not the stale playing anchor.
    if (playing) {
      playing = false;
      videoEl.pause();
    }
    if (!precise && typeof videoEl.fastSeek === "function") {
      videoEl.fastSeek(t);
    } else {
      videoEl.currentTime = t;
    }
    armSeekWatchdog();
    pump();
  }

  // Defensive: some browsers may not fire `seeked` (e.g. seeking to ~the current
  // time, or a torn-down element). If one is dropped, don't wedge `seekBusy`.
  function armSeekWatchdog(): void {
    if (seekWatchdog !== null) {
      clearTimeout(seekWatchdog);
    }
    seekWatchdog = setTimeout(() => {
      seekWatchdog = null;
      if (seekBusy) {
        seekBusy = false;
        if (seekPendingTime !== null) {
          issueSeek();
        }
      }
    }, 500);
  }

  function onVideoSeeked(): void {
    if (seekWatchdog !== null) {
      clearTimeout(seekWatchdog);
      seekWatchdog = null;
    }
    if (seekBusy) {
      seekBusy = false;
      if (seekPendingTime !== null) {
        issueSeek();
      }
    }
    syncFromCurrentTime();
  }

  function seekToIndex(target: number): void {
    requestSeek(target, true);
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
    // Playback supersedes any held scrub target.
    pendingDisplayIndex = null;
    pump();
  }

  function onPause(): void {
    playing = false;
    // Once paused, the paused filler anchors at the visible frame.
    intendedIndex = presentedIndex;
    pump();
  }

  // --- native scrub bar -----------------------------------------------------
  // The scrub bar is a native <input type=range> (its thumb drags fluidly for
  // free, decoupled from the slower video seek) over a separate "analyzed" strip
  // canvas. Both share the same value->position mapping (idx / (total-1)) and the
  // strip is inset by half the thumb width (CSS), so thumb and strip line up at
  // every position, including the ends.
  function onSeekPointerDown(): void {
    if (totalFrames <= 0) {
      return;
    }
    scrubbing = true;
    scrubIndex = displayIndex;
  }

  function onSeekInput(event: Event): void {
    if (totalFrames <= 0) {
      return;
    }
    const value = clamp(
      Math.round(Number((event.currentTarget as HTMLInputElement).value)),
      0,
      totalFrames - 1,
    );
    scrubIndex = value;
    if (scrubbing) {
      // Live drag: cheap approximate preview, thumb already follows scrubIndex.
      requestSeek(value, false);
    } else {
      // Keyboard / discrete change (Home/End/PageUp/Down): precise landing.
      requestSeek(value, true);
    }
  }

  function onSeekCommit(): void {
    if (!scrubbing) {
      return;
    }
    scrubbing = false;
    // Land exactly on the released frame.
    requestSeek(scrubIndex, true);
  }

  function setModel(value: string): void {
    if (value === selectedModel) return;
    selectedModel = value;
    resetScope();
  }

  // Dropdown label: ".../yolo11m.mlpackage" -> "yolo11m (CoreML)"; ".pt" -> basename.
  function modelOptionLabel(model: string): string {
    const base = model.split("/").pop() ?? model;
    return base.endsWith(".mlpackage")
      ? `${base.slice(0, -".mlpackage".length)} (CoreML)`
      : base;
  }

  function modelStem(model: string): string {
    const base = model.split("/").pop() ?? model;
    return base.replace(/\.(pt|mlpackage)$/, "");
  }

  // The export button only applies to a .pt source; if its CoreML twin already
  // exists the button just switches to it instead of re-exporting.
  let canExport = $derived(selectedModel.endsWith(".pt"));
  let existingCoreml = $derived(
    selectedModel.endsWith(".pt")
      ? (models.find(
          (m) => m.endsWith(".mlpackage") && modelStem(m) === modelStem(selectedModel),
        ) ?? null)
      : null,
  );

  async function onExportCoreml(): Promise<void> {
    if (!canExport || exporting) return;
    if (existingCoreml) {
      setModel(existingCoreml);
      return;
    }
    exporting = true;
    exportError = null;
    try {
      const result = await exportCoreml(selectedModel);
      models = result.models;
      setModel(result.model);
    } catch (err) {
      exportError = err instanceof Error ? err.message : "CoreML export failed";
    } finally {
      exporting = false;
    }
  }

  function setOverlay(mode: OverlayMode): void {
    if (mode === overlayMode) {
      return;
    }
    overlayMode = mode;
    // No buffer reset: box detections are already valid for this (clip, model)
    // scope. Toggling the overlay only changes whether pose is *also*
    // fetched/drawn; syncView re-reads the current frame and pump() tops up any
    // missing pose additively, so boxes never re-analyze from scratch.
    syncView();
    pump();
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

  // A frame is "complete" once we hold a scope-valid entry for it that also
  // carries pose when the current overlay wants pose. This lets a boxes->pose
  // switch re-enrich existing entries with pose instead of refetching boxes.
  function frameComplete(idx: number): boolean {
    const entry = buffer.get(idx);
    if (!entry || entry.scopeKey !== currentScopeKey()) {
      return false;
    }
    return entry.posed || !poseWanted();
  }

  function nextNeeded(): number | null {
    for (const idx of candidateOrder()) {
      if (!inFlight.has(idx) && !frameComplete(idx)) {
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
        posed: pose,
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

  // Redraw the per-detection zoom crops when the frame, detections, pose, or
  // threshold change. Cheap: a few small drawImage() pulls from the <video>.
  $effect(() => {
    void zoomCards;
    void presentedIndex;
    void showZoom;
    if (showZoom) {
      drawZoom();
    }
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

  // The "analyzed" strip under the scrub bar: which frames have detections
  // buffered yet. It's a separate canvas inset (in CSS) by half the range thumb
  // width and uses the SAME value->position mapping as the native range —
  // idx / (total - 1) across the canvas width — so a buffered column sits exactly
  // under the thumb for that frame, at every position including the two ends.
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
    const span = Math.max(1, totalFrames - 1);
    const colW = Math.max(1, Math.ceil(w / totalFrames));
    ctx.fillStyle = "#3f7d5a";
    const key = currentScopeKey();
    for (const [idx, entry] of buffer) {
      if (entry.scopeKey !== key) continue;
      const x = Math.min(w - colW, Math.floor((idx / span) * w));
      ctx.fillRect(x, 0, colW, h);
    }
  }

  // --- zoom crops -----------------------------------------------------------

  const ZOOM_TARGET = 220; // longest-side px of a zoom crop

  function boxIou(a: TuneDetection, b: number[]): number {
    const ix1 = Math.max(a.x1, b[0]);
    const iy1 = Math.max(a.y1, b[1]);
    const ix2 = Math.min(a.x2, b[2]);
    const iy2 = Math.min(a.y2, b[3]);
    const iw = Math.max(0, ix2 - ix1);
    const ih = Math.max(0, iy2 - iy1);
    const inter = iw * ih;
    if (inter <= 0) {
      return 0;
    }
    const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
    const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
    const union = areaA + areaB - inter;
    return union > 0 ? inter / union : 0;
  }

  // Pose payloads are 1:1 with detection boxes server-side (each pose carries
  // the detector bbox), so the best IoU match recovers the dog for a crop.
  function matchPose(det: TuneDetection, poses: TunePose[]): TunePose | null {
    let best: TunePose | null = null;
    let bestIou = 0.1; // require a little overlap to associate
    for (const pose of poses) {
      if (!pose.bbox || pose.bbox.length < 4) {
        continue;
      }
      const score = boxIou(det, pose.bbox);
      if (score > bestIou) {
        bestIou = score;
        best = pose;
      }
    }
    return best;
  }

  function buildZoomCards(
    dets: TuneDetection[],
    poses: TunePose[],
    thr: number,
  ): ZoomCard[] {
    return dets
      .map((det) => ({
        det,
        pose: matchPose(det, poses),
        kept: det.confidence >= thr,
      }))
      .sort((a, b) => b.det.confidence - a.det.confidence)
      .slice(0, 8);
  }

  function drawZoom(): void {
    if (!videoEl || !meta) {
      return;
    }
    for (let i = 0; i < zoomCards.length; i++) {
      const canvas = zoomCanvases[i];
      if (canvas) {
        drawZoomCard(canvas, zoomCards[i]);
      }
    }
  }

  function drawZoomCard(canvas: HTMLCanvasElement, card: ZoomCard): void {
    if (!videoEl || !meta) {
      return;
    }
    const fw = meta.width;
    const fh = meta.height;
    const det = card.det;
    // Pad the crop so keypoints near the box edge stay visible.
    const padX = (det.x2 - det.x1) * 0.12;
    const padY = (det.y2 - det.y1) * 0.12;
    const sx = clamp(det.x1 - padX, 0, fw);
    const sy = clamp(det.y1 - padY, 0, fh);
    const sw = Math.max(1, clamp(det.x2 + padX, 0, fw) - sx);
    const sh = Math.max(1, clamp(det.y2 + padY, 0, fh) - sy);
    const scale = clamp(ZOOM_TARGET / Math.max(sw, sh), 0.1, 8);
    const cw = Math.max(1, Math.round(sw * scale));
    const ch = Math.max(1, Math.round(sh * scale));
    if (canvas.width !== cw) canvas.width = cw;
    if (canvas.height !== ch) canvas.height = ch;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.clearRect(0, 0, cw, ch);
    try {
      ctx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, cw, ch);
    } catch {
      return; // video not sampleable this tick
    }
    ctx.lineWidth = 2;
    ctx.strokeStyle = card.kept ? "#28d17c" : "#e0556b";
    ctx.strokeRect(
      (det.x1 - sx) * scale,
      (det.y1 - sy) * scale,
      (det.x2 - det.x1) * scale,
      (det.y2 - det.y1) * scale,
    );
    if (card.pose) {
      drawPoseScaled(ctx, card.pose, sx, sy, scale);
    }
  }

  // Like drawPose, but offset+scaled into a zoom-crop canvas with fixed
  // card-space dot/line sizes so points read clearly even for tiny boxes.
  function drawPoseScaled(
    ctx: CanvasRenderingContext2D,
    pose: TunePose,
    sx: number,
    sy: number,
    scale: number,
  ): void {
    const byName = new Map(pose.keypoints.map((kp) => [kp.name, kp]));
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#5ad1ff";
    for (const [a, b] of POSE_EDGES) {
      const pa = byName.get(a);
      const pb = byName.get(b);
      if (!pa || !pb) {
        continue;
      }
      ctx.beginPath();
      ctx.moveTo((pa.x - sx) * scale, (pa.y - sy) * scale);
      ctx.lineTo((pb.x - sx) * scale, (pb.y - sy) * scale);
      ctx.stroke();
    }
    ctx.fillStyle = "#ffd166";
    for (const kp of pose.keypoints) {
      ctx.beginPath();
      ctx.arc((kp.x - sx) * scale, (kp.y - sy) * scale, 3, 0, Math.PI * 2);
      ctx.fill();
    }
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

<div class="tune" class:has-zoom={selectedPath && showZoom}>
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
            onseeked={onVideoSeeked}
            ontimeupdate={syncFromCurrentTime}
          ></video>
          <canvas bind:this={canvasEl}></canvas>
          {#if metaError}
            <div class="frame-error">{metaError}</div>
          {/if}
        </div>

        <div class="hud mono">
          <span class="clip" title={selectedPath}>{selectedName}</span>
          <span>frame {displayIndex}{totalFrames ? ` / ${totalFrames - 1}` : ""}</span>
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
            value={displayIndex}
            disabled={totalFrames <= 0}
            aria-label="Timeline"
            onpointerdown={onSeekPointerDown}
            oninput={onSeekInput}
            onpointerup={onSeekCommit}
            onpointercancel={onSeekCommit}
            onchange={onSeekCommit}
          />
          <div class="strip-wrap">
            <canvas bind:this={stripEl} class="strip"></canvas>
          </div>
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
                  <option value={model}>{modelOptionLabel(model)}</option>
                {/each}
              </select>
            </label>
          {/if}

          {#if canExport}
            <button
              type="button"
              class="coreml-btn"
              onclick={onExportCoreml}
              disabled={exporting}
              title={existingCoreml
                ? "Use the already-exported CoreML (GPU) model"
                : "Export this model to a GPU-safe CoreML model (runs on the GPU, ~2x faster). Takes ~1 min."}
            >
              {exporting
                ? "Exporting… (~1 min)"
                : existingCoreml
                  ? "Switch to CoreML (GPU)"
                  : "Export to CoreML (GPU)"}
            </button>
          {/if}
          {#if exportError}
            <span class="export-error mono small" role="alert">{exportError}</span>
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

          <button
            type="button"
            class="zoom-toggle"
            class:active={showZoom}
            onclick={() => (showZoom = !showZoom)}
            title="Show zoomed crops of each detection"
          >
            ⛶ zoom
          </button>
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

  {#if selectedPath && showZoom}
    <aside class="zoom-col">
      <div class="zoom-head">
        <span class="eyebrow">DETECTIONS</span>
        <span class="mono muted small">{zoomCards.length}</span>
      </div>
      {#if zoomCards.length > 0}
        <div class="zoom">
          {#each zoomCards as card, i (card.det.x1 + ":" + card.det.y1 + ":" + i)}
            <figure class="zoom-card" class:dropped={!card.kept}>
              <canvas bind:this={zoomCanvases[i]}></canvas>
              <figcaption class="mono">
                {card.det.class_name}
                {card.det.confidence.toFixed(2)}{card.pose ? " · pose" : ""}
              </figcaption>
            </figure>
          {/each}
        </div>
      {:else}
        <div class="zoom-empty muted small">No detections on this frame.</div>
      {/if}
    </aside>
  {/if}
</div>

<style>
  .tune {
    display: grid;
    grid-template-columns: 280px minmax(0, 1fr);
    grid-template-areas: "browser stage";
    gap: 1rem;
    min-height: 0;
    height: 100%;
    padding: 1rem 1.25rem;
    box-sizing: border-box;
  }

  /* Wide + zoom on: file list | player | zoom column (crops stack down). */
  .tune.has-zoom {
    grid-template-columns: 260px minmax(0, 1fr) 320px;
    grid-template-areas: "browser stage zoom";
  }

  .browser {
    grid-area: browser;
  }

  .stage {
    grid-area: stage;
  }

  .zoom-col {
    grid-area: zoom;
  }

  /* Medium: drop the zoom to a full-width row beneath the player (crops wrap).
     The stacked layout is content-sized and scrolls the tune-main as a whole, so
     the player is never squeezed/clipped by the detections row. */
  @media (max-width: 1280px) {
    .tune,
    .tune.has-zoom {
      height: auto;
      min-height: 100%;
    }

    .tune.has-zoom {
      grid-template-columns: 260px minmax(0, 1fr);
      grid-template-rows: auto auto;
      grid-template-areas:
        "browser stage"
        "zoom zoom";
    }

    .tune.has-zoom .zoom-col {
      min-width: 0;
      max-height: 32vh;
    }

    .tune.has-zoom .zoom {
      flex-direction: row;
      flex-wrap: wrap;
      align-items: flex-start;
    }

    .tune.has-zoom .zoom-card {
      width: 200px;
    }
  }

  /* Narrow: single-column stack (file list, player, zoom). */
  @media (max-width: 860px) {
    .tune,
    .tune.has-zoom {
      grid-template-columns: minmax(0, 1fr);
      grid-template-rows: auto auto auto;
      grid-template-areas:
        "browser"
        "stage"
        "zoom";
    }
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
    /* Keep the player pinned to the top so a taller detections column never
       stretches/centers it and pushes the transport controls down. */
    align-items: flex-start;
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
    /* Single source of truth for the thumb width: the native range thumb and
       the analyzed-strip inset both derive from it, so they stay aligned. */
    --seek-thumb-w: 14px;
    --seek-track-h: 6px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .seek {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: var(--seek-thumb-w);
    margin: 0;
    background: transparent;
    cursor: pointer;
  }

  .seek:disabled {
    cursor: default;
    opacity: 0.5;
  }

  .seek::-webkit-slider-runnable-track {
    height: var(--seek-track-h);
    border-radius: 3px;
    background: var(--line-strong, #324056);
  }

  .seek::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    box-sizing: border-box;
    width: var(--seek-thumb-w);
    height: var(--seek-thumb-w);
    border: none;
    border-radius: 50%;
    background: var(--amber, #f0b35a);
    /* Centre the thumb vertically on the track. */
    margin-top: calc((var(--seek-track-h) - var(--seek-thumb-w)) / 2);
  }

  .seek::-moz-range-track {
    height: var(--seek-track-h);
    border-radius: 3px;
    background: var(--line-strong, #324056);
  }

  .seek::-moz-range-thumb {
    box-sizing: border-box;
    width: var(--seek-thumb-w);
    height: var(--seek-thumb-w);
    border: none;
    border-radius: 50%;
    background: var(--amber, #f0b35a);
  }

  .seek:focus-visible {
    outline: 2px solid var(--accent, #3f7d5a);
    outline-offset: 3px;
    border-radius: 6px;
  }

  /* Inset by half the thumb width so the strip's [0..width] spans exactly the
     thumb's centre-travel range; the strip then uses idx/(total-1) like the
     range, lining up at every position including both ends. */
  .strip-wrap {
    padding-inline: calc(var(--seek-thumb-w) / 2);
  }

  .strip-wrap .strip {
    display: block;
    width: 100%;
    height: 8px;
    border-radius: 3px;
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

  .coreml-btn {
    background: var(--bg-1, #141a24);
    border: 1px solid var(--accent, #3f7d5a);
    color: var(--text, #d8e0ec);
    border-radius: 6px;
    padding: 0.25rem 0.55rem;
    font-size: 0.74rem;
    font-family: ui-monospace, monospace;
    cursor: pointer;
    white-space: nowrap;
  }

  .coreml-btn:hover:not(:disabled) {
    border-color: var(--amber, #f0b35a);
  }

  .coreml-btn:disabled {
    opacity: 0.6;
    cursor: progress;
  }

  .export-error {
    color: var(--amber, #f0b35a);
    max-width: 22ch;
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

  .zoom-toggle {
    background: var(--bg-1, #141a24);
    border: 1px solid var(--line-strong, #324056);
    color: var(--muted, #8a97a8);
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    cursor: pointer;
    font-size: 0.78rem;
  }

  .zoom-toggle.active {
    background: var(--accent-dim, #1d3346);
    color: #fff;
  }

  .zoom-col {
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
    border: 1px solid var(--line, #243042);
    border-radius: 10px;
    background: var(--bg-1, #141a24);
    overflow: hidden;
  }

  /* Keep the panel a stable width so toggling between frames with and without
     detections (e.g. while scrubbing) doesn't reflow / resize the player. */
  .tune.has-zoom .zoom-col {
    min-width: 320px;
  }

  .zoom-empty {
    padding: 0.75rem;
  }

  .zoom-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0.7rem;
    border-bottom: 1px solid var(--line, #243042);
  }

  .zoom-head .eyebrow {
    font-size: 0.62rem;
    letter-spacing: 0.08em;
    color: var(--muted, #8a97a8);
  }

  /* Wide: crops stack vertically down the column. */
  .zoom {
    display: flex;
    flex-direction: column;
    flex: 1;
    gap: 0.6rem;
    align-items: stretch;
    overflow-y: auto;
    min-height: 0;
    padding: 0.6rem;
  }

  .zoom-card {
    margin: 0;
    flex: 0 0 auto;
    border: 2px solid #28d17c;
    border-radius: 8px;
    overflow: hidden;
    background: #000;
    display: flex;
    flex-direction: column;
  }

  .zoom-card.dropped {
    border-color: #e0556b;
  }

  .zoom-card canvas {
    display: block;
    width: 100%;
    max-width: 100%;
    height: auto;
  }

  .zoom-card figcaption {
    font-size: 0.66rem;
    color: var(--text, #d8e0ec);
    padding: 0.2rem 0.4rem;
    background: var(--bg-1, #141a24);
    white-space: nowrap;
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
