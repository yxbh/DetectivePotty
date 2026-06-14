<script lang="ts">
  import { onDestroy } from "svelte";
  import {
    exportCoreml,
    fetchTuneDetect,
    fetchTuneDetectRange,
    fetchTuneFiles,
    fetchTuneMeta,
    fetchTuneModels,
    fetchTunePoseRange,
    fetchTuneScene,
    streamTuneTrackRange,
    tuneClipUrl,
  } from "./api";
  import { errMsg } from "./errors";
  import { isTypingTarget } from "./keys";
  import type {
    TuneDetection,
    TuneListing,
    TuneMeta,
    TunePose,
    TuneSceneObject,
    TuneTracker,
    TuneTrackStats,
    TuneTrackedDetection,
    TuneUltralyticsTrackerParams,
  } from "./types";
  import TuneClipBrowser from "./TuneClipBrowser.svelte";
  import TuneTimelineStrip from "./TuneTimelineStrip.svelte";
  import Transport from "./Transport.svelte";
  import {
    classBoxColor,
    trackColor,
    drawCanvasBoxLabel,
    boxLabelFontPx,
    formatDetLabel,
    formatTrackLabel,
    BOX_DOG,
    BOX_SIBLING,
    BOX_WEAK,
  } from "./overlayStyle";
  import { loadTuneLastDir, saveTuneLastDir } from "./prefs";
  import {
    DEFAULT_FLOOR,
    MAX_INFLIGHT,
    MAX_POSE_INFLIGHT,
    POSE_EDGES,
    POSE_RANGE_BATCH,
    RANGE_BATCH,
    SKIP_N,
    ULTRA_APPEARANCE_DEFAULT,
    ULTRA_MATCH_DEFAULT,
    ULTRA_NEW_TRACK_DEFAULT,
    ULTRA_PROXIMITY_DEFAULT,
    ULTRA_TRACK_BUFFER_DEFAULT,
    ULTRA_TRACK_HIGH_DEFAULT,
    ULTRA_TRACK_LOW_DEFAULT,
    URGENT_WINDOW,
    buildZoomCards,
    clamp,
  } from "./tuneDetectCore";
  import type { BufferEntry, OverlayMode, ZoomCard } from "./tuneDetectCore";

  const hasRvfc =
    typeof HTMLVideoElement !== "undefined" &&
    "requestVideoFrameCallback" in HTMLVideoElement.prototype;

  let listing = $state<TuneListing | null>(null);
  let listingLoading = $state(false);
  let listingError = $state<string | null>(null);

  let models = $state<string[]>([]);
  let coremlBatch = $state<Record<string, number>>({});
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
  // Frames with >=1 detection (a dog present) and frames whose pose pass has run.
  // Maintained by transition-guarded increments in the fillers (never recomputed
  // from an effect) and used both for the HUD and as strip-redraw triggers.
  let detectedCount = $state(0);
  let posedCount = $state(0);

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

  // --- tracking (P1j) -------------------------------------------------------
  // Tracking is a stateful, order-dependent batch pass: the user picks a tracker
  // + knobs and hits "Track range", the server decodes+detects+tracks the whole
  // clip in frame order (exactly like the harvest scan), and we cache the
  // per-frame track-id assignments. Scrub/play then draws persistent colored
  // boxes from the cache — the client stays stateless.
  let tracker = $state<TuneTracker>("off");
  // Harvest `ours` (IoU + center-gate) knobs — tuning these here tunes the
  // harvest segmentation, since both replay through the same `Tracker`.
  let trackSampleEvery = $state(5);
  let trackIou = $state(0.3);
  let trackMaxAge = $state(15);
  let trackCenterGate = $state(1.5);
  let ultraConf = $state(DEFAULT_FLOOR);
  let ultraTrackHigh = $state(ULTRA_TRACK_HIGH_DEFAULT);
  let ultraTrackLow = $state(ULTRA_TRACK_LOW_DEFAULT);
  let ultraNewTrack = $state(ULTRA_NEW_TRACK_DEFAULT);
  let ultraTrackBuffer = $state(ULTRA_TRACK_BUFFER_DEFAULT);
  let ultraMatch = $state(ULTRA_MATCH_DEFAULT);
  let ultraProximity = $state(ULTRA_PROXIMITY_DEFAULT);
  let ultraAppearance = $state(ULTRA_APPEARANCE_DEFAULT);
  let tracking = $state(false);
  let trackError = $state<string | null>(null);
  let trackStats = $state<TuneTrackStats | null>(null);
  // Streaming progress: how many sampled frames the forward pass has filled, and
  // the total expected for the bar. Tracking is sequential + stateful, so the pass
  // is always a single 0→end sweep we can show a progress readout for.
  let trackDone = $state(0);
  let trackExpected = $state(0);
  // AbortController for the in-flight streaming pass; switching clip/model/tracker
  // or clearing aborts it.
  let trackController: AbortController | null = null;
  // Debounce timer for the `ours` auto-run (knob typing shouldn't flood passes).
  let trackAutoTimer: ReturnType<typeof setTimeout> | null = null;
  // Tracked boxes per *sampled* source frame index. Non-reactive (like `buffer`);
  // the presented frame's boxes are mirrored into `frameTracks` by syncView.
  const trackByIndex = new Map<number, TuneTrackedDetection[]>();
  // The sample stride the cached track pass was run at (for nearest-sample snap).
  let trackedStride = 1;
  // The tracked boxes for the currently presented frame (reactive draw input).
  let frameTracks = $state<TuneTrackedDetection[]>([]);
  // True once a track pass has populated the cache for this clip/model scope.
  let tracked = $state(false);

  // --- Objects-in-scene diagnostic (top-N all-class detections) --------------
  // A read-only list (no boxes drawn) of what the detector sees on the current
  // frame, INCLUDING non-dog classes. Answers "is this gap an empty scene, a
  // sub-threshold dog, or the dog classed as a cat?". Fetched on the presented
  // frame (debounced, stale-aborted) only while the panel is open.
  let sceneOpen = $state(false);
  let sceneBoxes = $state(false);
  let sceneObjects = $state<TuneSceneObject[]>([]);
  let sceneLoading = $state(false);
  let sceneError = $state<string | null>(null);
  let sceneIndex = $state<number | null>(null);
  let sceneController: AbortController | null = null;
  let sceneTimer: ReturnType<typeof setTimeout> | null = null;
  const SCENE_TOP_N = 8;

  const trackingActive = $derived(tracker !== "off" && tracked);
  // Ultralytics native trackers are .pt-only; CoreML packages use `ours`.
  const isMlpackage = $derived(selectedModel.endsWith(".mlpackage"));

  // Async overlay buffer: detections (+pose) per frame index, each tagged with the
  // (path, model) scope it was fetched under so a stale draw is impossible.
  const buffer = new Map<number, BufferEntry>();
  const inFlight = new Map<number, AbortController>();
  // In-flight YOLO *requests* (a range request covers many indices but counts
  // once). The filler budgets on this, not `inFlight.size`, so a batched backfill
  // request occupies a single slot and an urgent near-cursor frame still gets one.
  let detectRequests = 0;
  // Decoupled pose passes in flight, keyed by frame index (for per-index dedup +
  // abort). A single pose *request* now covers a run of frames.
  const poseInFlight = new Map<number, AbortController>();
  // In-flight pose *requests* (a range request covers many indices but counts
  // once). Pose gates on this, not `poseInFlight.size`, so one batched pose
  // request occupies a single slot and never starves an urgent YOLO fetch.
  let poseRequests = 0;
  // Bumped on every scope change (clip / model). In-flight fetches whose
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

  void loadInitialListing();
  void loadModels();

  onDestroy(() => {
    teardownClip();
  });

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
  // is filled in proactively by the decoupled pose pass (see fetchPoseRangeInto),
  // never rebuffering on toggle.
  function currentScopeKey(): string {
    return `${selectedPath}|${selectedModel}`;
  }

  // --- listing + models -----------------------------------------------------

  async function loadListing(path: string): Promise<void> {
    listingLoading = true;
    listingError = null;
    try {
      listing = await fetchTuneFiles(path);
      // Remember the dir we actually landed on (server-resolved path) so reopening
      // the tab resumes here. "" = root list, which clears the saved pref.
      saveTuneLastDir(listing.path);
    } catch (err) {
      listingError = errMsg(err);
    } finally {
      listingLoading = false;
    }
  }

  // Mount: resume the browser at the last viewed directory. If that path is gone
  // (deleted, moved, or now outside the roots), the request 400s — clear the
  // stale pref and fall back to the root list so the browser is never stuck.
  async function loadInitialListing(): Promise<void> {
    const remembered = loadTuneLastDir();
    if (!remembered) {
      void loadListing("");
      return;
    }
    listingLoading = true;
    listingError = null;
    try {
      listing = await fetchTuneFiles(remembered);
      saveTuneLastDir(listing.path);
      listingLoading = false;
    } catch {
      saveTuneLastDir("");
      listingLoading = false;
      void loadListing("");
    }
  }

  async function loadModels(): Promise<void> {
    try {
      const data = await fetchTuneModels();
      models = data.models;
      coremlBatch = data.coreml_batch ?? {};
      selectedModel = data.default || data.models[0] || "";
    } catch {
      // Non-fatal: the picker just stays empty; detection still uses the default.
    }
  }

  // --- clip selection / lifecycle ------------------------------------------

  async function selectVideo(path: string, name: string): Promise<void> {
    if (path === selectedPath && meta) {
      return;
    }
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
      metaError = errMsg(err);
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
    clearBuffers();
    // Setting selectedPath renders the <video> with the new src; onLoadedMetadata
    // then seeks to frame 0, registers rVFC, and starts the filler.
    selectedPath = path;
  }

  // Abort + drop every in-flight/queued YOLO and pose fetch and the buffered
  // results, resetting all derived counts. The single place buffer state is
  // cleared so a clip/model change can never leak stale entries or counters.
  function clearBuffers(): void {
    for (const controller of inFlight.values()) {
      controller.abort();
    }
    inFlight.clear();
    detectRequests = 0;
    for (const controller of poseInFlight.values()) {
      controller.abort();
    }
    poseInFlight.clear();
    poseRequests = 0;
    buffer.clear();
    bufferedCount = 0;
    detectedCount = 0;
    posedCount = 0;
    clearTracks();
    clearScene();
  }

  // Drop the cached track pass (clip/model change invalidates track ids).
  function clearTracks(): void {
    if (trackAutoTimer !== null) {
      clearTimeout(trackAutoTimer);
      trackAutoTimer = null;
    }
    if (trackController) {
      trackController.abort();
      trackController = null;
    }
    trackByIndex.clear();
    trackStats = null;
    trackError = null;
    tracked = false;
    tracking = false;
    trackedStride = 1;
    trackDone = 0;
    trackExpected = 0;
    frameTracks = [];
  }

  // Drop the objects-in-scene result + cancel any in-flight fetch/debounce.
  function clearScene(): void {
    if (sceneTimer !== null) {
      clearTimeout(sceneTimer);
      sceneTimer = null;
    }
    sceneController?.abort();
    sceneController = null;
    sceneObjects = [];
    sceneLoading = false;
    sceneError = null;
    sceneIndex = null;
  }

  // Fetch the top-N all-class objects for `index` (stale-aborted). One inference
  // per frame; the debounced effect throttles scrubbing so we don't flood it.
  async function loadScene(index: number): Promise<void> {
    if (!selectedPath) return;
    const path = selectedPath;
    const reqModel = selectedModel;
    sceneController?.abort();
    const controller = new AbortController();
    sceneController = controller;
    sceneLoading = true;
    sceneError = null;
    try {
      const res = await fetchTuneScene(
        path,
        index,
        reqModel,
        SCENE_TOP_N,
        controller.signal,
      );
      // Ignore a result that raced past a clip/model/frame change.
      if (
        controller.signal.aborted ||
        path !== selectedPath ||
        reqModel !== selectedModel ||
        index !== presentedIndex
      ) {
        return;
      }
      sceneObjects = res.objects;
      sceneIndex = res.index;
      sceneLoading = false;
    } catch (err) {
      if (controller.signal.aborted) return;
      sceneError = errMsg(err);
      sceneLoading = false;
    }
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
    clearBuffers();
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
    syncTracks();
  }

  // Mirror the presented frame's tracked boxes from the cache. The track pass
  // only sampled every `trackedStride` frames, so snap the playhead to the
  // nearest sampled index and hold those boxes (persistent IDs across scrub).
  function syncTracks(): void {
    if (!tracked || tracker === "off") {
      frameTracks = [];
      return;
    }
    const stride = Math.max(1, trackedStride);
    const snapped = Math.round(presentedIndex / stride) * stride;
    frameTracks = trackByIndex.get(snapped) ?? [];
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
    seekToIndex(intendedIndex + delta);
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

  // Dropdown label: ".../yolo11m.mlpackage" -> "yolo11m (CoreML ×16)" (batched)
  // or "yolo11m (CoreML)" when single-frame; ".pt" -> basename.
  function modelOptionLabel(model: string): string {
    const base = model.split("/").pop() ?? model;
    if (!base.endsWith(".mlpackage")) return base;
    const stem = base.slice(0, -".mlpackage".length);
    const batch = coremlBatch[model] ?? 1;
    return batch > 1 ? `${stem} (CoreML ×${batch})` : `${stem} (CoreML)`;
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
      coremlBatch = result.coreml_batch ?? coremlBatch;
      setModel(result.model);
    } catch (err) {
      exportError = errMsg(err, "CoreML export failed");
    } finally {
      exporting = false;
    }
  }

  function setOverlay(mode: OverlayMode): void {
    if (mode === overlayMode) {
      return;
    }
    overlayMode = mode;
    // No buffer reset and no new fetches needed: boxes are valid for this
    // (clip, model) scope and pose is precomputed proactively regardless of the
    // overlay. Toggling only changes what's *drawn*, so syncView re-reads the
    // current frame's already-buffered boxes/pose. (pump() is a harmless no-op
    // here unless there's still outstanding fill work.)
    syncView();
    pump();
  }

  // Invalidate + rebuild the detection buffer after a model change.
  function resetScope(): void {
    scopeToken++;
    clearBuffers();
    syncView();
    pump();
  }

  // --- tracking (P1j) -------------------------------------------------------

  function setTracker(value: TuneTracker): void {
    if (value === tracker) return;
    tracker = value;
    // Track IDs are backend-specific; switching tracker invalidates the cache.
    clearTracks();
    syncTracks();
  }

  function isUltralyticsTracker(value: TuneTracker = tracker): boolean {
    return value === "bytetrack" || value === "botsort" || value === "botsort_reid";
  }

  function isBotsortTracker(value: TuneTracker = tracker): boolean {
    return value === "botsort" || value === "botsort_reid";
  }

  function ultralyticsParams(): TuneUltralyticsTrackerParams {
    return {
      conf: clamp(ultraConf, 0, 1),
      track_high_thresh: clamp(ultraTrackHigh, 0, 1),
      track_low_thresh: clamp(ultraTrackLow, 0, 1),
      new_track_thresh: clamp(ultraNewTrack, 0, 1),
      track_buffer: Math.max(0, Math.round(ultraTrackBuffer)),
      match_thresh: clamp(ultraMatch, 0, 1),
      proximity_thresh: isBotsortTracker() ? clamp(ultraProximity, 0, 1) : null,
      appearance_thresh: isBotsortTracker() ? clamp(ultraAppearance, 0, 1) : null,
      with_reid: tracker === "botsort_reid",
    };
  }

  function invalidateButtonDrivenTrack(): void {
    if (!isUltralyticsTracker()) return;
    clearTracks();
    syncTracks();
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

  // Run a stateful track pass over the whole clip (server caps the range) and
  // fill the per-frame track-id cache incrementally as the forward pass streams,
  // so the timeline lights up and a progress bar advances live. Tracking is
  // sequential + stateful, so this is always a single 0→end sweep — never
  // cursor-first like detect/pose. The `ours` backend auto-runs; the Ultralytics
  // backends stay button-driven (slower, hold the inference lock for the whole
  // pass). Switching clip/model/tracker aborts any in-flight pass.
  async function runTrackRange(): Promise<void> {
    if (!selectedPath || !meta || totalFrames <= 0) return;
    if (tracker === "off") return;
    // Cancel any pass already in flight before starting a fresh one.
    if (trackController) trackController.abort();
    const controller = new AbortController();
    trackController = controller;
    const seq = selectSeq;
    const stride = Math.max(1, trackSampleEvery);

    tracking = true;
    trackError = null;
    trackByIndex.clear();
    trackedStride = stride;
    tracked = false;
    trackDone = 0;
    // Approx number of sampled frames in the (capped) pass, for the progress bar.
    trackExpected = Math.max(1, Math.ceil(totalFrames / stride));
    frameTracks = [];

    try {
      await streamTuneTrackRange(
        selectedPath,
        0,
        totalFrames,
        selectedModel,
        tracker,
        {
          sampleEvery: trackSampleEvery,
          iouThreshold: trackIou,
          maxAgeFrames: trackMaxAge,
          centerDistGate: trackCenterGate,
          ultralytics: ultralyticsParams(),
        },
        {
          onFrames: (frames) => {
            if (seq !== selectSeq || controller.signal.aborted) return;
            for (const frame of frames) {
              trackByIndex.set(frame.index, frame.detections);
            }
            trackDone += frames.length;
            // Light up the playhead as the sweep passes it; first batch flips
            // `tracked` so the overlay switches from per-frame to tracked boxes.
            if (!tracked) tracked = true;
            syncTracks();
          },
          onDone: (done) => {
            if (seq !== selectSeq || controller.signal.aborted) return;
            trackedStride = done.stats.sample_every || stride;
            trackStats = done.stats;
            trackDone = done.stats.n_sampled_frames;
            trackExpected = done.stats.n_sampled_frames;
            tracked = true;
            syncTracks();
          },
          onError: (message) => {
            if (seq !== selectSeq || controller.signal.aborted) return;
            trackError = message || "Track pass failed";
          },
        },
        controller.signal,
      );
    } catch (err) {
      if (seq !== selectSeq || controller.signal.aborted) return;
      trackError = errMsg(err, "Track pass failed");
    } finally {
      if (trackController === controller) {
        trackController = null;
        if (seq === selectSeq) tracking = false;
      }
    }
  }

  // The Track button doubles as Cancel while a pass streams.
  function cancelTrackRange(): void {
    if (trackController) {
      trackController.abort();
      trackController = null;
    }
    tracking = false;
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

  // Anchor the urgency test tracks: where the user is looking (or, while playing,
  // a touch ahead of the playhead). YOLO work within URGENT_WINDOW of it must
  // never wait behind a pose pass.
  function anchorIndex(): number {
    const total = Math.max(1, totalFrames);
    return playing
      ? clamp(presentedIndex + leadFrames(), 0, total - 1)
      : clamp(intendedIndex, 0, total - 1);
  }

  function isUrgentYolo(idx: number): boolean {
    return Math.abs(idx - anchorIndex()) <= URGENT_WINDOW;
  }

  function hasEntry(idx: number): boolean {
    const entry = buffer.get(idx);
    return !!entry && entry.scopeKey === currentScopeKey();
  }

  // Next frame still missing its YOLO detections, in priority order.
  function nextNeeded(): number | null {
    for (const idx of candidateOrder()) {
      if (!inFlight.has(idx) && !hasEntry(idx)) {
        return idx;
      }
    }
    return null;
  }

  // Next DETECTED frame (>=1 box) whose pose pass hasn't run yet, in priority
  // order. Frames with no boxes need no pose, so they're skipped (they never
  // appear in the pose lane). Mirrors candidateOrder so pose fills cursor-first.
  function nextPoseNeeded(): number | null {
    const key = currentScopeKey();
    for (const idx of candidateOrder()) {
      if (poseInFlight.has(idx)) {
        continue;
      }
      const entry = buffer.get(idx);
      if (
        entry &&
        entry.scopeKey === key &&
        entry.detections.length > 0 &&
        !entry.posed
      ) {
        return idx;
      }
    }
    return null;
  }

  // Schedule a single fetch into one free slot with strict priority:
  //   urgent YOLO  >  pose (spare capacity, capped)  >  backfill YOLO.
  // Returns false when there's nothing left to do.
  function scheduleOne(): boolean {
    const y = nextNeeded();
    if (y !== null && isUrgentYolo(y)) {
      void fetchInto(y);
      return true;
    }
    if (poseAvailable && poseRequests < MAX_POSE_INFLIGHT) {
      const p = nextPoseNeeded();
      if (p !== null) {
        void fetchPoseRangeInto(p);
        return true;
      }
    }
    if (y !== null) {
      // Non-urgent backfill: pull a contiguous window in one batched request so
      // the server runs a single multi-frame forward (urgent frames above still
      // go single for latency).
      void fetchRangeInto(y);
      return true;
    }
    return false;
  }

  function pump(): void {
    if (!selectedPath || !meta) {
      return;
    }
    while (detectRequests + poseRequests < MAX_INFLIGHT) {
      if (!scheduleOne()) {
        break;
      }
    }
  }

  // The contiguous run of still-needed indices starting at `start` (which
  // nextNeeded already proved missing + not in flight), capped at RANGE_BATCH and
  // the clip end. Stops at the first index already buffered or in flight so a
  // batch never re-decodes/re-fetches work another slot owns.
  function rangeRunFrom(start: number): number[] {
    const total = totalFrames;
    const run: number[] = [];
    for (let idx = start; idx < total && run.length < RANGE_BATCH; idx++) {
      if (inFlight.has(idx) || hasEntry(idx)) {
        break;
      }
      run.push(idx);
    }
    return run;
  }

  // YOLO detect pass — boxes only (always pose=0). Pose is filled separately by
  // fetchPoseRangeInto so flipping the overlay never rebuffers boxes.
  async function fetchInto(idx: number): Promise<void> {
    if (!selectedPath) return;
    const path = selectedPath;
    const model = selectedModel;
    const scopeKey = currentScopeKey();
    const token = scopeToken;
    const controller = new AbortController();
    inFlight.set(idx, controller);
    detectRequests++;
    const started = performance.now();
    try {
      const res = await fetchTuneDetect(path, idx, model, false, controller.signal);
      if (token !== scopeToken) {
        return; // scope changed while in flight -> drop
      }
      detectMsEma = detectMsEma * 0.7 + (performance.now() - started) * 0.3;
      if (threshold < res.detection_floor) {
        threshold = res.detection_floor;
      }
      floor = res.detection_floor;
      // poseAvailable is sticky-down: the backend disables pose app-wide on a
      // failure, so never flip it back to true here.
      if (!res.pose_available) {
        poseAvailable = false;
      }
      // nextNeeded only yields un-buffered indices, so this is always a fresh
      // entry in this scope -> a plain detected-count increment (no decrement).
      buffer.set(res.index, {
        scopeKey,
        detections: res.detections,
        pose: [],
        posed: false,
      });
      bufferedCount = buffer.size;
      if (res.detections.length > 0) {
        detectedCount++;
      }
      if (res.index === presentedIndex) {
        syncView();
      }
      updateStrip();
    } catch {
      // Aborted or a transient single-frame error: skip it quietly rather than
      // surfacing per-frame noise. The frame can be retried on the next pass.
    } finally {
      inFlight.delete(idx);
      detectRequests = Math.max(0, detectRequests - 1);
      if (token === scopeToken) {
        pump();
      }
    }
  }

  // Batched backfill: detect a contiguous window in one request, buffering each
  // returned frame exactly as fetchInto does. The server caps the count and may
  // return fewer frames (smaller cap / EOF); any unreturned indices are simply
  // re-picked on the next pump. Falls back to a single fetch for a 1-frame run.
  async function fetchRangeInto(start: number): Promise<void> {
    if (!selectedPath) return;
    const run = rangeRunFrom(start);
    if (run.length === 0) return;
    if (run.length === 1) {
      void fetchInto(start);
      return;
    }
    const path = selectedPath;
    const model = selectedModel;
    const scopeKey = currentScopeKey();
    const token = scopeToken;
    const controller = new AbortController();
    for (const idx of run) {
      inFlight.set(idx, controller);
    }
    detectRequests++;
    const started = performance.now();
    try {
      const res = await fetchTuneDetectRange(
        path,
        run[0],
        run.length,
        model,
        controller.signal,
      );
      if (token !== scopeToken) {
        return; // scope changed while in flight -> drop
      }
      // Amortize the batch latency per frame so the play-ahead EMA still tracks
      // per-frame inference cost (it leads the playhead by frames, not requests).
      const elapsed = performance.now() - started;
      const per = res.frames.length > 0 ? elapsed / res.frames.length : elapsed;
      detectMsEma = detectMsEma * 0.7 + per * 0.3;
      for (const frame of res.frames) {
        if (threshold < frame.detection_floor) {
          threshold = frame.detection_floor;
        }
        floor = frame.detection_floor;
        if (!frame.pose_available) {
          poseAvailable = false;
        }
        if (hasEntry(frame.index)) {
          continue; // already buffered in this scope -> don't double-count
        }
        buffer.set(frame.index, {
          scopeKey,
          detections: frame.detections,
          pose: [],
          posed: false,
        });
        if (frame.detections.length > 0) {
          detectedCount++;
        }
        if (frame.index === presentedIndex) {
          syncView();
        }
      }
      bufferedCount = buffer.size;
      updateStrip();
    } catch {
      // Aborted or transient: the run's indices get retried on the next pass.
    } finally {
      for (const idx of run) {
        if (inFlight.get(idx) === controller) {
          inFlight.delete(idx);
        }
      }
      detectRequests = Math.max(0, detectRequests - 1);
      if (token === scopeToken) {
        pump();
      }
    }
  }

  // Decoupled pose pass: run pose for a contiguous run of detected frames'
  // already-buffered boxes (no YOLO re-run) in ONE batched request, so the server
  // runs a single multi-frame forward. Mutates each buffer entry in place and
  // marks it posed (even with no keypoints) so it isn't retried forever.
  async function fetchPoseRangeInto(start: number): Promise<void> {
    if (!selectedPath) return;
    const run = poseRunFrom(start);
    if (run.length === 0) return;
    const path = selectedPath;
    const scopeKey = currentScopeKey();
    const token = scopeToken;
    const frames = run.map((idx) => ({
      index: idx,
      boxes: (buffer.get(idx)?.detections ?? []).map((d) => [d.x1, d.y1, d.x2, d.y2]),
    }));
    const controller = new AbortController();
    for (const idx of run) {
      poseInFlight.set(idx, controller);
    }
    poseRequests++;
    try {
      const res = await fetchTunePoseRange(path, frames, controller.signal);
      if (token !== scopeToken) {
        return; // scope changed while in flight -> drop
      }
      for (const frame of res.frames) {
        if (!frame.pose_available) {
          poseAvailable = false;
        }
        const cur = buffer.get(frame.index);
        if (cur && cur.scopeKey === scopeKey && !cur.posed) {
          cur.pose = frame.pose;
          cur.posed = true;
          posedCount++;
          if (frame.index === presentedIndex) {
            syncView();
          }
        }
      }
      updateStrip();
    } catch {
      // Aborted or a transient pose error: leave the run un-posed; it'll be
      // retried on the next pass (unless pose got disabled above).
    } finally {
      for (const idx of run) {
        if (poseInFlight.get(idx) === controller) {
          poseInFlight.delete(idx);
        }
      }
      poseRequests = Math.max(0, poseRequests - 1);
      if (token === scopeToken) {
        pump();
      }
    }
  }

  // The contiguous run of pose-needed indices starting at `start` (which
  // nextPoseNeeded already proved buffered + detected + un-posed + not in flight),
  // capped at POSE_RANGE_BATCH. Stops at the first index that isn't pose-needed
  // (no entry, wrong scope, no detections, already posed, or in flight) so a batch
  // never re-poses work another slot owns.
  function poseRunFrom(start: number): number[] {
    const key = currentScopeKey();
    const total = totalFrames;
    const run: number[] = [];
    for (let idx = start; idx < total && run.length < POSE_RANGE_BATCH; idx++) {
      const entry = buffer.get(idx);
      if (
        poseInFlight.has(idx) ||
        !entry ||
        entry.scopeKey !== key ||
        entry.detections.length === 0 ||
        entry.posed
      ) {
        break;
      }
      run.push(idx);
    }
    return run;
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
    void frameTracks;
    void trackingActive;
    void sceneBoxes;
    void sceneObjects;
    void sceneIndex;
    void sceneOpen;
    drawOverlay();
  });

  // Redraw the two-lane "analyzed/pose" strip when the buffer grows, the
  // playhead moves, or anything affecting lane shading changes. The YOLO lane's
  // bright fill reacts live to `threshold`; the pose lane reacts to `posedCount`.
  $effect(() => {
    void bufferedCount;
    void detectedCount;
    void posedCount;
    void presentedIndex;
    void totalFrames;
    void threshold;
    void poseAvailable;
    void trackDone;
    void trackingActive;
    void tracker;
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

  // Fetch objects-in-scene for the presented frame while the panel is open
  // (debounced so scrubbing doesn't flood inference). Clip/model change clears
  // the cache via clearScene(); this just refreshes on the settled frame.
  $effect(() => {
    void sceneOpen;
    void presentedIndex;
    void selectedPath;
    void selectedModel;
    if (sceneTimer !== null) {
      clearTimeout(sceneTimer);
      sceneTimer = null;
    }
    if (!sceneOpen || !selectedPath) return;
    const idx = presentedIndex;
    sceneTimer = setTimeout(() => {
      sceneTimer = null;
      void loadScene(idx);
    }, 180);
  });

  // Auto-run the `ours` track pass when the clip/model/tracker or a knob settles.
  // Tracking with `ours` is fast + batched (~8–12 s), so it can feel as live as
  // detect/pose. The Ultralytics backends are slower + hold the inference lock for
  // the whole pass, so they stay button-driven (no auto-run). Debounced so typing
  // a knob value doesn't kick off a pass per keystroke.
  $effect(() => {
    void selectedPath;
    void selectedModel;
    void tracker;
    void trackSampleEvery;
    void trackIou;
    void trackMaxAge;
    void trackCenterGate;
    void totalFrames;
    if (trackAutoTimer !== null) {
      clearTimeout(trackAutoTimer);
      trackAutoTimer = null;
    }
    if (tracker !== "ours" || !selectedPath || totalFrames <= 0) return;
    trackAutoTimer = setTimeout(() => {
      trackAutoTimer = null;
      void runTrackRange();
    }, 350);
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
    const fontPx = boxLabelFontPx(Math.max(w, h));

    if (overlayMode === "boxes" || overlayMode === "both") {
      ctx.lineWidth = lineW;
      if (trackingActive) {
        // Persistent track-id boxes: color hashed from the id, id label drawn at
        // the corner. Below-threshold boxes are drawn dashed so the conf gate is
        // still legible.
        for (const det of frameTracks) {
          const keep = det.confidence >= threshold;
          const color = trackColor(det.track_id);
          ctx.strokeStyle = color;
          ctx.setLineDash(keep ? [] : [Math.max(4, lineW * 2), lineW * 2]);
          ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
          ctx.setLineDash([]);
          const label = formatTrackLabel(det.track_id, det.confidence);
          drawCanvasBoxLabel(ctx, label, det.x1, det.y1, color, fontPx);
        }
      } else {
        for (const det of frameDetections) {
          const keep = det.confidence >= threshold;
          // Alias-sourced boxes (a dog read as sheep/zebra/cow/... and accepted as a
          // dog) keep their real class name; kept aliases draw in the shared teal so
          // the reviewer can see the box came from an alias read, not a true "dog" box.
          const color = classBoxColor(det.class_name, keep);
          ctx.strokeStyle = color;
          ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
          const label = formatDetLabel(det.class_name, det.confidence);
          drawCanvasBoxLabel(ctx, label, det.x1, det.y1, color, fontPx);
        }
      }
    }

    if (overlayMode === "pose" || overlayMode === "both") {
      drawPose(ctx, w, framePose);
    }

    // Independent diagnostic layer: the "objects in scene" boxes (any class, no
    // dog filter). Drawn dashed amber so they never read as a real dog-detection
    // box. Only painted when the scene list matches the presented frame, since the
    // scene list is fetched per-settled-frame and not buffered across playback.
    if (
      sceneOpen &&
      sceneBoxes &&
      sceneObjects.length > 0 &&
      sceneIndex === presentedIndex
    ) {
      ctx.lineWidth = lineW;
      ctx.setLineDash([Math.max(4, lineW * 2), lineW * 2]);
      for (const obj of sceneObjects) {
        const isDog = obj.class_name.toLowerCase() === "dog";
        const color = isDog ? BOX_DOG : BOX_SIBLING;
        ctx.strokeStyle = color;
        ctx.strokeRect(obj.x1, obj.y1, obj.x2 - obj.x1, obj.y2 - obj.y1);
        const label = formatDetLabel(obj.class_name, obj.confidence);
        drawCanvasBoxLabel(ctx, label, obj.x1, obj.y1, color, fontPx);
      }
      ctx.setLineDash([]);
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

  // The two-lane timeline strip under the scrub bar:
  //   Lane 1 (YOLO):  faint track = frame analyzed (swept); bright green = >=1
  //     detection ABOVE the current confidence threshold, opacity ~ top kept
  //     confidence. Reacts live to the slider.
  //   Lane 2 (pose):  faint track = frame has >=1 detection (a dog present);
  //     bright cyan = pose pass ran AND produced keypoints, opacity ~ mean
  //     keypoint confidence.
  // It's a separate canvas inset (in CSS) by half the range thumb width and uses
  // the SAME value->position mapping as the native range (idx / (total - 1)
  // across the canvas width), so a column sits exactly under the thumb for that
  // frame at every position including the ends. Multiple frames can collapse to
  // one pixel column, so values are aggregated per-column by max and drawn
  // background -> faint -> bright (bright last) so a faint frame can never paint
  // over a brighter neighbour.
  // Opaque blend between two RGB triples (avoids translucent-overlap moiré on the strip).
  function lerpColor(
    a: readonly [number, number, number],
    b: readonly [number, number, number],
    t: number,
  ): string {
    const u = clamp(t, 0, 1);
    const r = Math.round(a[0] + (b[0] - a[0]) * u);
    const g = Math.round(a[1] + (b[1] - a[1]) * u);
    const bl = Math.round(a[2] + (b[2] - a[2]) * u);
    return `rgb(${r}, ${g}, ${bl})`;
  }

  const LANE1_TRACK: readonly [number, number, number] = [51, 80, 107]; // #33506b
  const LANE1_BRIGHT: readonly [number, number, number] = [40, 209, 124]; // #28d17c
  const LANE2_TRACK: readonly [number, number, number] = [44, 85, 102]; // #2c5566
  const LANE2_BRIGHT: readonly [number, number, number] = [90, 209, 255]; // #5ad1ff
  const LANE3_TRACK: readonly [number, number, number] = [58, 53, 96]; // #3a3560
  const LANE3_BRIGHT: readonly [number, number, number] = [179, 136, 255]; // #b388ff

  function updateStrip(): void {
    const c = stripEl;
    if (!c || totalFrames <= 0) {
      return;
    }
    const w = c.clientWidth || 300;
    const laneH = 8;
    const gap = 2;
    const h = laneH * 3 + gap * 2;
    if (c.width !== w) c.width = w;
    if (c.height !== h) c.height = h;
    const ctx = c.getContext("2d");
    if (!ctx) {
      return;
    }
    const lane2Y = laneH + gap;
    const lane3Y = (laneH + gap) * 2;
    ctx.clearRect(0, 0, w, h);
    // Lane backgrounds.
    ctx.fillStyle = "#1b2532";
    ctx.fillRect(0, 0, w, laneH);
    ctx.fillRect(0, lane2Y, w, laneH);
    ctx.fillRect(0, lane3Y, w, laneH);

    const span = Math.max(1, totalFrames - 1);
    const colW = Math.max(1, Math.ceil(w / totalFrames));
    const maxX = Math.max(0, w - colW);
    // Per-column aggregates (max). -1 marks "no bright signal" for the conf lanes.
    const analyzed = new Uint8Array(w);
    const detected = new Uint8Array(w);
    const tracked = new Uint8Array(w);
    const keptConf = new Float32Array(w).fill(-1);
    const poseConf = new Float32Array(w).fill(-1);
    const trackConf = new Float32Array(w).fill(-1);
    const key = currentScopeKey();
    for (const [idx, entry] of buffer) {
      if (entry.scopeKey !== key) continue;
      const x = clamp(Math.floor((idx / span) * w), 0, maxX);
      analyzed[x] = 1;
      let top = -1;
      for (const d of entry.detections) {
        if (d.confidence >= threshold && d.confidence > top) top = d.confidence;
      }
      if (top > keptConf[x]) keptConf[x] = top;
      if (entry.detections.length > 0) {
        detected[x] = 1;
        if (entry.posed && entry.pose.length > 0) {
          let sum = 0;
          let n = 0;
          for (const p of entry.pose) {
            for (const kp of p.keypoints) {
              sum += kp.confidence;
              n++;
            }
          }
          const mc = n > 0 ? sum / n : 0;
          if (mc > poseConf[x]) poseConf[x] = mc;
        }
      }
    }

    // Lane 3 (track): the forward 0→end sweep fills `trackByIndex` in frame order,
    // so painting each sampled frame's full stride span makes the lane fill
    // left→right as the pass streams — the lane IS the progress bar. Bright fill
    // (violet) where a track sits above the conf threshold, shaded by top track
    // confidence (matches lane 1's threshold gating). Cleared on tracker=off / reset
    // because `trackByIndex` is emptied there.
    if (trackingActive || tracking) {
      const tStride = Math.max(1, trackedStride);
      for (const [idx, dets] of trackByIndex) {
        const x0 = clamp(Math.floor((idx / span) * w), 0, maxX);
        const x1 = clamp(Math.floor(((idx + tStride - 1) / span) * w), 0, maxX);
        let top = -1;
        for (const d of dets) {
          if (d.confidence >= threshold && d.confidence > top) top = d.confidence;
        }
        for (let x = x0; x <= x1; x++) {
          tracked[x] = 1;
          if (top > trackConf[x]) trackConf[x] = top;
        }
      }
    }

    // Faint tracks first.
    ctx.fillStyle = "#33506b";
    for (let x = 0; x < w; x++) if (analyzed[x]) ctx.fillRect(x, 0, colW, laneH);
    ctx.fillStyle = "#2c5566";
    for (let x = 0; x < w; x++) if (detected[x]) ctx.fillRect(x, lane2Y, colW, laneH);
    ctx.fillStyle = "#3a3560";
    for (let x = 0; x < w; x++) if (tracked[x]) ctx.fillRect(x, lane3Y, colW, laneH);

    // Bright fills on top, opaque color (not alpha) encoding confidence so
    // overlapping 1px columns overwrite instead of double-compositing (no moiré).
    for (let x = 0; x < w; x++) {
      if (keptConf[x] >= 0) {
        ctx.fillStyle = lerpColor(LANE1_TRACK, LANE1_BRIGHT, 0.4 + 0.6 * keptConf[x]);
        ctx.fillRect(x, 0, colW, laneH);
      }
    }
    for (let x = 0; x < w; x++) {
      if (poseConf[x] >= 0) {
        ctx.fillStyle = lerpColor(LANE2_TRACK, LANE2_BRIGHT, 0.4 + 0.6 * poseConf[x]);
        ctx.fillRect(x, lane2Y, colW, laneH);
      }
    }
    for (let x = 0; x < w; x++) {
      if (trackConf[x] >= 0) {
        ctx.fillStyle = lerpColor(LANE3_TRACK, LANE3_BRIGHT, 0.4 + 0.6 * trackConf[x]);
        ctx.fillRect(x, lane3Y, colW, laneH);
      }
    }
  }

  // --- zoom crops -----------------------------------------------------------

  const ZOOM_TARGET = 220; // longest-side px of a zoom crop

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
    ctx.strokeStyle = card.kept ? BOX_DOG : BOX_WEAK;
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
    if (isTypingTarget(event.target, { allowRange: true })) {
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
  <TuneClipBrowser
    {listing}
    loading={listingLoading}
    error={listingError}
    {selectedPath}
    onloadpath={loadListing}
    onselectvideo={selectVideo}
  />

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
            loop
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
          {#if totalFrames}<span class="hud-yolo">dogs {detectedCount}</span>{/if}
          {#if totalFrames}
            {#if poseAvailable}
              <span class="hud-pose">pose {posedCount}/{detectedCount}</span>
            {:else}
              <span class="muted">pose n/a</span>
            {/if}
          {/if}
        </div>

        <TuneTimelineStrip
          bind:canvas={stripEl}
          {displayIndex}
          {totalFrames}
          onpointerdown={onSeekPointerDown}
          oninput={onSeekInput}
          oncommit={onSeekCommit}
          onresize={updateStrip}
        />

        <div class="controls">
          <Transport
            playing={playing}
            frame={displayIndex}
            total={totalFrames}
            fps={fps}
            skipN={SKIP_N}
            showReadout={false}
            onTogglePlay={togglePlay}
            onStep={step}
          />

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
                : "Export this model to a GPU-safe, batched CoreML model (runs on the GPU, ~3x faster on recorded clips). Takes ~1 min."}
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

          <button
            type="button"
            class="zoom-toggle"
            class:active={sceneOpen}
            onclick={() => (sceneOpen = !sceneOpen)}
            title="List the top objects the detector sees on this frame, including non-dog classes (cat/person/…) — helps explain frames with no dog box"
          >
            🔎 objects
          </button>
        </div>

        <div class="track-row">
          <label class="tracker">
            <span class="mono muted small">tracker</span>
            <select
              value={tracker}
              onchange={(e) =>
                setTracker((e.currentTarget as HTMLSelectElement).value as TuneTracker)}
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
              <label title="Sample every N source frames before tracking. Affects both Ours and Ultralytics; changing it requires a new track pass.">
                stride
                <input
                  type="number"
                  min="1"
                  max="60"
                  step="1"
                  bind:value={trackSampleEvery}
                  oninput={invalidateButtonDrivenTrack}
                />
              </label>
              {#if tracker === "ours"}
                <label title="Minimum box overlap to associate a detection with an existing Ours track. Lower joins more jumps; higher splits more tracks.">
                  iou
                  <input type="number" min="0" max="1" step="0.05" bind:value={trackIou} />
                </label>
                <label title="Sampled frames an unmatched Ours track survives before it dies. Higher bridges longer gaps.">
                  max-age
                  <input type="number" min="0" max="300" step="1" bind:value={trackMaxAge} />
                </label>
                <label title="Center-distance OR-gate in box diagonals for Ours. 0 means IoU-only; higher reconnects larger jumps.">
                  gate
                  <input type="number" min="0" max="20" step="0.1" bind:value={trackCenterGate} />
                </label>
              {:else if isUltralyticsTracker() && !isMlpackage}
                <label title="YOLO confidence floor passed to Ultralytics model.track(). Lower can expose more detections to the tracker; requires Re-track range.">
                  det-conf
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    bind:value={ultraConf}
                    oninput={invalidateButtonDrivenTrack}
                  />
                </label>
                <label title="Ultralytics track_high_thresh: high-confidence detections used for the primary association pass.">
                  track-high
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    bind:value={ultraTrackHigh}
                    oninput={invalidateButtonDrivenTrack}
                  />
                </label>
                <label title="Ultralytics track_low_thresh: lower-confidence detections still eligible for secondary association.">
                  track-low
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    bind:value={ultraTrackLow}
                    oninput={invalidateButtonDrivenTrack}
                  />
                </label>
                <label title="Ultralytics new_track_thresh: minimum confidence required to start a new track ID.">
                  new-track
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    bind:value={ultraNewTrack}
                    oninput={invalidateButtonDrivenTrack}
                  />
                </label>
                <label title="Ultralytics track_buffer: sampled frames an unmatched track stays alive before removal.">
                  buffer
                  <input
                    type="number"
                    min="0"
                    max="10000"
                    step="1"
                    bind:value={ultraTrackBuffer}
                    oninput={invalidateButtonDrivenTrack}
                  />
                </label>
                <label title="Ultralytics match_thresh: association matching threshold. Higher is stricter; lower can bridge more uncertain matches.">
                  match
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    bind:value={ultraMatch}
                    oninput={invalidateButtonDrivenTrack}
                  />
                </label>
                {#if isBotsortTracker()}
                  <label title="BoT-SORT proximity_thresh: spatial proximity gate before appearance matching. Lower is more permissive.">
                    prox
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.01"
                      bind:value={ultraProximity}
                      oninput={invalidateButtonDrivenTrack}
                    />
                  </label>
                  <label title="BoT-SORT appearance_thresh: appearance/ReID similarity threshold. Higher requires stronger visual match.">
                    appear
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.01"
                      bind:value={ultraAppearance}
                      oninput={invalidateButtonDrivenTrack}
                    />
                  </label>
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
              onclick={tracking ? cancelTrackRange : runTrackRange}
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
            <div
              class="track-stats mono small"
              title={trackStatsTitle(trackStats)}
            >
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

        {#if sceneOpen}
          <div class="scene-panel">
            <div class="scene-head">
              <span class="eyebrow">OBJECTS IN SCENE</span>
              <div class="scene-head-right">
                <span class="mono muted small">
                  {#if sceneLoading}…{:else}frame {sceneIndex ?? presentedIndex}{/if}
                </span>
                <button
                  type="button"
                  class="zoom-toggle"
                  class:active={sceneBoxes}
                  onclick={() => (sceneBoxes = !sceneBoxes)}
                  title="Overlay these objects' boxes on the frame (dashed amber) so you can see where each class — e.g. a 'sheep' read — actually sits. Best paused or stepping; boxes draw on the frame the list was fetched on."
                >
                  ▢ boxes
                </button>
              </div>
            </div>
            {#if sceneError}
              <span class="export-error mono small" role="alert">{sceneError}</span>
            {:else if sceneObjects.length > 0}
              <ul class="scene-list mono small">
                {#each sceneObjects as obj, i (obj.class_name + ":" + i)}
                  <li class:non-dog={obj.class_name.toLowerCase() !== "dog"}>
                    <span class="scene-cls">{obj.class_name}</span>
                    <span class="scene-conf">{obj.confidence.toFixed(2)}</span>
                  </li>
                {/each}
              </ul>
            {:else if !sceneLoading}
              <div class="scene-empty muted small">
                Nothing detected on this frame (above the detector floor).
              </div>
            {/if}
          </div>
        {/if}

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

  .eyebrow {
    font-size: 0.6rem;
    letter-spacing: 0.28em;
    color: var(--amber, #f0b35a);
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

  .hud .hud-yolo {
    color: #28d17c;
  }

  .hud .hud-pose {
    color: #5ad1ff;
  }

  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 1rem;
  }

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

  .track-knobs label {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    color: var(--muted, #8a97a8);
  }

  .track-knobs input {
    width: 4.2rem;
    background: var(--bg-1, #141a24);
    border: 1px solid var(--line-strong, #324056);
    color: var(--text, #d8e0ec);
    border-radius: 5px;
    padding: 0.2rem 0.3rem;
    font-family: ui-monospace, monospace;
    font-size: 0.74rem;
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

  .scene-panel {
    margin-top: 0.6rem;
    padding-top: 0.6rem;
    border-top: 1px solid var(--line, #243042);
  }

  .scene-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.6rem;
    margin-bottom: 0.4rem;
  }

  .scene-head-right {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .scene-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem 0.5rem;
  }

  .scene-list li {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--bg-1, #141a24);
    border: 1px solid var(--line-strong, #324056);
    border-radius: 6px;
    padding: 0.15rem 0.45rem;
  }

  .scene-list li.non-dog {
    border-color: var(--amber, #f0b35a);
  }

  .scene-cls {
    color: var(--text, #d8e0ec);
  }

  .scene-conf {
    color: var(--muted, #8a97a8);
  }

  .scene-empty {
    padding: 0.2rem 0;
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

  .pose-hint code,
  .empty code {
    background: var(--bg-2, #1b2330);
    padding: 0.05rem 0.3rem;
    border-radius: 4px;
  }
</style>
