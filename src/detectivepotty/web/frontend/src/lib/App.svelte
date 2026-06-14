<script lang="ts">
  import type { EventSummary } from "./types";
  import { onMount } from "svelte";
  import { get } from "svelte/store";
  import { fetchDogs } from "./api";
  import { autoAdvance, liveNotifications, liveSound } from "./prefs";
  import {
    liveConnected,
    liveEvents,
    liveNewCount,
    onLiveEvent,
    startLive,
    stopLive,
  } from "./live";
  import { playChime, showEventNotification } from "./notify";
  import LiveFeed from "./LiveFeed.svelte";
  import TuneDetect from "./TuneDetect.svelte";
  import LabelReview from "./LabelReview.svelte";
  import HelpOverlay from "./HelpOverlay.svelte";
  import ReviewConsole from "./review/ReviewConsole.svelte";
  import type { ReviewHeaderState, ReviewOpenRequest } from "./review/types";
  import { navigate, route, routeToView } from "./router";
  import { errMsg } from "./errors";
  import { isTypingTarget } from "./keys";

  const STATUS_FILTERS: Array<[string, string]> = [
    ["", "All"],
    ["unlabeled", "Unlabeled"],
    ["labeled", "Labeled"],
    ["rejected", "Rejected"],
    ["uncertain", "Uncertain"],
  ];

  let dogs = $state<string[]>([]);
  let dogError = $state<string | null>(null);
  let cameraInput = $state<HTMLInputElement | null>(null);
  let helpOpen = $state(false);

  // The active view is derived from the URL so a refresh restores it (and deep
  // links work). `?event=<id>` on the review route restores the open event.
  let view = $derived(routeToView($route.path));
  let tuneMounted = $state(false);
  let labelMounted = $state(false);
  let toasts = $state<Array<{ id: string; summary: EventSummary }>>([]);
  let reviewHeader = $state<ReviewHeaderState | null>(null);
  let reviewOpenRequest = $state<ReviewOpenRequest | null>(null);
  let reviewOpenSeq = 0;
  let liveStarted = false;

  init();

  // Real-time alerts: a successful first load seeds the "known" baseline before
  // the stream opens, so initial events never trigger a notification/banner.
  onMount(() => {
    const off = onLiveEvent(handleLiveEvent);
    return () => {
      off();
      stopLive();
    };
  });

  $effect(() => {
    if (view === "tune") {
      tuneMounted = true;
    } else if (view === "label") {
      labelMounted = true;
    }
  });

  function handleLiveEvent(summary: EventSummary): void {
    pushToast(summary);
    if (get(liveNotifications)) {
      showEventNotification(summary);
    }
    if (get(liveSound)) {
      playChime();
    }
  }

  function pushToast(summary: EventSummary): void {
    const id = `${summary.event_id}:${Date.now()}`;
    toasts = [...toasts, { id, summary }].slice(-4);
    setTimeout(() => {
      toasts = toasts.filter((t) => t.id !== id);
    }, 6000);
  }

  function dismissToast(id: string): void {
    toasts = toasts.filter((t) => t.id !== id);
  }

  // Open a live/banner event in the Review console: refresh the list (which also
  // acknowledges the new arrivals and clears the banner), then select it.
  async function openLiveEvent(eventId: string): Promise<void> {
    if (!navigateView("/")) {
      return;
    }
    reviewOpenRequest = { seq: ++reviewOpenSeq, eventId };
  }

  async function init(): Promise<void> {
    try {
      dogs = await fetchDogs();
      dogError = null;
    } catch (err) {
      dogs = [];
      dogError = errMsg(err);
    }
  }

  function startLiveOnce(): void {
    if (liveStarted) {
      return;
    }
    liveStarted = true;
    startLive();
  }

  function confirmUnsavedReviewChanges(): boolean {
    return confirm("Continue without saving label changes?");
  }

  function navigateView(path: string): boolean {
    if (view === "review" && path === "/") {
      return true;
    }
    if (view === "review" && reviewHeader?.dirty && !confirmUnsavedReviewChanges()) {
      return false;
    }
    navigate(path);
    return true;
  }

  function focusCameraFilter(): void {
    cameraInput?.focus();
    cameraInput?.select();
  }

  function setReviewHeader(state: ReviewHeaderState): void {
    reviewHeader = state;
  }

  function openHelp(): void {
    helpOpen = true;
  }

  function closeHelp(): void {
    helpOpen = false;
  }

  function onKey(event: KeyboardEvent): void {
    if (event.defaultPrevented) {
      return;
    }
    if (helpOpen) {
      if (event.key === "Escape" || event.key === "?") {
        event.preventDefault();
        helpOpen = false;
      }
      return;
    }
    if (view === "review") {
      return;
    }
    if (event.metaKey || event.ctrlKey || event.altKey) {
      return;
    }
    if (isTypingTarget(event.target)) {
      if (event.key === "Escape") {
        (event.target as HTMLElement).blur();
      }
      return;
    }
    // The Tune page owns its own keyboard (Space, ←/→, Shift+←/→); don't let the
    // review keymap swallow those (e.g. Space toggling a non-existent video).
    if (view === "tune") {
      if (event.key === "Escape") {
        event.preventDefault();
        void navigateView("/");
      }
      return;
    }
    // The Label page owns its own keyboard (Space, ←/→, I/O, 1-4, Enter, S, j/k);
    // let its window handler act and only catch Escape here to return to Review.
    if (view === "label") {
      if (event.key === "Escape") {
        event.preventDefault();
        void navigateView("/");
      }
      return;
    }
    // The Live feed has its own minimal keymap: v / Esc return to Review.
    if (view === "live") {
      if (event.key === "v" || event.key === "Escape") {
        event.preventDefault();
        void navigateView("/");
      } else if (event.key === "?") {
        event.preventDefault();
        helpOpen = true;
      }
      return;
    }
  }
</script>

<svelte:window onkeydown={onKey} />

<div class="app">
  <header class="bar">
    <div class="brand">
      <span class="eyebrow mono">REVIEW CONSOLE</span>
      <h1>DetectivePotty</h1>
    </div>

    <div class="viewtabs" role="tablist" aria-label="View">
      <button
        type="button"
        role="tab"
        aria-selected={view === "review"}
        class:active={view === "review"}
        onclick={() => navigateView("/")}
      >
        Review
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={view === "live"}
        class:active={view === "live"}
        onclick={() => navigateView("/live")}
        title="Real-time feed of new events (v)"
      >
        Live
        <span class="live-dot" class:on={$liveConnected} aria-hidden="true"></span>
        {#if $liveNewCount > 0}<span class="live-badge">{$liveNewCount}</span>{/if}
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={view === "tune"}
        class:active={view === "tune"}
        onclick={() => navigateView("/tune")}
        title="Tune YOLO detection on a clip"
      >
        Tune
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={view === "label"}
        class:active={view === "label"}
        onclick={() => navigateView("/label")}
        title="Label harvested clips for training"
      >
        Label
      </button>
    </div>

    {#if view === "review" && reviewHeader}
    <div class="controls">
      <div class="segmented" role="tablist" aria-label="Filter by label status">
        {#each STATUS_FILTERS as [value, text] (value)}
          <button
            type="button"
            role="tab"
            aria-selected={reviewHeader.statusFilter === value}
            class:active={reviewHeader.statusFilter === value}
            onclick={() => reviewHeader?.applyFilter(value)}
          >
            {text}
          </button>
        {/each}
      </div>

      <div class="camera-field">
        <span class="slash mono">/</span>
        <input
          bind:this={cameraInput}
          type="search"
          placeholder="Filter camera"
          value={reviewHeader.cameraFilter}
          oninput={(event) =>
            reviewHeader?.setCameraFilter((event.currentTarget as HTMLInputElement).value)}
          onkeydown={(event) => {
            if (event.key === "Enter") {
              reviewHeader?.commitCameraFilter();
            }
          }}
        />
        {#if reviewHeader.cameraFilter}
          <button type="button" class="clear" onclick={() => reviewHeader?.clearCamera()} aria-label="Clear camera filter">×</button>
        {/if}
      </div>
    </div>
    {/if}

    <div class="bar-right">
      {#if view === "review" && reviewHeader}
      <div class="progress" title="{reviewHeader.labeledCount} of {reviewHeader.eventCount} loaded events labeled">
        <div class="progress-text mono">
          <strong>{reviewHeader.labeledCount}</strong>/{reviewHeader.eventCount}
          {#if reviewHeader.unfilteredTotal != null && reviewHeader.unfilteredTotal !== reviewHeader.eventCount}
            <span class="muted">· {reviewHeader.unfilteredTotal} on disk</span>
          {/if}
        </div>
        <div class="progress-track"><div class="progress-fill" style="width: {reviewHeader.progressPct}%"></div></div>
      </div>

      <button
        type="button"
        class="toggle"
        class:on={$autoAdvance}
        onclick={() => autoAdvance.update((v) => !v)}
        title="Auto-advance to next unlabeled after save (a)"
      >
        auto-advance
      </button>
      {/if}

      <button type="button" class="help-btn" onclick={() => (helpOpen = true)} aria-label="Keyboard shortcuts">
        <kbd>?</kbd>
      </button>
    </div>
  </header>

  <main class="live-main" hidden={view !== "live"}>
    <LiveFeed events={$liveEvents} connected={$liveConnected} onpick={openLiveEvent} />
  </main>

  <main class="tune-main" hidden={view !== "tune"}>
    {#if tuneMounted}
      <TuneDetect />
    {/if}
  </main>

  <main class="tune-main" hidden={view !== "label"}>
    {#if labelMounted}
      <LabelReview />
    {/if}
  </main>

  <ReviewConsole
    active={view === "review"}
    {dogs}
    {dogError}
    {helpOpen}
    liveNewCount={$liveNewCount}
    openRequest={reviewOpenRequest}
    onheader={setReviewHeader}
    onready={startLiveOnce}
    onfocuscamera={focusCameraFilter}
    onrequesthelp={openHelp}
    onclosehelp={closeHelp}
  />
</div>

{#if toasts.length > 0}
  <div class="toaster" aria-live="polite">
    {#each toasts as toast (toast.id)}
      <button type="button" class="toast" onclick={() => { dismissToast(toast.id); void openLiveEvent(toast.summary.event_id); }}>
        <span class="toast-dot"></span>
        <span class="toast-body">
          <strong>New potty event</strong>
          <span class="toast-sub">{toast.summary.camera || "camera"} · click to review</span>
        </span>
        <span
          class="toast-x"
          role="button"
          tabindex="-1"
          aria-label="Dismiss"
          onclick={(e) => { e.stopPropagation(); dismissToast(toast.id); }}
          onkeydown={(e) => { if (e.key === "Enter") { e.stopPropagation(); dismissToast(toast.id); } }}
        >×</span>
      </button>
    {/each}
  </div>
{/if}

<HelpOverlay open={helpOpen} onclose={() => (helpOpen = false)} />

<style>
  .app {
    display: flex;
    flex-direction: column;
    height: 100dvh;
    min-height: 0;
  }

  main[hidden] {
    display: none !important;
  }

  .bar {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 1.25rem;
    padding: 0.7rem 1.25rem;
    border-bottom: 1px solid var(--line-strong);
    background: linear-gradient(180deg, var(--bg-1), var(--bg));
    position: relative;
  }

  /* faint surveillance scanlines */
  .bar::after {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background-image: repeating-linear-gradient(
      0deg,
      rgba(255, 255, 255, 0.015) 0px,
      rgba(255, 255, 255, 0.015) 1px,
      transparent 1px,
      transparent 3px
    );
  }

  .brand {
    display: grid;
    gap: 0.05rem;
    flex: 0 0 auto;
  }

  .eyebrow {
    font-size: 0.6rem;
    letter-spacing: 0.32em;
    color: var(--amber);
  }

  .brand h1 {
    margin: 0;
    font-family: var(--font-display);
    font-size: 1.32rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1;
  }

  .viewtabs {
    display: inline-flex;
    flex: 0 0 auto;
    padding: 0.2rem;
    gap: 0.15rem;
    background: var(--bg-inset);
    border: 1px solid var(--line);
    border-radius: 999px;
  }

  .viewtabs button {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    border: 0;
    background: transparent;
    color: var(--text-dim);
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
  }

  .viewtabs button:hover {
    color: var(--text);
  }

  .viewtabs button.active {
    background: var(--teal);
    color: #04201d;
    font-weight: 700;
  }

  .live-dot {
    width: 0.45rem;
    height: 0.45rem;
    border-radius: 50%;
    background: var(--text-faint);
  }

  .live-dot.on {
    background: var(--teal);
    box-shadow: 0 0 5px color-mix(in srgb, var(--teal) 80%, transparent);
  }

  .viewtabs button.active .live-dot.on {
    background: #04201d;
    box-shadow: none;
  }

  .live-badge {
    min-width: 1.1rem;
    padding: 0 0.3rem;
    border-radius: 999px;
    background: var(--amber);
    color: #1a1204;
    font-size: 0.64rem;
    font-weight: 800;
    line-height: 1.1rem;
    text-align: center;
  }

  .live-main {
    flex: 1 1 auto;
    min-height: 0;
    overflow: hidden;
  }

  .tune-main {
    flex: 1 1 auto;
    min-height: 0;
    /* Scrolls only when the player itself can't fit a short window; the
       detections panel scrolls internally so its height never grows this. */
    overflow: auto;
  }

  .toaster {
    position: fixed;
    right: 1rem;
    bottom: 1rem;
    z-index: 50;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    max-width: min(22rem, calc(100vw - 2rem));
  }

  .toast {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.65rem 0.75rem;
    text-align: left;
    border: 1px solid var(--line-strong);
    border-left: 3px solid var(--teal);
    border-radius: var(--radius-sm);
    background: var(--bg-1);
    box-shadow: var(--shadow);
    cursor: pointer;
    animation: toast-in 0.22s cubic-bezier(0.2, 0.9, 0.3, 1) both;
  }

  .toast:hover {
    background: var(--bg-2);
  }

  .toast-dot {
    flex: 0 0 auto;
    width: 0.55rem;
    height: 0.55rem;
    border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 6px color-mix(in srgb, var(--teal) 80%, transparent);
  }

  .toast-body {
    display: grid;
    gap: 0.1rem;
    min-width: 0;
  }

  .toast-body strong {
    font-size: 0.84rem;
  }

  .toast-sub {
    font-size: 0.72rem;
    color: var(--text-dim);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .toast-x {
    margin-left: auto;
    flex: 0 0 auto;
    color: var(--text-faint);
    font-size: 1rem;
    line-height: 1;
    padding: 0 0.2rem;
  }

  .toast-x:hover {
    color: var(--text);
  }

  @keyframes toast-in {
    from {
      opacity: 0;
      transform: translateY(8px);
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .toast {
      animation: none;
    }
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex: 1 1 auto;
    min-width: 0;
    flex-wrap: wrap;
  }

  .segmented {
    display: inline-flex;
    padding: 0.2rem;
    gap: 0.15rem;
    background: var(--bg-inset);
    border: 1px solid var(--line);
    border-radius: 999px;
  }

  .segmented button {
    border: 0;
    background: transparent;
    color: var(--text-dim);
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
  }

  .segmented button:hover {
    color: var(--text);
    background: transparent;
  }

  .segmented button.active {
    background: var(--amber);
    color: #1a1204;
    font-weight: 700;
  }

  .camera-field {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0 0.5rem;
    background: var(--bg-inset);
    border: 1px solid var(--line-strong);
    border-radius: var(--radius-sm);
    min-width: 11rem;
  }

  .camera-field:focus-within {
    border-color: var(--teal);
    box-shadow: 0 0 0 3px var(--teal-soft);
  }

  .camera-field .slash {
    color: var(--text-faint);
    font-size: 0.8rem;
  }

  .camera-field input {
    border: 0;
    background: transparent;
    padding: 0.45rem 0;
  }

  .camera-field input:focus-visible {
    box-shadow: none;
  }

  .clear {
    border: 0;
    background: transparent;
    color: var(--text-faint);
    padding: 0 0.2rem;
    font-size: 1.1rem;
    line-height: 1;
  }

  .clear:hover {
    color: var(--text);
    background: transparent;
  }

  .bar-right {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    flex: 0 0 auto;
  }

  .progress {
    display: grid;
    gap: 0.28rem;
    min-width: 7rem;
  }

  .progress-text {
    font-size: 0.74rem;
    color: var(--text-dim);
  }

  .progress-text strong {
    color: var(--green);
  }

  .progress-track {
    height: 3px;
    border-radius: 999px;
    background: var(--bg-3);
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--green), var(--teal));
    transition: width 0.3s ease;
  }

  .toggle {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 0.4rem 0.6rem;
    color: var(--text-faint);
    background: var(--bg-inset);
  }

  .toggle.on {
    color: var(--teal);
    border-color: rgba(84, 210, 196, 0.4);
    background: var(--teal-soft);
  }

  .help-btn {
    padding: 0.3rem 0.45rem;
    background: var(--bg-inset);
  }

  .help-btn kbd {
    pointer-events: none;
  }

  @media (max-width: 900px) {
    .bar {
      flex-wrap: wrap;
    }
  }
</style>
