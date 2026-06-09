<script lang="ts">
  import type { EventDetail, EventSummary, LabelDraft } from "./types";
  import { onMount } from "svelte";
  import { get } from "svelte/store";
  import { fetchDogs, fetchEventDetail, fetchEvents, saveLabel } from "./api";
  import { autoAdvance, liveNotifications, liveSound } from "./prefs";
  import {
    acknowledgeEvents,
    liveConnected,
    liveEvents,
    liveNewCount,
    onLiveEvent,
    startLive,
    stopLive,
  } from "./live";
  import { playChime, showEventNotification } from "./notify";
  import EventList from "./EventList.svelte";
  import EventDetailView from "./EventDetail.svelte";
  import LiveFeed from "./LiveFeed.svelte";
  import TuneDetect from "./TuneDetect.svelte";
  import HelpOverlay from "./HelpOverlay.svelte";
  import { navigate, route, routeToView } from "./router";

  const STATUS_FILTERS: Array<[string, string]> = [
    ["", "All"],
    ["unlabeled", "Unlabeled"],
    ["labeled", "Labeled"],
    ["rejected", "Rejected"],
    ["uncertain", "Uncertain"],
  ];

  let dogs = $state<string[]>([]);
  let events = $state<EventSummary[]>([]);
  let unfilteredTotal = $state<number | null>(null);
  let selectedId = $state<string | null>(null);

  let statusFilter = $state("");
  let cameraFilter = $state("");
  let cameraInput = $state<HTMLInputElement | null>(null);

  let listLoading = $state(true);
  let listError = $state<string | null>(null);

  let detail = $state<EventDetail | null>(null);
  let detailLoading = $state(false);
  let detailError = $state<string | null>(null);

  let helpOpen = $state(false);
  let saving = $state(false);
  let saveStatus = $state("");

  // The active view is derived from the URL so a refresh restores it (and deep
  // links work). `?event=<id>` on the review route restores the open event.
  let view = $derived(routeToView($route.path));
  let toasts = $state<Array<{ id: string; summary: EventSummary }>>([]);

  // Single source of truth for the label editor (lifted out of EventDetail so a
  // keyboard shortcut and the visible form can never disagree).
  let draft = $state<LabelDraft>({ label: "unknown", status: "labeled", dog: "", note: "" });
  let baseline = $state<LabelDraft>({ label: "unknown", status: "labeled", dog: "", note: "" });

  // Monotonic tokens guard against slow fetches resolving after a newer
  // selection (fast j/k navigation) or a newer filter change.
  let detailToken = 0;
  let listToken = 0;

  let dirty = $derived(
    detail != null &&
      (draft.label !== baseline.label ||
        draft.status !== baseline.status ||
        draft.dog !== baseline.dog ||
        draft.note !== baseline.note),
  );
  let labeledCount = $derived(
    events.filter((e) => e.label_status && e.label_status !== "unlabeled").length,
  );
  let progressPct = $derived(
    events.length ? Math.round((labeledCount / events.length) * 100) : 0,
  );

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

  // Reconcile selection FROM the route: opening `/?event=<id>` (deep link, back/
  // forward, or refresh) selects that event. No-ops once it matches the current
  // selection, so it cannot loop with selectEvent's URL write-back.
  $effect(() => {
    const current = $route;
    if (routeToView(current.path) !== "review") {
      return;
    }
    const ev = current.query.get("event");
    if (ev && ev !== selectedId) {
      void selectEvent(ev);
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
    navigate("/");
    await loadEvents();
    await selectEvent(eventId);
  }

  async function init(): Promise<void> {
    dogs = await fetchDogs();
    await loadEvents();
    startLive();
  }

  async function loadEvents(): Promise<void> {
    const token = ++listToken;
    listLoading = true;
    listError = null;
    try {
      const page = await fetchEvents({ labelStatus: statusFilter, camera: cameraFilter });
      if (token !== listToken) {
        return; // a newer filter/search superseded this fetch
      }
      events = page.events;
      unfilteredTotal = page.unfilteredTotal;
      // Everything now shown in Review is "known" — clears the live banner and
      // prevents these ids from re-counting/re-notifying via the stream.
      acknowledgeEvents(events.map((event) => event.event_id));
      if (selectedId && !events.some((event) => event.event_id === selectedId)) {
        selectedId = null;
        detail = null;
      }
      // Don't auto-select the first event when the URL deep-links a specific one
      // (?event=…) — the route-reconciliation effect will open it instead.
      const deepLinked = get(route).query.get("event");
      if (events.length > 0 && !selectedId && !deepLinked) {
        await selectEvent(events[0].event_id);
      }
    } catch (err) {
      if (token !== listToken) {
        return;
      }
      listError = err instanceof Error ? err.message : String(err);
      events = [];
      unfilteredTotal = null;
    } finally {
      if (token === listToken) {
        listLoading = false;
      }
    }
  }

  async function selectEvent(eventId: string): Promise<void> {
    selectedId = eventId;
    saveStatus = "";
    // Reflect the open event in the URL (replace, so list navigation doesn't
    // flood history) so a refresh restores it. navigate() is a no-op when the
    // URL already matches, which keeps the route-reconciliation effect from
    // looping.
    if (view === "review") {
      navigate(`/?event=${encodeURIComponent(eventId)}`, { replace: true });
    }
    const token = ++detailToken;
    detailLoading = true;
    detailError = null;
    try {
      const loaded = await fetchEventDetail(eventId);
      if (token !== detailToken) {
        return; // a newer selection superseded this fetch
      }
      detail = loaded;
      resetDraft(loaded);
    } catch (err) {
      if (token !== detailToken) {
        return;
      }
      detail = null;
      detailError = err instanceof Error ? err.message : String(err);
    } finally {
      if (token === detailToken) {
        detailLoading = false;
      }
    }
  }

  function resetDraft(loaded: EventDetail): void {
    const m = loaded.metadata as Record<string, unknown>;
    const status = (m.label_status as string) || "unlabeled";
    const extra = (m.extra as Record<string, unknown> | undefined) ?? {};
    const next: LabelDraft = {
      label: (m.label as string) || "unknown",
      // An unsaved event edits toward "labeled" by default; the user can still
      // switch to rejected/uncertain.
      status: status === "unlabeled" ? "labeled" : status,
      dog: (m.dog as string) || "",
      note: (extra.label_note as string) || "",
    };
    draft = next;
    baseline = { ...next };
  }

  function applySummary(updated: EventSummary): void {
    const index = events.findIndex((event) => event.event_id === updated.event_id);
    if (index !== -1) {
      events[index] = updated;
    }
  }

  async function save(): Promise<void> {
    if (!detail || saving || !dirty) {
      return;
    }
    const targetId = detail.summary.event_id;
    const saveToken = detailToken;
    const snapshot: LabelDraft = { ...draft };
    saving = true;
    saveStatus = "Saving…";
    try {
      const updated = await saveLabel(targetId, {
        label: snapshot.label,
        label_status: snapshot.status,
        note: snapshot.note,
        dog: snapshot.dog || null,
      });
      applySummary(updated);
      // Only mutate editor state if the user is still on the very same selection
      // (the token also rejects a navigate-away-and-back to the same id).
      if (detail && detail.summary.event_id === targetId && detailToken === saveToken) {
        baseline = { ...snapshot };
        patchDetailMetadata(snapshot);
        detail.summary = updated;
        saveStatus = "Saved";
        if (statusFilter !== "" && updated.label_status !== statusFilter) {
          // The event no longer belongs to the active filter — drop it from the
          // list and advance to its neighbour so the list stays consistent.
          removeFromListAndAdvance(targetId);
        } else if ($autoAdvance) {
          jumpUnlabeled(1, targetId);
        }
      }
    } catch (err) {
      if (detail && detail.summary.event_id === targetId && detailToken === saveToken) {
        saveStatus = `Save failed: ${err instanceof Error ? err.message : String(err)}`;
      }
    } finally {
      saving = false;
    }
  }

  function removeFromListAndAdvance(targetId: string): void {
    const idx = events.findIndex((event) => event.event_id === targetId);
    events = events.filter((event) => event.event_id !== targetId);
    if (events.length === 0) {
      selectedId = null;
      detail = null;
      return;
    }
    const nextIdx = Math.min(idx < 0 ? 0 : idx, events.length - 1);
    void selectEvent(events[nextIdx].event_id);
  }

  function patchDetailMetadata(snap: LabelDraft): void {
    if (!detail) {
      return;
    }
    const m = detail.metadata as Record<string, unknown>;
    m.label = snap.label;
    m.label_status = snap.status;
    m.dog = snap.dog || null;
    const extra =
      m.extra && typeof m.extra === "object" ? (m.extra as Record<string, unknown>) : {};
    // Mirror the backend, which always writes label_note from the payload.
    extra.label_note = snap.note;
    m.extra = extra;
  }

  function indexOfSelected(): number {
    return events.findIndex((event) => event.event_id === selectedId);
  }

  function move(delta: number): void {
    if (events.length === 0) {
      return;
    }
    const current = indexOfSelected();
    const next = current < 0 ? 0 : Math.min(events.length - 1, Math.max(0, current + delta));
    void selectEvent(events[next].event_id);
  }

  function edge(end: 0 | 1): void {
    if (events.length === 0) {
      return;
    }
    void selectEvent(events[end ? events.length - 1 : 0].event_id);
  }

  function jumpUnlabeled(dir: 1 | -1, fromId: string | null = selectedId): void {
    const n = events.length;
    if (n === 0) {
      return;
    }
    const start = events.findIndex((event) => event.event_id === fromId);
    const base = start < 0 ? (dir === 1 ? -1 : n) : start;
    for (let step = 1; step <= n; step += 1) {
      const j = ((base + dir * step) % n + n) % n;
      const candidate = events[j];
      if (candidate.event_id === fromId) {
        continue; // never re-select the current event (e.g. only one unlabeled)
      }
      if (candidate.label_status === "unlabeled") {
        void selectEvent(candidate.event_id);
        return;
      }
    }
  }

  function toggleVideo(): void {
    const video = document.querySelector<HTMLVideoElement>(".detail video");
    if (video) {
      if (video.paused) {
        void video.play().catch(() => undefined);
      } else {
        video.pause();
      }
    }
  }

  function isTyping(target: EventTarget | null): boolean {
    const el = target as HTMLElement | null;
    if (!el) {
      return false;
    }
    return (
      el.tagName === "INPUT" ||
      el.tagName === "TEXTAREA" ||
      el.tagName === "SELECT" ||
      el.isContentEditable
    );
  }

  function applyFilter(value: string): void {
    statusFilter = value;
    void loadEvents();
  }

  function onCameraKeydown(event: KeyboardEvent): void {
    if (event.key === "Enter") {
      void loadEvents();
    }
  }

  function clearCamera(): void {
    cameraFilter = "";
    void loadEvents();
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
    if (event.metaKey || event.ctrlKey || event.altKey) {
      return;
    }
    if (isTyping(event.target)) {
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
        navigate("/");
      }
      return;
    }
    // The Live feed has its own minimal keymap: v / Esc return to Review.
    if (view === "live") {
      if (event.key === "v" || event.key === "Escape") {
        event.preventDefault();
        navigate("/");
      } else if (event.key === "?") {
        event.preventDefault();
        helpOpen = true;
      }
      return;
    }
    // Let native activation handle Enter/Space when a button, link, or media
    // element is focused (e.g. tab to a control and press Space to click it).
    if (event.key === " " || event.key === "Enter") {
      const el = event.target as HTMLElement | null;
      if (el?.closest("button, a[href], video, audio, [role='button'], [role='tab']")) {
        return;
      }
    }

    // Dog roster shortcuts: Shift+1…N picks the Nth configured dog, Shift+0
    // unassigns. Uses event.code so it is layout-robust (Shift+1 = "!" on US
    // keyboards) and never clashes with the plain-digit label keys below.
    if (event.shiftKey && detail && /^Digit[0-9]$/.test(event.code)) {
      const digit = Number(event.code.slice(5));
      if (digit === 0) {
        event.preventDefault();
        draft.dog = "";
        return;
      }
      const pick = dogs[digit - 1];
      if (pick) {
        event.preventDefault();
        draft.dog = pick;
      }
      return;
    }

    switch (event.key) {
      case "j":
      case "ArrowDown":
        event.preventDefault();
        move(1);
        break;
      case "k":
      case "ArrowUp":
        event.preventDefault();
        move(-1);
        break;
      case "n":
        event.preventDefault();
        jumpUnlabeled(1);
        break;
      case "N":
        event.preventDefault();
        jumpUnlabeled(-1);
        break;
      case "g":
        event.preventDefault();
        edge(0);
        break;
      case "G":
        event.preventDefault();
        edge(1);
        break;
      case "1":
        if (detail) draft.label = "pee";
        break;
      case "2":
        if (detail) draft.label = "poop";
        break;
      case "3":
        if (detail) draft.label = "not_potty";
        break;
      case "0":
        if (detail) draft.label = "unknown";
        break;
      case "r":
        if (detail) draft.status = "rejected";
        break;
      case "u":
        if (detail) draft.status = "uncertain";
        break;
      case "s":
      case "Enter":
        event.preventDefault();
        void save();
        break;
      case "/":
        event.preventDefault();
        cameraInput?.focus();
        cameraInput?.select();
        break;
      case "?":
        event.preventDefault();
        helpOpen = true;
        break;
      case "a":
        event.preventDefault();
        autoAdvance.update((v) => !v);
        break;
      case "v":
        event.preventDefault();
        navigate("/live");
        break;
      case " ":
        event.preventDefault();
        toggleVideo();
        break;
      default:
        break;
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
        onclick={() => navigate("/")}
      >
        Review
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={view === "live"}
        class:active={view === "live"}
        onclick={() => navigate("/live")}
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
        onclick={() => navigate("/tune")}
        title="Tune YOLO detection on a clip"
      >
        Tune
      </button>
    </div>

    {#if view === "review"}
    <div class="controls">
      <div class="segmented" role="tablist" aria-label="Filter by label status">
        {#each STATUS_FILTERS as [value, text] (value)}
          <button
            type="button"
            role="tab"
            aria-selected={statusFilter === value}
            class:active={statusFilter === value}
            onclick={() => applyFilter(value)}
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
          bind:value={cameraFilter}
          onkeydown={onCameraKeydown}
        />
        {#if cameraFilter}
          <button type="button" class="clear" onclick={clearCamera} aria-label="Clear camera filter">×</button>
        {/if}
      </div>
    </div>
    {/if}

    <div class="bar-right">
      {#if view === "review"}
      <div class="progress" title="{labeledCount} of {events.length} loaded events labeled">
        <div class="progress-text mono">
          <strong>{labeledCount}</strong>/{events.length}
          {#if unfilteredTotal != null && unfilteredTotal !== events.length}
            <span class="muted">· {unfilteredTotal} on disk</span>
          {/if}
        </div>
        <div class="progress-track"><div class="progress-fill" style="width: {progressPct}%"></div></div>
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

  {#if view === "review" && $liveNewCount > 0}
    <button type="button" class="new-banner" onclick={() => void loadEvents()}>
      <span class="pip"></span>
      {$liveNewCount} new event{$liveNewCount === 1 ? "" : "s"} · click to load
    </button>
  {/if}

  {#if view === "live"}
    <main class="live-main">
      <LiveFeed events={$liveEvents} connected={$liveConnected} onpick={openLiveEvent} />
    </main>
  {:else if view === "tune"}
    <main class="tune-main">
      <TuneDetect />
    </main>
  {:else}
    <main class="layout">
      <section class="sidebar">
        <div class="list-scroll">
          <EventList
            {events}
            {selectedId}
            loading={listLoading}
            error={listError}
            onselect={selectEvent}
          />
        </div>
      </section>

      <section class="detail-scroll detail">
        <EventDetailView
          {detail}
          {dogs}
          {draft}
          {dirty}
          {saving}
          {saveStatus}
          loading={detailLoading}
          error={detailError}
          onsave={save}
        />
      </section>
    </main>
  {/if}
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

  .new-banner {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    width: 100%;
    padding: 0.5rem 1rem;
    border: 0;
    border-bottom: 1px solid var(--line);
    background: color-mix(in srgb, var(--teal) 14%, var(--bg-1));
    color: var(--text);
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
  }

  .new-banner:hover {
    background: color-mix(in srgb, var(--teal) 22%, var(--bg-1));
  }

  .new-banner .pip {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 6px color-mix(in srgb, var(--teal) 80%, transparent);
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

  .layout {
    flex: 1 1 auto;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(19rem, 26rem) minmax(0, 1fr);
  }

  /* Each pane owns its own scrollbar so the list and detail scroll
     independently (bounded grid height + min-height:0 children). */
  .sidebar {
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    border-right: 1px solid var(--line-strong);
    background: var(--bg);
  }

  .list-scroll {
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
  }

  .detail-scroll {
    min-height: 0;
    overflow-y: auto;
  }

  @media (max-width: 900px) {
    .bar {
      flex-wrap: wrap;
    }

    .layout {
      grid-template-columns: 1fr;
      grid-template-rows: minmax(0, 42vh) minmax(0, 1fr);
    }

    .sidebar {
      border-right: 0;
      border-bottom: 1px solid var(--line-strong);
    }
  }
</style>
