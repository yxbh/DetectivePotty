<script lang="ts">
  import type { EventSummary } from "./types";
  import EventCard from "./EventCard.svelte";
  import { liveNotifications, liveSound } from "./prefs";
  import { notificationsSupported, playChime, requestNotificationPermission } from "./notify";

  interface Props {
    events: EventSummary[];
    connected: boolean;
    onpick: (eventId: string) => void;
  }

  let { events, connected, onpick }: Props = $props();

  let note = $state<{ text: string; kind: "ok" | "warn" } | null>(null);
  let noteTimer: ReturnType<typeof setTimeout> | undefined;
  const supported = notificationsSupported();

  function setNote(text: string, kind: "ok" | "warn"): void {
    note = { text, kind };
    clearTimeout(noteTimer);
    // Confirmations are transient; actionable warnings stay until changed.
    if (kind === "ok") {
      noteTimer = setTimeout(() => {
        note = null;
      }, 3500);
    }
  }

  async function toggleNotifications(): Promise<void> {
    if ($liveNotifications) {
      $liveNotifications = false;
      note = null;
      return;
    }
    const perm = await requestNotificationPermission();
    if (perm === "granted") {
      $liveNotifications = true;
      setNote("Notifications enabled — you'll get a popup for each new event.", "ok");
    } else {
      $liveNotifications = false;
      setNote(
        perm === "denied"
          ? "Notifications are blocked in your browser settings."
          : "Notification permission was not granted.",
        "warn",
      );
    }
  }

  function toggleSound(): void {
    $liveSound = !$liveSound;
    // Play the chime right away so enabling sound gives instant feedback (and the
    // click gesture unlocks the AudioContext) instead of staying silent until the
    // next event arrives.
    if ($liveSound) {
      playChime();
    }
  }
</script>

<div class="live">
  <header class="live-head">
    <div class="title">
      <span class="dot" class:on={connected}></span>
      <h2>Live feed</h2>
      <span class="status mono">{connected ? "streaming" : "reconnecting…"}</span>
    </div>

    <div class="live-toggles">
      <button
        type="button"
        role="switch"
        class="toggle"
        class:on={$liveNotifications}
        aria-checked={$liveNotifications}
        disabled={!supported}
        onclick={toggleNotifications}
        title={supported
          ? "Browser notification for each new event"
          : "This browser does not support notifications"}
      >
        <span class="switch" aria-hidden="true"><span class="knob"></span></span>
        <span class="label">notify</span>
      </button>
      <button
        type="button"
        role="switch"
        class="toggle"
        class:on={$liveSound}
        aria-checked={$liveSound}
        onclick={toggleSound}
        title="Play a chime for each new event"
      >
        <span class="switch" aria-hidden="true"><span class="knob"></span></span>
        <span class="label">sound</span>
      </button>
    </div>
  </header>

  {#if note}
    <p class="perm-note" class:ok={note.kind === "ok"}>{note.text}</p>
  {/if}

  {#if events.length === 0}
    <div class="live-empty">
      <p class="big">Waiting for new events…</p>
      <p class="sub">
        New potty events stream in here the moment the pipeline finalizes them.
        Keep this tab open — events also surface as a banner on the Review page.
      </p>
    </div>
  {:else}
    <div class="live-list">
      {#each events as event, i (event.event_id)}
        <EventCard {event} index={i} active={false} onselect={onpick} />
      {/each}
    </div>
  {/if}
</div>

<style>
  .live {
    display: flex;
    flex-direction: column;
    min-height: 0;
    height: 100%;
  }

  .live-head {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.85rem 1.1rem;
    border-bottom: 1px solid var(--line);
    background: var(--bg-1);
  }

  .title {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }

  .title h2 {
    margin: 0;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
  }

  .status {
    font-size: 0.7rem;
    color: var(--text-faint);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .dot {
    width: 0.6rem;
    height: 0.6rem;
    border-radius: 50%;
    background: var(--amber);
    box-shadow: 0 0 0 2px var(--bg-1);
  }

  .dot.on {
    background: var(--teal);
    animation: pulse 1.8s ease-in-out infinite;
  }

  .live-toggles {
    display: flex;
    gap: 0.4rem;
  }

  /* On/off pill switches: the track turns teal and the knob slides right when
     enabled, so the state is unambiguous at a glance (not just a colour change). */
  .toggle {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.6rem 0.3rem 0.45rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text-faint);
    background: var(--bg-inset);
    border: 1px solid var(--line-strong);
    border-radius: 999px;
  }

  .toggle.on {
    color: var(--teal);
    border-color: rgba(84, 210, 196, 0.4);
    background: var(--teal-soft);
  }

  .toggle:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .switch {
    position: relative;
    flex: 0 0 auto;
    width: 1.7rem;
    height: 0.95rem;
    border-radius: 999px;
    background: rgba(236, 233, 227, 0.18);
    transition: background 0.16s ease;
  }

  .toggle.on .switch {
    background: var(--teal);
  }

  .knob {
    position: absolute;
    top: 50%;
    left: 0.13rem;
    width: 0.68rem;
    height: 0.68rem;
    border-radius: 50%;
    background: var(--text-dim);
    transform: translateY(-50%);
    transition: left 0.16s ease, background 0.16s ease;
  }

  .toggle.on .knob {
    left: calc(100% - 0.81rem);
    background: var(--bg-inset);
  }

  .perm-note {
    margin: 0;
    padding: 0.5rem 1.1rem;
    font-size: 0.78rem;
    color: var(--amber);
    background: color-mix(in srgb, var(--amber) 10%, transparent);
    border-bottom: 1px solid var(--line);
  }

  .perm-note.ok {
    color: var(--teal);
    background: color-mix(in srgb, var(--teal) 10%, transparent);
  }

  .live-list {
    display: grid;
    gap: 0.5rem;
    padding: 0.75rem;
    overflow-y: auto;
    align-content: start;
  }

  .live-empty {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 0.5rem;
    padding: 2rem;
    color: var(--text-dim);
  }

  .live-empty .big {
    margin: 0;
    font-size: 1.1rem;
    color: var(--text);
  }

  .live-empty .sub {
    margin: 0;
    max-width: 42ch;
    font-size: 0.85rem;
    line-height: 1.5;
    color: var(--text-faint);
  }

  @keyframes pulse {
    0%,
    100% {
      box-shadow: 0 0 0 2px var(--bg-1), 0 0 0 0 color-mix(in srgb, var(--teal) 55%, transparent);
    }
    50% {
      box-shadow: 0 0 0 2px var(--bg-1), 0 0 0 5px color-mix(in srgb, var(--teal) 0%, transparent);
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .dot.on {
      animation: none;
    }

    .switch,
    .knob {
      transition: none;
    }
  }
</style>
