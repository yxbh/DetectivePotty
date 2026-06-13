<script lang="ts">
  import type { EventSummary } from "./types";
  import { versioned } from "./api";
  import {
    basisHint,
    formatClock,
    formatGuess,
    formatRelative,
    formatTime,
    labelText,
  } from "./format";

  interface Props {
    event: EventSummary;
    active: boolean;
    index: number;
    onselect: (eventId: string) => void;
  }

  let { event, active, index, onselect }: Props = $props();

  let el = $state<HTMLButtonElement | null>(null);
  let thumbnailUrl = $derived(versioned(event.thumbnail_url, event.media_version));
  let hint = $derived(basisHint(event.time_basis));
  let status = $derived(event.label_status || "unlabeled");
  let animationIndex = $derived(Math.min(index, 12));
  let hasLabel = $derived(
    event.label && event.label !== "unknown" && status !== "unlabeled",
  );

  // Keep the keyboard-selected card within the scroll viewport.
  $effect(() => {
    if (active && el) {
      el.scrollIntoView({ block: "nearest" });
    }
  });
</script>

<button
  bind:this={el}
  class="event-card"
  class:active
  type="button"
  style="--i: {animationIndex}"
  onclick={() => onselect(event.event_id)}
>
  <span class="thumb-wrap">
    {#if thumbnailUrl}
      <img class="thumb" src={thumbnailUrl} alt="" loading="lazy" />
    {:else}
      <span class="thumb thumb--empty" aria-hidden="true">no frame</span>
    {/if}
    <span class="chip status-{status}">{status}</span>
  </span>

  <span class="meta">
    <span class="top">
      <strong class="camera">{event.camera || "Unknown camera"}</strong>
      {#if event.dog}<span class="dog">{event.dog}</span>{/if}
    </span>

    <span class="time mono" title={formatTime(event.utc_ts)}>
      {formatClock(event.utc_ts)}
      {#if hint}<span class="time-hint">{hint}</span>{/if}
    </span>

    <span class="foot">
      <span class="guess mono">{formatGuess(event.classifier_guess, event.classifier_confidence)}</span>
      {#if hasLabel}
        <span class="chip label-{event.label}">{labelText(event.label)}</span>
      {/if}
      <span class="gen" title={formatTime(event.recorded_at)}>{formatRelative(event.recorded_at)}</span>
    </span>
  </span>
</button>

<style>
  .event-card {
    display: grid;
    grid-template-columns: 5.5rem 1fr;
    gap: 0.7rem;
    width: 100%;
    padding: 0.5rem;
    text-align: left;
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    background: var(--bg-1);
    position: relative;
    transition: border-color 0.14s ease, background 0.14s ease, transform 0.1s ease;
    animation: card-in 0.32s ease both;
    animation-delay: calc(var(--i) * 22ms);
  }

  .event-card::before {
    content: "";
    position: absolute;
    left: -1px;
    top: 0.5rem;
    bottom: 0.5rem;
    width: 3px;
    border-radius: 999px;
    background: transparent;
    transition: background 0.14s ease;
  }

  .event-card:hover {
    background: var(--bg-2);
    border-color: var(--line-strong);
  }

  .event-card.active {
    background: var(--bg-2);
    border-color: rgba(245, 165, 36, 0.45);
  }

  .event-card.active::before {
    background: var(--amber);
  }

  .thumb-wrap {
    position: relative;
    line-height: 0;
  }

  .thumb {
    width: 5.5rem;
    height: 4.1rem;
    object-fit: cover;
    border-radius: var(--radius-sm);
    background: var(--bg-inset);
    display: block;
  }

  .thumb--empty {
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--text-faint);
    line-height: 1;
  }

  .thumb-wrap .chip {
    position: absolute;
    left: 0.2rem;
    bottom: 0.2rem;
    padding: 0.08rem 0.34rem;
    font-size: 0.6rem;
    backdrop-filter: blur(4px);
  }

  .meta {
    display: grid;
    gap: 0.28rem;
    min-width: 0;
    align-content: start;
  }

  .top {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    min-width: 0;
  }

  .camera {
    font-weight: 600;
    font-size: 0.92rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .dog {
    flex: 0 0 auto;
    font-size: 0.72rem;
    color: var(--amber);
    font-weight: 600;
  }

  .time {
    font-size: 0.8rem;
    color: var(--text);
    display: flex;
    align-items: baseline;
    gap: 0.45rem;
  }

  .time-hint {
    font-family: var(--font-ui);
    font-size: 0.64rem;
    color: var(--text-faint);
    letter-spacing: 0.02em;
  }

  .foot {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    min-width: 0;
  }

  .guess {
    font-size: 0.72rem;
    color: var(--text-dim);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .foot .chip {
    flex: 0 0 auto;
    padding: 0.08rem 0.4rem;
    font-size: 0.62rem;
  }

  .gen {
    margin-left: auto;
    flex: 0 0 auto;
    font-size: 0.68rem;
    color: var(--text-faint);
    font-family: var(--font-mono);
  }

  @keyframes card-in {
    from {
      opacity: 0;
      transform: translateY(4px);
    }
  }
</style>
