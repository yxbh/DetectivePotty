<script lang="ts">
  import type { EventSummary } from "./types";
  import EventCard from "./EventCard.svelte";

  interface Props {
    events: EventSummary[];
    selectedId: string | null;
    loading: boolean;
    error: string | null;
    onselect: (eventId: string) => void;
  }

  let { events, selectedId, loading, error, onselect }: Props = $props();
</script>

<div class="event-list" aria-live="polite">
  {#if loading}
    <div class="empty-state">Scanning dataset…</div>
  {:else if error}
    <div class="error-state">Failed to load events: {error}</div>
  {:else if events.length === 0}
    <div class="empty-state">No events match the current filter.</div>
  {:else}
    {#each events as event, i (event.event_id)}
      <EventCard
        {event}
        index={i}
        active={event.event_id === selectedId}
        {onselect}
      />
    {/each}
  {/if}
</div>

<style>
  .event-list {
    display: grid;
    gap: 0.5rem;
    padding: 0.75rem;
    align-content: start;
  }
</style>
