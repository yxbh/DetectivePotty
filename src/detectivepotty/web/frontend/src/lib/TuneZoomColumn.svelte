<script lang="ts">
  import type { ZoomCard } from "./tuneDetectCore";

  interface Props {
    zoomCards: ZoomCard[];
    canvases: HTMLCanvasElement[];
  }

  let { zoomCards, canvases = $bindable<HTMLCanvasElement[]>([]) }: Props = $props();
</script>

<aside class="zoom-col">
  <div class="zoom-head">
    <span class="eyebrow">DETECTIONS</span>
    <span class="mono muted small">{zoomCards.length}</span>
  </div>
  {#if zoomCards.length > 0}
    <div class="zoom">
      {#each zoomCards as card, i (card.det.x1 + ":" + card.det.y1 + ":" + i)}
        <figure class="zoom-card" class:dropped={!card.kept}>
          <canvas bind:this={canvases[i]}></canvas>
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

<style>
  .zoom-col {
    grid-area: zoom;
    min-height: 0;
    display: flex;
    flex-direction: column;
    border: 1px solid var(--line, #243042);
    border-radius: 10px;
    background: var(--bg-1, #141a24);
    overflow: hidden;
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

  @media (max-width: 1280px) {
    .zoom-col {
      min-width: 0;
      max-height: 32vh;
    }

    .zoom {
      flex-direction: row;
      flex-wrap: wrap;
      align-items: flex-start;
    }

    .zoom-card {
      width: 200px;
    }
  }
</style>
