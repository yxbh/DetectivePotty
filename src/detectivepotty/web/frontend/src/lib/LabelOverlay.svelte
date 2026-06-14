<script lang="ts">
  import type { ResolvedBox } from "./labelBox";
  import type { LabelPresentTrack } from "./types";
  import { formatDetLabel, formatTrackLabel, isAliasClass } from "./overlayStyle";

  interface SiblingBox {
    track: LabelPresentTrack;
    bbox: { x1: number; y1: number; x2: number; y2: number };
    class_name: string;
    confidence: number;
  }

  interface Props {
    width: number;
    height: number;
    siblingBoxes: SiblingBox[];
    activeBox: ResolvedBox | null;
    labelFont: number;
    onselectclip: (spanId: string) => void;
  }

  let { width, height, siblingBoxes, activeBox, labelFont, onselectclip }: Props = $props();
  const isAlias = isAliasClass;
</script>

<svg class="overlay" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet">
  {#each siblingBoxes as sibling (sibling.track.span_id + ':' + sibling.track.track_id)}
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <rect
      x={sibling.bbox.x1}
      y={sibling.bbox.y1}
      width={sibling.bbox.x2 - sibling.bbox.x1}
      height={sibling.bbox.y2 - sibling.bbox.y1}
      class="box sibling"
      role="button"
      tabindex="-1"
      onclick={() => onselectclip(sibling.track.span_id)}
      ><title
        >Other segment (Track {sibling.track.track_id}) — may be the same dog; click to open
        its clip</title
      ></rect
    >
    <text
      x={sibling.bbox.x1 + labelFont * 0.2}
      y={sibling.bbox.y1 - labelFont * 0.3 < labelFont
        ? sibling.bbox.y1 + labelFont
        : sibling.bbox.y1 - labelFont * 0.3}
      class="box-label sibling"
      class:alias={isAlias(sibling.class_name)}
      font-size={labelFont}
      >{formatTrackLabel(sibling.track.track_id, sibling.confidence, sibling.class_name)}</text
    >
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
      y={activeBox.bbox.y1 - labelFont * 0.3 < labelFont
        ? activeBox.bbox.y1 + labelFont
        : activeBox.bbox.y1 - labelFont * 0.3}
      class="box-label active"
      class:alias={isAlias(activeBox.class_name)}
      font-size={labelFont}
      >{formatDetLabel(activeBox.class_name, activeBox.confidence)}</text
    >
  {/if}
</svg>
{#if activeBox && activeBox.extrapolated}
  <div
    class="no-detect"
    title="This frame is in the clip's padding, before/after this track's first/last detection. No box is drawn rather than freeze a stale one."
  >
    no detection at this frame
  </div>
{/if}

<style>
  .overlay {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }
  .overlay .box {
    fill: none;
    vector-effect: non-scaling-stroke;
  }
  .overlay .box.active {
    stroke: var(--green);
    stroke-width: 3;
  }
  .overlay .box.sibling {
    stroke: var(--amber);
    stroke-width: 2;
    stroke-dasharray: 5 4;
    opacity: 0.8;
    pointer-events: auto;
    cursor: pointer;
    fill: rgba(240, 169, 58, 0.06);
  }
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
  .overlay .box-label.active {
    fill: var(--green);
  }
  .overlay .box-label.sibling {
    fill: var(--amber);
    opacity: 0.85;
  }
  .overlay .box-label.alias {
    fill: var(--teal);
  }
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
</style>
