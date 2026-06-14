<script lang="ts">
  import { observeResize } from "./resize";

  interface Props {
    canvas: HTMLCanvasElement | null;
    displayIndex: number;
    totalFrames: number;
    onpointerdown: () => void;
    oninput: (event: Event) => void;
    oncommit: () => void;
    onresize: () => void;
  }

  let {
    canvas = $bindable<HTMLCanvasElement | null>(null),
    displayIndex,
    totalFrames,
    onpointerdown,
    oninput,
    oncommit,
    onresize,
  }: Props = $props();

  $effect(() => observeResize(canvas, onresize));
</script>

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
    onpointerdown={onpointerdown}
    oninput={oninput}
    onpointerup={oncommit}
    onpointercancel={oncommit}
    onchange={oncommit}
  />
  <div class="strip-wrap">
    <canvas bind:this={canvas} class="strip"></canvas>
    <div class="strip-legend mono small muted">
      <span><i class="sw yolo"></i> YOLO (analyzed · detected)</span>
      <span><i class="sw pose"></i> pose</span>
      <span><i class="sw track"></i> track</span>
    </div>
  </div>
</div>

<style>
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
    height: 28px;
    border-radius: 3px;
  }

  .strip-legend {
    display: flex;
    gap: 1rem;
    margin-top: 3px;
    padding-inline: 1px;
  }

  .strip-legend .sw {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 2px;
    margin-right: 4px;
    vertical-align: -1px;
  }

  .strip-legend .sw.yolo {
    background: #28d17c;
  }

  .strip-legend .sw.pose {
    background: #5ad1ff;
  }

  .strip-legend .sw.track {
    background: #b388ff;
  }

  .small {
    font-size: 0.74rem;
  }

  .muted {
    color: var(--muted, #8a97a8);
  }
</style>
