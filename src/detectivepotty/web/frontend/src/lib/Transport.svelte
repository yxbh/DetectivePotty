<script lang="ts">
  interface Props {
    playing: boolean;
    frame: number;
    total: number;
    fps: number;
    skipN?: number;
    showSkip?: boolean;
    showReadout?: boolean;
    onTogglePlay: () => void;
    onStep: (delta: number) => void;
  }

  let {
    playing,
    frame,
    total,
    fps,
    skipN = 10,
    showSkip = true,
    showReadout = false,
    onTogglePlay,
    onStep,
  }: Props = $props();

  const lastFrame = $derived(Math.max(0, total - 1));
  const seconds = $derived(fps > 0 ? frame / fps : 0);
</script>

<div class="transport">
  {#if showSkip}
    <button type="button" onclick={() => onStep(-skipN)} title="Back {skipN} frames (Shift+←)">⏮</button>
  {/if}
  <button type="button" onclick={() => onStep(-1)} title="Back 1 frame (←)">◀</button>
  <button type="button" class="play" onclick={onTogglePlay} title="Play / Pause (Space)">
    {playing ? "⏸" : "▶"}
  </button>
  <button type="button" onclick={() => onStep(1)} title="Forward 1 frame (→)">▶</button>
  {#if showSkip}
    <button type="button" onclick={() => onStep(skipN)} title="Forward {skipN} frames (Shift+→)">⏭</button>
  {/if}
  {#if showReadout}
    <span class="frame-readout mono" title="Current frame / last frame · time">
      f{frame} / {lastFrame} · {seconds.toFixed(2)}s
    </span>
  {/if}
</div>

<style>
  .transport {
    display: flex;
    align-items: center;
    gap: 0.25rem;
  }
  .transport button {
    background: var(--bg-1);
    border: 1px solid var(--line-strong);
    color: var(--text);
    border-radius: 6px;
    padding: 0.3rem 0.55rem;
    cursor: pointer;
    font-size: 0.9rem;
    min-width: 2.1rem;
    line-height: 1;
  }
  .transport button:hover {
    background: var(--bg-3);
  }
  .transport button.play {
    background: var(--amber);
    border-color: var(--amber-bright);
    color: #1a1204;
    min-width: 2.8rem;
  }
  .transport button.play:hover {
    background: var(--amber-bright);
  }
  .frame-readout {
    margin-left: auto;
    font-size: 0.76rem;
    color: var(--text-dim);
  }
</style>
