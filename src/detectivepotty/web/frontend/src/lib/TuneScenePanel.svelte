<script lang="ts">
  import type { TuneSceneObject } from "./types";

  interface Props {
    sceneBoxes: boolean;
    sceneLoading: boolean;
    sceneIndex: number | null;
    presentedIndex: number;
    sceneError: string | null;
    sceneObjects: TuneSceneObject[];
  }

  let {
    sceneBoxes = $bindable(),
    sceneLoading,
    sceneIndex,
    presentedIndex,
    sceneError,
    sceneObjects,
  }: Props = $props();
</script>

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
    <div class="scene-empty muted small">Nothing detected on this frame (above the detector floor).</div>
  {/if}
</div>

<style>
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

  .eyebrow {
    font-size: 0.6rem;
    letter-spacing: 0.28em;
    color: var(--amber, #f0b35a);
  }

  .export-error {
    color: var(--amber, #f0b35a);
    max-width: 22ch;
  }

  .muted {
    color: var(--muted, #8a97a8);
  }

  .small {
    font-size: 0.74rem;
  }
</style>
