<script lang="ts">
  interface FilmstripCrop {
    frame: number;
    url: string | null;
  }

  interface Props {
    crops: FilmstripCrop[];
    currentFrame: number;
    fps: number;
    onseek: (frame: number) => void;
  }

  let { crops, currentFrame, fps, onseek }: Props = $props();
</script>

<div class="filmstrip" title="Every sampled detection of the followed track — click a crop to seek there.">
  {#if crops.length === 0}
    <span class="muted small">No detections sampled for this track.</span>
  {:else}
    {#each crops as c, i (i)}
      <button
        type="button"
        class="film-card"
        class:cur={Math.abs(c.frame - currentFrame) < fps / 2}
        onclick={() => onseek(c.frame)}
        title={`Frame ${c.frame} · ${(c.frame / fps).toFixed(2)}s`}
      >
        {#if c.url}
          <img src={c.url} alt={`detection at frame ${c.frame}`} />
        {:else}
          <span class="film-ph"></span>
        {/if}
        <span class="film-f mono">f{c.frame}</span>
      </button>
    {/each}
  {/if}
</div>

<style>
  .filmstrip {
    display: flex;
    gap: 0.3rem;
    overflow-x: auto;
    padding-bottom: 0.2rem;
  }
  .film-card {
    flex: none;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.1rem;
    background: var(--bg-3);
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 0.15rem;
    cursor: pointer;
  }
  .film-card.cur {
    border-color: var(--green);
  }
  .film-card img {
    width: 72px;
    height: 54px;
    object-fit: cover;
    border-radius: 3px;
    background: #000;
    display: block;
  }
  .film-ph {
    width: 72px;
    height: 54px;
    border-radius: 3px;
    background: #0a0e16;
    display: block;
  }
  .film-f {
    font-size: 0.6rem;
    color: var(--text-dim);
  }
  .small {
    font-size: 0.72rem;
  }
  .muted {
    color: var(--text-dim);
  }
</style>
