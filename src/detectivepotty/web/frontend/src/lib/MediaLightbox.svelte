<script lang="ts">
  interface MediaLightboxItem {
    src: string;
    alt: string;
    caption?: string;
    eyebrow?: string;
  }

  interface Props {
    open: boolean;
    items: MediaLightboxItem[];
    index: number;
    onclose: () => void;
    onindexchange: (index: number) => void;
  }

  let { open, items, index, onclose, onindexchange }: Props = $props();

  let safeIndex = $derived(items.length ? Math.min(Math.max(index, 0), items.length - 1) : 0);
  let active = $derived(items[safeIndex] ?? null);
  let hasMany = $derived(items.length > 1);

  function setIndex(next: number): void {
    if (!items.length) {
      return;
    }
    onindexchange(Math.min(Math.max(next, 0), items.length - 1));
  }

  function step(delta: number): void {
    if (!items.length) {
      return;
    }
    onindexchange((safeIndex + delta + items.length) % items.length);
  }

  function handleKey(event: KeyboardEvent): void {
    if (!open) {
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      onclose();
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      step(-1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      step(1);
    } else if (event.key === "Home") {
      event.preventDefault();
      setIndex(0);
    } else if (event.key === "End") {
      event.preventDefault();
      setIndex(items.length - 1);
    }
  }

  function handleBackdropClick(event: MouseEvent): void {
    if (event.target === event.currentTarget) {
      onclose();
    }
  }
</script>

<svelte:window onkeydown={handleKey} />

{#if open && active}
  <div class="lightbox">
    <button
      type="button"
      class="backdrop"
      aria-label="Close lightbox"
      tabindex="-1"
      onclick={handleBackdropClick}
    ></button>
    <div class="scan-frame" role="dialog" aria-modal="true" aria-label="Media inspection lightbox" tabindex="-1">
      <header class="topbar">
        <div class="title">
          <span class="eyebrow">{active.eyebrow ?? "Media inspection"}</span>
          <strong>{active.caption ?? active.alt}</strong>
        </div>
        <div class="status mono" aria-live="polite">
          {String(safeIndex + 1).padStart(2, "0")} / {String(items.length).padStart(2, "0")}
        </div>
        <button type="button" class="close" onclick={onclose} aria-label="Close lightbox">
          esc
        </button>
      </header>

      <div class="stage">
        {#if hasMany}
          <button
            type="button"
            class="nav prev"
            onclick={() => step(-1)}
            aria-label="Previous media item"
          >
            ←
          </button>
        {/if}

        <figure>
          {#key active.src}
            <img src={active.src} alt={active.alt} />
          {/key}
          <figcaption>
            <span>{active.caption ?? active.alt}</span>
            <span class="hint">Esc closes · ← / → scan</span>
          </figcaption>
        </figure>

        {#if hasMany}
          <button
            type="button"
            class="nav next"
            onclick={() => step(1)}
            aria-label="Next media item"
          >
            →
          </button>
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  .lightbox {
    position: fixed;
    inset: 0;
    z-index: 70;
    display: grid;
    place-items: center;
    padding: clamp(0.75rem, 2vw, 1.5rem);
    background:
      radial-gradient(circle at 20% 8%, rgba(245, 165, 36, 0.14), transparent 26rem),
      radial-gradient(circle at 80% 92%, rgba(84, 210, 196, 0.11), transparent 30rem),
      rgba(3, 5, 7, 0.88);
    backdrop-filter: blur(8px);
    animation: fade 0.14s ease-out both;
  }

  .lightbox::before {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background-image:
      linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255, 255, 255, 0.026) 1px, transparent 1px);
    background-size: 48px 48px;
    mask-image: radial-gradient(circle at center, black 0 58%, transparent 78%);
  }

  .backdrop {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    border: 0;
    border-radius: 0;
    background: transparent;
    cursor: zoom-out;
  }

  .scan-frame {
    position: relative;
    z-index: 1;
    width: min(100%, 94rem);
    height: min(100%, 90dvh);
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    overflow: hidden;
    border: 1px solid color-mix(in srgb, var(--line-strong) 70%, var(--amber));
    border-radius: calc(var(--radius) + 4px);
    background:
      linear-gradient(180deg, rgba(18, 24, 31, 0.96), rgba(6, 9, 13, 0.98)),
      var(--bg);
    box-shadow:
      0 30px 90px rgba(0, 0, 0, 0.58),
      inset 0 1px 0 rgba(255, 255, 255, 0.05);
    animation: raise 0.18s cubic-bezier(0.2, 0.9, 0.3, 1) both;
  }

  .scan-frame::after {
    content: "";
    position: absolute;
    inset: 0.55rem;
    pointer-events: none;
    border: 1px solid rgba(245, 165, 36, 0.14);
    border-radius: calc(var(--radius) - 1px);
  }

  .topbar {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto auto;
    align-items: center;
    gap: 1rem;
    padding: 0.85rem 1rem;
    border-bottom: 1px solid var(--line);
    background:
      repeating-linear-gradient(
        90deg,
        rgba(255, 255, 255, 0.018) 0 1px,
        transparent 1px 6px
      ),
      var(--bg-1);
  }

  .title {
    display: grid;
    gap: 0.16rem;
    min-width: 0;
  }

  .eyebrow {
    font-family: var(--font-mono);
    color: var(--amber);
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
  }

  .title strong {
    min-width: 0;
    overflow: hidden;
    color: var(--text);
    font-size: clamp(0.9rem, 1vw, 1.05rem);
    font-weight: 650;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .status {
    color: var(--teal);
    font-size: 0.78rem;
    letter-spacing: 0.1em;
  }

  .close {
    font-family: var(--font-mono);
    color: var(--text-dim);
    padding: 0.32rem 0.55rem;
    background: var(--bg-inset);
  }

  .close:hover {
    color: var(--amber);
    border-color: color-mix(in srgb, var(--amber) 50%, var(--line-strong));
  }

  .stage {
    position: relative;
    min-height: 0;
    display: grid;
    place-items: center;
    padding: clamp(1rem, 2vw, 1.6rem);
  }

  figure {
    min-width: 0;
    min-height: 0;
    width: 100%;
    height: 100%;
    display: grid;
    grid-template-rows: minmax(0, 1fr) auto;
    gap: 0.8rem;
    margin: 0;
  }

  img {
    align-self: center;
    justify-self: center;
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-sm);
    background: #000;
    box-shadow: 0 16px 50px rgba(0, 0, 0, 0.48);
  }

  figcaption {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    color: var(--text-dim);
    font-size: 0.78rem;
  }

  .hint {
    flex: 0 0 auto;
    color: var(--text-faint);
    font-family: var(--font-mono);
  }

  .nav {
    position: absolute;
    z-index: 2;
    top: 50%;
    width: 2.4rem;
    height: 3.8rem;
    transform: translateY(-50%);
    border-color: rgba(84, 210, 196, 0.28);
    background: color-mix(in srgb, var(--bg-inset) 82%, transparent);
    color: var(--teal);
    font-size: 1.35rem;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.32);
  }

  .nav:hover {
    border-color: var(--teal);
    background: var(--teal-soft);
  }

  .prev {
    left: clamp(0.8rem, 2vw, 1.4rem);
  }

  .next {
    right: clamp(0.8rem, 2vw, 1.4rem);
  }

  @keyframes fade {
    from {
      opacity: 0;
    }
  }

  @keyframes raise {
    from {
      opacity: 0;
      transform: translateY(10px) scale(0.985);
    }
  }

  @media (max-width: 720px) {
    .topbar {
      grid-template-columns: minmax(0, 1fr) auto;
    }

    .status {
      display: none;
    }

    figcaption {
      display: grid;
    }

    .nav {
      top: auto;
      bottom: 3.4rem;
      height: 2.4rem;
      transform: none;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .lightbox,
    .scan-frame {
      animation: none;
    }
  }
</style>
