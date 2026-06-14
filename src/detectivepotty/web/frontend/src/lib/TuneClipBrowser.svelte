<script lang="ts">
  import type { TuneEntry, TuneListing } from "./types";

  interface Props {
    listing: TuneListing | null;
    loading: boolean;
    error: string | null;
    selectedPath: string | null;
    onloadpath: (path: string) => void | Promise<void>;
    onselectvideo: (path: string, name: string) => void | Promise<void>;
  }

  let { listing, loading, error, selectedPath, onloadpath, onselectvideo }: Props = $props();

  function openEntry(entry: TuneEntry): void {
    if (entry.kind === "dir") {
      void onloadpath(entry.path);
      return;
    }
    void onselectvideo(entry.path, entry.name);
  }

  function goUp(): void {
    if (!listing || listing.parent === null) {
      return;
    }
    void onloadpath(listing.parent);
  }
</script>

<aside class="browser">
  <div class="browser-head">
    <span class="eyebrow mono">CLIPS</span>
    {#if listing && listing.parent !== null}
      <button type="button" class="up" onclick={goUp} title="Up one level">↑ up</button>
    {/if}
  </div>
  {#if listing}
    <div class="crumb mono" title={listing.path}>
      {listing.path || "roots"}
    </div>
  {/if}
  <div class="entries">
    {#if loading}
      <p class="muted">Loading…</p>
    {:else if error}
      <p class="error">{error}</p>
    {:else if listing && listing.entries.length === 0}
      <p class="muted">No clips or folders here.</p>
    {:else if listing}
      {#each listing.entries as entry (entry.path)}
        <button
          type="button"
          class="entry"
          class:active={entry.path === selectedPath}
          onclick={() => openEntry(entry)}
        >
          <span class="icon">{entry.kind === "dir" ? "📁" : "🎬"}</span>
          <span class="name">{entry.name}</span>
        </button>
      {/each}
    {/if}
  </div>
</aside>

<style>
  .browser {
    grid-area: browser;
    display: flex;
    flex-direction: column;
    min-height: 0;
    border: 1px solid var(--line, #243042);
    border-radius: 10px;
    background: var(--bg-1, #141a24);
    overflow: hidden;
  }

  .browser-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0.75rem;
    border-bottom: 1px solid var(--line, #243042);
  }

  .eyebrow {
    font-size: 0.6rem;
    letter-spacing: 0.28em;
    color: var(--amber, #f0b35a);
  }

  .up {
    font-size: 0.72rem;
    background: transparent;
    border: 1px solid var(--line-strong, #324056);
    color: var(--text, #d8e0ec);
    border-radius: 6px;
    padding: 0.15rem 0.45rem;
    cursor: pointer;
  }

  .crumb {
    font-size: 0.68rem;
    color: var(--muted, #8a97a8);
    padding: 0.4rem 0.75rem;
    border-bottom: 1px solid var(--line, #243042);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    direction: rtl;
    text-align: left;
  }

  .entries {
    overflow-y: auto;
    min-height: 0;
    padding: 0.35rem;
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .entry {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    color: var(--text, #d8e0ec);
    padding: 0.4rem 0.5rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.82rem;
  }

  .entry:hover {
    background: var(--bg-2, #1b2330);
  }

  .entry.active {
    background: var(--accent-dim, #1d3346);
    color: #fff;
  }

  .entry .name {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .muted {
    color: var(--muted, #8a97a8);
  }

  .error {
    color: var(--red, #ff6b6b);
  }
</style>
