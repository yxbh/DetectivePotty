<script lang="ts">
  import { formatClock } from "./format";
  import type { LabelClipSummary } from "./types";

  interface Props {
    clips: LabelClipSummary[];
    selectedId: string | null;
    loading: boolean;
    error: string | null;
    filter: "all" | "unlabeled" | "labeled";
    totalCount: number;
    labeledCount: number;
    unlabeledCount: number;
    selectedPosition: number | null;
    onreload: () => void | Promise<void>;
    onselect: (spanId: string) => void;
    onfilter: (filter: "all" | "unlabeled" | "labeled") => void;
    onnextunlabeled: () => void;
  }

  interface ClipGroup {
    key: string;
    scene: string | null;
    size: number;
    camera: string | null;
    items: LabelClipSummary[];
  }

  let {
    clips,
    selectedId,
    loading,
    error,
    filter,
    totalCount,
    labeledCount,
    unlabeledCount,
    selectedPosition,
    onreload,
    onselect,
    onfilter,
    onnextunlabeled,
  }: Props = $props();

  const progressPct = $derived(totalCount ? Math.round((labeledCount / totalCount) * 100) : 0);

  // Group siblings by scene so overlapping track segments stay visually linked.
  const clipGroups = $derived.by<ClipGroup[]>(() => {
    const groups: ClipGroup[] = [];
    const byKey = new Map<string, ClipGroup>();
    for (const c of clips) {
      const key = c.scene_size > 1 && c.scene_id ? `scene:${c.scene_id}` : `solo:${c.span_id}`;
      let group = byKey.get(key);
      if (!group) {
        group = {
          key,
          scene: c.scene_size > 1 ? c.scene_id : null,
          size: c.scene_size,
          camera: c.camera_name,
          items: [],
        };
        byKey.set(key, group);
        groups.push(group);
      }
      group.items.push(c);
    }
    return groups;
  });
</script>

<aside class="clip-list">
  <div class="list-chrome">
    <div class="list-head">
      <h2>Harvested clips</h2>
      <button type="button" class="ghost" onclick={() => void onreload()} title="Reload clip list">↻</button>
    </div>
    <div class="list-tools">
      <div class="filter-tabs" role="tablist" aria-label="Filter harvested clips">
        <button type="button" role="tab" class:active={filter === "all"} aria-selected={filter === "all"} onclick={() => onfilter("all")}>
          All <span>{totalCount}</span>
        </button>
        <button type="button" role="tab" class:active={filter === "unlabeled"} aria-selected={filter === "unlabeled"} onclick={() => onfilter("unlabeled")}>
          Todo <span>{unlabeledCount}</span>
        </button>
        <button type="button" role="tab" class:active={filter === "labeled"} aria-selected={filter === "labeled"} onclick={() => onfilter("labeled")}>
          Done <span>{labeledCount}</span>
        </button>
      </div>
      <button
        type="button"
        class="next-unlabeled"
        disabled={unlabeledCount === 0}
        onclick={onnextunlabeled}
        title="Open the next unlabeled clip (N)"
      >
        next unlabeled <kbd>N</kbd>
      </button>
      <div class="list-progress mono" title={`${labeledCount} of ${totalCount} clips have at least one label range`}>
        <span>{labeledCount}/{totalCount} labeled</span>
        <span>
          {#if selectedPosition != null}
            {selectedPosition + 1}/{clips.length} shown
          {:else}
            {clips.length} shown
          {/if}
        </span>
        <div class="progress-track" aria-hidden="true">
          <div class="progress-fill" style="width: {progressPct}%"></div>
        </div>
      </div>
    </div>
  </div>
  {#if loading}
    <p class="muted pad">Loading clips…</p>
  {:else if error}
    <p class="error pad">{error}</p>
  {:else if clips.length === 0 && totalCount === 0}
    <p class="muted pad">
      No harvested clips found. Run <code>detectivepotty harvest</code> to populate the harvest dir.
    </p>
  {:else if clips.length === 0}
    <p class="muted pad">No clips match this label filter.</p>
  {:else}
    {#each clipGroups as group (group.key)}
      {#if group.scene}
        <div
          class="scene-head"
          title="Same camera + overlapping time window — {group.size} detection segments (often the same dog re-detected after the tracker lost it, not confirmed separate dogs). Label each on its own clip."
        >
          <span class="scene-cam">{group.camera ?? "camera"}</span>
          <span class="scene-when">{formatClock(group.items[0].span_start_utc)}</span>
          <span class="scene-badge">×{group.size} segments</span>
        </div>
      {/if}
      <ul class:scene-group={group.scene}>
        {#each group.items as clip (clip.span_id)}
          <li>
            <button
              type="button"
              class:active={clip.span_id === selectedId}
              onclick={() => onselect(clip.span_id)}
              title={`${clip.camera_name ?? clip.camera_id ?? "unknown camera"}\n${clip.source_id}\n${formatClock(clip.span_start_utc)} → ${formatClock(clip.span_end_utc)}`}
            >
              <span class="row1">
                <span class="cam">{clip.camera_name ?? clip.camera_id ?? clip.source_id}</span>
                <span
                  class="badge"
                  class:done={clip.labeled}
                  title={clip.labeled
                    ? `${clip.n_trainable_ranges} trainable / ${clip.n_ranges} ranges`
                    : "Not labeled yet"}
                >
                  {clip.labeled ? `✓${clip.n_trainable_ranges}` : "·"}
                </span>
              </span>
              <span class="row2">
                <span class="when" title="Clip start (local time)"
                  >{formatClock(clip.span_start_utc)}</span
                >
                <span class="dur" title="Clip duration">{clip.duration_s.toFixed(1)}s</span>
                <span
                  class="trk"
                  title="Track segment this clip follows — its boxes/labels bind to this track"
                  >T{clip.track_id ?? "?"}</span
                >
              </span>
              {#if clip.labeled && (clip.behaviors.length || clip.dogs.length)}
                <span class="row3">
                  {#each clip.behaviors as b (b)}
                    <span class="chip b-{b}">{b}</span>
                  {/each}
                  {#each clip.dogs as d (d)}
                    <span class="chip dog">{d}</span>
                  {/each}
                </span>
              {/if}
            </button>
          </li>
        {/each}
      </ul>
    {/each}
  {/if}
</aside>

<style>
  .clip-list {
    border-right: 1px solid var(--line-strong);
    overflow-y: auto;
    min-height: 0;
  }
  .list-chrome {
    position: sticky;
    top: 0;
    background: var(--bg, #0c1018);
    border-bottom: 1px solid var(--line-strong);
    z-index: 1;
  }
  .list-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.4rem 0.6rem 0.25rem;
  }
  .list-head h2 {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0;
    color: var(--text-dim);
  }
  .list-tools {
    display: grid;
    gap: 0.45rem;
    padding: 0 0.6rem 0.55rem;
  }
  .filter-tabs {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.2rem;
    padding: 0.16rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-pill);
    background: var(--bg-inset);
  }
  .filter-tabs button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.25rem;
    min-width: 0;
    padding: 0.28rem 0.3rem;
    border: 0;
    border-radius: var(--radius-pill);
    background: transparent;
    color: var(--text-dim);
    font-size: 0.68rem;
  }
  .filter-tabs button.active {
    background: var(--amber);
    color: #1a1204;
  }
  .filter-tabs span {
    font-family: var(--font-mono);
    font-size: 0.62rem;
  }
  .next-unlabeled {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.35rem;
    padding: 0.34rem 0.5rem;
    border-color: color-mix(in srgb, var(--teal) 40%, var(--line-strong));
    background: var(--teal-soft);
    color: var(--teal);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .next-unlabeled kbd {
    background: rgba(0, 0, 0, 0.18);
    color: inherit;
  }
  .list-progress {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 0.25rem 0.5rem;
    color: var(--text-faint);
    font-size: 0.64rem;
  }
  .progress-track {
    grid-column: 1 / -1;
    height: 3px;
    border-radius: var(--radius-pill);
    background: var(--bg-3);
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--beh-pee), var(--teal));
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .scene-head {
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
    padding: 0.3rem 0.6rem 0.15rem;
    font-size: 0.68rem;
    color: var(--text-dim);
    border-top: 1px solid var(--line-strong);
  }
  .scene-cam {
    font-weight: 600;
    color: #b9c6d6;
  }
  .scene-when {
    margin-left: auto;
  }
  .scene-badge {
    background: #3a2c12;
    color: #f0c869;
    border-radius: 999px;
    padding: 0.02rem 0.4rem;
  }
  ul.scene-group {
    border-left: 2px solid #3a2c12;
    margin-left: 0.35rem;
  }
  li {
    margin: 0;
  }
  li button {
    display: flex;
    flex-direction: column;
    gap: 0.12rem;
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    border-bottom: 1px solid var(--line-strong);
    color: inherit;
    padding: 0.32rem 0.6rem;
    cursor: pointer;
  }
  li button:hover {
    background: var(--hover, #131c28);
  }
  li button.active {
    background: var(--hover, #16202e);
    box-shadow: inset 3px 0 0 var(--amber);
  }
  .row1 {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.4rem;
  }
  .cam {
    font-size: 0.82rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .row2 {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.7rem;
    color: var(--text-dim);
  }
  .row2 .trk {
    margin-left: auto;
  }
  .row3 {
    display: flex;
    flex-wrap: wrap;
    gap: 0.2rem;
    margin-top: 0.1rem;
  }
  .chip {
    font-size: 0.62rem;
    padding: 0.02rem 0.32rem;
    border-radius: 4px;
    background: var(--bg-3);
    color: var(--text-dim);
  }
  .chip.b-pee {
    background: var(--beh-pee);
    color: #1a1204;
  }
  .chip.b-poop {
    background: var(--beh-poop);
    color: #1a1204;
  }
  .chip.b-not_potty {
    background: color-mix(in srgb, var(--beh-not-potty) 42%, var(--bg-3));
    color: var(--text);
  }
  .chip.b-excluded {
    background: color-mix(in srgb, var(--beh-excluded) 48%, var(--bg-3));
    color: #fdd;
  }
  .chip.dog {
    background: var(--beh-dog);
    color: #dfe;
  }
  .badge {
    font-size: 0.66rem;
    padding: 0.03rem 0.38rem;
    border-radius: 999px;
    background: var(--bg-3);
    color: var(--text-dim);
    flex: none;
  }
  .badge.done {
    background: #1f7a3f;
    color: #d6ffe2;
  }
  .pad {
    padding: 1rem;
  }
  .muted {
    color: var(--text-dim);
  }
  .error {
    color: #ff6b6b;
  }
  .ghost {
    background: transparent;
    border: 1px solid var(--line-strong);
    color: inherit;
    border-radius: 6px;
    cursor: pointer;
    padding: 0.12rem 0.38rem;
  }
  code {
    background: var(--bg-3);
    padding: 0.05rem 0.3rem;
    border-radius: 4px;
    font-size: 0.85em;
  }
</style>
