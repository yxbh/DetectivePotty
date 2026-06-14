<script lang="ts">
  import { formatClock } from "./format";
  import type { LabelClipSummary } from "./types";

  interface Props {
    clips: LabelClipSummary[];
    selectedId: string | null;
    loading: boolean;
    error: string | null;
    onreload: () => void | Promise<void>;
    onselect: (spanId: string) => void;
  }

  interface ClipGroup {
    key: string;
    scene: string | null;
    size: number;
    camera: string | null;
    items: LabelClipSummary[];
  }

  let { clips, selectedId, loading, error, onreload, onselect }: Props = $props();

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
  <div class="list-head">
    <h2>Harvested clips</h2>
    <button type="button" class="ghost" onclick={() => void onreload()} title="Reload clip list">↻</button>
  </div>
  {#if loading}
    <p class="muted pad">Loading clips…</p>
  {:else if error}
    <p class="error pad">{error}</p>
  {:else if clips.length === 0}
    <p class="muted pad">
      No harvested clips found. Run <code>detectivepotty harvest</code> to populate the harvest dir.
    </p>
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
  .list-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.4rem 0.6rem;
    position: sticky;
    top: 0;
    background: var(--bg, #0c1018);
    border-bottom: 1px solid var(--line-strong);
    z-index: 1;
  }
  .list-head h2 {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0;
    color: var(--text-dim);
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
    background: #f1cf5b;
    color: #1a1204;
  }
  .chip.b-poop {
    background: #c08a55;
    color: #1a1204;
  }
  .chip.b-not_potty {
    background: #3a4150;
    color: var(--text);
  }
  .chip.b-excluded {
    background: #5a2f42;
    color: #fdd;
  }
  .chip.dog {
    background: #2f5d4a;
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
