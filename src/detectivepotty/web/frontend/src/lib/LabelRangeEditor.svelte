<script lang="ts">
  import type { LabelPresentTrack, LabelRangeItem, LabelVocabulary } from "./types";
  import { frameToSeconds } from "./video/frameTime";

  interface Props {
    siblingTracks: LabelPresentTrack[];
    markIn: number | null;
    markOut: number | null;
    pendingBehavior: string;
    pendingDog: string;
    vocabulary: LabelVocabulary;
    ranges: LabelRangeItem[];
    fps: number;
    dirty: boolean;
    saving: boolean;
    saveStatus: string | null;
    currentFrame: number;
    onselectclip: (spanId: string) => void;
    onsetmarkin: () => void;
    onsetmarkout: () => void;
    onbehavior: (behavior: string) => void;
    ondog: (dog: string) => void;
    onaddrange: () => void;
    onsave: () => void | Promise<void>;
    onseekrange: (range: LabelRangeItem) => void;
    onupdaterange: (idx: number, patch: Partial<LabelRangeItem>) => void;
    onretimerange: (idx: number, edge: "start" | "end") => void;
    ondeleterange: (idx: number) => void;
  }

  let {
    siblingTracks,
    markIn,
    markOut,
    pendingBehavior,
    pendingDog,
    vocabulary,
    ranges,
    fps,
    dirty,
    saving,
    saveStatus,
    currentFrame,
    onselectclip,
    onsetmarkin,
    onsetmarkout,
    onbehavior,
    ondog,
    onaddrange,
    onsave,
    onseekrange,
    onupdaterange,
    onretimerange,
    ondeleterange,
  }: Props = $props();

  const dogKeyHint = $derived.by<Record<string, string>>(() => {
    const map: Record<string, string> = {};
    vocabulary.dogs.forEach((dog, i) => {
      if (i < 9) {
        map[dog] = `\u21e7${i + 1}`;
      }
    });
    return map;
  });

  function fmtFrame(frame: number): string {
    return `${frame} \u00b7 ${frameToSeconds(frame, fps).toFixed(2)}s`;
  }
</script>

<div class="editor-col">
  {#if siblingTracks.length}
    <div class="siblings">
      <span
        class="lbl"
        title="Other track segments overlapping this clip's window — often the same dog re-detected, not confirmed separate dogs."
        >Other segments here</span
      >
      <div class="sib-chips">
        {#each siblingTracks as track (track.span_id + ':' + track.track_id)}
          <button
            type="button"
            class="sib-chip"
            onclick={() => onselectclip(track.span_id)}
            title="Open the clip that follows this segment (labels bind to a clip's own track)"
          >
            → Track {track.track_id}
          </button>
        {/each}
      </div>
    </div>
  {/if}

  <div class="marks">
    <button type="button" onclick={onsetmarkin} title="Mark range start at current frame (I)">
      In <span class="mono">{markIn ?? "—"}</span>
    </button>
    <button type="button" onclick={onsetmarkout} title="Mark range end at current frame (O)">
      Out <span class="mono">{markOut ?? "—"}</span>
    </button>
  </div>

  <div class="pickers">
    <div class="picker">
      <span class="lbl">Behavior</span>
      <div class="seg">
        {#each vocabulary.behaviors as behavior, i (behavior)}
          <button
            type="button"
            class:active={pendingBehavior === behavior}
            onclick={() => onbehavior(behavior)}
            title={`Set behavior to ${behavior}${i < 9 ? ` (${i + 1})` : ""}`}
          >
            {behavior}{#if i < 9}<span class="kh">{i + 1}</span>{/if}
          </button>
        {/each}
      </div>
    </div>
    <div class="picker">
      <span class="lbl">Dog</span>
      <div class="seg">
        {#each vocabulary.dogs as dog (dog)}
          <button
            type="button"
            class:active={pendingDog === dog}
            onclick={() => ondog(dog)}
            title={`Set dog to ${dog}${dogKeyHint[dog] ? ` (${dogKeyHint[dog]})` : ""}`}
          >
            {dog}{#if dogKeyHint[dog]}<span class="kh">{dogKeyHint[dog]}</span>{/if}
          </button>
        {/each}
      </div>
    </div>
  </div>

  <div class="actions">
    <button type="button" class="primary" onclick={onaddrange} title="Add a range from In→Out (Enter)">
      + Add range
    </button>
    <button
      type="button"
      class="save"
      class:dirty
      disabled={saving || !dirty}
      onclick={() => void onsave()}
      title="Save labels.json (S)"
    >
      {saving ? "Saving…" : dirty ? "Save (S)" : "Saved"}
    </button>
    {#if saveStatus && saveStatus !== "saved"}
      <span class="error small">{saveStatus}</span>
    {:else if saveStatus === "saved"}
      <span class="ok small">✓ saved</span>
    {/if}
  </div>

  <div class="ranges">
    <h3>Ranges ({ranges.length})</h3>
    {#if ranges.length === 0}
      <p class="muted small">No ranges yet. Mark In/Out, pick behavior + dog, then Add.</p>
    {:else}
      <ul>
        {#each ranges as range, idx (idx)}
          <li>
            <button
              type="button"
              class="seek"
              onclick={() => onseekrange(range)}
              title="Seek to range start"
            >
              <span class="r-frames mono"
                >{fmtFrame(range.start_frame)} → {fmtFrame(range.end_frame)}</span
              >
            </button>
            <div class="retime" aria-label="Retime range">
              <button
                type="button"
                onclick={() => onretimerange(idx, "start")}
                title={`Set range start to current frame ${currentFrame}`}
              >
                start now
              </button>
              <button
                type="button"
                onclick={() => onretimerange(idx, "end")}
                title={`Set range end to current frame ${currentFrame}`}
              >
                end now
              </button>
            </div>
            <select
              class="r-sel"
              value={range.behavior}
              onchange={(event) =>
                onupdaterange(idx, { behavior: (event.target as HTMLSelectElement).value })}
              title="Behavior for this range"
            >
              {#each vocabulary.behaviors as behavior (behavior)}
                <option value={behavior}>{behavior}</option>
              {/each}
            </select>
            <select
              class="r-sel"
              value={range.dog}
              onchange={(event) =>
                onupdaterange(idx, { dog: (event.target as HTMLSelectElement).value })}
              title="Dog for this range"
            >
              {#each vocabulary.dogs as dog (dog)}
                <option value={dog}>{dog}</option>
              {/each}
            </select>
            <button
              type="button"
              class="del"
              onclick={() => ondeleterange(idx)}
              aria-label="Delete range"
              title="Delete range"
              >×</button
            >
          </li>
        {/each}
      </ul>
    {/if}
  </div>

  <p class="legend mono">
    Space play · ←/→ step (⇧×10) · I/O mark · 1-4 behavior · ⇧1-4 dog · Enter add · S save ·
    j/k clip · n/N unlabeled
  </p>
</div>

<style>
  .editor-col {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    min-height: 0;
    overflow-y: auto;
  }
  .siblings {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  .sib-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
  }
  .sib-chip {
    background: #3a2c12;
    border: 1px solid #5a4520;
    color: #f0c869;
    border-radius: 6px;
    padding: 0.2rem 0.5rem;
    cursor: pointer;
    font-size: 0.74rem;
  }
  .marks {
    display: flex;
    gap: 0.4rem;
  }
  .marks button {
    flex: 1;
    background: var(--bg-3);
    border: none;
    color: inherit;
    border-radius: 6px;
    padding: 0.35rem;
    cursor: pointer;
  }
  .pickers {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .picker {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .picker .lbl {
    width: 4rem;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-dim);
    flex: none;
  }
  .lbl {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-dim);
  }
  .seg {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
  }
  .seg button {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    background: var(--bg-3);
    border: 1px solid transparent;
    color: inherit;
    border-radius: 6px;
    padding: 0.22rem 0.5rem;
    cursor: pointer;
    font-size: 0.78rem;
  }
  .seg button.active {
    background: var(--amber);
    border-color: var(--amber-bright);
    color: #1a1204;
  }
  .kh {
    font-size: 0.6rem;
    opacity: 0.7;
    background: rgba(0, 0, 0, 0.25);
    border-radius: 3px;
    padding: 0 0.2rem;
  }
  .actions {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  .actions .primary {
    background: #2f7d4f;
    border: none;
    color: #eafff2;
    border-radius: 6px;
    padding: 0.4rem 0.7rem;
    cursor: pointer;
    font-weight: 600;
  }
  .actions .save {
    background: var(--bg-3);
    border: none;
    color: inherit;
    border-radius: 6px;
    padding: 0.4rem 0.7rem;
    cursor: pointer;
  }
  .actions .save.dirty {
    background: var(--amber);
    color: #1a1204;
  }
  .actions .save:disabled {
    opacity: 0.6;
    cursor: default;
  }
  .ranges {
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .ranges h3 {
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-dim);
    margin: 0.2rem 0;
  }
  .ranges ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  .ranges li {
    display: flex;
    align-items: stretch;
    flex-wrap: wrap;
    gap: 0.25rem;
  }
  .ranges .seek {
    flex: 1 1 100%;
    display: flex;
    align-items: center;
    background: var(--hover, #131c28);
    border: 1px solid var(--line-strong);
    border-radius: 6px;
    padding: 0.3rem 0.45rem;
    cursor: pointer;
    color: inherit;
    min-width: 0;
  }
  .retime {
    flex: 1 1 7rem;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.15rem;
  }
  .retime button {
    background: var(--accent-soft);
    border: 1px solid color-mix(in srgb, var(--accent) 35%, var(--line-strong));
    color: var(--accent);
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.62rem;
    padding: 0.15rem 0.2rem;
    white-space: nowrap;
  }
  .r-frames {
    font-size: 0.7rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .r-sel {
    flex: 1 1 5rem;
    background: var(--bg-3);
    color: inherit;
    border: 1px solid var(--line-strong);
    border-radius: 6px;
    font-size: 0.7rem;
    padding: 0.15rem 0.2rem;
    min-width: 0;
  }
  .ranges .del {
    background: transparent;
    border: 1px solid var(--line-strong);
    color: var(--red);
    border-radius: 6px;
    width: 1.8rem;
    cursor: pointer;
    font-size: 1rem;
  }
  .legend {
    font-size: 0.68rem;
    color: var(--text-dim);
    border-top: 1px solid var(--line-strong);
    padding-top: 0.4rem;
    margin-top: auto;
  }
  .small {
    font-size: 0.72rem;
  }
  .muted {
    color: var(--text-dim);
  }
  .error {
    color: #ff6b6b;
  }
  .ok {
    color: var(--green);
  }
</style>
