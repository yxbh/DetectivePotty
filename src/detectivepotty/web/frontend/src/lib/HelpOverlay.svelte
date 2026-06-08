<script lang="ts">
  interface Props {
    open: boolean;
    onclose: () => void;
  }

  let { open, onclose }: Props = $props();

  const SECTIONS: Array<{ title: string; rows: Array<[string[], string]> }> = [
    {
      title: "Navigate",
      rows: [
        [["j", "↓"], "Next event"],
        [["k", "↑"], "Previous event"],
        [["n"], "Next unlabeled"],
        [["N"], "Previous unlabeled"],
        [["g"], "Jump to top"],
        [["G"], "Jump to bottom"],
      ],
    },
    {
      title: "Label",
      rows: [
        [["1"], "Pee"],
        [["2"], "Poop"],
        [["3"], "Not potty"],
        [["0"], "Unknown"],
        [["r"], "Stage status · rejected"],
        [["u"], "Stage status · uncertain"],
        [["s", "↵"], "Save label"],
      ],
    },
    {
      title: "Dog",
      rows: [
        [["⇧1", "⇧2", "…"], "Assign Nth roster dog"],
        [["⇧0"], "Unassign dog"],
      ],
    },
    {
      title: "General",
      rows: [
        [["/"], "Focus camera filter"],
        [["Space"], "Play / pause clip"],
        [["a"], "Toggle auto-advance"],
        [["v"], "Toggle Live feed"],
        [["?"], "Toggle this help"],
        [["Esc"], "Close / blur / leave Live"],
      ],
    },
  ];
</script>

{#if open}
  <div
    class="scrim"
    role="button"
    tabindex="-1"
    aria-label="Close help"
    onclick={onclose}
    onkeydown={(e) => e.key === "Enter" && onclose()}
  ></div>
  <div class="sheet" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts">
    <header>
      <h2>Keyboard shortcuts</h2>
      <button type="button" class="close" onclick={onclose} aria-label="Close">esc</button>
    </header>
    <div class="cols">
      {#each SECTIONS as section (section.title)}
        <section>
          <h3>{section.title}</h3>
          <dl>
            {#each section.rows as [keys, desc] (desc)}
              <div class="row">
                <dt>
                  {#each keys as key, i (key)}
                    {#if i > 0}<span class="or">/</span>{/if}
                    <kbd>{key}</kbd>
                  {/each}
                </dt>
                <dd>{desc}</dd>
              </div>
            {/each}
          </dl>
        </section>
      {/each}
    </div>
  </div>
{/if}

<style>
  .scrim {
    position: fixed;
    inset: 0;
    background: rgba(4, 5, 6, 0.72);
    backdrop-filter: blur(3px);
    z-index: 40;
    animation: fade 0.16s ease;
  }

  .sheet {
    position: fixed;
    z-index: 41;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: min(46rem, calc(100vw - 2rem));
    max-height: calc(100vh - 3rem);
    overflow-y: auto;
    background: var(--bg-1);
    border: 1px solid var(--line-strong);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    animation: pop 0.18s cubic-bezier(0.2, 0.9, 0.3, 1);
  }

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.25rem;
    border-bottom: 1px solid var(--line);
  }

  header h2 {
    margin: 0;
    font-size: 1.05rem;
  }

  .close {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    padding: 0.25rem 0.5rem;
  }

  .cols {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr));
    gap: 1.25rem;
    padding: 1.25rem;
  }

  section h3 {
    margin: 0 0 0.6rem;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--amber);
  }

  dl {
    margin: 0;
    display: grid;
    gap: 0.45rem;
  }

  .row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
  }

  dt {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    flex: 0 0 auto;
  }

  .or {
    color: var(--text-faint);
    font-size: 0.7rem;
  }

  dd {
    margin: 0;
    color: var(--text-dim);
    font-size: 0.85rem;
    text-align: right;
  }

  @keyframes fade {
    from {
      opacity: 0;
    }
  }

  @keyframes pop {
    from {
      opacity: 0;
      transform: translate(-50%, -48%) scale(0.98);
    }
  }
</style>
