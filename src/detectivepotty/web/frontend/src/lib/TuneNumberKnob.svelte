<script lang="ts">
  interface Props {
    label: string;
    value: number;
    min: number | string;
    max: number | string;
    step: number | string;
    title: string;
    oninput?: () => void;
  }

  let { label, value = $bindable(), min, max, step, title, oninput }: Props = $props();
</script>

<label {title}>
  {label}
  <input
    type="number"
    {min}
    {max}
    {step}
    bind:value
    oninput={() => oninput?.()}
  />
</label>

<style>
  label {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    color: var(--muted, #8a97a8);
  }

  input {
    width: 4.2rem;
    background: var(--bg-1, #141a24);
    border: 1px solid var(--line-strong, #324056);
    color: var(--text, #d8e0ec);
    border-radius: 5px;
    padding: 0.2rem 0.3rem;
    font-family: ui-monospace, monospace;
    font-size: 0.74rem;
  }
</style>
