/**
 * Shared visual language for detection-box overlays across the three surfaces
 * that draw them in different render tech: Tune (`<canvas>`), Label (SVG) and
 * Review (HTML crop badge). Each surface keeps its own renderer — this module
 * only centralises the *palette*, *label format* and *legibility* so a "dog
 * 0.82" box reads identically everywhere.
 *
 * The hex values intentionally mirror the `app.css` `:root` tokens (canvas
 * can't read CSS custom properties, so it needs literals; SVG/HTML CSS should
 * use the matching `var(--…)` token). Keep them in sync.
 */

/** Kept dog box / followed (own) track — matches `--green`. */
export const BOX_DOG = "#4fc480";
/** Kept non-dog alias box (sheep/zebra/cow/… accepted as a dog) — `--teal`. */
export const BOX_ALIAS = "#54d2c4";
/** Below-threshold / weak box — `--red`. */
export const BOX_WEAK = "#e5544b";
/** Sibling track (Label) and scene-only objects (Tune) — `--amber`. */
export const BOX_SIBLING = "#f5a524";

/** Dark halo stroked behind canvas label text so it stays legible on any frame. */
const LABEL_OUTLINE = "rgba(0, 0, 0, 0.82)";

/** A detected class that isn't a literal "dog" — an accepted alias read. */
export function isAliasClass(className: string): boolean {
  return className.toLowerCase() !== "dog";
}

/**
 * Canonical stroke/label colour for a single-class detection box: weak (red)
 * when gated out, teal for an accepted alias, green for a true dog.
 */
export function classBoxColor(className: string, kept: boolean): string {
  if (!kept) return BOX_WEAK;
  return isAliasClass(className) ? BOX_ALIAS : BOX_DOG;
}

/** `"dog 0.82"` — the class + confidence label shared by every surface. */
export function formatDetLabel(className: string, confidence: number): string {
  return `${className} ${confidence.toFixed(2)}`;
}

/** `"#10 0.82"` or `"#10 dog 0.82"` for a tracked box (class optional). */
export function formatTrackLabel(
  trackId: string | number,
  confidence: number,
  className?: string,
): string {
  const cls = className ? `${className} ` : "";
  return `#${trackId} ${cls}${confidence.toFixed(2)}`;
}

/**
 * Box-label font size in pixels (canvas) / user-space units (SVG viewBox),
 * scaled off the larger image edge so it reads consistently regardless of
 * source resolution. Both Tune and Label previously hand-rolled ≈edge/45.
 */
export function boxLabelFontPx(maxEdge: number): number {
  return Math.max(12, Math.round(maxEdge / 44));
}

/**
 * Deterministic vivid colour per track id (hash → HSL hue) so the same id keeps
 * its colour across frames and runs. Shared by any surface that colours boxes
 * by track.
 */
export function trackColor(id: string): string {
  let h = 0;
  for (let i = 0; i < id.length; i++) {
    h = (h * 31 + id.charCodeAt(i)) >>> 0;
  }
  return `hsl(${h % 360}, 85%, 58%)`;
}

/**
 * Draw an outlined box label on a canvas at the box's top-left, flipping below
 * the top edge when there's no room above. Strokes a dark halo then fills the
 * colour, matching the legibility of the SVG (paint-order stroke) and HTML
 * (pill background) labels. Self-contained: saves/restores the context state it
 * touches so callers can keep their own stroke/dash settings.
 */
export function drawCanvasBoxLabel(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  boxTop: number,
  color: string,
  fontPx: number,
): void {
  ctx.save();
  ctx.font = `${fontPx}px ui-monospace, monospace`;
  ctx.textBaseline = "bottom";
  ctx.lineJoin = "round";
  const ty = boxTop > fontPx + 4 ? boxTop - 2 : boxTop + fontPx + 2;
  ctx.lineWidth = Math.max(2, Math.round(fontPx / 7));
  ctx.strokeStyle = LABEL_OUTLINE;
  ctx.strokeText(text, x, ty);
  ctx.fillStyle = color;
  ctx.fillText(text, x, ty);
  ctx.restore();
}
