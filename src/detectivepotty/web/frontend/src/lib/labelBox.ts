import type { LabelTrackBox } from "./types";

export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** A box resolved for an arbitrary clip frame from sparse sampled detections. */
export interface ResolvedBox {
  bbox: BBox;
  /** True when the box was linearly interpolated between two samples. */
  interpolated: boolean;
  /** True when `frame` sits before the first / after the last sample (held
   *  at the nearest end box — draw it dashed/faint, it is a guess). */
  extrapolated: boolean;
  /** Confidence of the nearest underlying sample (for tinting). */
  confidence: number;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Resolve the dog box for `frame` from `boxes` (sampled detections, ascending
 * by `clip_frame_idx`).
 *
 * Harvest records a detection only every `sample_every` frames, so naively
 * snapping to the nearest sample makes the overlay lag/lead the moving dog —
 * worst at a clip's start/end. This linearly interpolates between the two
 * surrounding samples instead, and flags frames outside the sampled range as
 * `extrapolated` so the caller can de-emphasise a held end box rather than
 * present a stale box as ground truth.
 *
 * Pure + side-effect free so it is trivially testable offline.
 */
export function boxAtFrame(
  boxes: readonly LabelTrackBox[] | undefined,
  frame: number,
): ResolvedBox | null {
  if (!boxes || boxes.length === 0) return null;

  const first = boxes[0];
  const last = boxes[boxes.length - 1];

  if (frame <= first.clip_frame_idx) {
    return {
      bbox: { ...first.bbox },
      interpolated: false,
      extrapolated: frame < first.clip_frame_idx,
      confidence: first.confidence,
    };
  }
  if (frame >= last.clip_frame_idx) {
    return {
      bbox: { ...last.bbox },
      interpolated: false,
      extrapolated: frame > last.clip_frame_idx,
      confidence: last.confidence,
    };
  }

  // Find the bracketing pair [lo, hi] with lo.frame < frame <= hi.frame.
  let lo = first;
  let hi = last;
  for (let i = 1; i < boxes.length; i += 1) {
    if (boxes[i].clip_frame_idx >= frame) {
      hi = boxes[i];
      lo = boxes[i - 1];
      break;
    }
  }

  if (frame === lo.clip_frame_idx) {
    return { bbox: { ...lo.bbox }, interpolated: false, extrapolated: false, confidence: lo.confidence };
  }
  if (frame === hi.clip_frame_idx) {
    return { bbox: { ...hi.bbox }, interpolated: false, extrapolated: false, confidence: hi.confidence };
  }

  const span = hi.clip_frame_idx - lo.clip_frame_idx;
  const t = span > 0 ? (frame - lo.clip_frame_idx) / span : 0;
  return {
    bbox: {
      x1: lerp(lo.bbox.x1, hi.bbox.x1, t),
      y1: lerp(lo.bbox.y1, hi.bbox.y1, t),
      x2: lerp(lo.bbox.x2, hi.bbox.x2, t),
      y2: lerp(lo.bbox.y2, hi.bbox.y2, t),
    },
    interpolated: true,
    extrapolated: false,
    confidence: t < 0.5 ? lo.confidence : hi.confidence,
  };
}
