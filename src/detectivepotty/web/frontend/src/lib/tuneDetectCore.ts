import type { TuneDetection, TunePose } from "./types";

// How many frames Shift+Arrow skips.
export const SKIP_N = 10;
// Cap concurrent detection fetches. Server serializes inference under a lock,
// so a deep queue just delays interactive seeks; 2 keeps the pipe busy while
// a fresh seek's frame can still jump the queue within one inference slot.
export const MAX_INFLIGHT = 2;
export const RANGE_BATCH = 8;
export const MAX_POSE_INFLIGHT = 1;
export const POSE_RANGE_BATCH = 8;
export const URGENT_WINDOW = 30;
export const DEFAULT_FLOOR = 0.05;
export const ULTRA_TRACK_HIGH_DEFAULT = 0.25;
export const ULTRA_TRACK_LOW_DEFAULT = 0.1;
export const ULTRA_NEW_TRACK_DEFAULT = 0.25;
export const ULTRA_TRACK_BUFFER_DEFAULT = 30;
export const ULTRA_MATCH_DEFAULT = 0.8;
export const ULTRA_PROXIMITY_DEFAULT = 0.5;
export const ULTRA_APPEARANCE_DEFAULT = 0.25;

// Skeleton edges by raw DeepLabCut keypoint name. Drawn only when both
// endpoints are present, so the mock subset and full SuperAnimal backend both
// render sensibly.
export const POSE_EDGES: Array<[string, string]> = [
  ["nose", "upper_jaw"],
  ["nose", "neck_base"],
  ["neck_base", "back_base"],
  ["back_base", "neck_end"],
  ["back_base", "back_middle"],
  ["back_middle", "back_end"],
  ["back_end", "tail_base"],
  ["tail_base", "tail_end"],
  ["back_base", "front_left_paw"],
  ["back_base", "front_right_paw"],
  ["front_left_thai", "front_left_knee"],
  ["front_left_knee", "front_left_paw"],
  ["front_right_thai", "front_right_knee"],
  ["front_right_knee", "front_right_paw"],
  ["back_end", "back_left_paw"],
  ["back_end", "back_right_paw"],
  ["back_left_thai", "back_left_knee"],
  ["back_left_knee", "back_left_paw"],
  ["back_right_thai", "back_right_knee"],
  ["back_right_knee", "back_right_paw"],
];

export type OverlayMode = "boxes" | "pose" | "both";

export interface BufferEntry {
  scopeKey: string;
  detections: TuneDetection[];
  pose: TunePose[];
  // Whether the decoupled pose pass has RUN for this frame (regardless of
  // whether it produced keypoints — empty is a valid result, marked posed so it
  // isn't retried forever).
  posed: boolean;
}

export interface ZoomCard {
  det: TuneDetection;
  pose: TunePose | null;
  kept: boolean;
}

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function boxIou(a: TuneDetection, b: number[]): number {
  const ix1 = Math.max(a.x1, b[0]);
  const iy1 = Math.max(a.y1, b[1]);
  const ix2 = Math.min(a.x2, b[2]);
  const iy2 = Math.min(a.y2, b[3]);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) {
    return 0;
  }
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

// Pose payloads are 1:1 with detection boxes server-side (each pose carries
// the detector bbox), so the best IoU match recovers the dog for a crop.
export function matchPose(det: TuneDetection, poses: TunePose[]): TunePose | null {
  let best: TunePose | null = null;
  let bestIou = 0.1; // require a little overlap to associate
  for (const pose of poses) {
    if (!pose.bbox || pose.bbox.length < 4) {
      continue;
    }
    const score = boxIou(det, pose.bbox);
    if (score > bestIou) {
      bestIou = score;
      best = pose;
    }
  }
  return best;
}

export function buildZoomCards(
  dets: TuneDetection[],
  poses: TunePose[],
  thr: number,
): ZoomCard[] {
  return dets
    .map((det) => ({
      det,
      pose: matchPose(det, poses),
      kept: det.confidence >= thr,
    }))
    .sort((a, b) => b.det.confidence - a.det.confidence)
    .slice(0, 8);
}
