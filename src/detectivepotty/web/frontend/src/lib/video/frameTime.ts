const DEFAULT_FPS = 30;
const FRAME_EPSILON = 1e-6;

export interface FrameTimeline {
  fps: number;
  totalFrames: number;
  frameTimes: number[] | null;
}

export function effectiveFps(
  fps: number | null | undefined,
  duration: number | null | undefined,
  totalFrames: number | null | undefined,
  fallback = DEFAULT_FPS,
): number {
  if (fps && fps > 0) return fps;
  if (duration && duration > 0 && totalFrames && totalFrames > 0) {
    return totalFrames / duration;
  }
  return fallback;
}

export function makeFrameTimeline(
  fps: number | null | undefined,
  duration: number | null | undefined,
  totalFrames: number | null | undefined,
  frameTimes?: number[] | null,
): FrameTimeline {
  const cleanTimes =
    frameTimes && frameTimes.length > 0
      ? frameTimes.map((v) => Number(v)).filter((v) => Number.isFinite(v))
      : null;
  const normalizedTimes = cleanTimes ? normalizeFrameTimes(cleanTimes) : null;
  const frameCount = normalizedTimes?.length ?? Math.max(0, Math.floor(totalFrames ?? 0));
  return {
    fps: effectiveFps(fps, duration, frameCount),
    totalFrames: Math.max(1, frameCount),
    frameTimes: normalizedTimes && normalizedTimes.length === frameCount ? normalizedTimes : null,
  };
}

export function lastFrame(totalFrames: number | null | undefined): number {
  return Math.max(0, Math.floor(totalFrames ?? 0) - 1);
}

export function clampFrame(frame: number, totalFrames: number | null | undefined): number {
  return Math.max(0, Math.min(lastFrame(totalFrames), frame));
}

export function frameToSeconds(frame: number, fps: number): number {
  return fps > 0 ? frame / fps : 0;
}

export function frameToVideoTime(frame: number, fps: number): number {
  return fps > 0 ? (frame + 0.5) / fps : 0;
}

export function videoTimeToFrameFloor(
  time: number,
  fps: number,
  totalFrames: number | null | undefined,
): number {
  return clampFrame(Math.floor(time * fps + FRAME_EPSILON), totalFrames);
}

export function videoTimeToFrameRounded(
  time: number,
  fps: number,
  totalFrames: number | null | undefined,
): number {
  return clampFrame(Math.round(time * fps), totalFrames);
}

export function frameToRatio(
  frame: number,
  totalFrames: number | null | undefined,
): number {
  const span = Math.max(1, lastFrame(totalFrames));
  return clampFrame(frame, totalFrames) / span;
}

export function ratioToFrame(
  ratio: number,
  totalFrames: number | null | undefined,
): number {
  const clamped = Math.max(0, Math.min(1, ratio));
  return clampFrame(Math.round(clamped * lastFrame(totalFrames)), totalFrames);
}

export function timelineFrameToSeconds(timeline: FrameTimeline, frame: number): number {
  const target = clampFrame(frame, timeline.totalFrames);
  if (timeline.frameTimes) return timeline.frameTimes[target] ?? 0;
  return frameToSeconds(target, timeline.fps);
}

export function timelineFrameToVideoTime(timeline: FrameTimeline, frame: number): number {
  const target = clampFrame(frame, timeline.totalFrames);
  if (!timeline.frameTimes) return frameToVideoTime(target, timeline.fps);
  const current = timeline.frameTimes[target] ?? 0;
  const next =
    target + 1 < timeline.frameTimes.length
      ? timeline.frameTimes[target + 1]
      : current + finalFrameDuration(timeline);
  return current + Math.max(0, next - current) / 2;
}

export function timelineVideoTimeToFrameFloor(timeline: FrameTimeline, time: number): number {
  if (!timeline.frameTimes) {
    return videoTimeToFrameFloor(time, timeline.fps, timeline.totalFrames);
  }
  const idx = upperBound(timeline.frameTimes, Math.max(0, time)) - 1;
  return clampFrame(idx, timeline.totalFrames);
}

export function timelineRatioToFrame(timeline: FrameTimeline, ratio: number): number {
  return ratioToFrame(ratio, timeline.totalFrames);
}

function normalizeFrameTimes(values: number[]): number[] | null {
  const base = values[0] ?? 0;
  const out = values.map((v) => Math.max(0, v - base));
  for (let i = 1; i < out.length; i += 1) {
    if (out[i] + FRAME_EPSILON < out[i - 1]) return null;
  }
  return out;
}

function finalFrameDuration(timeline: FrameTimeline): number {
  const times = timeline.frameTimes;
  if (!times || times.length < 2) return timeline.fps > 0 ? 1 / timeline.fps : 0;
  return Math.max(0, times[times.length - 1] - times[times.length - 2]);
}

function upperBound(values: number[], target: number): number {
  let lo = 0;
  let hi = values.length;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (values[mid] <= target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
