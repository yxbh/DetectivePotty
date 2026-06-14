const DEFAULT_FPS = 30;
const FRAME_EPSILON = 1e-6;

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
