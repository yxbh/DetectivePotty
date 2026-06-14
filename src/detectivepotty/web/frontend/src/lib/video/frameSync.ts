import {
  clampFrame,
  type FrameTimeline,
  timelineFrameToVideoTime,
  timelineVideoTimeToFrameFloor,
} from "./frameTime";

const hasRvfc =
  typeof HTMLVideoElement !== "undefined" &&
  "requestVideoFrameCallback" in HTMLVideoElement.prototype;

export interface VideoFrameSyncOptions {
  timeline: () => FrameTimeline;
  onFrame: (frame: number) => void;
  onPlayingChange?: (playing: boolean) => void;
}

export class VideoFrameSync {
  private video: HTMLVideoElement | null = null;
  private rvfcHandle: number | null = null;

  constructor(private readonly options: VideoFrameSyncOptions) {}

  setVideo(video: HTMLVideoElement | null): void {
    if (this.video === video) return;
    this.cancelFrameLoop();
    this.video = video;
  }

  destroy(): void {
    this.cancelFrameLoop();
    this.video = null;
  }

  loadedMetadata(startFrame = 0): void {
    this.seekToFrame(startFrame);
    this.registerFrameLoop();
  }

  seekToFrame(frame: number): number {
    const timeline = this.options.timeline();
    const target = clampFrame(frame, timeline.totalFrames);
    this.options.onFrame(target);
    if (this.video) {
      this.video.currentTime = timelineFrameToVideoTime(timeline, target);
    }
    return target;
  }

  stepFrame(currentFrame: number, delta: number): number {
    if (this.video && !this.video.paused) this.video.pause();
    return this.seekToFrame(currentFrame + delta);
  }

  togglePlay(): void {
    if (!this.video) return;
    if (this.video.paused) void this.video.play().catch(() => undefined);
    else this.video.pause();
  }

  syncFromCurrentTime(): void {
    if (!this.video) return;
    this.options.onFrame(
      timelineVideoTimeToFrameFloor(this.options.timeline(), this.video.currentTime),
    );
  }

  handlePlay(): void {
    this.options.onPlayingChange?.(true);
  }

  handlePause(): void {
    this.options.onPlayingChange?.(false);
  }

  cancelFrameLoop(): void {
    if (hasRvfc && this.video && this.rvfcHandle !== null) {
      try {
        this.video.cancelVideoFrameCallback(this.rvfcHandle);
      } catch {
        /* element may be detaching */
      }
    }
    this.rvfcHandle = null;
  }

  private registerFrameLoop(): void {
    if (!hasRvfc || !this.video) return;
    this.cancelFrameLoop();
    const cb = (): void => {
      this.rvfcHandle = null;
      this.syncFromCurrentTime();
      if (this.video) {
        this.rvfcHandle = this.video.requestVideoFrameCallback(cb);
      }
    };
    this.rvfcHandle = this.video.requestVideoFrameCallback(cb);
  }
}
