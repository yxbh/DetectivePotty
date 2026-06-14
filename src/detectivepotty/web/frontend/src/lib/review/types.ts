export interface ReviewHeaderState {
  statusFilter: string;
  cameraFilter: string;
  labeledCount: number;
  eventCount: number;
  unfilteredTotal: number | null;
  progressPct: number;
  dirty: boolean;
  applyFilter: (value: string) => void;
  setCameraFilter: (value: string) => void;
  commitCameraFilter: () => void;
  clearCamera: () => void;
}

export interface ReviewOpenRequest {
  seq: number;
  eventId: string;
}
