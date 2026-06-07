"""Keypoint pose estimation for richer, view-robust dog posture features.

Pose is additive and config-gated (``pose.enabled``); when pose is disabled or
keypoints are missing/low-confidence the system falls back to the existing
bbox-derived posture heuristics. All keypoint coordinates are in ORIGINAL-source
pixel space, matching the coordinate discipline used by detections and crops.
"""
