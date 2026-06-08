"""Configuration schema and secret-free hashing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

Device = Literal["auto", "cuda", "mps", "cpu"]
SubstreamChoice = Literal["low", "medium", "high"]
SourceKind = Literal["protect", "file", "rtsp"]
PoseBackend = Literal["superanimal", "mock"]
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class RuntimeSecrets(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DETECTIVEPOTTY_", extra="ignore")

    nvr_api_key: SecretStr | None = None
    nvr_username: SecretStr | None = None
    nvr_password: SecretStr | None = None


class GlobalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_dir: Path = Path("dataset")
    model_name: str = "models/yolo11m.pt"
    inference_long_edge_px: int = Field(default=640, gt=0)
    device: Device = "auto"
    log_level: str = "INFO"
    dogs: list[str] = Field(default_factory=list)
    dedupe_reruns: bool = True
    rerun_match_tolerance_s: float = Field(default=5.0, ge=0)

    @field_validator("dogs")
    @classmethod
    def clean_dog_roster(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for name in value:
            stripped = name.strip()
            if stripped and stripped not in cleaned:
                cleaned.append(stripped)
        return cleaned


class PoseConfig(BaseModel):
    """Keypoint-pose settings.

    Pose is additive and OFF by default until validated end-to-end. Quality
    thresholds (how much pose is trustworthy) are kept separate from the per-feature
    behavior thresholds so one global confidence value does not silently do
    everything. ``enable_pose_classifier``/``enable_pose_gate`` gate the two
    consumers independently because the detection gate is the riskiest change.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    backend: PoseBackend = "superanimal"
    model_name: str = "hrnet_w32"
    device: Device = "auto"
    crop_margin_frac: float = Field(default=0.4, ge=0.0)
    min_keypoint_conf: float = Field(default=0.5, ge=0.0, le=1.0)
    min_required_frames: int = Field(default=3, ge=1)
    min_pose_coverage: float = Field(default=0.5, ge=0.0, le=1.0)
    min_torso_keypoints: int = Field(default=3, ge=0)
    max_pose_gap_s: float = Field(default=1.0, gt=0.0)
    candidate_only: bool = True
    # Temporal box union: pose crops are built from the union of a dog's detector
    # boxes over this trailing ``mono_ts`` window (seconds) to recover full extent
    # when a single frame under-segments on low-contrast IR. 0.0 disables it (the
    # pose crop is then byte-identical to the raw detector box). Only affects pose;
    # tracking/posture/recorder boxes are untouched.
    box_union_window_s: float = Field(default=0.0, ge=0.0)
    enable_pose_classifier: bool = False
    # Experimental: additive pose augmentation of the detection gate. Validated only
    # with the mock backend so far (wiring + gate-OFF byte-identical baseline); keep
    # off until validated end-to-end against the real superanimal backend.
    enable_pose_gate: bool = False


class ProtectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nvr_host: str | None = None
    api_key_env: str | None = "DETECTIVEPOTTY_NVR_API_KEY"
    username_env: str | None = "DETECTIVEPOTTY_NVR_USERNAME"
    password_env: str | None = "DETECTIVEPOTTY_NVR_PASSWORD"
    verify_tls: bool = True

    @field_validator("api_key_env", "username_env", "password_env")
    @classmethod
    def env_names_only(cls, value: str | None) -> str | None:
        if value is not None and not ENV_NAME_RE.fullmatch(value):
            raise ValueError("secret references must be environment variable names")
        return value

    @field_validator("nvr_host")
    @classmethod
    def host_must_not_contain_credentials(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parts = urlsplit(value)
        if parts.username or parts.password:
            raise ValueError("nvr_host must not contain credentials")
        return value


class ZoneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    points: list[tuple[float, float]] = Field(default_factory=list)

    @field_validator("points")
    @classmethod
    def _validate_normalized(
        cls, value: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        # Zone points are normalized [0.0, 1.0] image coordinates (see README), so
        # they are resolution-independent and can be compared against normalized
        # detection centers. Pixel coordinates here silently filtered out every
        # detection, so reject them up front instead of failing open at runtime.
        for x, y in value:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(
                    "zone points must be normalized to [0.0, 1.0] image "
                    f"coordinates; got ({x}, {y})"
                )
        return value


class CameraInputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: SourceKind = "protect"
    path: Path | None = None
    source_id: str | None = None
    url_env: str | None = None

    @field_validator("url_env")
    @classmethod
    def url_env_name_only(cls, value: str | None) -> str | None:
        if value is not None and not ENV_NAME_RE.fullmatch(value):
            raise ValueError("url_env must be an environment variable name")
        return value

    @model_validator(mode="after")
    def validate_kind_fields(self) -> Self:
        if self.kind == "rtsp":
            if not self.url_env:
                raise ValueError("rtsp input requires 'url_env'")
            if self.path is not None:
                raise ValueError("rtsp input must not set 'path'")
        elif self.url_env is not None:
            raise ValueError("'url_env' is only valid when kind == 'rtsp'")
        return self

    def resolve_url(self) -> str | None:
        """Return the direct RTSP URL from the configured env var, if set.

        The full ``rtsp://user:pass@host/path`` (credentials included) lives in
        an environment variable so secrets never touch YAML, mirroring
        ``ProtectConfig`` secret handling.
        """

        if not self.url_env:
            return None
        return os.environ.get(self.url_env)


class CameraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    enabled: bool = True
    input: CameraInputConfig = Field(default_factory=CameraInputConfig)
    substream_choice: SubstreamChoice = "medium"
    animal_supported: bool = True
    roi: list[ZoneConfig] = Field(default_factory=list)
    ignore_zones: list[ZoneConfig] = Field(default_factory=list)
    detection_conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    event_duration_s: float = Field(default=8.0, gt=0.0)
    stationary_threshold_s: float = Field(default=2.0, ge=0.0)
    squat_threshold: float = Field(default=0.35, ge=0.0)
    sample_rate_fps: float = Field(default=5.0, gt=0.0)
    pre_roll_s: float = Field(default=15.0, ge=0.0)
    post_roll_s: float = Field(default=30.0, ge=0.0)
    retention_days: int = Field(default=30, ge=1)
    retention_max_gb: float | None = Field(default=None, gt=0.0)


class Config(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    global_settings: GlobalSettings = Field(default_factory=GlobalSettings, alias="global")
    protect: ProtectConfig = Field(default_factory=ProtectConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    cameras: list[CameraConfig] = Field(default_factory=list)
    runtime_secrets: RuntimeSecrets = Field(
        default_factory=RuntimeSecrets,
        exclude=True,
        repr=False,
    )

    def config_hash(self) -> str:
        return config_hash(self)

    def resolve_secret(self, env_field: Literal["api_key", "username", "password"]) -> str | None:
        env_name = getattr(self.protect, f"{env_field}_env")
        if env_name is None:
            return None
        return os.environ.get(env_name)


def load_config(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)


def config_hash(config: Config) -> str:
    payload: dict[str, Any] = config.model_dump(
        mode="json",
        by_alias=True,
        exclude={"runtime_secrets"},
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
