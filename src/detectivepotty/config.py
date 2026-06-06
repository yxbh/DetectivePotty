"""Configuration schema and secret-free hashing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

Device = Literal["auto", "mps", "cpu"]
SubstreamChoice = Literal["low", "medium", "high"]
SourceKind = Literal["protect", "file"]
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class RuntimeSecrets(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DETECTIVEPOTTY_", extra="ignore")

    nvr_api_key: SecretStr | None = None
    nvr_username: SecretStr | None = None
    nvr_password: SecretStr | None = None


class GlobalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_dir: Path = Path("dataset")
    model_name: str = "yolo11n.pt"
    inference_long_edge_px: int = Field(default=1280, gt=0)
    device: Device = "auto"
    log_level: str = "INFO"
    dogs: list[str] = Field(default_factory=list)

    @field_validator("dogs")
    @classmethod
    def clean_dog_roster(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for name in value:
            stripped = name.strip()
            if stripped and stripped not in cleaned:
                cleaned.append(stripped)
        return cleaned


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


class CameraInputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: SourceKind = "protect"
    path: Path | None = None
    source_id: str | None = None


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
