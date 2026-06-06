"""UniFi Protect integration."""

from detectivepotty.protect.client import (
    ProtectCameraChannel,
    ProtectCameraInfo,
    ProtectClient,
)
from detectivepotty.protect.trigger import ProtectAnimalTrigger, parse_smartdetect_event

__all__ = [
    "ProtectAnimalTrigger",
    "ProtectCameraChannel",
    "ProtectCameraInfo",
    "ProtectClient",
    "parse_smartdetect_event",
]
