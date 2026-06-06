from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from detectivepotty.events import TriggerReason
from detectivepotty.geometry import BBox
from detectivepotty.protect.trigger import ProtectAnimalTrigger, parse_smartdetect_event


RECEIVED_AT = datetime(2026, 6, 6, 0, 0, 3, tzinfo=timezone.utc)


def _animal_packet(action: str = "add") -> dict[str, object]:
    return {
        "action": action,
        "new_obj": {
            "id": "evt-animal-1",
            "type": "smartDetectZone",
            "camera": "cam-1",
            "start": "2026-06-06T00:00:00Z",
            "score": 87,
            "smartDetectTypes": ["animal"],
            "metadata": {
                "detectedThumbnails": [
                    {"type": "animal", "coord": [10, 20, 110, 220], "confidence": 92}
                ]
            },
        },
    }


def test_parse_animal_event_returns_trigger_event() -> None:
    event = parse_smartdetect_event(_animal_packet(), received_at=RECEIVED_AT)

    assert event is not None
    assert event.camera_id == "cam-1"
    assert event.reason is TriggerReason.PROTECT_ANIMAL
    assert event.detection_ts == datetime(2026, 6, 6, 0, 0, tzinfo=timezone.utc)
    assert event.notification_ts == RECEIVED_AT
    assert event.score == 0.87
    assert event.bbox == BBox(10, 20, 110, 220)
    assert event.raw == {
        "protect_event_id": "evt-animal-1",
        "action": "add",
        "smart_detect_types": ["animal"],
        "score": 0.87,
    }


def test_parse_non_animal_event_returns_none() -> None:
    packet = _animal_packet()
    packet["new_obj"]["smartDetectTypes"] = ["person"]  # type: ignore[index]

    assert parse_smartdetect_event(packet, received_at=RECEIVED_AT) is None


def test_parse_remove_event_returns_none() -> None:
    assert parse_smartdetect_event(_animal_packet("remove"), received_at=RECEIVED_AT) is None


def test_parse_uiprotect_style_object_packet() -> None:
    payload = SimpleNamespace(
        id="evt-animal-2",
        camera_id="cam-2",
        start=datetime(2026, 6, 6, 0, 1, tzinfo=timezone.utc),
        smart_detect_types=[SimpleNamespace(value="animal")],
        score=55,
        metadata=SimpleNamespace(
            detected_thumbnails=[SimpleNamespace(type="animal", coord=[1, 2, 3, 4])]
        ),
    )
    packet = SimpleNamespace(action=SimpleNamespace(value="add"), new_obj=payload)

    event = parse_smartdetect_event(packet, received_at=RECEIVED_AT)

    assert event is not None
    assert event.camera_id == "cam-2"
    assert event.score == 0.55
    assert event.bbox == BBox(1, 2, 3, 4)


def test_trigger_deduplicates_lifecycle_packets() -> None:
    trigger = ProtectAnimalTrigger(client=SimpleNamespace())  # type: ignore[arg-type]
    first = parse_smartdetect_event(_animal_packet("add"), received_at=RECEIVED_AT)
    update = parse_smartdetect_event(_animal_packet("update"), received_at=RECEIVED_AT)

    assert first is not None
    assert update is not None
    assert trigger._remember(first) is True
    assert trigger._remember(update) is False
