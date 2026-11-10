from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

# data logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("InputIngestion")


# data contract models

# frame metadata
@dataclass
class Metadata:
    timestamp: float | int = 0
    sensor_id: str | None = None
    frame_id: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Metadata:
        if not data:
            return cls()
        known = {"timestamp", "sensor_id", "frame_id"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            timestamp=data.get("timestamp", 0),
            sensor_id=data.get("sensor_id"),
            frame_id=data.get("frame_id"),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"timestamp": self.timestamp}
        if self.sensor_id is not None:
            d["sensor_id"] = self.sensor_id
        if self.frame_id is not None:
            d["frame_id"] = self.frame_id
        d.update(self.extra)
        return d


# scene global context
@dataclass
class GlobalContext:
    scene_type: str = "urban"
    surface: str = "sidewalk"
    main_hazard: str | None = None
    overall_risk_level: float | str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GlobalContext:
        if not data:
            return cls()
        known = {"scene_type", "surface", "main_hazard", "overall_risk_level"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            scene_type=data.get("scene_type", "urban"),
            surface=data.get("surface", "sidewalk"),
            main_hazard=data.get("main_hazard"),
            overall_risk_level=data.get("overall_risk_level"),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "scene_type": self.scene_type,
            "surface": self.surface,
        }
        if self.main_hazard is not None:
            d["main_hazard"] = self.main_hazard
        if self.overall_risk_level is not None:
            d["overall_risk_level"] = self.overall_risk_level
        d.update(self.extra)
        return d


# object classification
@dataclass
class Classification:
    object_class: str = "unknown"
    confidence: float = 0.0
    category: str = "static"  # "static" or "dynamic"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Classification:
        if not data:
            return cls()
        return cls(
            object_class=data.get("class", "unknown"),
            confidence=float(data.get("confidence", 0.0)),
            category=str(data.get("category", "static")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.object_class,
            "confidence": self.confidence,
            "category": self.category,
        }


# spatial metrics relative to the pedestrian
@dataclass
class SpatialMetrics:
    estimated_distance: float = 0.0
    azimuth_deg: float = 0.0
    bbox_normalized: list[float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SpatialMetrics:
        if not data:
            return cls()
        return cls(
            estimated_distance=float(data.get("estimated_distance", 0.0)),
            azimuth_deg=float(data.get("azimuth_deg", 0.0)),
            bbox_normalized=data.get("bbox_normalized"),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "estimated_distance": self.estimated_distance,
            "azimuth_deg": self.azimuth_deg,
        }
        if self.bbox_normalized is not None:
            d["bbox_normalized"] = self.bbox_normalized
        return d


# movement and kinematics
@dataclass
class Movement:
    approach_delta: float | None = None
    movement_state: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Movement:
        if not data:
            return cls()
        delta = data.get("approach_delta")
        return cls(
            approach_delta=float(delta) if delta is not None else None,
            movement_state=data.get("movement_state"),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.approach_delta is not None:
            d["approach_delta"] = self.approach_delta
        if self.movement_state is not None:
            d["movement_state"] = self.movement_state
        return d


# scene graph relations and contextual risk
@dataclass
class SemanticRelations:
    risk_level: str = "low"  # "high", "medium", "low"
    spatial_relation: str | None = None
    surface_relation: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SemanticRelations:
        if not data:
            return cls()
        return cls(
            risk_level=data.get("risk_level", "low"),
            spatial_relation=data.get("spatial_relation"),
            surface_relation=data.get("surface_relation"),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"risk_level": self.risk_level}
        if self.spatial_relation is not None:
            d["spatial_relation"] = self.spatial_relation
        if self.surface_relation is not None:
            d["surface_relation"] = self.surface_relation
        return d


# obstacle detection
@dataclass
class Detection:
    track_id: int | None = None
    classification: Classification = field(default_factory=Classification)
    spatial_metrics: SpatialMetrics = field(default_factory=SpatialMetrics)
    movement: Movement = field(default_factory=Movement)
    semantic_relations: SemanticRelations = field(default_factory=SemanticRelations)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Detection:
        known = {"track_id", "classification", "spatial_metrics", "movement", "semantic_relations"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            track_id=data.get("track_id"),
            classification=Classification.from_dict(data.get("classification")),
            spatial_metrics=SpatialMetrics.from_dict(data.get("spatial_metrics")),
            movement=Movement.from_dict(data.get("movement")),
            semantic_relations=SemanticRelations.from_dict(data.get("semantic_relations")),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.track_id is not None:
            d["track_id"] = self.track_id
        d["classification"] = self.classification.to_dict()
        d["spatial_metrics"] = self.spatial_metrics.to_dict()
        mov = self.movement.to_dict()
        if mov:
            d["movement"] = mov
        sem = self.semantic_relations.to_dict()
        if sem:
            d["semantic_relations"] = sem
        d.update(self.extra)
        return d


# frame input payload
@dataclass
class FrameInput:
    metadata: Metadata = field(default_factory=Metadata)
    global_context: GlobalContext = field(default_factory=GlobalContext)
    detections: list[Detection] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameInput:
        meta = Metadata.from_dict(data.get("metadata"))
        context = GlobalContext.from_dict(data.get("global_context"))
        raw_dets = data.get("detections", [])
        dets = [Detection.from_dict(d) if isinstance(d, dict) else d for d in raw_dets]
        return cls(metadata=meta, global_context=context, detections=dets)

    @classmethod
    def from_json(cls, json_str: str) -> FrameInput:
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "global_context": self.global_context.to_dict(),
            "detections": [d.to_dict() for d in self.detections],
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# validation and normalization utils

# checks if frame conforms to expected contract
def validate_frame(data: dict[str, Any] | FrameInput) -> tuple[bool, list[str]]:
    errors: list[str] = []
    payload = data.to_dict() if isinstance(data, FrameInput) else data

    if not isinstance(payload, dict):
        return False, ["Payload must be a dictionary or FrameInput instance."]

    detections = payload.get("detections")
    if detections is not None and not isinstance(detections, list):
        errors.append("Field 'detections' must be a list.")
    elif isinstance(detections, list):
        for idx, det in enumerate(detections):
            if not isinstance(det, dict):
                errors.append(f"Detection at index {idx} is not a dictionary.")
                continue
            if "track_id" in det and det["track_id"] is not None and not isinstance(det["track_id"], int):
                errors.append(f"Detection #{idx}: 'track_id' must be an integer if provided.")
            cls_info = det.get("classification")
            if cls_info is not None and not isinstance(cls_info, dict):
                errors.append(f"Detection #{idx}: 'classification' must be a dictionary.")

    return len(errors) == 0, errors


# normalizes payload into standard dictionary format
def normalize_payload(data: dict[str, Any] | str | FrameInput) -> dict[str, Any]:
    if isinstance(data, FrameInput):
        return data.to_dict()
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise TypeError(f"Unsupported input type for normalization: {type(data).__name__}")

    normalized = dict(data)
    if "metadata" not in normalized or not isinstance(normalized["metadata"], dict):
        normalized["metadata"] = {}
    if "global_context" not in normalized or not isinstance(normalized["global_context"], dict):
        normalized["global_context"] = {}
    if "detections" not in normalized or not isinstance(normalized["detections"], list):
        normalized["detections"] = []

    return normalized


# parses json string or dict into FrameInput
def parse_frame_input(data: dict[str, Any] | str) -> FrameInput:
    if isinstance(data, str):
        return FrameInput.from_json(data)
    if isinstance(data, dict):
        return FrameInput.from_dict(data)
    raise TypeError(f"Cannot parse input of type {type(data).__name__}")


# input sources and handler

# base input source
class BaseInputSource(ABC):

    @abstractmethod
    def get_frame(self) -> dict[str, Any] | None:
        pass


# direct input source for pushing frames
class DirectInputSource(BaseInputSource):

    def __init__(self) -> None:
        self._current_frame: dict[str, Any] | None = None

    def push(self, frame: dict[str, Any] | str | FrameInput) -> None:
        self._current_frame = normalize_payload(frame)

    def get_frame(self) -> dict[str, Any] | None:
        frame = self._current_frame
        self._current_frame = None
        return frame


# stream input source wrapping an iterable
class StreamInputSource(BaseInputSource):

    def __init__(self, stream: Sequence[dict[str, Any]] | Iterator[dict[str, Any]]) -> None:
        self._iterator = iter(stream)

    def get_frame(self) -> dict[str, Any] | None:
        try:
            raw = next(self._iterator)
            return normalize_payload(raw)
        except StopIteration:
            return None


# input handler for receiving and normalizing frames
class InputHandler:

    def __init__(self, source: BaseInputSource | None = None) -> None:
        self.source = source or DirectInputSource()

    def set_source(self, source: BaseInputSource) -> None:
        self.source = source

    # ingests, validates, and normalizes an incoming payload
    def ingest(self, raw_data: dict[str, Any] | str | FrameInput) -> dict[str, Any]:
        normalized = normalize_payload(raw_data)
        is_valid, errors = validate_frame(normalized)
        if not is_valid:
            logger.warning("Input frame validation warnings: %s", errors)
        return normalized

    # fetches next frame from source
    def fetch_next(self) -> dict[str, Any] | None:
        frame = self.source.get_frame()
        if frame is None:
            return None
        return self.ingest(frame)


# test
if __name__ == "__main__":
    sample_json = '''{
      "metadata": {"timestamp": 1714567890},
      "global_context": {
        "scene_type": "urban_intersection",
        "surface": "crosswalk",
        "main_hazard": "approaching_vehicle",
        "overall_risk_level": 0.85
      },
      "detections": [
        {
          "track_id": 14,
          "classification": {"class": "car", "confidence": 0.92, "category": "dynamic"},
          "spatial_metrics": {"bbox_normalized": [0.45, 0.55, 0.30, 0.40], "estimated_distance": 4.0, "azimuth_deg": -15.0},
          "movement": {"movement_state": "approaching", "approach_delta": 0.05},
          "semantic_relations": {"risk_level": "high", "spatial_relation": "blocking_path", "surface_relation": "on_crosswalk"}
        }
      ]
    }'''

    parsed = parse_frame_input(sample_json)
    valid, errs = validate_frame(parsed)
    print("Input Module Test")
    print(f"Validation successful: {valid} (Errors: {errs})")
    print(f"Parsed Frame - Detections: {len(parsed.detections)}")
    det0 = parsed.detections[0]
    print(f"Detection 0: class={det0.classification.object_class}, dist={det0.spatial_metrics.estimated_distance}m, risk={det0.semantic_relations.risk_level}")
    print("Normalized output payload:")
    print(json.dumps(parsed.to_dict(), indent=2))
