from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Iterator, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# data logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("InputIngestion")


# data contract models

# frame metadata
class Metadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: float | int = 0
    sensor_id: str | None = None
    frame_id: int | None = None


# scene global context
class GlobalContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    scene_type: str = "urban"
    surface: str = "sidewalk"
    main_hazard: str | None = None
    overall_risk_level: float | str | None = None


# object classification
class Classification(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    object_class: str = Field(default="unknown", alias="class")
    confidence: float = 0.0
    category: str = "static"  # "static" or "dynamic"


# spatial metrics relative to the pedestrian
class SpatialMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    estimated_distance: float = 0.0
    azimuth_deg: float = 0.0
    bbox_normalized: list[float] | None = None


# movement and kinematics
class Movement(BaseModel):
    model_config = ConfigDict(extra="allow")

    approach_delta: float | None = None
    movement_state: str | None = None


# scene graph relations and contextual risk
class SemanticRelations(BaseModel):
    model_config = ConfigDict(extra="allow")

    risk_level: str = "low"  # "high", "medium", "low"
    spatial_relation: str | None = None
    surface_relation: str | None = None


# obstacle detection
class Detection(BaseModel):
    model_config = ConfigDict(extra="allow")

    track_id: int | None = None
    classification: Classification = Field(default_factory=Classification)
    spatial_metrics: SpatialMetrics = Field(default_factory=SpatialMetrics)
    movement: Movement = Field(default_factory=Movement)
    semantic_relations: SemanticRelations = Field(default_factory=SemanticRelations)


# frame input payload
class FrameInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    metadata: Metadata = Field(default_factory=Metadata)
    global_context: GlobalContext = Field(default_factory=GlobalContext)
    detections: list[Detection] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, by_alias=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameInput:
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> FrameInput:
        return cls.model_validate_json(json_str)

    def to_json(self, indent: int | None = None) -> str:
        return self.model_dump_json(exclude_none=True, by_alias=True, indent=indent)


# validation and normalization utils

# parses json string or dict into FrameInput
def parse_frame_input(data: dict[str, Any] | str) -> FrameInput:
    if isinstance(data, str):
        return FrameInput.model_validate_json(data)
    return FrameInput.model_validate(data)


# normalizes payload into standard dictionary format
def normalize_payload(data: dict[str, Any] | str | FrameInput) -> dict[str, Any]:
    if isinstance(data, FrameInput):
        frame_obj = data
    elif isinstance(data, str):
        frame_obj = FrameInput.model_validate_json(data)
    elif isinstance(data, dict):
        frame_obj = FrameInput.model_validate(data)
    else:
        raise TypeError(f"Unsupported input type for normalization: {type(data).__name__}")
    return frame_obj.model_dump(exclude_none=True, by_alias=True)


# checks if frame conforms to expected contract
def validate_frame(data: dict[str, Any] | str | FrameInput) -> tuple[bool, list[str]]:
    try:
        if isinstance(data, FrameInput):
            return True, []
        if isinstance(data, str):
            FrameInput.model_validate_json(data)
        else:
            FrameInput.model_validate(data)
        return True, []
    except ValidationError as e:
        return False, [str(err) for err in e.errors()]


# input sources and handler

# base input source
class BaseInputSource(ABC):

    @abstractmethod
    def get_frame(self) -> dict[str, Any] | str | None:
        pass


# direct input source for pushing frames
class DirectInputSource(BaseInputSource):

    def __init__(self) -> None:
        self._current_frame: dict[str, Any] | str | None = None

    def push(self, frame: dict[str, Any] | str | FrameInput) -> None:
        if isinstance(frame, FrameInput):
            self._current_frame = frame.model_dump(exclude_none=True, by_alias=True)
        else:
            self._current_frame = frame

    def get_frame(self) -> dict[str, Any] | str | None:
        frame = self._current_frame
        self._current_frame = None
        return frame


# stream input source wrapping an iterable
class StreamInputSource(BaseInputSource):

    def __init__(self, stream: Sequence[Any] | Iterator[Any]) -> None:
        self._iterator = iter(stream)

    def get_frame(self) -> dict[str, Any] | str | None:
        try:
            return next(self._iterator)
        except StopIteration:
            return None


# input handler for receiving and normalizing frames
class InputHandler:

    def __init__(self, source: BaseInputSource | None = None) -> None:
        self.source = source or DirectInputSource()

    def set_source(self, source: BaseInputSource) -> None:
        self.source = source

    # ingests, validates, and normalizes an incoming payload
    def ingest(self, raw_data: dict[str, Any] | str | FrameInput) -> dict[str, Any] | None:
        try:
            return normalize_payload(raw_data)
        except (ValidationError, TypeError) as e:
            logger.error("Falha na validação do frame. Frame descartado: %s", e)
            return None

    # fetches next frame from source
    def fetch_next(self) -> dict[str, Any] | None:
        raw_frame = self.source.get_frame()
        if raw_frame is None:
            return None
        return self.ingest(raw_frame)


# test
if __name__ == "__main__":
    import json

    sample_json = '''{
      "metadata": {"timestamp": 1714567890, "unknown_camera_type": "infra_red"},
      "global_context": {
        "scene_type": "urban_intersection",
        "surface": "crosswalk",
        "main_hazard": "approaching_vehicle",
        "overall_risk_level": 0.85
      },
      "detections": [
        {
          "track_id": 14,
          "classification": {"class": "car", "confidence": "0.92", "category": "dynamic"},
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
    print(f"Detection 0: class={det0.classification.object_class}, dist={det0.spatial_metrics.estimated_distance}m")

    handler = InputHandler()
    handler.source = DirectInputSource()
    handler.source.push(sample_json)

    resultado_final = handler.fetch_next()
    print("Normalized output payload:")
    print(json.dumps(resultado_final, indent=2))
