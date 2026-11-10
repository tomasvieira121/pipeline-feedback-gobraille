# core modules

from .input import (
    Classification,
    Detection,
    FrameInput,
    GlobalContext,
    InputHandler,
    Metadata,
    Movement,
    SemanticRelations,
    SpatialMetrics,
    normalize_payload,
    parse_frame_input,
    validate_frame,
)
from .validation_prioritization import (
    DetectionPrioritizer,
    DetectionValidator,
    ValidationPrioritizationPipeline,
)

__all__ = [
    "Metadata",
    "GlobalContext",
    "Classification",
    "SpatialMetrics",
    "Movement",
    "SemanticRelations",
    "Detection",
    "FrameInput",
    "validate_frame",
    "normalize_payload",
    "parse_frame_input",
    "InputHandler",
    "DetectionValidator",
    "DetectionPrioritizer",
    "ValidationPrioritizationPipeline",
]

try:
    from .slm_processing import SLMProcessor
    __all__.append("SLMProcessor")
except ImportError:
    pass

try:
    from .tts_module import PiperStreamingTTS
    __all__.append("PiperStreamingTTS")
except ImportError:
    pass
