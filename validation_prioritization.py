import json
import math
import logging
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ValidationPrioritization")


# parameter

# detections with confidence lower that this are discarded
CONFIDENCE_THRESHOLD: float = 0.60

# number of detections kept after prioritization
TOP_N_DETECTIONS: int = 3

# objects with higher distance are ignored
MAX_DISTANCE_STATIC: float = 10.0  
MAX_DISTANCE_DYNAMIC: float = 20.0 

AZIMUTH_DIRECT_PATH: float = 30.0

RISK_LEVEL_WEIGHTS: dict[str, float] = {
    "high":   1.0,
    "medium": 0.6,
    "low":    0.2,
}



def _get_nested(obj: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(key, default)
        if obj is None:
            return default
    return obj

# calc ttc in seconds
def _compute_ttc(distance: float, approach_delta: float | None) -> float | None:
    if approach_delta is None or approach_delta <= 0.0:
        return None
    if distance <= 0.0:
        return 0.0
    return distance / approach_delta

# convert ttc into urgency score
def _ttc_score(ttc: float | None) -> float:
    if ttc is None:
        return 0.0
    TTC_CRITICAL = 2.0
    TTC_SAFE = 15.0
    if ttc <= TTC_CRITICAL:
        return 1.0
    if ttc >= TTC_SAFE:
        return 0.0
    return (TTC_SAFE - ttc) / (TTC_SAFE - TTC_CRITICAL)

# converts metric distance into proximity score
def _distance_score(distance: float, max_dist: float) -> float:
    if distance <= 0.0:
        return 1.0
    score = 1.0 - (distance / max_dist)
    return max(0.0, min(1.0, score))


#converts azimuth degrees int o centrality score
def _azimuth_score(azimuth_deg: float) -> float:
    abs_az = abs(azimuth_deg)
    if abs_az >= 90.0:
        return 0.0
    return 1.0 - (abs_az / 90.0)


# validation stage where unreliable dettections are removed

class DetectionValidator:

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        max_distance_static: float = MAX_DISTANCE_STATIC,
        max_distance_dynamic: float = MAX_DISTANCE_DYNAMIC,
        min_persistence_frames: int = 2,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.max_distance_static = max_distance_static
        self.max_distance_dynamic = max_distance_dynamic
        self.min_persistence_frames = min_persistence_frames
        self.track_history: dict[int, int] = {}

    def reset(self) -> None:

        self.track_history.clear()
        logger.info("Tracking history reset.")

    def validate(self, payload: dict) -> list[dict]:

        raw_detections: list[dict] = payload.get("detections", [])
        valid: list[dict] = []

        # First, increment history for all track_ids present in this frame
        # so we can check their persistence.
        for det in raw_detections:
            track_id = det.get("track_id")
            if track_id is not None:
                self.track_history[track_id] = self.track_history.get(track_id, 0) + 1

        for det in raw_detections:
            track_id = det.get("track_id")
            obj_class = _get_nested(det, "classification", "class", default="unknown")
            confidence = _get_nested(det, "classification", "confidence", default=0.0)
            category = _get_nested(det, "classification", "category", default="")
            distance = _get_nested(det, "spatial_metrics", "estimated_distance", default=math.inf)

            # verify presence of track id
            if track_id is None:
                logger.info(
                    "Detection '%s' dropped: missing track_id (single-frame noise).",
                    obj_class,
                )
                continue

            # verify temporal consistency
            seen_count = self.track_history.get(track_id, 0)
            if seen_count < self.min_persistence_frames:
                logger.info(
                    "Detection '%s' (track=%s) dropped: not persistent (appeared in %d frame(s), min: %d).",
                    obj_class, track_id, seen_count, self.min_persistence_frames,
                )
                continue

            # verify confidence score
            if confidence < self.confidence_threshold:
                logger.info(
                    "Detection '%s' (track=%s) dropped: confidence %.2f < threshold %.2f.",
                    obj_class, track_id, confidence, self.confidence_threshold,
                )
                continue

            # verify maximum distance
            is_dynamic = "dynamic" in category.lower()
            max_dist = self.max_distance_dynamic if is_dynamic else self.max_distance_static

            if distance > max_dist:
                logger.info(
                    "Detection '%s' (track=%s) dropped: distance %.1f m > maximum %.1f m for category '%s'.",
                    obj_class, track_id, distance, max_dist, category,
                )
                continue

            valid.append(det)
            logger.debug("Detection '%s' (track=%s) passed validation.", obj_class, track_id)

        logger.info(
            "Validation complete: %d/%d detections passed.",
            len(valid), len(raw_detections),
        )
        return valid


# prioritiztion stage

class DetectionPrioritizer:
 
# the calculated scores are multiplied by the respective weights
    def __init__(
        self,
        w_risk: float = 0.35,
        w_ttc: float = 0.35,
        w_dist: float = 0.20,
        w_az: float = 0.10,
        top_n: int = TOP_N_DETECTIONS,
        max_distance_static: float = MAX_DISTANCE_STATIC,
        max_distance_dynamic: float = MAX_DISTANCE_DYNAMIC,
    ) -> None:
        self.w_risk = w_risk
        self.w_ttc = w_ttc
        self.w_dist = w_dist
        self.w_az = w_az
        self.top_n = top_n
        self.max_distance_static = max_distance_static
        self.max_distance_dynamic = max_distance_dynamic

    def _compute_score(self, det: dict) -> float:
        category = _get_nested(det, "classification", "category", default="")
        risk_label = _get_nested(det, "semantic_relations", "risk_level", default="low")
        distance = _get_nested(det, "spatial_metrics", "estimated_distance", default=math.inf)
        azimuth = _get_nested(det, "spatial_metrics", "azimuth_deg", default=90.0)
        approach_delta = _get_nested(det, "movement", "approach_delta", default=None)

        is_dynamic = "dynamic" in category.lower()
        max_dist = self.max_distance_dynamic if is_dynamic else self.max_distance_static

        risk_w = RISK_LEVEL_WEIGHTS.get(risk_label, 0.0)
        ttc = _compute_ttc(distance, approach_delta)
        ttc_s = _ttc_score(ttc)
        dist_s = _distance_score(distance, max_dist)
        az_s = _azimuth_score(azimuth)

        score = (
            self.w_risk * risk_w
            + self.w_ttc  * ttc_s
            + self.w_dist * dist_s
            + self.w_az   * az_s
        )

        logger.debug(
            "Detection score '%s': risk=%.2f | ttc=%.2f (TTC=%s s) | dist=%.2f | az=%.2f → total=%.4f",
            _get_nested(det, "classification", "class", default="?"),
            risk_w, ttc_s,
            f"{ttc:.1f}" if ttc is not None else "N/A",
            dist_s, az_s, score,
        )
        return score

    #sorts the detection list
    def prioritize(self, valid_detections: list[dict]) -> list[dict]:

        if not valid_detections:
            logger.info("No valid detections to prioritize.")
            return []

        scored = []
        for det in valid_detections:
            score = self._compute_score(det)

            enriched = dict(det)
            enriched["_priority_score"] = round(score, 4)
            scored.append(enriched)

        # Sort by descending score
        scored.sort(key=lambda d: d["_priority_score"], reverse=True)

        # Truncate to top-N
        top = scored[: self.top_n]

        logger.info(
            "Prioritization complete: %d detections retained (out of %d valid, top-%d).",
            len(top), len(valid_detections), self.top_n,
        )
        for i, d in enumerate(top, start=1):
            logger.info(
                "  #%d → class='%s' | score=%.4f | risk='%s' | dist=%.1f m",
                i,
                _get_nested(d, "classification", "class", default="?"),
                d["_priority_score"],
                _get_nested(d, "semantic_relations", "risk_level", default="?"),
                _get_nested(d, "spatial_metrics", "estimated_distance", default=-1.0),
            )

        return top



class ValidationPrioritizationPipeline:


    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        top_n: int = TOP_N_DETECTIONS,
        max_distance_static: float = MAX_DISTANCE_STATIC,
        max_distance_dynamic: float = MAX_DISTANCE_DYNAMIC,
        min_persistence_frames: int = 2,
    ) -> None:
        self.validator = DetectionValidator(
            confidence_threshold=confidence_threshold,
            max_distance_static=max_distance_static,
            max_distance_dynamic=max_distance_dynamic,
            min_persistence_frames=min_persistence_frames,
        )
        self.prioritizer = DetectionPrioritizer(
            top_n=top_n,
            max_distance_static=max_distance_static,
            max_distance_dynamic=max_distance_dynamic,
        )

    def reset(self) -> None:
        self.validator.reset()

    def process(self, raw_payload: dict) -> dict:

        logger.info("=== Starting Validation and Prioritization pipeline ===")
        logger.info(
            "Received payload: %d raw detections.",
            len(raw_payload.get("detections", [])),
        )

        # Step 1 — Validation
        valid_detections = self.validator.validate(raw_payload)

        # Step 2 — Prioritization
        prioritized_detections = self.prioritizer.prioritize(valid_detections)

        # Assemble the output payload preserving metadata and global_context
        output_payload = {
            "metadata": raw_payload.get("metadata", {}),
            "global_context": raw_payload.get("global_context", {}),
            "detections": prioritized_detections,
        }

        logger.info(
            "=== Pipeline complete: %d detection(s) in final payload ===",
            len(prioritized_detections),
        )
        return output_payload


if __name__ == "__main__":
    from input_simulation import loadScenarios

    scenarios = loadScenarios("scene_dataset.json")
    pipeline = ValidationPrioritizationPipeline()

    for sc in scenarios:
        pipeline.reset()
        logger.info("=== Scenario: %s ===", sc["name"])
        for frame in sc["frames"]:
            out = pipeline.process(frame)
            logger.info("Output detections: %d", len(out["detections"]))
