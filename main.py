from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any

from modules.input import FrameInput, normalize_payload, validate_frame
from modules.validation_prioritization import ValidationPrioritizationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FeedbackPipeline")


# Env util

# reads a .env file and injects variables into the environment
def load_env_file(filepath: str = ".env") -> None:
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    val = v.strip().strip("'").strip('"')
                    os.environ[k.strip()] = val
        logger.info("Environment variables loaded from: %s", filepath)
    else:
        logger.debug(".env file not found at: %s", filepath)


# sanitizes a string so it can be safely used as a filename
def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)


# pipeline orchestrator

class FeedbackPipeline:

    def __init__(
        self,
        ollama_host: str | None = None,
        ollama_model: str | None = None,
        run_baseline: bool = False,
        use_tts: bool = False,
        confidence_threshold: float = 0.60,
        min_persistence_frames: int = 2,
        top_n: int = 3,
    ) -> None:
        load_env_file()

        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        self.run_baseline = run_baseline
        self.use_tts = use_tts
        self.confidence_threshold = confidence_threshold
        self.min_persistence_frames = min_persistence_frames
        self.top_n = top_n

        # Initialize validation and prioritization 
        self.validation_pipeline = ValidationPrioritizationPipeline(
            confidence_threshold=self.confidence_threshold,
            min_persistence_frames=self.min_persistence_frames,
            top_n=self.top_n,
        )

        # Initialize SLM processor
        try:
            from modules.slm_processing import SLMProcessor
            self.processor = SLMProcessor(host=self.ollama_host, model=self.ollama_model)
        except Exception as e:
            logger.error("Failed to initialize SLMProcessor: %s", e)
            raise

        # Initialize TTS
        self.tts_engine = None
        if self.use_tts:
            try:
                from modules.tts_module import PiperStreamingTTS
                self.tts_engine = PiperStreamingTTS()
                logger.info("PiperStreamingTTS initialized successfully.")
            except Exception as e:
                logger.warning(
                    "Could not initialize PiperStreamingTTS: %s. Continuing without audio synthesis.",
                    e,
                )
                self.use_tts = False
                self.tts_engine = None

    # clears tracking history and resets state
    def reset(self) -> None:
        self.validation_pipeline.reset()
        if self.tts_engine is not None:
            self.tts_engine.reset_audio_duration()

    # processes a single input frame through the pipeline
    def process_frame(self, raw_frame: dict[str, Any] | str | FrameInput) -> dict[str, Any]:
        normalized_frame = normalize_payload(raw_frame)

        if self.tts_engine is not None:
            self.tts_engine.reset_audio_duration()

        if self.run_baseline:
            pipeline_time_ms = 0.0
            payload_to_slm = normalized_frame
        else:
            t_pipeline_start = time.perf_counter()
            prioritized_payload = self.validation_pipeline.process(normalized_frame)
            pipeline_time_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
            payload_to_slm = prioritized_payload

        warning_text = ""
        metrics: dict[str, Any] = {}

        try:
            warning_text, metrics = self.processor.process_with_metrics(
                payload_to_slm, tts=self.tts_engine
            )
            metrics["pipeline_processing_time_ms"] = round(pipeline_time_ms, 1)
            metrics["end_to_end_latency_ms"] = round(
                metrics.get("latency_ms", 0.0) + pipeline_time_ms, 1
            )
        except Exception as e:
            logger.error("Error during SLM inference: %s", e)
            warning_text = f"[INFERENCE ERROR: {e}]"
            metrics = {
                "ttft_ms": 0.0,
                "ttfa_ms": 0.0,
                "slm_latency_ms": 0.0,
                "tts_latency_ms": 0.0,
                "audio_duration_s": 0.0,
                "latency_ms": 0.0,
                "pipeline_processing_time_ms": round(pipeline_time_ms, 1),
                "end_to_end_latency_ms": 0.0,
                "error": str(e),
            }

        return {
            "raw_input": normalized_frame,
            "prioritized_payload": payload_to_slm,
            "warning_text": warning_text,
            "metrics": metrics,
        }

    # releases pipeline resources
    def shutdown(self) -> None:
        if self.tts_engine is not None:
            self.tts_engine.shutdown()
            self.tts_engine = None

    def __enter__(self) -> FeedbackPipeline:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()




def main() -> None:
    parser = argparse.ArgumentParser(
        description="GoBraille Feedback Pipeline Entrypoint"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run in baseline mode (bypasses validation and prioritization)",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Enable streaming text-to-speech audio synthesis",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a JSON file containing a frame payload to process",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Ollama server host (default from .env or http://localhost:11434)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model name (default from .env or qwen2.5:1.5b)",
    )
    args = parser.parse_args()

    if not args.input:
        logger.error("No input payload provided. Please specify a JSON frame file using --input <path>.")
        parser.print_help()
        sys.exit(1)

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            frame_data = json.load(f)
        logger.info("Processing frame from file: %s", args.input)
    except Exception as e:
        logger.error("Failed to read input JSON file '%s': %s", args.input, e)
        sys.exit(1)

    pipeline = FeedbackPipeline(
        ollama_host=args.host,
        ollama_model=args.model,
        run_baseline=args.baseline,
        use_tts=args.tts,
    )

    try:
        result = pipeline.process_frame(frame_data)

        print("\n" + "=" * 60)
        print(" PIPELINE EXECUTION RESULT ")
        print("=" * 60)
        print(f"Warning Generated : \"{result['warning_text']}\"")
        print(f"Dominant Risk     : {result['metrics'].get('dominant_risk', 'N/A')}")
        print(f"End-to-End Latency: {result['metrics'].get('end_to_end_latency_ms', 0.0)} ms")
        print(f"TTFT (First Token): {result['metrics'].get('ttft_ms', 0.0)} ms")
        if pipeline.use_tts:
            print(f"TTFA (First Audio): {result['metrics'].get('ttfa_ms', 0.0)} ms")
        print("=" * 60 + "\n")

    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
