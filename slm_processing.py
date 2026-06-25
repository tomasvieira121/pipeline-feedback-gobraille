import json
import logging
import os
import time
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SLMProcessing")


# ollama model parameters

OLLAMA_HOST: str  = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL")

# max tokens per risk, so that higher risk gets shorter text
MAX_TOKENS_BY_RISK: dict[str, int] = {
    "high":    40,   
    "medium":  70,  
    "low":     90,  
    "default": 60,   
}

# prompt guidelines
# system prompt is assembled from a risk-specific instruction + shared rules
# the instruction tone scales with urgency: imperative for high, calm for low

# risk specific instruct set
# one entry per risk level; keys must match the values used in semantic_relations
_RISK_INSTRUCTIONS: dict[str, str] = {
    "high": (
        "You are an emergency navigation assistant for a visually impaired pedestrian. "
        "There is an IMMEDIATE DANGER. Issue a SHORT, URGENT WARNING in one or two sentences. "
        "Be direct and imperative. Example: 'Stop immediately. A car is approaching fast from the front.' "
        "Do NOT explain or add extra context."
    ),
    "medium": (
        "You are a navigation assistant for a visually impaired pedestrian. "
        "There is a moderate obstacle nearby. Describe it concisely in one or two sentences, "
        "including its type, approximate direction, and any recommended action. "
        "Use a calm but attentive tone."
    ),
    "low": (
        "You are a navigation assistant for a visually impaired pedestrian. "
        "The environment is mostly clear. Briefly mention any minor obstacle detected "
        "in a reassuring tone. Keep the message to one sentence."
    ),
}

# rules appended to every prompt regardless of risk level
_COMMON_RULES = (
    "\n\nRules you MUST follow:\n"
    "- Reply ONLY with the spoken warning, no JSON, no explanations, no bullet points.\n"
    "- Use natural spoken language suitable for Text-to-Speech output.\n"
    "- Never mention bounding boxes, coordinates, confidence scores, or track IDs.\n"
    "- Base your warning exclusively on the JSON data provided by the user.\n"
    "- If the detections list is empty, say: 'Path appears clear. Proceed with caution.'"
)


# assembles the final system prompt from the risk instruction + common rules
def build_system_prompt(dominant_risk: str) -> str:

    instruction = _RISK_INSTRUCTIONS.get(
        dominant_risk.lower(),
        _RISK_INSTRUCTIONS["low"],  # fallback to low if risk level is unknown
    )
    return instruction + _COMMON_RULES


# utils

# returns the highest risk level found across all detections
def _get_dominant_risk(detections: list[dict]) -> str:

    priority_order = ["high", "medium", "low"]

    for level in priority_order:
        for det in detections:
            risk = (
                det.get("semantic_relations", {})
                   .get("risk_level", "")
                   .lower()
            )
            if risk == level:
                return level

    return "default"


# removes internal pipeline fields (prefixed with _) before sending to the model
def _strip_internal_fields(detections: list[dict]) -> list[dict]:

    cleaned = []
    for det in detections:
        cleaned.append({k: v for k, v in det.items() if not k.startswith("_")})
    return cleaned


# serializes the payload to JSON, stripping internal fields first
def _build_user_message(payload: dict) -> str:

    clean_payload = dict(payload)
    clean_payload["detections"] = _strip_internal_fields(
        payload.get("detections", [])
    )
    return json.dumps(clean_payload, indent=2, ensure_ascii=False)


# ollama client
# wraps the /api/chat endpoint; supports both single-shot and streaming inference

class OllamaClient:

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = OLLAMA_MODEL,
        timeout: int = 30,
    ) -> None:
        if not host:
            raise ValueError(
                "OLLAMA_HOST is not set. Define it in your .env file or pass it explicitly."
            )
        if not model:
            raise ValueError(
                "OLLAMA_MODEL is not set. Define it in your .env file or pass it explicitly."
            )
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._chat_url = f"{self.host}/api/chat"

    # single-shot inference — returns text only, no streaming
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 60,
    ) -> str:

        payload = {
            "model": self.model,
            "stream": False,
            "options": {
                "num_predict": max_tokens,  # token budget
                "temperature": 0.2,         # low randomness for consistent warnings
                "top_p": 0.9,
            },
            "messages": [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_message},
            ],
        }

        logger.debug("Sending request to Ollama: model=%s, max_tokens=%d", self.model, max_tokens)

        t_start = time.perf_counter()
        response = requests.post(
            self._chat_url,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        data: dict[str, Any] = response.json()
        generated_text: str = (
            data.get("message", {})
                .get("content", "")
                .strip()
        )

        logger.info(
            "Inference completed in %.0f ms | tokens generated: %s | model: %s",
            elapsed_ms,
            data.get("eval_count", "N/A"),
            self.model,
        )

        return generated_text

    # streaming inference — returns text + latency/token metrics dict
    def generate_with_metrics(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 60,
    ) -> tuple[str, dict]:

        payload = {
            "model": self.model,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.2,
                "top_p": 0.9,
            },
            "messages": [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_message},
            ],
        }

        logger.debug("Sending streaming request to Ollama: model=%s", self.model)

        t_start = time.perf_counter()
        response = requests.post(
            self._chat_url,
            json=payload,
            stream=True,
            timeout=self.timeout,
        )
        response.raise_for_status()

        t_first_token = None
        generated_text = ""
        final_data = {}

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8"))

                # records the first generated token (TTFT)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    generated_text += content

                if chunk.get("done", False):
                    final_data = chunk

        t_end = time.perf_counter()

        # time calculations (in milliseconds)
        ttft_ms = (t_first_token - t_start) * 1000 if t_first_token else (t_end - t_start) * 1000
        latency_ms = (t_end - t_start) * 1000

        # convert Ollama durations (nanoseconds -> milliseconds)
        prompt_eval_duration_ms = final_data.get("prompt_eval_duration", 0) / 1_000_000.0
        eval_duration_ms = final_data.get("eval_duration", 0) / 1_000_000.0
        total_duration_ms = final_data.get("total_duration", 0) / 1_000_000.0

        metrics = {
            "ttft_ms": round(ttft_ms, 1),
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": final_data.get("prompt_eval_count", 0),
            "response_tokens": final_data.get("eval_count", 0),
            "ollama_prompt_eval_duration_ms": round(prompt_eval_duration_ms, 1),
            "ollama_eval_duration_ms": round(eval_duration_ms, 1),
            "ollama_total_duration_ms": round(total_duration_ms, 1),
        }

        logger.info(
            "Streaming inference completed: TTFT=%.1f ms | Latency=%.1f ms | Tokens=%d | model: %s",
            ttft_ms,
            latency_ms,
            metrics["response_tokens"],
            self.model,
        )

        return generated_text.strip(), metrics


# high-level processor
# orchestrates prompt building, token budgeting, and model inference

class SLMProcessor:

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = OLLAMA_MODEL,
        max_tokens_by_risk: dict[str, int] | None = None,
    ) -> None:
        self.client = OllamaClient(host=host, model=model)
        self.max_tokens_by_risk = max_tokens_by_risk or MAX_TOKENS_BY_RISK

    # prepares all inputs needed for inference (shared by process and process_with_metrics)
    def _prepare_inference(self, prioritized_payload: dict) -> tuple[str, str, str, int]:

        detections = prioritized_payload.get("detections", [])

        dominant_risk = _get_dominant_risk(detections)
        max_tokens = self.max_tokens_by_risk.get(
            dominant_risk,
            self.max_tokens_by_risk["default"],
        )

        logger.info(
            "Dominant risk: '%s' | token budget: %d | detections: %d",
            dominant_risk, max_tokens, len(detections),
        )

        system_prompt = build_system_prompt(dominant_risk)
        user_message  = _build_user_message(prioritized_payload)

        logger.debug("System prompt:\n%s", system_prompt)
        logger.debug("User message (payload):\n%s", user_message)

        return dominant_risk, system_prompt, user_message, max_tokens

    # runs inference without collecting metrics — suitable for live use
    def process(self, prioritized_payload: dict) -> str:

        _, system_prompt, user_message, max_tokens = self._prepare_inference(prioritized_payload)

        warning_text = self.client.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
        )

        logger.info("Generated warning: \"%s\"", warning_text)
        return warning_text

    # runs inference with streaming and returns full latency/token metrics
    def process_with_metrics(self, prioritized_payload: dict) -> tuple[str, dict]:

        dominant_risk, system_prompt, user_message, max_tokens = self._prepare_inference(prioritized_payload)

        warning_text, metrics = self.client.generate_with_metrics(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
        )

        metrics["dominant_risk"] = dominant_risk
        metrics["max_tokens_budget"] = max_tokens

        logger.info("Generated warning with metrics: \"%s\"", warning_text)
        return warning_text, metrics


if __name__ == "__main__":
    from input_simulation import loadScenarios
    from validation_prioritization import ValidationPrioritizationPipeline

    scenarios  = loadScenarios("scene_dataset.json")
    val_pipe   = ValidationPrioritizationPipeline()
    slm        = SLMProcessor()

    for sc in scenarios:
        val_pipe.reset()
        print("\n" + "=" * 65)
        print(f"  {sc['name']}")
        print("=" * 65)

        last_frame = sc["frames"][-1]
        prioritized = val_pipe.process(last_frame)

        print("\n[Prioritized payload sent to SLM]")
        preview = dict(prioritized)
        preview["detections"] = _strip_internal_fields(prioritized.get("detections", []))
        print(json.dumps(preview, indent=2, ensure_ascii=False))

        print("\n[Warning Generated by SLM → TTS]")
        try:
            warning = slm.process(prioritized)
            print(f"  ▶ {warning}")
        except requests.ConnectionError:
            print( "  Ollama server not accessible. " )
        except requests.HTTPError as exc:
            print(f" http error {exc}")
