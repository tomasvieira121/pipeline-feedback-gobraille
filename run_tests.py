import json
import os
import re
import sys
import time
from datetime import datetime

from input_simulation import loadScenarios
from validation_prioritization import ValidationPrioritizationPipeline
from slm_processing import SLMProcessor


# env with slm config
# reads a .env file and injects variables into the environment
def load_env_file(filepath=".env"):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    val = v.strip().strip("'").strip('"')
                    os.environ[k.strip()] = val
        print(f"Environment variables loaded from: {filepath}")
    else:
        print("env file not found.")


# sanitizes a string so it can be safely used as a filename
def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def main():
    load_env_file()

    ollama_host = os.getenv("OLLAMA_HOST")
    ollama_model = os.getenv("OLLAMA_MODEL")

    # baseline configuration from env or command line arguments
    env_baseline = os.getenv("RUN_BASELINE", "false").strip().lower() in ("true", "1", "yes")
    cli_baseline = "--baseline" in sys.argv
    cli_pipeline = "--pipeline" in sys.argv

    if cli_baseline:
        run_baseline = True
    elif cli_pipeline:
        run_baseline = False
    else:
        run_baseline = env_baseline

    mode_label = "BASELINE (Raw Unfiltered Input)" if run_baseline else "PIPELINE (Validated & Prioritized)"

    print("\n" + "=" * 80)
    print(f" STARTING ASSISTED NAVIGATION TEST ")
    print(f" Mode:        {mode_label}")
    print(f" SLM Model:   {ollama_model}")
    print(f" Ollama Host: {ollama_host}")
    print("=" * 80 + "\n")

    # start pipeline components
    try:
        scenarios_definition = loadScenarios("scene_dataset.json")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Failed to load scenarios: {e}")
        sys.exit(1)

    pipeline = ValidationPrioritizationPipeline(
        confidence_threshold=0.60,
        min_persistence_frames=2,
        top_n=3
    )

    try:
        processor = SLMProcessor(host=ollama_host, model=ollama_model)
    except Exception as e:
        print(f"Failed to initialize SLMProcessor: {e}")
        sys.exit(1)

    # results container
    results_data = {
        "test_metadata": {
            "timestamp": datetime.now().isoformat(),
            "ollama_model": ollama_model,
            "ollama_host": ollama_host,
            "run_baseline": run_baseline,
            "execution_mode": "baseline" if run_baseline else "pipeline",
            "confidence_threshold": 0.60,
            "min_persistence_frames": 2,
            "top_n_detections": 3
        },
        "scenarios": []
    }

    # flat list for aggregate metric calculation at the end
    flat_frames_metrics = []

    # scenario execution
    for idx, sc_def in enumerate(scenarios_definition, start=1):
        print(f"\nScenario {idx}: {sc_def['name']}")
        print(f"   Description: {sc_def['description']}")
        print("-" * 80)

        pipeline.reset()

        frames = sc_def["frames"]
        scenario_log = {
            "scenario_index": idx,
            "scenario_name": sc_def["name"],
            "description": sc_def["description"],
            "ground_truth": sc_def.get("ground_truth", {}),
            "frames": []
        }

        for f_idx, raw_frame in enumerate(frames, start=1):
            print(f"\n  [Frame {f_idx}/{len(frames)}] Processing...")

            if run_baseline:
                pipeline_time_ms = 0.0
                payload_to_slm = raw_frame
            else:
                t_pipeline_start = time.perf_counter()
                prioritized_payload = pipeline.process(raw_frame)
                pipeline_time_ms = (time.perf_counter() - t_pipeline_start) * 1000
                payload_to_slm = prioritized_payload

            # SLM processing and metric measurement
            raw_dets_count = len(raw_frame.get("detections", []))
            sent_dets_count = len(payload_to_slm.get("detections", []))

            warning_text = ""
            metrics = {}

            try:
                warning_text, metrics = processor.process_with_metrics(payload_to_slm)
                metrics["pipeline_processing_time_ms"] = round(pipeline_time_ms, 1)
                metrics["end_to_end_latency_ms"] = round(metrics["latency_ms"] + pipeline_time_ms, 1)
                flat_frames_metrics.append(metrics)
            except Exception as e:
                print(f"  ❌ Error during SLM inference: {e}")
                warning_text = f"[INFERENCE ERROR: {e}]"
                metrics = {
                    "ttft_ms": 0.0,
                    "latency_ms": 0.0,
                    "pipeline_processing_time_ms": round(pipeline_time_ms, 1),
                    "end_to_end_latency_ms": 0.0,
                    "error": str(e)
                }

            # log frame data
            frame_log = {
                "frame_index": f_idx,
                "raw_input": raw_frame,
                "prioritized_payload": payload_to_slm,
                "warning_text": warning_text,
                "metrics": metrics
            }
            scenario_log["frames"].append(frame_log)

            # displays frame summary in the terminal
            if run_baseline:
                print(f"     └─ Detections: {raw_dets_count} raw (baseline unfiltered)")
            else:
                print(f"     └─ Detections: {raw_dets_count} raw ➔ {sent_dets_count} validated/prioritized")
            if "dominant_risk" in metrics:
                print(f"     └─ Dominant Risk: '{metrics['dominant_risk'].upper()}'")
            print(f"     └─ TTFT (Client): {metrics.get('ttft_ms')} ms")
            print(f"     └─ E2E Latency: {metrics.get('end_to_end_latency_ms')} ms")
            print(f"     └─ Generated Warning: \"{warning_text}\"")

        results_data["scenarios"].append(scenario_log)

    # test metrics calculation
    successful_frames = [f for f in flat_frames_metrics if "error" not in f]
    if successful_frames:
        avg_ttft = sum(f["ttft_ms"] for f in successful_frames) / len(successful_frames)
        avg_latency = sum(f["end_to_end_latency_ms"] for f in successful_frames) / len(successful_frames)
        total_tokens = sum(f["response_tokens"] for f in successful_frames)
    else:
        avg_ttft = 0.0
        avg_latency = 0.0
        total_tokens = 0

    results_data["test_summary"] = {
        "total_scenarios": len(scenarios_definition),
        "total_frames": len(flat_frames_metrics),
        "avg_ttft_ms": round(avg_ttft, 1),
        "avg_end_to_end_latency_ms": round(avg_latency, 1),
        "total_tokens_generated": total_tokens,
        "successful_inferences": len(successful_frames),
        "failed_inferences": len(flat_frames_metrics) - len(successful_frames)
    }

    # output results file
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_model = sanitize_filename(ollama_model)
    baseline_suffix = "_baseline" if run_baseline else ""
    filename = f"results_{sanitized_model}{baseline_suffix}_{timestamp_str}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    # final terminal output
    print("\n" + "=" * 80)
    print(" TEST EXECUTION REPORT ")
    print("=" * 80)
    print(f"Log File: {filepath}")
    print(f"Mode:     {mode_label}")
    print(f"Model:    {ollama_model}")
    print(f"Average TTFT (Time-to-First-Token): {results_data['test_summary']['avg_ttft_ms']} ms")
    print(f"Average End-to-End Latency:      {results_data['test_summary']['avg_end_to_end_latency_ms']} ms")
    print(f"Total Generated Tokens:          {results_data['test_summary']['total_tokens_generated']}")
    print(f"Successful / Total Inferences:   {results_data['test_summary']['successful_inferences']} / {results_data['test_summary']['total_frames']}")
    print("-" * 80)

    det_col_header = "Raw (Base)" if run_baseline else "Raw ➔ Val"
    print(f"{'Scenario':<10} | {'Frame':<5} | {det_col_header:<12} | {'Risk':<8} | {'TTFT (ms)':<10} | {'E2E Lat (ms)':<12}")
    print("-" * 80)
    for sc in results_data["scenarios"]:
        sc_num = sc["scenario_index"]
        for fr in sc["frames"]:
            f_num = fr["frame_index"]
            raw_c = len(fr["raw_input"].get("detections", []))
            val_c = len(fr["prioritized_payload"].get("detections", []))
            det_str = f"{raw_c} (raw)" if run_baseline else f"{raw_c} ➔ {val_c}"
            m = fr["metrics"]
            risk = m.get("dominant_risk", "N/A")
            ttft = m.get("ttft_ms", 0.0)
            e2e = m.get("end_to_end_latency_ms", 0.0)
            print(f"Scenario {sc_num:<2} | {f_num:<5} | {det_str:<12} | {risk:<8} | {ttft:<10.1f} | {e2e:<12.1f}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()