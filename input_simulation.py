import json
import logging

# data logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DatasetLoader")


# loads all scenarios from the dataset file into a list of dicts
def loadScenarios(filepath: str = "scene_dataset.json") -> list[dict]:
    # load scene data from scene_dataset.json
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            scenarios = json.load(f)
        logger.info("Successfully loaded %d scenarios from '%s'.", len(scenarios), filepath)
        return scenarios
    except FileNotFoundError:
        logger.error("Dataset file not found at path: %s", filepath)
        raise
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON from '%s'. Error: %s", filepath, e)
        raise


# load test
if __name__ == "__main__":
    try:
        all_scenarios = loadScenarios()
        print(f"Successfully loaded {len(all_scenarios)} scenarios.")

        if all_scenarios:
            first_scenario = all_scenarios[0]
            print(f"\n--- First Scenario Summary ---")
            print(f"ID: {first_scenario.get('scenario_id')}")
            print(f"Name: {first_scenario.get('name')}")
            print(f"Frames: {len(first_scenario.get('frames', []))}")
            print(f"Ground Truth: {first_scenario.get('ground_truth')}")

    except Exception as e:
        print(f"An error occurred: {e}")