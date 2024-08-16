import json
import sys
import os

def extract_component_results(json_data):
    component_results = []
    if "results" in json_data and "results" in json_data["results"]:
        results = json_data["results"]["results"]
        for result in results:
            if "gradingResult" in result and "componentResults" in result["gradingResult"]:
                component_results.extend(result["gradingResult"]["componentResults"])
    return component_results

def extract_generation_results(json_data):
    generation_results = []
    if "results" in json_data and "results" in json_data["results"]:
        results = json_data["results"]["results"]
        for result in results:
            if "response" in result and "output" in result["response"]:
                generation_results.append({
                    **result["response"]["output"],
                    'question': result["prompt"]["raw"],
                    'latency_ms': result["latencyMs"]
                })
    return generation_results

def create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")
    except OSError as e:
        print(f"Error creating directory '{path}': {e}")

def main(timestamp):
    base_path = '/home/tpllmws23/Chatbot-LLama-Pruefungsamt/llm_eval/output'
    input_json_file_path = os.path.join(base_path, 'promptfoo_data', 'raw', f'output_{timestamp}.json')
    output_component_results_path = os.path.join(base_path, 'promptfoo_data', 'processed', 'componentResults')
    output_generation_results_path = os.path.join(base_path, 'promptfoo_data', 'processed', 'generationResults')

    create_directory(output_component_results_path)
    create_directory(output_generation_results_path)

    output_json_file_path_componentResults = os.path.join(output_component_results_path, f'output_cR_{timestamp}.json')
    output_json_file_path_generationResults = os.path.join(output_generation_results_path, f'output_gR_{timestamp}.json')

    with open(input_json_file_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    all_component_results = extract_component_results(json_data)

    with open(output_json_file_path_componentResults, "w", encoding="utf-8") as output_file:
        json.dump(all_component_results, output_file, indent=2)

    print("Extrahierte componentResults wurden in die Datei", output_json_file_path_componentResults, "geschrieben.")

    all_generation_results = extract_generation_results(json_data)

    with open(output_json_file_path_generationResults, "w", encoding="utf-8") as output_file:
        json.dump(all_generation_results, output_file, indent=2)

    print("Extrahierte generationResults wurden in die Datei", output_json_file_path_generationResults, "geschrieben.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Bitte geben Sie einen Zeitstempel als Argument an.")
        sys.exit(1)

    timestamp = sys.argv[1]
    main(timestamp)
