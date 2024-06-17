import json

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
                generation_results.append({**result["response"]["output"], **{
                    'question': result["prompt"]["raw"],
                    'latency_ms': result["latencyMs"]
                }})
    return generation_results 

input_json_file_path = '/home/tpllmws23/Chatbot-LLama-Pruefungsamt/llm_eval/output/output2.json'
output_json_file_path_componentResults = '/home/tpllmws23/Chatbot-LLama-Pruefungsamt/llm_eval/output/output_componentResults2.json'
output_json_file_path_generation_results = '/home/tpllmws23/Chatbot-LLama-Pruefungsamt/llm_eval/output/output_generation_results2.json'

with open(input_json_file_path, "r", encoding="utf-8") as file:
    json_data = json.load(file)

all_component_results = extract_component_results(json_data)

with open(output_json_file_path_componentResults, "w", encoding="utf-8") as output_file:
    json.dump(all_component_results, output_file, indent=2)

print("Extrahierte componentResults wurden in die Datei", output_json_file_path_componentResults, "geschrieben.")




with open(input_json_file_path, "r", encoding="utf-8") as file:
    json_data = json.load(file)

all_generation_results = extract_generation_results(json_data)



with open(output_json_file_path_generation_results, "w", encoding="utf-8") as output_file:
    json.dump(all_generation_results, output_file, indent=2)

print("Extrahierte generation_results wurden in die Datei", output_json_file_path_generation_results, "geschrieben.")
