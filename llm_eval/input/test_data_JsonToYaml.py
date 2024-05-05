import json
import os
import yaml

def convert_json_to_yaml(json_file, yaml_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    yaml_tests = []

    for test in data['tests']:
        for question in test['questions']:
            for answer in test['answers']:
                yaml_test = f"""
- vars:
    ids: {{"test": "{test['id']}", "question": "{question['id']}", "answer": "{answer['id']}"}}
    question: "{question['text']}"
    answer: "{answer['text']}"
    quality: {{"specificity": {question['quality']['specificity']}, "relevance": {question['quality']['relevance']}, "answer_correctness": {answer['correctness']}}}
    citation: {{ "document": "{answer['cite']['document']}", "page": {answer['cite']['page']}}}
  assert:"""

                for metric in data['metrics']:
                    yaml_test += f"""
    - type: "python"
      value: "file://{metric}" """

                yaml_tests.append(yaml_test)

    with open(yaml_file, 'w') as f:
        f.write('\n'.join(yaml_tests))


json_file = '/home/tpllmws23/Chatbot-LLama-Pruefungsamt/llm_eval/input/test_data.json'
yaml_file = '/home/tpllmws23/Chatbot-LLama-Pruefungsamt/llm_eval/input/tests.yaml'

if os.path.exists(yaml_file):
    os.remove(yaml_file)

convert_json_to_yaml(json_file, yaml_file)
