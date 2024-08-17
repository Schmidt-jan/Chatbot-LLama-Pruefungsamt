import pandas as pd
from typing import List
import json

class AssertionConfig:
    def __init__(self, type: str, value: str, threshold: float):
        self.type = type
        self.value = value
        self.threshold = threshold


def string_to_array(string: str) -> list[str]:
    elements  = string.split(';')
    elements = [element.strip() for element in elements]
    elements = [element.replace('\n', '') for element in elements]
    return elements

def string_to_int_array(input) -> list[int]:
    if isinstance(input, int):
        return [input]
    string = str(input)
    elements  = string.split(';')
    elements = [int(element.strip()) for element in elements]
    return elements

    
def generate_yaml_entry(row, asserts: List[AssertionConfig]):
    # create json object for the keywords, page and answer
    keywords = row['keywords']
    answer = row['answer']
    page = row['page']

    jsonObj = {
        "keywords": keywords,
        "page": page,
        "answer": answer
    }

    single_test =  f"- vars:\n"
    single_test += f"    question: \"{row['question']}\"\n"
    single_test += f"    expected_response_data: { json.dumps(jsonObj) }\n"
    
    if len(asserts) == 0:
        return single_test
    
    single_test += f"  assert:\n"

    for assert_ in asserts:
        single_test += f"    - type: {assert_.type}\n"
        single_test += f"      value: {assert_.value}\n"
        single_test += f"      threshold: {assert_.threshold}\n\n"

    return single_test

def excel_to_test_yaml(excel_file, yaml_file, asserts, num_rows=None):
    df = pd.read_excel(excel_file, nrows=num_rows)
    df['keywords'] = df['keywords'].apply(string_to_array)
    df['page'] = df['page'].apply(string_to_int_array)

    with open(yaml_file, 'w') as f:
        for _, row in df.iterrows():
            f.write(generate_yaml_entry(row, asserts))
    

if __name__ == "__main__":
    excel_file = "QA.xlsx"
    yaml_file = "../rag/tests.yaml"
    num_rows_to_use = 30

    # assert for the test generation
    # the key is mapped to the 'type', and the value is mapped to the 'value' in the test config
    asserts = [
#        AssertionConfig(type="python", value="file://metrics/cosine_similarity.py", threshold=0.75),
#        AssertionConfig(type="python", value="file://metrics/chat_gpt_context_relevance.py", threshold=0.75),
#        AssertionConfig(type="python", value="file://metrics/jaccard_score.py", threshold=0.75),
#        AssertionConfig(type="python", value="file://metrics/rouge-1_score.py", threshold=0.75),
#        AssertionConfig(type="python", value="file://metrics/rouge-2_score.py", threshold=0.75),
#        AssertionConfig(type="python", value="file://metrics/rouge-l_score.py", threshold=0.75),
        AssertionConfig(type="python", value="file://metrics/keywords_score.py", threshold=0.75)
    ]

    excel_to_test_yaml(excel_file, yaml_file, asserts, num_rows_to_use)
