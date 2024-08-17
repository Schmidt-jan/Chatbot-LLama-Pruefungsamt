import pandas as pd
from rag_test_output import RagTestOutput
import uuid
from langchain_core.documents import Document
import argparse
import uuid
import promptfoo
import datetime
import json

from sympy import comp

def create_df(test_id: str, embedding_model: str, question: str, expected_answer: str, avg_score: float, returned_document: Document, document_score: float, threshold: float, metric: str):
    return pd.DataFrame({
        'test_id': test_id,
        'embedding_model': embedding_model,
        'question': question,
        'expected_answer': expected_answer,
        'avg_score': avg_score,
        'returned_document': returned_document,
        'document_score': document_score,
        'threshold': threshold,
        'metric': metric
    })




# def main(input_path, output_file):
#     # load json input 
#     with open(input_path, 'r') as file:
#         input_str = file.read()
#         promptfoo_output = json.loads(input_str)
# 
# 
#     res_df = pd.DataFrame(columns=['test_id', 'embedding_model', 'knn', 'chunk_size', 'chunk_overlap', 'question', 'expected_answer', 'avg_test_score', 'returned_document', 'document_score', 'threshold', 'metric'])
#     
#     
#     for idx, result in enumerate(results.results):
#         rag_test_output = RagTestOutput.from_json(result.response.output)
#         embedding_model = rag_test_output.embedding_model
#         question = result.vars.question
#         expected_answer = result.vars.answer
#         test_id = str(uuid.uuid4())
#         
#         for componentResult in result.gradingResult.componentResults:
#             avg_score = componentResult.score
#             threshold = componentResult.assertion.threshold
#             metric = componentResult.assertion.metric
# 
#             for doc_idx, doc in enumerate(rag_test_output.documents):
#                 document_score = componentResult.componentResults[doc_idx].score
#                 new_row = [test_id, embedding_model, question, expected_answer, avg_score, doc, document_score, threshold, metric]
#                 res_df.loc[len(res_df)] = new_row
# 
#     # save as csv
#     res_df.to_json(output_file, orient='records')
# 
# timestamp = datetime.datetime.now().isoformat()
# output_file = 'output_' + timestamp + '.json'
# main("/home/tpllmws23/Chatbot-LLama-Pruefungsamt/Chatbot-Jan/tests/_output/output.json", output_file)

# 

def transform_json(original_json):
    unique_providers = []
    output_format = []

    for result in original_json['results']['results']:
        if result['provider']['id'] in unique_providers:
            provider_id = unique_providers.index(result['provider']['id'])
        else:
            provider_id = len(unique_providers)
            unique_providers.append(result['provider']['id'])

        extracted_provider_info = json.loads(result['provider']['id'])
        result['provider'] = extracted_provider_info

        tests_response = {}

        tests_response['question'] = result['vars']['question']
        tests_response['provider'] = {
                'id': provider_id,
                'name': result['provider']['retriever'],
                'knn': result['provider']['knn'],
                'search_metric': result['provider']['search_metric'],
                'chunk_size': result['provider']['chunk_size'],
                'chunk_overlap': result['provider']['chunk_overlap'],
                'tests': [],
        }
        tests_response['expected_response'] = result['vars']['expected_response_data']

        test_id = str(uuid.uuid4())
        returned_docs = result['response']['context']['documents']
        for test_output in result['gradingResult']['componentResults']:
            test_info = {
                'test_id': test_id,
                'metric': test_output['componentResults'][0]['assertion']['metric'],
                'threshold': test_output['assertion']['threshold'],
                'score': test_output['score'],
                'pass': test_output['pass'],
                'question': result['vars']['question'],
                'subResults': []
            }

            for doc_idx, component_result in enumerate(test_output['componentResults']):
                doc_data = {}
                doc_data['doc'] = returned_docs[doc_idx]
                doc_data['score'] = component_result['score']
                doc_data['pass'] = component_result['pass']

                test_info['subResults'].append(doc_data)

            tests_response['provider']['tests'].append(test_info)

        output_format.append(tests_response)
    return output_format


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process input string and write to output file')
    parser.add_argument('-f', '--input_str', type=str, help='Input string to process', default='/home/tpllmws23/Chatbot-LLama-Pruefungsamt/Chatbot-Jan/tests/_output/output.json')
    parser.add_argument('-o', '--output_file', type=str, help='Output file to write processed string')
    args = parser.parse_args()

    input_str = args.input_str
    output_file = args.output_file

    if output_file is None:
        # get the timestamp int the ISO format
        timestamp = datetime.datetime.now().isoformat()
        output_file = 'output_' + timestamp + '.json'

    with open(input_str, 'r') as file:
        input_str = file.read()
    promptfoo_output = json.loads(input_str)
    transformed_json = transform_json(promptfoo_output)
    with open(output_file, 'w') as file:
        json.dump(transformed_json, file)

