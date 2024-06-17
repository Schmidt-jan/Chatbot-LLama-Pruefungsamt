from math import exp
from langchain_core.documents import Document
from typing import List
import json 
from promptfoo import Assertion

class RagTestOutput:
    documents: List[Document]
    embedding_model: str
    tags: List[str]
    
    def __init__(self, documents: List[Document], embedding_model: str, tags: List[str] = []):
        self.documents = documents
        self.embedding_model = embedding_model
        self.tags = tags
    
    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    
    @classmethod
    def from_json(cls, json_str):
        json_dict = json.loads(json_str)
        documents = [Document(**doc) for doc in json_dict["documents"]]
        return cls(documents, json_dict["embedding_model"], json_dict["tags"])
    
    def to_dict(self):
        return {
            "documents": [doc.__dict__ for doc in self.documents],
            "embedding_model": self.embedding_model,
            "tags": self.tags
        }


class ExpextedDataInResponse:
    def __init__(self, answer, page, keywords):
        self.answer = answer
        self.page = page
        self.keywords = keywords

class ContextLoad:
    def __init__(self, question: str, expected_data_in_response: ExpextedDataInResponse, rag_test_output: RagTestOutput, threshold: float):
        self.question = question
        self.expected_data_in_response = expected_data_in_response
        self.rag_test_output = rag_test_output
        self.threshold = threshold

def get_required_vals(output, context, fallback_threshold, metric_name: str) -> ContextLoad:

    prompt = context['vars']['question']
    expexted_response_data = context['vars']['expected_response_data']

    expected_data_in_response = ExpextedDataInResponse(**expexted_response_data)
    assertions = [Assertion(**a) for a in context['test']['assert']]
    
    # Get the correct assertion configurations
    metric_assertions = [a for a in assertions if a.value and metric_name in a.value]
    if len(metric_assertions) > 1:
        raise ValueError(f"Only one assertion with value containing '{metric_name}' is allowed")
    
    if (len(metric_assertions) == 0):
        raise ValueError(f"No assertion with value containing '{metric_name}' found")
    
    if (metric_assertions[0].threshold) is None:
        metric_assertions[0].threshold = fallback_threshold
    
    threshold = metric_assertions[0].threshold
    rag_test_output = RagTestOutput.from_json(output)

    return ContextLoad(prompt, expected_data_in_response, rag_test_output, threshold)