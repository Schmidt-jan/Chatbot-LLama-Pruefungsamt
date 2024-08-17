from openai import OpenAI
import os
from typing import Any, Dict, Union, Tuple
from rag_test_output import RagTestOutput
import json
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from promptfoo import Assertion
from rag_test_output import get_required_vals


class DocumentRating:
    def __init__(self, idx: int, rating: float):
        self.idx = idx
        self.rating = rating

class ContextRating:
    def __init__(self, documents: list[DocumentRating]):
        self.documents = documents

    @classmethod
    def from_json(cls, data: dict) -> "ContextRating":
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")

        if "documents" not in data:
            raise ValueError("Missing 'documents' key in the JSON data")

        documents = [DocumentRating(doc["idx"], doc["rating"]) for doc in data["documents"]]
        return cls(documents)


def generate_prompt(question: str, documents_page_content: list[str]) -> str:
    return  f"""You should evaluate if a given context contains the required information to answer a question. 
                The context is given with an array of strings. Now you should create a rating between 0.0 (irrelevant) 
                and 1.0 (important) for each element in the context.
                Read the texts carefully and think about your answer!

                ONLY answer with a JSON object in the following format:
                "gpt":{{
                    "documents": [
                        {{ 
                            "idx": number,
                            "rating": number
                        }},
                        {{ 
                            "idx": number,
                            "rating": number
                        }},  
                }}

                
                Question:
                {question}

                Context:
                {documents_page_content}"""

def get_assert(output, context) -> Union[bool, float, Dict[str, Any]]:
    test_config = get_required_vals(output, context, 1.0, 'chat_gpt_context_relevance')
    documents_page_content = [doc.page_content for doc in test_config.rag_test_output.documents]
    gpt_prompt = generate_prompt(test_config.question, documents_page_content)
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{
        "role": "system",
        "content": gpt_prompt
        }
     ])
    response = completion.choices[0].message.content

    try:
        response = json.loads(response)
        context_rating = ContextRating.from_json(response["gpt"])
    except json.JSONDecodeError:
        return {
            'pass': False,
            'score': 0.0,
            'reason': 'Invalid JSON response from GPT-3.5-turbo',
        }
    
    assertion = Assertion(type='context relevance', value=None, provider=test_config.rag_test_output.embedding_model, metric='gpt-3.5-turbo')
    pass_ = any(doc.rating > test_config.threshold for doc in context_rating.documents)
    similarity = max(doc.rating for doc in context_rating.documents)
    return {
        'pass': pass_,
        'score': similarity,
        'componentResults': [componentResult.__dict__ for componentResult in context_rating.documents],
        'assertion': assertion.__dict__
    }