from enum import Enum
from typing import List, Tuple, Callable, Optional
from promptfoo import GradingResult, Assertion, ComponentResult
from langchain_core.documents import Document
from pyparsing import C
from rag_test_output import ContextLoad
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from torch import embedding
from typing import Any, Dict, Union

# enum for min max or avg
class ScoreCalculationType(Enum):
    MIN= 'min'
    MAX = 'max'
    AVG = 'avg'

def create_assertion_and_grading_result(context: ContextLoad, score: float, threshold: float, metric: str, components_results: List[Dict[str, Any]]) -> Union[bool, float, Dict[str, Any]]:
    assertion = Assertion(
        type='similarity', 
        threshold=threshold, 
        provider=context.rag_test_output.embedding_model, 
        metric=metric)
    
    return {
        'pass': bool(score > threshold),
        'score': score,
        'reason': f"Similarity score: {score}, threshold: {threshold}",
        'componentResults': components_results,
        'assertion': assertion.__dict__
    }
        


def calculate_documents_score(context: ContextLoad, 
                     score_function: Callable[[str, str, SentenceTransformerEmbeddings | None, list[str] | None], float],
                     metric: str,
                     score_calculation_type: ScoreCalculationType = ScoreCalculationType.AVG,
                     embedding: SentenceTransformerEmbeddings | None = None
                     ) -> Union[bool, float, Dict[str, Any]]:

    components_results: List[Dict[str, Any]] = []
    scores = []

    for doc in context.rag_test_output.documents:
        score = score_function(doc.page_content, context.expected_data_in_response.answer, embedding, context.expected_data_in_response.keywords)
        scores.append(score)
        pass_ = bool(score > context.threshold)
        reason = f"Score: {score}, Threshold: {context.threshold}"
        assertion = {
            'type': 'similarity',
            'threshold': context.threshold,
            'provider': context.rag_test_output.embedding_model,
            'metric': metric
        }
        
        grading_result = {
            'assertion': assertion,
            'pass': pass_,
            'score': score,
            'reason': reason
        }
        components_results.append(grading_result)

    if score_calculation_type == ScoreCalculationType.MAX:
        score = max(scores)
    elif score_calculation_type == ScoreCalculationType.AVG:
        score = sum(scores) / len(scores)
    else:
        score = min(scores)

    return create_assertion_and_grading_result(context, score, context.threshold, metric, components_results)
