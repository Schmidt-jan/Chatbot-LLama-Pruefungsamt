from typing import Any, Dict, Union
from rag_test_output import get_required_vals
import evaluate
from promptfoo import Assertion, GradingResult
from typing import List, Tuple
from rouge import Rouge
from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from helper import calculate_documents_score, ScoreCalculationType

METRIC = 'rouge-2'

def rouge2_score(retrieved: str, expexted: str, embedding: SentenceTransformerEmbeddings | None = None, keywords: list[str] | None = None) -> float:
    rouge = Rouge()
    results = rouge.get_scores(retrieved, expexted, avg=True)
    return results[METRIC]['f']


def get_assert(output, context) -> Union[bool, float, Dict[str, Any]]:
    test_config = get_required_vals(output, context, 0.75, METRIC)

    return calculate_documents_score(test_config, rouge2_score, METRIC, ScoreCalculationType.AVG)
