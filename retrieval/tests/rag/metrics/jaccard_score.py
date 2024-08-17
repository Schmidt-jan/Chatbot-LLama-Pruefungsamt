from typing import Any, Dict, Union
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from rag_test_output import get_required_vals
from helper import calculate_documents_score, ScoreCalculationType

METRIC = 'jaccard'

def jaccard(retrieved: str, expexted: str, embedding: SentenceTransformerEmbeddings | None = None, keywords: list[str] | None = None) -> float:
    words_t1 = set(retrieved.split())
    words_t2 = set(expexted.split())
    intersection = words_t1.intersection(words_t2)
    union = words_t1.union(words_t2)
    return float(len(intersection)) / len(union)


def get_assert(output, context) -> Union[bool, float, Dict[str, Any]]:
    test_config = get_required_vals(output, context, 0.75, METRIC)

    return calculate_documents_score(test_config, jaccard, METRIC, ScoreCalculationType.AVG)
