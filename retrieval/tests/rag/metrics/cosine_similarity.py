from typing import Any, Dict, Union
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from rag_test_output import get_required_vals
from helper import calculate_documents_score, ScoreCalculationType

METRIC = 'cosine_similarity'

def cosine_score(retrieved: str, expexted: str, embedding: SentenceTransformerEmbeddings | None = None, keywords: list[str] | None = None) -> float:
    if embedding is None:
        raise ValueError("Embedding model is not provided")
    embedded_retrieved = embedding.embed_query(retrieved)
    embedded_expected = embedding.embed_query(expexted)
    return cosine_similarity([embedded_retrieved], [embedded_expected])[0][0]


def get_assert(output, context) -> Union[bool, float, Dict[str, Any]]:
    test_config = get_required_vals(output, context, 0.75, METRIC)

    model_name = test_config.rag_test_output.embedding_model
    embedding = SentenceTransformerEmbeddings(model_name=model_name)

    return calculate_documents_score(test_config, cosine_score, METRIC, ScoreCalculationType.AVG, embedding)
