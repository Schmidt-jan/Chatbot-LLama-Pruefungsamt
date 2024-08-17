from typing import List, Union
import custom_rag_loader as crl
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from rag_test_output import RagTestOutput
from chromadb.utils import embedding_functions

class MyEmbeddingFunction(Embeddings):
    def __init__(self, model_name):
        self.model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.call_embedding_function(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.call_embedding_function(text)[0]
    
    def call_embedding_function(self, texts: Union[str, List[str]]) -> List[List[float]]:
        text_list = texts if isinstance(texts, list) else [texts]
        embedding =  self.model._model.encode(
                text_list,
                convert_to_numpy=True
            ).tolist()
        return embedding


def load_db(retriever, chunk_size, chunk_overlap):
    config = crl.RagConfig(
        model=crl.SupportedModels.Mistral,
        db_embedding_model=retriever,
        db_chunk_size=chunk_size,
        db_chunk_overlap=chunk_overlap,
        version="v2",
        distance="l2",
    )
    
    sentence_transformer_ef = MyEmbeddingFunction(retriever.value)
    db = Chroma(
        persist_directory=config.db_path, 
        collection_name=config.db_collection, 
        embedding_function=sentence_transformer_ef
    )

    return db

def load_config(config):

    retriever = config.get('retriever', None)
    if (retriever is None):
        raise Exception("retriever not provided")
    retriever = crl.DbSupportedEmbeddingModels(retriever)
    k = config.get('knn', 4)
    search_metric = config.get('search_metric', 'similar')
    chunk_size = config.get('chunk_size', 1024)
    chunk_size = crl.DbSupportedChunkSizes(chunk_size)
    chunk_overlap = config.get('chunk_overlap', 128)
    chunk_overlap = crl.DbSupportedChunkOverlap(chunk_overlap)

    return retriever, k, search_metric, chunk_size, chunk_overlap

def clean_output(documents: List[Document]) -> List[Document]:
    # drop metadata._node_content
    for doc in documents:
        doc.metadata.pop('_node_content', None)

    return documents

def call_api(prompt, options, context):
    prompt = context['vars']['question']
    config = options.get('config', None)
    retriever, k, search_metric, chunk_size, chunk_overlap = load_config(config)
    
    db = load_db(retriever, chunk_size, chunk_overlap)
    documents = db.search(prompt, search_metric, k=k)
    documents = clean_output(documents)

    result = RagTestOutput(documents, retriever.value)
    result = {
        "output": result.to_json(),
        "context": result.to_dict()
    }

    return result