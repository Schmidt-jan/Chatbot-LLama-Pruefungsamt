import chunk
from math import dist
import os
from typing import Tuple

from langchain.llms.llamacpp import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
from langchain.vectorstores.chroma import Chroma
from enum import Enum
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

LLM_DIR = "/home/tpllmws23/llms"
DB_DIR = "/home/tpllmws23/Chatbot-LLama-Pruefungsamt/Chatbot-Jan/databases"

class SupportedModels(Enum):
    Mistral = "mistralai/Mistral-7B-Instruct-v0.1"
    Llama2 = "meta-llama/Llama-2-13b-chat-hf"
    Vicuna = "lmsys/vicuna-13b-v1.5"
    Mixtral_Q3 ="mistralai/Mixtral-8x7B-Instruct-v0.1"
    Llama3 = "meta-llama/Meta-Llama-3-8B-Instruct.Q8_0"
    
class DbSupportedEmbeddingModels(Enum):
    All_MiniLM_L6_v2 = "sentence-transformers/all-MiniLM-L6-v2"
    All_mpnet_base_v2 = "sentence-transformers/all-mpnet-base-v2"
    Paraphrase_MiniLM_L6_v2 = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    Paraphrase_multilingual_MiniLM_L12_v2 = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    Multi_qa_mpnet_base_dot_v1 = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    Mxbai_embed_large_v1 = "mixedbread-ai/mxbai-embed-large-v1"
    German_roberta_sentence_transformer_v2 = "T-Systems-onsite/german-roberta-sentence-transformer-v2"

class DbSupportedCtxSizes(Enum):
    Ctx_1024 = 1024

class DbSupportedChunkSizes(Enum):
    Chunk_128 = 128
    Chunk_256 = 256
    Chunk_512 = 512
    Chunk_1024 = 1024
    Chunk_2048 = 2048

class DbSupportedChunkOverlap(Enum):
    Overlap_0 = 0
    Overlap_64 = 64
    Overlap_128 = 128
    Overlap_256 = 256
    Overlap_512 = 512

class RagConfig:
    def __init__(self, 
                 model: SupportedModels, 
                 n_ctx: int = 4096, 
                 db_embedding_model: DbSupportedEmbeddingModels | SupportedModels  = DbSupportedEmbeddingModels.All_MiniLM_L6_v2,
                 db_n_ctx: DbSupportedCtxSizes = DbSupportedCtxSizes.Ctx_1024,
                 db_chunk_size: DbSupportedChunkSizes = DbSupportedChunkSizes.Chunk_512,
                 db_chunk_overlap: DbSupportedChunkOverlap = DbSupportedChunkOverlap.Overlap_64,
                 db_collection: str = "Pruefungsamt", 
                 version: str = "v1",
                 distance: str = "l2",
                 use_streaming = True):

            
        self.model_path, self.db_path = self.generate_model_path_and_db_path(model,
                                                                             db_embedding_model,
                                                                             db_n_ctx,
                                                                             db_chunk_size,
                                                                             db_chunk_overlap,
                                                                             version,
                                                                             distance)
        self.db_collection = db_collection
        self.n_ctx = n_ctx
        self.use_streaming = use_streaming
        self.db_embedding_model = db_embedding_model




    def generate_model_path_and_db_path(self, 
                                        model_name: SupportedModels, 
                                        db_embedding_model: DbSupportedEmbeddingModels | SupportedModels = DbSupportedEmbeddingModels.All_MiniLM_L6_v2,
                                        db_n_ctx: DbSupportedCtxSizes = DbSupportedCtxSizes.Ctx_1024,
                                        db_chunk_size: DbSupportedChunkSizes = DbSupportedChunkSizes.Chunk_512,
                                        db_chunk_overlap: DbSupportedChunkOverlap = DbSupportedChunkOverlap.Overlap_64,
                                        version: str = "v1",
                                        distance: str = "l2",
                                        ) -> Tuple[str, str]:
        

            
        if db_chunk_overlap.value > db_chunk_size.value / 2:
            raise ValueError("Overlap must be smaller than half of the chunk size")

        if model_name == SupportedModels.Mistral:
            relative_model_path = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

        elif model_name == SupportedModels.Llama2:
            relative_model_path = "llama-2-13b-chat.Q4_K_M.gguf"

        elif model_name == SupportedModels.Vicuna:
            relative_model_path = "vicuna-13b-v1.5.Q4_K_M.gguf"

        elif model_name == SupportedModels.Mixtral_Q3:
            relative_model_path = "mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf"

        elif model_name == SupportedModels.Llama3:
            relative_model_path = "Meta-Llama-3-8B-Instruct.Q8_0.gguf"

        if version == "v2":
            if db_chunk_size == DbSupportedChunkSizes.Chunk_1024 and db_chunk_overlap != DbSupportedChunkOverlap.Overlap_128:
                raise ValueError("Only Chunk_1024 and Overlap_128 are supported in v2")
            if db_chunk_size == DbSupportedChunkSizes.Chunk_2048 and db_chunk_overlap != DbSupportedChunkOverlap.Overlap_256:
                raise ValueError("Only Overlap_256 is supported with Chunk_2048 in v2")
            if distance != "l2" and distance != "cosine":
                raise ValueError("Only l2 and cosine distance are supported in v2")

        short_model_name = db_embedding_model.value.split("/")[-1]
        file = f"{short_model_name}_{db_chunk_size.value}_{db_chunk_overlap.value}_{distance}"
        folder = f"{DB_DIR}/{version}/{db_embedding_model.value}"

        db_path = os.path.join(folder, file)        

        model_path = os.path.join(LLM_DIR, relative_model_path)
        return model_path, db_path

def destroy_old_session():
    if 'embedding' in globals():
        del embedding
    
    if 'llm' in globals():
        del llm

def load_llm(config: RagConfig)-> LlamaCpp:
    destroy_old_session()

    if config.use_streaming:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    else:
        callback_manager = None

    llm = LlamaCpp(
        model_path=config.model_path,
        n_ctx=config.n_ctx,
        n_gpu_layers=-1,
        n_batch=config.n_ctx,
        temperature=0.0,
        logits_all=True,
        logprobs=20,
        callback_manager=callback_manager,
        chat_format="llama-2"
    )

    return llm

def load_llm_rag_model(config: RagConfig)-> Tuple[LlamaCpp, Chroma]:
    destroy_old_session()

    if isinstance(config.db_embedding_model, SupportedModels):
        embedding = LlamaCppEmbeddings(
            model_path=config.model_path
        )

    elif config.db_embedding_model.value.startswith("sentence-transformers"):
        embedding = SentenceTransformerEmbeddings(model_name=config.db_embedding_model.value.split("/")[-1])

    llm = load_llm(config)  


    db = Chroma(persist_directory=config.db_path, collection_name=config.db_collection ,embedding_function=embedding)  

    return llm, db
    