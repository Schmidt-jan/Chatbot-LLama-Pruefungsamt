import json
from custom_rag_loader import DbSupportedEmbeddingModels, RagConfig, SupportedModels, load_llm_rag_model, DbSupportedChunkSizes, DbSupportedChunkOverlap
from helper.call_llm import USE_DB, call_llm

from langchain.prompts import ChatPromptTemplate


from torch import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine



template = """Answer the following question based only on the provided context. Always return the source of an information and it is mandatory to answer in GERMAN:
{context}

Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)

config = RagConfig(
        n_ctx=4096,
        model=SupportedModels.Llama3,
        db_embedding_model=DbSupportedEmbeddingModels.Paraphrase_multilingual_MiniLM_L12_v2,
        db_chunk_size=DbSupportedChunkSizes.Chunk_1024,
        db_chunk_overlap=DbSupportedChunkOverlap.Overlap_128,
        version="v3",
        distance="l2",
        use_streaming=False,
    )

llm, db = load_llm_rag_model(config)

retriever = db.as_retriever(k = 3)


def format_docs(docs):
    context =""
    for doc in docs:
        context += "Content: \n" + doc.page_content + "\n"
        context += "Source: \n" + str(doc.metadata['file_path']) + "\n\n\n"
    return context

def simple_rag_chain(question: str, answer:str, use_db = USE_DB.TRUE):

    final_prompt = ""

    docs = []
    if(use_db):
        docs = retriever.get_relevant_documents(question)
        formatted_docs = format_docs(docs)
        final_prompt = prompt.format(context=formatted_docs, question= question)
    else:
        final_prompt = prompt.format(context=answer, question= question)

    return docs, llm.invoke(final_prompt)

def call_api(prompt, options, context):

    return call_llm('Meta-Llama-3-8B-Instruct.Q8_0', simple_rag_chain, prompt, options, context)


"""
def simple_rag_chain(question: str, answer:str):
    #docs = retriever.get_relevant_documents(question)
    #formatted_docs = format_docs(docs)

    final_prompt = prompt.format(context=answer, question= question)

    print(final_prompt)

    return llm.invoke(final_prompt)
"""

