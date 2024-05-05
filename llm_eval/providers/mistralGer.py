import json
from custom_rag_loader import DbSupportedEmbeddingModels, RagConfig, SupportedModels, load_llm_rag_model
from helper.call_llm import call_llm

from langchain.prompts import ChatPromptTemplate


from torch import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine



template = """Answer the following question based only on the provided context. Always return the source of an information and it is mandatory to answer in GERMAN:
{context}

Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)

llm, db = load_llm_rag_model(RagConfig(
        model=SupportedModels.MistralGer, 
        db_embedding_model=DbSupportedEmbeddingModels.Paraphrase_multilingual_MiniLM_L12_v2, 
        use_streaming=False,
        n_ctx=4096,
        ))

retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5})


def format_docs(docs):
    context =""
    for doc in docs:
        context += "Content: \n" + doc.page_content + "\n"
        context += "Source: \n" + str(doc.metadata['file_path']) + "\n\n\n"
    return context

def simple_rag_chain(question: str, answer:str):

    final_prompt = prompt.format(context=answer, question= question)

    print(final_prompt)

    return llm.invoke(final_prompt)

def call_api(prompt, options, context):

    return call_llm('em_german_mistral_v01.Q8_0', simple_rag_chain, prompt, options, context)

#call_api("Wie viele Einwohner hat Deutschland?", None, {'vars': {'answer': 'Deutschland hat 80 Millionen Einwohner'}})