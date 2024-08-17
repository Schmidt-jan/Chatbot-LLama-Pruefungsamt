from custom_rag_loader import DbSupportedEmbeddingModels, RagConfig, SupportedModels, load_db, DbSupportedChunkSizes, DbSupportedChunkOverlap
from custom_rag_loader import DbSupportedEmbeddingModels, RagConfig, SupportedModels, DbSupportedChunkSizes, DbSupportedChunkOverlap
from helper.call_llm import USE_DB
from openai import OpenAI



def generate_prompt(context: str, question: str):

    return f"""
        Answer the following question based only on the provided context. Always return the source of an information and it is mandatory to answer in GERMAN

        Context:
        {context}

        Question:
        {question}
"""


def format_docs(docs):
    context =""
    for doc in docs:
        context += "Content: \n" + doc.page_content + "\n"
        context += "Source: \n" + str(doc.metadata['file_path']) + "\n\n\n"
    return context


def call_chatGPT_model(prompt):

    client = OpenAI()
    completion = client.chat.completions.create(model="gpt-3.5-turbo-0125", messages=[{
        "role": "system",
        "content": prompt
        }
     ])
    
    if completion.usage is not None:

        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens

        price_per_million_input_tokens = 0.5  # Dollar per 1M Tokens
        price_per_million_output_tokens = 1.5  # Dollar per 1M Tokens
        tokens_amount = 1_000_000

        price_prompt_tokens = (prompt_tokens * price_per_million_input_tokens) / tokens_amount
        price_completion_tokens = (completion_tokens * price_per_million_output_tokens) / tokens_amount

        total_cost = price_prompt_tokens + price_completion_tokens

        print(f"Model: {completion.model}")
        print(f"Price for prompt tokens: {price_prompt_tokens:.10f} USD")
        print(f"Price for completion tokens: {price_completion_tokens:.10f} USD")
        print(f"Total cost: {total_cost:.10f} USD")

    return completion.choices[0].message.content


def simple_rag_chain(question: str, answer:str, use_db = USE_DB.TRUE):

    db = load_db(
    RagConfig(
        model=SupportedModels.Mistral, 
        n_ctx=4096,
        db_embedding_model=DbSupportedEmbeddingModels.Paraphrase_multilingual_MiniLM_L12_v2,
        db_chunk_overlap=DbSupportedChunkOverlap.Overlap_256,
        db_chunk_size=DbSupportedChunkSizes.Chunk_2048,
        version="v3"
        )
    )
    retriever = db.as_retriever(k=3)

    final_prompt = ""
    docs = []

    if(use_db):
        docs = retriever.get_relevant_documents(question)
        formatted_docs = format_docs(docs)
        final_prompt = generate_prompt(context=formatted_docs, question= question)
    else:
        final_prompt = generate_prompt(context=answer, question= question)

    return docs, call_chatGPT_model(final_prompt)


def call_api(prompt, options, context):
    return call_llm('gpt-3.5-turbo-0125', prompt, options, context)


def call_llm(model: str, prompt, options, context):

    docs, openAi_result = simple_rag_chain(prompt, options, context)

    answer = context['vars']['answer']

    result = {
        'output': {
            'model': model,
            'output': openAi_result,
            'answer': answer,
            'keywords': context['vars']['keywords']['list'],
        }
    }

    return result
