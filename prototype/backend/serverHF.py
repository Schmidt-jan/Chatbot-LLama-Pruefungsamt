import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
import re
import json
import fitz  # PyMuPDF
from fuzzywuzzy import fuzz


from custom_rag_loader import DbSupportedEmbeddingModels, RagConfig, SupportedModels, load_llm_rag_model, load_llm_huggingface_rag_model, DbSupportedChunkSizes, DbSupportedChunkOverlap
from langchain.prompts import ChatPromptTemplate
from openai import Client

openai_client = Client()

llm, db = load_llm_huggingface_rag_model(
    RagConfig(
        model=SupportedModels.Mistral_Finetune_Huggingface_RAFTV2,
        n_ctx=4096,
        db_embedding_model=DbSupportedEmbeddingModels.Paraphrase_multilingual_MiniLM_L12_v2,
        db_chunk_overlap=DbSupportedChunkOverlap.Overlap_256,
        db_chunk_size=DbSupportedChunkSizes.Chunk_2048,
        version="v3",
        use_streaming=True
    )
)

template = """[INST]You are a smart helpful assistant for the HTWG Konstanz. Answer the following question based only on the provided context. Always return the source of an information and it is mandatory to answer in GERMAN:

Context: {context}

Question: {question}[/INST]"""

prompt = ChatPromptTemplate.from_template(template)
last_pages = None
last_link = None

def format_docs(docs):
    context = ""
    pages = []
    for doc in docs:
        context += "Content: \n" + doc.page_content + "\n"
        context += "Source: \n" + str(doc.metadata['file_path']) + "\n\n\n"
        source = doc.metadata['file_path']
        content = doc.page_content
        print('Source:', source)
        page = find_page(source, content)
        if not len(page) == 0:
            pages.append(find_page(source, content))
        print('Hier sind pages?', pages)
        link = correct_link(source)
    return context, pages, link

def simple_rag_chain(question: str):
    global last_pages, last_link
    docs = db.search(question, 'similarity', k=3)

    formatted_docs, pages, link = format_docs(docs)
    last_pages = pages
    last_link = link

    final_prompt = prompt.format(context=formatted_docs, question=question)

    for chunk in llm.stream(final_prompt, pipeline_kwargs={
            "max_new_tokens": 1024,
            "temperature": 0.1,  
            "top_p":0,
            "do_sample":True, 
            "repetition_penalty":1.2 
        }):
        yield chunk

def simple_rag_chain_openai(question: str):
    global last_pages, last_link
    docs = db.search(question, 'similarity', k=3)
    formatted_docs, pages, link = format_docs(docs)
    last_pages = pages
    last_link = link
    final_prompt = prompt.format(context=formatted_docs, question=question)
    response = openai_client.completions.create(model="gpt-3.5-turbo-instruct", prompt=final_prompt, temperature=0, max_tokens=1000, stream=True)
    for chunk in response:
        current_content = chunk.choices[0].text
        yield current_content


'''
this is very complicated and error prone, it works but can probably be replaced by getting the metadata of the context document
chunks / saving the page numbers in that metadata
'''
def find_page(file_path, search_text):
    # Öffne die PDF-Datei
    pdf_document = fitz.open(file_path)
    print('Hier ist der Filepath:', file_path)
    search_text = re.sub(r'\s+', ' ', search_text.strip())  # Bereinige den Suchtext
    #print("Hier ist der Suchtext:" + search_text)
    found_pages = []

  # Iteriere durch jede Seite
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        page_text = page.get_text("text")
        
        # Normalize the page text: remove extra spaces, convert to lowercase
        page_text = re.sub(r'\s+', ' ', page_text.strip()).lower()  
        #print(f"Seite {page_num + 1} Text: {page_text[100:]}...")  # Print first 100 characters for debugging

        # Suche nach dem Text
        if fuzz.ratio(search_text,page_text) >= 80:
            if not page_num + 1 == 1:
                found_pages.append(page_num + 1)  # Seitennummern sind 1-basiert

    if not found_pages:
        print(f"Suchtext wurde nicht gefunden.")
    else:
        print(f"Suchtext gefunden auf den Seiten: {found_pages}")

    return found_pages

def correct_link(filepath):
    print('ist das der richtige filepath?', filepath[62:])
    filepath = filepath[62:]
    print('ist das der richtige filepath?', filepath[:10])
    links = []
    link = None  # Initialisiere die Variable link mit None

    if filepath[:10] == "119_ZuSMa_":
        link = "https://www.htwg-konstanz.de/fileadmin/pub/allgemein/Dokumente/SPOs/119_ZuSMa_Senat_18012022.pdf"
    elif filepath[:10] == "124_SPOMa_":
        link = "https://www.htwg-konstanz.de/fileadmin/pub/allgemein/Dokumente/SPOs/124_SPOMa_AT_Senat_08112022.pdf"
    elif filepath[:10] == "Infoversta":
        link = "https://www.htwg-konstanz.de/fileadmin/pub/fk_in/stg_msi/Dokumente/Infoveranstaltung_Masterstudiengaenge-Informatik_HTWG-Konstanz.pdf"
    elif filepath[:10] == "Modulhandb":
        link = "https://www.htwg-konstanz.de/fileadmin/pub/fk_in/stg_msi/Dokumente/Modulhandbuch_MSI_SS23_Stand_10-Jan-2023.pdf"
    elif filepath[:10] == "SPO_MSI_SP":
        link = "https://www.htwg-konstanz.de/studium/pruefungsangelegenheiten/satzungenordnungenamtsblatt"
    elif filepath[:10] == "Wahlpflich":
        link = "https://www.htwg-konstanz.de/fileadmin/pub/fk_in/stg_msi/Dokumente/Wahlpflichtmodule_Master-Informatik_SS24_Stand_20-Mar-2024.pdf"

    if link is not None:
        links.append(link)
        print('Hier ist der Link:', link)
        return links[0]
    else:
        print('Kein passender Link gefunden.')
        return None




app = FastAPI()


origins = [
    "http://localhost:3000",
    "http://localhost:3006",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/requests")
async def ask_question(question: str):
    try:
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")
        
        # Stream the LLM response as a text event stream
        return StreamingResponse(simple_rag_chain(question), media_type='text/event-stream')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/requestsOAI")
async def ask_question_openai(question: str):
    try:
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")
        
        # Stream the LLM response as a text event stream
        return StreamingResponse(simple_rag_chain_openai(question), media_type='text/event-stream')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/pages')
async def pages():
    return Response(json.dumps({'pages': last_pages, 'link': last_link}), media_type='application/json')


if __name__ == '__main__':
    uvicorn.run(host="0.0.0.0", port=8000, app="serverHF:app", reload=False)