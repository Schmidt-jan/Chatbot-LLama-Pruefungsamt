from operator import index
from os import path
from typing import Sequence
from uuid import uuid4
from PyPDF2 import PdfReader
import chromadb
import langchain_community
import langchain_community.document_loaders
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents import Document
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
from chromadb.api.types import EmbeddingFunction, Documents, Embedding
import os
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
import re

VERSION = "v3"
DB_DIR = f"/home/tpllmws23/Chatbot-LLama-Pruefungsamt/Chatbot-Jan/databases/{VERSION}"
embedding_list = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
#    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
#    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "mixedbread-ai/mxbai-embed-large-v1",
#    "T-Systems-onsite/german-roberta-sentence-transformer-v2"
]


def get_pdfs_in_folder(folder_path):
    """
    Get all pdfs in a folder
    :param folder_path: path to folder
    :return: list of pdfs
    """
    import os
    pdfs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)
            pdfs.append(path)
    return pdfs


curr_filepath = ""
parts = []

def visitor_body(text, cm, tm, fontDict, fontSize):
    y = tm[5]

    if "119_ZuSMa_Senat_18012022.pdf" in curr_filepath:
        if y > 50 and y < 730:
            parts.append(text)
    elif "SPO_MSI_SPONr5_Senat_10122019.pdf" in curr_filepath:
        if y > 50 and y < 770:
            parts.append(text)
    else:
        parts.append(text)


def get_pdf_text(file_path) -> Document:
    """
    Get text from pdf and return for each page the page content and some metadata
    :param file_path: path to pdf
    :return: list of dictionaries with page content and metadata
    """
    global curr_filepath
    parts.clear()
    curr_filepath = file_path
    reader = PdfReader(file_path)
    for i in range(len(reader.pages)):
        reader.pages[i].extract_text(visitor_text=visitor_body)

    joined_text = " ".join(parts)
    lines = joined_text.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    cleaned_text = '\n'.join(cleaned_lines)

    return Document(page_content=cleaned_text, metadata={"file_path": file_path})




def get_pdf_as_text(file_path) -> str:
    """
    Get text from pdf
    :param file_path: path to pdf
    :return: text of pdf
    """
    reader = PdfReader(file_path)
    text = ""
    for i in range(len(reader.pages)):
        text += reader.pages[i].extract_text(visitor_text=visitor_body)
    return text


def get_text_from_pdfs_in_folder(folder_path) -> list[Document]:
    """
    Get text from all pdfs in a folder
    :param folder_path: path to folder
    :return: list of dictionaries with page content and metadata
    """
    pdf_paths = get_pdfs_in_folder(folder_path)
    documents = []
    for pdf_path in pdf_paths:
        documents.append(get_pdf_text(pdf_path))
    return documents


def split_documents(documents: list[Document], chunk_size: int = 1024, chunk_overlap: int = 128) -> list[Document]:
    """
    Split documents into chunks of size chunk_size
    :param documents: list of documents
    :param chunk_size: size of chunks
    :param chunk_overlap: overlap between chunks
    :return: list of documents with chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size = chunk_size,
                        chunk_overlap  = chunk_overlap,
                        separators = ["\n\n", "\n§", "  §"]
                    )
    return text_splitter.split_documents(documents)


def store_into_database(documents: list[Document], database_path: str, collection_name: str, model_name: str, distance : str = "cosine"):
    """
    Store documents into database
    :param documents: list of documents
    :param database_path: path to database
    """
    client = chromadb.PersistentClient(database_path)
    collection = client.get_or_create_collection(collection_name, metadata={"hnsw:space": distance})
    
    model = SentenceTransformer(model_name)

    for doc in documents:
        embedding = model.encode(doc.page_content, convert_to_tensor=True).tolist()
        collection.add(str(uuid.uuid4()), embedding, doc.metadata, doc.page_content)

    
def index_data():
    documents = get_text_from_pdfs_in_folder("/home/tpllmws23/Chatbot-LLama-Pruefungsamt/main_data")
    documents = split_documents(documents)
    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    db_path = os.path.join(DB_DIR, model_name)
    store_into_database(documents, db_path, "Pruefungsamt", model_name)

def query_data(path, model_name):
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    client = chromadb.PersistentClient(db_path)
    collection = client.get_collection("Pruefungsamt", embedding_function=sentence_transformer_ef)
    query = collection.query(query_texts=["Ausländer"])
    print(query)

if __name__ == "__main__":
    documents = get_text_from_pdfs_in_folder("/home/tpllmws23/Chatbot-LLama-Pruefungsamt/main_data_filtered")
    
    # Uncomment, if you want the raw data
    # dir = "/home/tpllmws23/Chatbot-LLama-Pruefungsamt/main_data_filtered"
    # documents = []
    # for file in os.listdir(dir):
    #     loader = PyPDFLoader(path.join(dir, file))
    #     documents += loader.load()

    for embedding_model in embedding_list:
        for chunk_size, chunk_overlap in [[1024, 128], [2048, 256], [4096, 256]]:
            for distance in ["cosine", "l2"]:
                db_path = os.path.join(DB_DIR, embedding_model)
                db_path = db_path + f"_{chunk_size}_{chunk_overlap}_{distance}_no_edit"
                #query_data(db_path, embedding_model)

                documents2 = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                store_into_database(documents2, db_path, "Pruefungsamt", embedding_model, distance)
                print(f"Finished indexing {embedding_model} with chunk size {chunk_size} and overlap {chunk_overlap} and distance {distance}")