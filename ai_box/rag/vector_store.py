from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

DATA_PATH = "rag/data.txt"
FAISS_PATH = "rag/faiss_index"


def build_vector_store():

    loader = TextLoader(DATA_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(FAISS_PATH)

    return vectorstore


def load_vector_store():    

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(FAISS_PATH):
        return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        return build_vector_store()
