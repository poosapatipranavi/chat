
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


_embed_model = SentenceTransformer(MODEL_NAME)

def get_embeddings(text):
    return _embed_model.encode(text).tolist()

def load_and_split_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    docs = splitter.split_documents(pages)
    return docs
