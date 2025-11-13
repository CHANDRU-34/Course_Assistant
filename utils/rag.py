import os
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from models.embeddings import get_embedding_model
from config.config import VECTOR_STORE_PATH

def process_pdf(file_path):
    """Load and split PDF into chunks"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def build_vector_store(docs):
    """Create FAISS vector store"""
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store

def load_vector_store():
    """Load FAISS index if exists"""
    if not os.path.exists(VECTOR_STORE_PATH):
        return None
    embeddings = get_embedding_model()
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

def retrieve_relevant_docs(query, top_k=3):
    """Retrieve relevant chunks for query"""
    vector_store = load_vector_store()
    if not vector_store:
        return []
    return vector_store.similarity_search(query, k=top_k)