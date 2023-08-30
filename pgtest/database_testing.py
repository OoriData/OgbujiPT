from ogbujipt.embedding_helper import pgvector_connection
import asyncio
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()



def load_embedding_model(embedding_model_name):
    # LLM will be downloaded from HuggingFace automatically
    return SentenceTransformer(embedding_model_name)

db = pgvector_connection(SentenceTransformer("all-MiniLM-L12-v2"))
