import os
from dotenv import load_dotenv

load_dotenv()

# Pinecone / OpenAI
PINECONE_INDEX  = os.getenv("PINECONE_INDEX", "derma-chatbot")
PINECONE_CLOUD  = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
OPENAI_MODEL_EMBED = os.getenv("EMBED_MODEL", "text-embedding-3-small")
OPENAI_MODEL_CHAT  = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Embedding dimension must match the embedding model
EMBED_DIM = 1536  # text-embedding-3-small

# Namespacing & data
NAMESPACE     = "dermatology"
DATA_DIR      = "data"       # put PDFs/TXT/MD here
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
