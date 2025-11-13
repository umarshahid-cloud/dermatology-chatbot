import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredFileLoader,
)
from app.config import (
    PINECONE_INDEX, PINECONE_CLOUD, PINECONE_REGION,
    OPENAI_MODEL_EMBED, EMBED_DIM, NAMESPACE,
    DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, OPENAI_API_KEY
)
from app.vectorstores.pinecone_store import PineconeStore

load_dotenv()

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Please set it in your .env or environment.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
DATA_DIR_PATH = Path(DATA_DIR)

def load_docs():
    docs = []
    for fp in DATA_DIR_PATH.glob("**/*"):
        if fp.is_dir():
            continue
        suffix = fp.suffix.lower()
        if suffix == ".pdf":
            docs.extend(PyPDFLoader(str(fp)).load())
        elif suffix in {".txt", ".md"}:
            docs.extend(TextLoader(str(fp), encoding="utf-8").load())
        else:
            docs.extend(UnstructuredFileLoader(str(fp)).load())
    return docs

def get_embedding(text, model=OPENAI_MODEL_EMBED):
    text = (text or "").replace("\n", " ")
    return openai_client.embeddings.create(
        input=[text], model=model
    ).data[0].embedding

def main():
    if not DATA_DIR_PATH.exists():
        raise SystemExit(f"Put your source files in ./{DATA_DIR} first.")

    print("Loading documents")
    raw_docs = load_docs()
    if not raw_docs:
        raise SystemExit(f"No documents found in ./{DATA_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    print(f"Split into {len(chunks)} chunks")

    print("Embedding with OpenAI")
    vectors, ids, metadatas = [], [], []

    for i, c in enumerate(chunks):
        src = os.path.basename(str(c.metadata.get("source", "unknown"))) or "unknown"
        ids.append(f"{src}::{i}")
        metadatas.append({
            "source": src,
            "book": "Oxford Handbook of Dermatology",
            "text": c.page_content    
        })
        vectors.append(get_embedding(c.page_content))

    print("Setting up Pinecone")
    store = PineconeStore(
        index_name=PINECONE_INDEX,
        namespace=NAMESPACE,
        dimension=EMBED_DIM,
        cloud=PINECONE_CLOUD,
        region=PINECONE_REGION,
        metric="cosine",
    )

    print("Upserting to Pinecone")
    store.upsert_vectors(ids=ids, vectors=vectors, metadatas=metadatas, batch_size=50)

    stats = store.describe_stats()
    print("Index stats:", stats)
    print(f"Done. Indexed {len(chunks)} chunks into '{PINECONE_INDEX}'.")

if __name__ == "__main__":
    main()
