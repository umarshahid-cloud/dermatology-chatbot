import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredFileLoader,
)

from app.config import (
    PINECONE_INDEX, PINECONE_CLOUD, PINECONE_REGION,
    OPENAI_MODEL_EMBED, EMBED_DIM, NAMESPACE,
    DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
)
from app.vectorstores.pinecone_store import PineconeStore

load_dotenv()

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

def main():
    if not DATA_DIR_PATH.exists():
        raise SystemExit(f"Put your source files in ./{DATA_DIR} first.")

    print("Loading documents...")
    raw_docs = load_docs()
    if not raw_docs:
        raise SystemExit(f"No documents found in ./{DATA_DIR}")

    print("Chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    # Normalize metadata
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        src = os.path.basename(str(src)) if src else "unknown"
        c.metadata = {"source": src, "book": "Oxford Handbook of Dermatology"}

    print("Setting up embeddings + Pinecone...")
    embeddings = OpenAIEmbeddings(model=OPENAI_MODEL_EMBED)

    store = PineconeStore(
        index_name=PINECONE_INDEX,
        embeddings=embeddings,
        namespace=NAMESPACE,
        dimension=EMBED_DIM,
        cloud=PINECONE_CLOUD,
        region=PINECONE_REGION,
    )

    print("Upserting to Pinecone...")
    store.upsert_documents(chunks)

    stats = store.describe_stats()
    print("Index stats:", stats)
    print(f"Done. Indexed {len(chunks)} chunks into '{PINECONE_INDEX}'.")

if __name__ == "__main__":
    # Allow running as a script (python -m app.ingest.ingest_derm)
    main()
