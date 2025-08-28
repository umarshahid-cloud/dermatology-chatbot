import os
from dotenv import load_dotenv
from app.rag.rag_chain import RAGChain

load_dotenv()

def main():
    rag = RAGChain(
        index_name=os.getenv("PINECONE_INDEX", "derma-chatbot"),
        namespace=os.getenv("NAMESPACE", "dermatology"),
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
    )

    # Test retrieve only
    docs = rag.retrieve("What is eczema?", k=2)
    print("Retrieved docs:")
    for d in docs:
        print(d)

    # Test single answer
    ans = rag.answer("What is eczema?", k=2)
    print("\nAnswer:\n", ans)

    # Test streaming answer
    print("\nStreaming answer:")
    for tok in rag.stream_answer("What is eczema?", k=2):
        print(tok, end="", flush=True)

if __name__ == "__main__":
    main()
