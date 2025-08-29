from openai import OpenAI
from pinecone import Pinecone
from app.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    NAMESPACE,
    OPENAI_MODEL_EMBED,
    OPENAI_MODEL_CHAT,
)

class RAGChain:
    SYSTEM_PROMPT = """You are a careful dermatology study assistant.
    Answer ONLY from the provided context (excerpts from 'Oxford Handbook of Dermatology').
    If the answer is not present, say you don't know.
    Cite like (Book — Source). This is not medical advice. Be concise.
    """

    def __init__(self, temperature=0.0):
        self.namespace = NAMESPACE
        self.embed_model = OPENAI_MODEL_EMBED
        self.chat_model = OPENAI_MODEL_CHAT
        self.temperature = temperature
        if not PINECONE_API_KEY:
                    raise RuntimeError("PINECONE_API_KEY is missing.")
        if not OPENAI_API_KEY:
                    raise RuntimeError("OPENAI_API_KEY is missing.")
        # API clients
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pinecone_client.Index(PINECONE_INDEX)

    def embed_text(self, text):
        response = self.openai_client.embeddings.create(
            model=self.embed_model,
            input=text
        )
        return response.data[0].embedding

    def search_index(self, query_vector, top_k=4):
        return self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            include_values=False,
            namespace=self.namespace,
        )

    def parse_results(self, result):
        matches = getattr(result, "matches", result.get("matches", []))
        docs = []
        for match in matches:
            metadata = getattr(match, "metadata", None)
            if metadata is None and isinstance(match, dict):
                metadata = match.get("metadata", {})
            if metadata is None:
                metadata = {}

            text = (metadata.get("text") or metadata.get("page_content") or "").strip()
            docs.append({
                "text": text,
                "book": metadata.get("book", "unknown"),
                "source": metadata.get("source", "unknown"),
                "score": getattr(match, "score", None) or (
                    match.get("score") if isinstance(match, dict) else None
                ),
            })
        return docs

    def format_context(self, documents, max_chars=4000):
        formatted, total = [], 0
        for doc in documents:
            line = f"- {doc['text']}\n  (Source: {doc['book']} — {doc['source']})\n"
            if total + len(line) > max_chars:
                break
            formatted.append(line)
            total += len(line)
        return "".join(formatted) if formatted else "(no relevant passages found)"

    def build_messages(self, question, context):
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"},
        ]

    def stream_answer(self, messages):
        return self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )