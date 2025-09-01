from openai import OpenAI
from app.config import (
    OPENAI_API_KEY,
    PINECONE_INDEX,
    NAMESPACE,
    OPENAI_MODEL_EMBED,
    OPENAI_MODEL_CHAT,
    PINECONE_CLOUD,
    PINECONE_REGION,
)
from app.vectorstores.pinecone_store import PineconeStore


class RAGChain:
    SYSTEM_PROMPT = """You are a knowledgeable and polite dermatology specialist acting as a study assistant. 
    Your role is to carefully teach and guide the user on dermatology topics using only the provided context 
    (excerpts from the 'Oxford Handbook of Dermatology'). 

    - Always explain concepts in a clear, structured, and easy-to-understand way. 
    - Maintain a professional yet friendly and encouraging tone, as if teaching a student. 
    - If the answer is not found in the provided context, respond politely and reassuringly, 
    making it clear that the information is not available, so the user does not feel discouraged. 
    - When citing, always reference the source in the format: (Book — Source). 
    - Keep your answers concise but helpful, focusing only on what the context supports. 
    - This is strictly an educational assistant role, not medical advice.
    """


    def __init__(
        self,
        temperature=0.0,
        index_name=PINECONE_INDEX,
        namespace=NAMESPACE,
        cloud=PINECONE_CLOUD,
        region=PINECONE_REGION,
        dimension=1536,
        metric="cosine",
        ):

        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing.")

        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.embed_model = OPENAI_MODEL_EMBED
        self.chat_model = OPENAI_MODEL_CHAT
        self.temperature = temperature

        self.store = PineconeStore(
            index_name=index_name,
            namespace=namespace,
            dimension=dimension,
            cloud=cloud,
            region=region,
            metric=metric,
        )
        self.namespace = namespace

    def embed_text(self, text):
        resp = self.openai_client.embeddings.create(
            model=self.embed_model,
            input=text,
        )
        return resp.data[0].embedding

    def search_index(self, query_vector, top_k = 10):
        return self.store.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            include_values=False,
        )

    def build_messages_from_result(self, question, result):
        """
        Assumes:
          - result.matches is a list
          - each match has .metadata and .score
          - metadata may contain 'text' or 'page_content'
        """
        matches = getattr(result, "matches", []) or []

        docs = []
        for match in matches:
            metadata = getattr(match, "metadata", {}) or {}
            text = (metadata.get("text") or metadata.get("page_content") or "").strip()

            docs.append({
            "text": text,
            "book": metadata.get("book", "unknown"),
            "source": metadata.get("source", "unknown"),
            })

        context_lines = [
            f"- {d['text']}\n  (Source: {d['book']} — {d['source']})\n"
            for d in docs if d["text"]
        ]
        context = "".join(context_lines) if context_lines else "(no relevant passages found)"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"},
        ]
        return messages

    def stream_answer(self, messages):
        return self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )
