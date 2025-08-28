import os
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import NAMESPACE, OPENAI_MODEL_EMBED, OPENAI_MODEL_CHAT

SYSTEM_PROMPT = """You are a careful dermatology study assistant.
Answer ONLY from the provided context (excerpts from 'Oxford Handbook of Dermatology').
If the answer is not present, say you don't know.
Cite like (Book — Source). This is not medical advice. Be concise.
"""

class RAGChain:
    def __init__(self, index_name,
                 namespace=NAMESPACE,
                 embed_model=OPENAI_MODEL_EMBED,
                 chat_model=OPENAI_MODEL_CHAT):
        self.namespace = namespace
        self.embedding_model = embed_model
        self.chat_model_name = chat_model

        # langchain_openai clients (use OPENAI_API_KEY from env)
        self.embedder = OpenAIEmbeddings(model=self.embedding_model)
        self.llm = ChatOpenAI(model=self.chat_model_name, temperature=0)

        # pinecone client
        self.pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = self.pinecone_client.Index(index_name)

    def _embed(self, text):
        # langchain_openai embedding for a single query
        return self.embedder.embed_query(text)

    def retrieve(self, question, k=4):
        query_vector = self._embed(question)
        result = self.index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True,
            include_values=False,
            namespace=self.namespace
        )
        matches = result.matches if hasattr(result, "matches") else result["matches"]

        retrieved_docs = []
        for match in matches:
            metadata = match.metadata if hasattr(match, "metadata") else match["metadata"]
            content_text = metadata.get("text") or metadata.get("page_content") or ""
            retrieved_docs.append({
                "text": content_text,
                "book": metadata.get("book", "unknown"),
                "source": metadata.get("source", "unknown"),
                "score": getattr(match, "score", None) or match.get("score"),
            })
        return retrieved_docs

    def _format_context(self, documents, max_chars=4000):
        formatted, total = [], 0
        for doc in documents:
            line = f"- {doc['text']}\n  (Source: {doc['book']} — {doc['source']})"
            if total + len(line) > max_chars:
                break
            formatted.append(line)
            total += len(line)
        return "\n".join(formatted) if formatted else "(no relevant passages found)"

    def answer(self, question, k=4):
        context = self._format_context(self.retrieve(question, k))
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nContext:\n{context}\n\nAnswer:")
        ]
        resp = self.llm.invoke(messages)
        return resp.content

    def stream_answer(self, question, k=4):
        context = self._format_context(self.retrieve(question, k))
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nContext:\n{context}\n\nAnswer:")
        ]
        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content
