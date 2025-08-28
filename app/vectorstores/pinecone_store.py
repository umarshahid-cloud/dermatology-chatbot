import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.docstore.document import Document

class PineconeStore:
    """
    Thin wrapper around Pinecone + LangChain's PineconeVectorStore.
    - Creates/ensures the index (serverless)
    - Holds a ready-to-use VectorStore

    NOTE: Ensure your document metadata ONLY contains str/number/bool/list[str]
    and no None values before calling upsert_documents.
    """
    def __init__(
        self,
        index_name,
        embeddings,
        namespace="dermatology",
        dimension=1536,
        cloud=None,
        region=None,
    ):
        self.index_name = index_name
        self.namespace = namespace
        self.embeddings = embeddings

        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

        if cloud and region:
            self._ensure_index(dimension=dimension, cloud=cloud, region=region)

        # Ready-to-use LangChain VectorStore wrapper
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=self.namespace,
        )

    def _ensure_index(self, dimension, cloud, region):
        existing = [i["name"] for i in self.pc.list_indexes()]
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

    def upsert_documents(self, docs, batch_size=None):
        return self.vectorstore.add_documents(docs, batch_size=batch_size)

    def describe_stats(self):
        idx = self.pc.Index(self.index_name)
        return idx.describe_index_stats()
