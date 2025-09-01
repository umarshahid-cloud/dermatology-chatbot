import math
from pinecone import Pinecone, ServerlessSpec
from app.config import PINECONE_API_KEY

class PineconeStore:
    """
    Try-first approach:
    - On upsert/query/stats, try using the index.
    - If it doesn't exist, create it (serverless) and retry once.
    """

    def __init__(self, index_name, namespace="dermatology", dimension=1536,
                 cloud=None, region=None, metric="cosine"):
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is missing.")

        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region

        self.pc = Pinecone(api_key=PINECONE_API_KEY)

    def _index(self):
        return self.pc.Index(self.index_name)

    def _create_index(self):
        if not (self.cloud and self.region):
            raise ValueError(
                f"Index '{self.index_name}' not found and no cloud/region provided."
            )
        self.pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.metric,
            spec=ServerlessSpec(cloud=self.cloud, region=self.region),
        )

    def upsert_vectors(self, ids, vectors, metadatas, batch_size=256):
        if not vectors:
            return {"upserted": 0, "batches": 0}
        if not (len(ids) == len(vectors) == len(metadatas)):
            raise ValueError("ids, vectors, metadatas must have equal length")

        total = 0
        n = len(vectors)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            items = [(ids[i], vectors[i], metadatas[i]) for i in range(start, end)]

            try:
                self._index().upsert(vectors=items, namespace=self.namespace)
            except Exception:
                self._create_index()
                self._index().upsert(vectors=items, namespace=self.namespace)

            total += len(items)

        return {"upserted": total, "batches": math.ceil(n / batch_size)}

    def describe_stats(self):
        try:
            return self._index().describe_index_stats()
        except Exception:
            self._create_index()
            return self._index().describe_index_stats()
