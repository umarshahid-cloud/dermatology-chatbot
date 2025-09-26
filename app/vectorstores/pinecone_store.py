import math
from pinecone import Pinecone, ServerlessSpec
from app.config import PINECONE_API_KEY

class PineconeStore:
    def __init__(self, index_name, namespace="dermatology", dimension=3072,
                 cloud="aws", region="us-east-1", metric="cosine"):
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is missing.")

        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud or "aws"
        self.region = region or "us-east-1"

        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        # Check if index exists and validate dimension
        self._validate_or_create_index()

    def _index(self):
        return self.pc.Index(self.index_name)

    def _validate_or_create_index(self):
        """Check if index exists and matches dimension; create if needed."""
        try:
            # List indexes and check if index_name exists
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                )
            else:
                # Validate dimension
                stats = self._index().describe_index_stats()
                index_dim = stats["dimension"]
                if index_dim != self.dimension:
                    raise ValueError(
                        f"Index '{self.index_name}' has dimension {index_dim}, "
                        f"but {self.dimension} was expected. Delete the index or use a new name."
                    )
        except Exception as e:
            raise RuntimeError(f"Failed to validate/create index: {str(e)}")

    def upsert_vectors(self, ids, vectors, metadatas, batch_size=100):
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
            except Exception as e:
                raise RuntimeError(f"Upsert failed: {str(e)}")

            total += len(items)

        return {"upserted": total, "batches": math.ceil(n / batch_size)}

    def describe_stats(self):
        return self._index().describe_index_stats()

    def query(self, vector, top_k=4, include_metadata=True, include_values=False):
        return self._index().query(
            vector=vector,
            top_k=top_k,
            include_metadata=include_metadata,
            include_values=include_values,
            namespace=self.namespace,
        )