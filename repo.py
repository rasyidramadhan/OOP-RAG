from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance



# Document Repository
class DocumentRepository:
    def __init__(self):
        self.use_qdrant = False
        self.qdrant_client: Optional[QdrantClient] = None
        self.memory_store: List[str] = []

    # Initialize for connected to qdrant
    def initialize_qdrant(self, server):
        try:
            self.qdrant_client = QdrantClient(url=server)
            self.qdrant_client.recreate_collection(
                collection_name="demo_collection",
                vectors_config=VectorParams(size=128, distance=Distance.COSINE)
            )
            self.use_qdrant = True
            print("Connected to Qdrant")

        except Exception:
            print("Qdrant not available. Using in-memory storage.")
            self.use_qdrant = False

    # Add qdrant or memory backend
    def add(self, text: str, vector: List[float]) -> int:
        doc_id = len(self.memory_store)

        if self.use_qdrant:
            payload = {"text": text}
            self.qdrant_client.upsert(
                collection_name="demo_collection",
                points=[PointStruct(id=doc_id, vector=vector, payload=payload)]
            )

        # Save memory
        self.memory_store.append(text)
        return doc_id

    # Makes a request to Qdrant to find the nearest vector.
    # Returns the text payload for each point.
    def search(self, vector: List[float], text_query: str) -> List[str]:
        if self.use_qdrant:
            demos = self.qdrant_client.query_points(
                    collection_name="demo_collection",
                    query=vector,
                    limit=2
                ).points # Retrieves points a list of search result objects.

            return [p.payload.get("text", "") for p in demos]

        else:
            results = [doc for doc in self.memory_store if text_query.lower() in doc.lower()]
            
            if not results and self.memory_store:
                return [self.memory_store[0]]
            
            return results

    # taking status
    def get_status(self) -> Dict[str, Any]:
        return {
            "backend": "qdrant" if self.use_qdrant else "memory",
            "doc_count": len(self.memory_store)
        }