import time
import random
import os
import json
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance



# Creating embedding classes
class Embedding:
    def fake_embed(self, text: str) -> List[float]:
        random.seed(abs(hash(text)) % 10000)
        return [random.random() for _ in range(128)]

# Document Repository
class DocumentRepository:
    def __init__(self):
        self.use_qdrant = False
        self.qdrant_client: Optional[QdrantClient] = None
        self.memory_store: List[str] = []
        self.initialize_qdrant()

    # Initialize for connected to qdrant
    def initialize_qdrant(self, server="http://localhost:6333"):
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

# RAG Work
class RAG_SERVICE:
    def __init__(self, repo: DocumentRepository, embedder: Embedding):
        self.repo = repo
        self.embedder = embedder
        self.chain = self.build_graph()

    # Questions are embedded using fake embedding.
    # The resulting embedding is used to search for documents in Qdrant or memory.
    # Relevant documents are stored as context.
    def retrieve(self, state):
        question = state["question"]
        vector = self.embedder.fake_embed(question)
        results = self.repo.search(vector=vector, text_query=question)
        state["context"] = results
        return state

    # Answer for context question
    def answer(self, state):
        ctx = state["context"]
    
        if ctx:
            state["answer"] = ctx[0]
        else:
            state["answer"] = "No matching document found."

        return state

    # flow map with build_graph
    def build_graph(self):
        workflow = StateGraph(dict)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("answer", self.answer)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", END)
        return workflow.compile()

    # Question
    def process_question(self, question: str) -> Dict[str, Any]:
        start = time.time()
        result = self.chain.invoke({"question": question})
        return {
            "question": question,
            "answer": result["answer"],
            "context_used": result.get("context", []),
            "latency_sec": f"{round(time.time() - start, 3)} second"
        }

    # Add document
    def add_document(self, text: str) -> Dict[str, Any]:
        vec = self.embedder.fake_embed(text)
        doc_id = self.repo.add(text, vec)
        return {"id": doc_id, "status": "added"}

# API END POINTS

class QuestionRequest(BaseModel):
    question: str

class DocumentRequest(BaseModel):
    text: str
        
service = {}

# lifespan manager
@asynccontextmanager
async def initialize(app: FastAPI):
    embedder = Embedding()
    repo = DocumentRepository()
    rag_service = RAG_SERVICE(repo, embedder)
    service["rag"] = rag_service
    service["repo"] = repo

    yield # waiting for server to run
    service.clear() # removes all objects from the service when the server stops.

app = FastAPI(title="Refractored RAG Demo", lifespan=initialize)

@app.post("/ask") # Ask question
def ask(req: QuestionRequest):
    try:
        return service["rag"].process_question(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add") # Add documents
def add(req: DocumentRequest):
    try:
        return service["rag"].add_document(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status") # Checking Service Status
def status():
    st = service["repo"].get_status()
    return {
        "qdrant_ready": st["backend"],
        "in_memory_docs_count": st["doc_count"],
        "service_status": "active"
    }