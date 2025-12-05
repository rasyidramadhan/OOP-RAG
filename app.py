from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from repo import DocumentRepository
from rag import Embedding, RAG_SERVICE

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
    repo.initialize_qdrant(server="http://localhost:6333")
    rag_service = RAG_SERVICE(repo, embedder)
    service["rag"] = rag_service
    service["repo"] = repo

    yield # waiting for server to run
    service.clear() # removes all objects from the service when the server stops.

# API End Points
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