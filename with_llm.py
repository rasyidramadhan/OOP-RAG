# This code is not recommended for products.


# required libraries
import time
import random
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate




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
                collection_name="demo_collection", vectors_config=VectorParams(size=128, distance=Distance.COSINE))
            self.use_qdrant = True
            print("Connected to Qdrant")

        except Exception:
                print("Qdrant not available. Falling back to in-memory list.")
                self.use_qdrant = False

    # Add qdrant or memory backend
    def add(self, text: str, vector: List[float]) -> int:
        docs_memory = len(self.memory_store)

        if self.use_qdrant:
            payload = {"text": text}
            self.qdrant_client.upsert(collection_name="demo_collection", points=[PointStruct(id=docs_memory, vector=vector, payload=payload)])
            self.memory_store.append(text) 

        else:
            self.memory_store.append(text)
            
        return docs_memory
    
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
    # LLM initialitation
    def __init__(self, repo: DocumentRepository, embedder: Embedding):
        self.repo = repo
        self.embedder = embedder
        model = pipeline(task="text-generation", 
                model="microsoft/Phi-3.5-mini-instruct",
                 max_length=64,
                 temperature=0.2,
                 repetition_penalty=1.2)

        self.llm = HuggingFacePipeline(pipeline=model)
        self.prompt = PromptTemplate.from_template(
                    "Answer the question in ONE short sentence."
                    "Do NOT explain. Do NOT continue writing."
                    "Give only the final answer.\n"
                    "Question: {question}\n"
                    "Answer:")

        self.chain = self.build_graph()

    # Questions are embedded using fake embedding.
    # The resulting embedding is used to search for documents in Qdrant or memory.
    # Relevant documents are stored as context.
    def retrieve(self, state):
        query = state["question"]
        embedd = self.embedder.fake_embed(query)
        results = self.repo.search(vector=embedd, text_query=query)
        state["context"] = results
        return state
    
    # Receive a prompt that already contains context.
    # Respond briefly according to the instructions you set.
    # The output is cleaned and returned
    def answers(self, state):
        ctx = state["context"]
        question = state["question"]

        ctx_joined = "\n".join(ctx) if ctx else "No context available."
        prompt = self.prompt.format(context=ctx_joined, question=question)
        answer = self.llm.invoke(prompt)

        if isinstance(answer, list):
            answer = answer[0]["generated_text"]

        answer = answer.replace(prompt, "").strip()
        state["answer"] = answer
        
        return state
    
    # flow map with build_graph
    def build_graph(self):
        workflow = StateGraph(dict)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("answer", self.answers)
        workflow.set_entry_point("retrieve") # init point
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", END)
        chain = workflow.compile() # compile graph
        return chain
    
    # Question
    def process_question(self, question: str) -> Dict[str, Any]:
        start = time.time()
        results = self.chain.invoke({"question": question})
        return {
            "question": question,
            "answer": results["answer"],
            "context_used": results.get("context", []),
            "latency_sec": f"{round(time.time() - start, 3)} second"
        }
    
    # Add document
    def add_document(self, text: str) -> Dict[str, Any]:
        embedd = self.embedder.fake_embed(text)
        doc_repo = self.repo.add(text, embedd)
        return {"id": doc_repo, "status": "added"}

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
    rag_service = RAG_SERVICE(repo, embedder=embedder)
    service["rag"] = rag_service
    service["repo"] = repo

    yield # waiting for server to run
    service.clear() # removes all objects from the service when the server stops.

app = FastAPI(title="Refractored RAG Demo", lifespan=initialize)

@app.post("/ask") # Ask question for LLM
def question(req: QuestionRequest):
    try:
        return service["rag"].process_question(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/add") # Add documents
def add_document(req: DocumentRequest):
    try:
        return service["rag"].add_document(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status") # Checking Service Status
def status():
    repo_status = service["repo"].get_status()
    return {
        "qdrant_ready": repo_status["backend"],
        "in_memory_docs_count": repo_status["doc_count"],
        "service_status": "active"
    }