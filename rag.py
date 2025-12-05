import time, random
from repo import DocumentRepository
from langgraph.graph import StateGraph, END
from typing import List, Dict, Any

# Creating embedding classes
class Embedding:
    def fake_embed(self, text: str) -> List[float]:
        random.seed(abs(hash(text)) % 10000)
        return [random.random() for _ in range(128)]

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