# OOP-RAG
OOP for RAG with qdrant flow and embedded memory.
# RAG Refactor (simple demo)

This repository contains a refactored Retrieval-Augmented-Generation (RAG) demo:
- deterministic fake embedder (stable for tests)
- Qdrant integration (optional)
- HuggingFace LLM optional (controlled with env var)
- short-answer defaults and cleaning to avoid noisy outputs

### pip install -r requirements.txt
### docker run -p 6333:6333 qdrant/qdrant
### uvicorn "with_llm or app":app --reload --port 8000

This code addresses technical bugs (self parameter, class initialization) and search logic issues (fallback if Qdrant doesn't find results due to fake embedding). Save it in 3 separate files: repo.py, rag.py, and app.py. Files: main.py  and with.py as the whole code.
- File repo.py (Data Acces Layer): handle storage.
- File rag.py (Business logic layer): This file contains the Embedding logic (already fixed with self) and the LangGraph flow.
- File app.py (API Layer): This file ties everything together. The main improvement is in the initialize function, where we properly instantiate the class before calling its methods.

Design Decisions: This refactoring prioritizes the Separation of Concerns principle by breaking the monolithic application into three logical layers: repo.py (Data Access), rag.py (Business Logic), and app.py (Interface). This approach allows components to be modified independently. For example, the Qdrant database logic is fully encapsulated within the DocumentRepository. The API (app.py) no longer knows whether data is stored in Qdrant or the memory list, but instead interacts only through abstract add and search methods.

Trade-off Considered: One major trade-off was maintaining dual storage (storing data in Qdrant and the memory list simultaneously). This decision was made to ensure the availability of the demo system. Given the use of "Fake Embedding," which generates random vectors, vector-based searches often failed to find semantic relevance. By keeping a copy of the data in memory, the system could fall back to simple keyword matching if the vector results were empty, ensuring users always received a useful response even if the vector accuracy was low.

Maintainability Improvements This version significantly improves testability and scalability. By using Dependency Injection on RAG_SERVICE (passing repo and embedder as parameters), we can easily create unit tests that use mock repositories without requiring a real database connection. Additionally, global variables (USING_QDRANT, chain) have been removed and replaced by FastAPI lifecycle management (lifespan), preventing unwanted side effects and making the application safer when running in a production environment.
