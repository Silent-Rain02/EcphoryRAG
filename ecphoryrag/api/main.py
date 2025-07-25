# main.py

import os
from typing import List, Optional, Dict, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# 修改导入路径
from ecphoryrag import EcphoryRAG

# Configure environment variables with defaults
RAG_INDEX_FILE = os.environ.get("RAG_INDEX_FILE", "ecphory_rag.faiss")
RAG_METADATA_FILE = os.environ.get("RAG_METADATA_FILE", "ecphory_rag_metadata.json")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3")
EXTRACTION_LLM_MODEL = os.environ.get("EXTRACTION_LLM_MODEL", "phi4")
GENERATION_LLM_MODEL = os.environ.get("GENERATION_LLM_MODEL", "phi4")

# Initialize the FastAPI app
app = FastAPI(
    title="EcphoryRAG API",
    description="API for a neurocognitive-inspired RAG system using entities as memory traces",
    version="1.0.0"
)

# Define Pydantic models for request/response
class DocumentModel(BaseModel):
    id: str
    text: str

class IndexRequest(BaseModel):
    documents: List[DocumentModel]

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    retrieved_trace_texts: List[str]

# Global EcphoryRAG instance
ecphory_rag_instance = None

@app.on_event("startup")
async def startup_event():
    """
    Initialize the EcphoryRAG instance on startup.
    If FAISS files exist, load from them.
    """
    global ecphory_rag_instance
    
    # Check if index files exist
    index_exists = os.path.exists(RAG_INDEX_FILE) and os.path.exists(RAG_METADATA_FILE)
    
    # Initialize EcphoryRAG
    ecphory_rag_instance = EcphoryRAG(
        embedding_model_name=EMBEDDING_MODEL,
        extraction_llm_model=EXTRACTION_LLM_MODEL,
        generation_llm_model=GENERATION_LLM_MODEL,
        ollama_host=OLLAMA_HOST,
        faiss_index_file=RAG_INDEX_FILE if index_exists else None,
        faiss_metadata_file=RAG_METADATA_FILE if index_exists else None
    )
    
    # Override the query method to return both answer and traces used
    original_query = ecphory_rag_instance.query
    original_generate_response = ecphory_rag_instance.generate_response
    
    def extended_generate_response(query_text: str, retrieved_traces: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Wrapper for generate_response that returns both the answer and the traces used.
        """
        answer = original_generate_response(query_text, retrieved_traces)
        return answer, retrieved_traces
    
    def extended_query(query_text: str, top_k_final_traces: int = 5) -> Tuple[str, List[Dict]]:
        """
        Wrapper for query that returns both the answer and the traces used.
        """
        relevant_traces = ecphory_rag_instance.retrieve_relevant_traces(
            query_text, 
            top_k_final=top_k_final_traces
        )
        
        answer, traces = extended_generate_response(query_text, relevant_traces)
        return answer, traces
    
    # Replace methods with extended versions
    ecphory_rag_instance.generate_response = extended_generate_response
    ecphory_rag_instance.query = extended_query
    
    print(f"EcphoryRAG initialized with models: {EMBEDDING_MODEL}, {EXTRACTION_LLM_MODEL}, {GENERATION_LLM_MODEL}")
    if index_exists:
        print(f"Loaded existing index from {RAG_INDEX_FILE} and {RAG_METADATA_FILE}")
    else:
        print("Created new index (no existing files found)")

@app.post("/index/", response_model=dict)
async def index_documents(request: IndexRequest):
    """
    Index a batch of documents in the EcphoryRAG system.
    """
    if not ecphory_rag_instance:
        raise HTTPException(status_code=500, detail="EcphoryRAG system not initialized")
    
    # Convert Pydantic models to dictionaries
    documents = [{"id": doc.id, "text": doc.text} for doc in request.documents]
    
    # Index the documents
    try:
        num_traces = ecphory_rag_instance.index_documents(documents)
        return {
            "status": "success", 
            "message": f"Indexed {len(documents)} documents, created {num_traces} memory traces"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing documents: {str(e)}")

@app.post("/query/", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the EcphoryRAG system.
    """
    if not ecphory_rag_instance:
        raise HTTPException(status_code=500, detail="EcphoryRAG system not initialized")
    
    try:
        # Get answer and traces
        answer, traces = ecphory_rag_instance.query(request.query, top_k_final_traces=request.top_k)
        
        # Extract trace texts for response
        trace_texts = [trace["text"] for trace in traces]
        
        return QueryResponse(
            answer=answer,
            retrieved_trace_texts=trace_texts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/save_index/", response_model=dict)
async def save_index():
    """
    Save the current state of the EcphoryRAG index.
    """
    if not ecphory_rag_instance:
        raise HTTPException(status_code=500, detail="EcphoryRAG system not initialized")
    
    try:
        ecphory_rag_instance.save(RAG_INDEX_FILE, RAG_METADATA_FILE)
        return {
            "status": "success", 
            "message": f"Saved index to {RAG_INDEX_FILE} and {RAG_METADATA_FILE}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving index: {str(e)}")

# Run the app if executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 