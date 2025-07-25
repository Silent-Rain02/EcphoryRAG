#!/usr/bin/env python
"""
Baseline RAG Implementation

This module provides a simpler baseline RAG implementation that only performs
vector similarity search on text chunks without entity extraction and depth-based retrieval.
It's designed as a comparison baseline for EcphoryRAG.
"""

import os
import json
import numpy as np
import time
import logging
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Use absolute imports as in EcphoryRAG
from ecphoryrag.src.ollama_clients import get_ollama_embedding, get_ollama_completion, get_token_usage, reset_token_counters
from ecphoryrag.src.faiss_store import FAISSChunkStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BaselineRAG')

class BaselineRAG:
    """
    A simple baseline RAG implementation that performs direct vector search
    on text chunks without entity extraction or depth-based retrieval.
    """
    
    def __init__(
        self,
        workspace_path: str = "workspace",
        embedding_model_name: str = "bge-m3",
        generation_llm_model: str = "phi4",
        ollama_host: str = "http://localhost:11434",
        embedding_dimension: int = 1024,
        enable_chunking: bool = True,
        chunk_size: int = 512,
        chunk_overlap: int = 128
    ):
        """
        Initialize the Baseline RAG system.
        
        Args:
            workspace_path: Path to the workspace directory for storing all persistent data
            embedding_model_name: Name of the Ollama embedding model
            generation_llm_model: Name of the Ollama LLM for response generation
            ollama_host: URL of the Ollama server
            embedding_dimension: Dimension of the embeddings
            enable_chunking: Whether to enable text chunking
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Define standard file paths within workspace
        self.chunks_idx_file = os.path.join(workspace_path, "chunks.faiss")
        self.chunks_meta_file = os.path.join(workspace_path, "chunks_meta.json")
        self.manifest_file = os.path.join(workspace_path, "workspace_manifest.json")
        
        # Store Ollama configuration
        self.ollama_host = ollama_host
        os.environ["OLLAMA_HOST"] = ollama_host
        
        # Store model names
        self.embedding_model_name = embedding_model_name
        self.generation_llm_model = generation_llm_model
        
        # Initialize Ollama client functions
        self.embed_func = get_ollama_embedding
        self.complete_func = get_ollama_completion
        
        # Store embedding dimension
        self.embedding_dimension = embedding_dimension
        
        # Initialize chunking parameters
        self.enable_chunking = enable_chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize indexed document manifest
        self.indexed_doc_manifest = {}
        
        # Initialize search parameters
        self.top_k = 5
        
        # Initialize available metrics
        self.available_metrics = ["exact_match", "f1_score", "rouge1", "rouge2", "rougeL"]
        
        # Initialize token usage and timing stats
        self.reset_usage_stats()
        
        # Try to load existing data
        try:
            # Load chunk store
            if os.path.exists(self.chunks_idx_file) and os.path.exists(self.chunks_meta_file):
                logger.info(f"Loading existing chunk store from {self.chunks_idx_file}")
                self.chunk_store = FAISSChunkStore.load_store(
                    self.chunks_idx_file,
                    self.chunks_meta_file,
                    embedding_dimension
                )
            else:
                logger.info("Creating new chunk store")
                self.chunk_store = FAISSChunkStore(embedding_dimension)
                
            # Load manifest
            if os.path.exists(self.manifest_file):
                logger.info(f"Loading workspace manifest from {self.manifest_file}")
                with open(self.manifest_file, 'r') as f:
                    self.indexed_doc_manifest = json.load(f)
            else:
                logger.info("Creating new workspace manifest")
                self.indexed_doc_manifest = {}
                
        except Exception as e:
            logger.error(f"Error loading workspace data: {str(e)}")
            # Initialize components as new if loading fails
            self.chunk_store = FAISSChunkStore(embedding_dimension)
            self.indexed_doc_manifest = {}
    
    def reset_usage_stats(self):
        """Reset token usage and timing statistics."""
        # Reset Ollama token counters
        reset_token_counters()
        
        # Initialize timing stats
        self.timing_stats = {
            "embedding_time": 0,
            "retrieval_time": 0,
            "generation_time": 0,
            "total_time": 0
        }
        
        # Initialize operation counts
        self.operation_counts = {
            "embedding_calls": 0,
            "retrieval_calls": 0,
            "generation_calls": 0
        }
    
    def get_usage_stats(self):
        """Get current token usage and timing statistics."""
        # Get token usage from ollama_clients
        token_usage = get_token_usage()
        
        # Combine with timing stats
        stats = {
            "token_usage": token_usage,
            "timing": self.timing_stats,
            "operations": self.operation_counts
        }
        
        return stats
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text string and reshape to the expected format.
        
        Args:
            text: The text to embed
            
        Returns:
            A NumPy array of shape (1, embedding_dimension)
        """
        try:
            # Track timing and count
            start_time = time.time()
            self.operation_counts["embedding_calls"] += 1
            
            embedding = self.embed_func(text, self.embedding_model_name)
            
            # Update timing stats
            self.timing_stats["embedding_time"] += time.time() - start_time
            
            if not embedding or not isinstance(embedding, (list, np.ndarray)):
                logger.warning("Embedding is empty or not a list/array, returning zeros.")
                return np.zeros((1, self.embedding_dimension), dtype=np.float32)
                
            embedding_np = np.array(embedding, dtype=np.float32).reshape(1, -1)
            
            if embedding_np.shape[1] != self.embedding_dimension:
                logger.warning(f"Embedding dimension mismatch: got {embedding_np.shape[1]}, expected {self.embedding_dimension}. Returning zeros.")
                return np.zeros((1, self.embedding_dimension), dtype=np.float32)
                
            return embedding_np
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {str(e)}")
            return np.zeros((1, self.embedding_dimension), dtype=np.float32)
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks of specified size with overlap.
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        return self.text_splitter.split_text(text)
    
    def index_documents(
        self, 
        documents: List[Dict[str, str]], 
        enable_chunking: Optional[bool] = None,
        chunk_size: Optional[int] = None, 
        chunk_overlap: Optional[int] = None,
        force_reindex_all: bool = False
    ) -> Tuple[int, int]:
        """
        Index documents for retrieval.
        
        Args:
            documents: List of document dictionaries, each with 'id', 'title', and 'text' keys
            enable_chunking: Whether to enable chunking (overrides instance setting)
            chunk_size: Size of text chunks (overrides instance setting)
            chunk_overlap: Overlap between chunks (overrides instance setting)
            force_reindex_all: Whether to force reindexing all documents
            
        Returns:
            Tuple of (number of entities indexed, number of chunks indexed)
        """
        # Override chunking parameters if provided
        use_chunking = enable_chunking if enable_chunking is not None else self.enable_chunking
        use_chunk_size = chunk_size if chunk_size is not None else self.chunk_size
        use_chunk_overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap
        
        # Update text splitter if chunking parameters changed
        if use_chunking and (use_chunk_size != self.chunk_size or use_chunk_overlap != self.chunk_overlap):
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=use_chunk_size,
                chunk_overlap=use_chunk_overlap
            )
        
        # Track number of chunks indexed
        total_chunks_indexed = 0
        
        # Process each document
        for doc in documents:
            doc_id = doc.get('id', '')
            doc_title = doc.get('title', '')
            doc_text = doc.get('text', '')
            
            # Skip if document is already indexed and not forcing reindex
            if not force_reindex_all and doc_id in self.indexed_doc_manifest:
                logger.debug(f"Skipping already indexed document: {doc_id}")
                continue
            
            # If chunking is enabled, split document text into chunks
            if use_chunking and len(doc_text) > use_chunk_size:
                chunks = self._split_text(doc_text)
                logger.debug(f"Split document {doc_id} into {len(chunks)} chunks")
            else:
                # Use the whole document as a single chunk
                chunks = [doc_text]
            
            # Index each chunk
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                
                # Get embedding for the chunk
                chunk_embedding = self._get_embedding(chunk_text)
                
                # Add chunk to chunk store
                self.chunk_store.add_chunks(
                    chunk_embeddings=chunk_embedding,
                    chunk_metadata_list=[{
                        'chunk_id': chunk_id,
                        'chunk_text': chunk_text,
                        'source_doc_id': doc_id,
                        'title': doc_title
                    }]
                )
                
                total_chunks_indexed += 1
            
            # Update manifest with document metadata
            self.indexed_doc_manifest[doc_id] = {
                "title": doc_title,
                "indexed_at": time.time(),
                "num_chunks": len(chunks)
            }
        
        # Save state after indexing
        self.save()
        
        logger.info(f"Indexed {total_chunks_indexed} chunks from {len(documents)} documents")
        return 0, total_chunks_indexed  # Return 0 for entities since we don't extract them
    
    def retrieve_relevant_chunks(
        self, 
        query_text: str, 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve the most relevant chunks for a query using vector similarity.
        
        Args:
            query_text: The query text
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of retrieved chunks with metadata
        """
        # Track timing
        start_time = time.time()
        self.operation_counts["retrieval_calls"] += 1
        
        # Get embedding for the query
        query_embedding = self._get_embedding(query_text)
        
        # Search for similar chunks
        retrieved_chunks = self.chunk_store.search_chunks(query_embedding, k=top_k)
        
        # Update timing stats
        self.timing_stats["retrieval_time"] += time.time() - start_time
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for generation")
        return retrieved_chunks
    
    def generate_response(self, query_text: str, retrieved_chunks: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Generate a response to the query using the retrieved chunks.
        
        Args:
            query_text: The original query text
            retrieved_chunks: List of retrieved chunk dictionaries
            
        Returns:
            Tuple of (generated response string, list of source documents)
        """
        # Track timing
        start_time = time.time()
        self.operation_counts["generation_calls"] += 1
        
        if not retrieved_chunks:
            logger.warning("No chunks retrieved, generating response with limited context")
            # Simple fallback prompt
            prompt = f"""
            Answer the following question to the best of your ability:
            
            Question: {query_text}
            
            Answer:
            """
            llm_raw_output, _ = self.complete_func(prompt, self.generation_llm_model)
            return llm_raw_output, []
        
        # Sort chunks by similarity for better reasoning flow
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Format chunks for the prompt
        chunk_texts = []
        for idx, chunk in enumerate(sorted_chunks, 1):
            chunk_id = chunk.get('chunk_id', 'unknown_id')
            source_doc_id = chunk.get('source_doc_id', 'unknown_doc')
            title = chunk.get('title', 'Untitled')
            chunk_text = chunk.get('chunk_text', '')
            
            # Format with chunk metadata
            formatted_text = f"Passage {idx} (ID: {chunk_id}, From Document: {source_doc_id}, Title: {title}):\n{chunk_text}"
            chunk_texts.append(formatted_text)
        
        chunk_context = "\n\n".join(chunk_texts)
        
        # Construct prompt for multi-hop reasoning
        prompt = f"""You are an expert research assistant specializing in multi-hop reasoning and connecting information across different sources. Your task is to answer a complex question by carefully analyzing the provided information.

QUESTION: {query_text}

AVAILABLE INFORMATION:

SUPPORTING TEXT PASSAGES:
{chunk_context}

INSTRUCTIONS:
1. This question requires multi-hop reasoning - you need to connect information from different passages.
2. Think step-by-step through the reasoning process before giving your final answer.
3. Your answer MUST be derived from the provided information only. Do not use external knowledge.
4. If the information provided is insufficient, clearly state what's missing.
5. After your reasoning, provide the final concise answer on a new line, prefixed with 'FINAL_ANSWER_TEXT:'.
6. Your answer MUST be a single word or short phrase that directly answers the question, with no explanation or extra text.
7. If the answer is not present in the provided information, output: "NOT FOUND".

REASONING STEPS:
(Work through your reasoning process here, connecting information across different passages)

FINAL_ANSWER_TEXT:
"""
        
        # Call the LLM for generation
        logger.info(f"Generating response using {len(retrieved_chunks)} chunks")
        llm_raw_output, _ = self.complete_func(prompt, self.generation_llm_model)

        # Extract final answer
        final_answer_marker = "FINAL_ANSWER_TEXT:"
        if final_answer_marker in llm_raw_output:
            llm_answer = llm_raw_output.split(final_answer_marker, 1)[-1].strip()
        else:
            logger.warning(f"Final answer marker '{final_answer_marker}' not found in LLM output. Using fallback.")
            # Fallback: use last non-empty line
            lines = [line.strip() for line in llm_raw_output.strip().split('\n') if line.strip()]
            llm_answer = lines[-1] if lines else llm_raw_output.strip()

        # Prepare source attribution
        cited_sources = []
        seen_doc_ids = set()
        
        # Get unique source documents from the retrieved chunks
        for chunk in retrieved_chunks:
            doc_id = chunk.get('source_doc_id')
            chunk_id = chunk.get('chunk_id')
            
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                source_doc = {
                    "id": doc_id,
                    "title": chunk.get('title', 'Unknown'),
                    "chunk_id": chunk_id
                }
                cited_sources.append(source_doc)
        
        # Update timing stats
        self.timing_stats["generation_time"] += time.time() - start_time
        
        return llm_answer, cited_sources
    
    def generate_response_with_raw(self, query_text: str, retrieved_chunks: List[Dict]) -> Tuple[str, List[Dict], str]:
        """
        Generate a response to the query and return the raw LLM output for evaluation.
        
        Args:
            query_text: The original query text
            retrieved_chunks: List of retrieved chunk dictionaries
            
        Returns:
            Tuple of (final answer, list of source documents, raw LLM output)
        """
        # Track timing
        start_time = time.time()
        self.operation_counts["generation_calls"] += 1
        
        if not retrieved_chunks:
            logger.warning("No chunks retrieved, generating response with limited context")
            # Simple fallback prompt
            prompt = f"""
            Answer the following question to the best of your ability:
            
            Question: {query_text}
            
            Answer:
            """
            llm_raw_output, _ = self.complete_func(prompt, self.generation_llm_model)
            return llm_raw_output, [], llm_raw_output
        
        # Sort chunks by similarity for better reasoning flow
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Format chunks for the prompt
        chunk_texts = []
        for idx, chunk in enumerate(sorted_chunks, 1):
            chunk_id = chunk.get('chunk_id', 'unknown_id')
            source_doc_id = chunk.get('source_doc_id', 'unknown_doc')
            title = chunk.get('title', 'Untitled')
            chunk_text = chunk.get('chunk_text', '')
            
            # Format with chunk metadata
            formatted_text = f"Passage {idx} (ID: {chunk_id}, From Document: {source_doc_id}, Title: {title}):\n{chunk_text}"
            chunk_texts.append(formatted_text)
        
        chunk_context = "\n\n".join(chunk_texts)
        
        # Construct prompt for multi-hop reasoning
        prompt = f"""You are an expert research assistant specializing in multi-hop reasoning and connecting information across different sources. Your task is to answer a complex question by carefully analyzing the provided information.

QUESTION: {query_text}

AVAILABLE INFORMATION:

SUPPORTING TEXT PASSAGES:
{chunk_context}

INSTRUCTIONS:
1. This question requires multi-hop reasoning - you need to connect information from different passages.
2. Think step-by-step through the reasoning process before giving your final answer.
3. Your answer MUST be derived from the provided information only. Do not use external knowledge.
4. If the information provided is insufficient, clearly state what's missing.
5. After your reasoning, provide the final concise answer on a new line, prefixed with 'FINAL_ANSWER_TEXT:'.
6. Your answer MUST be a single word or short phrase that directly answers the question, with no explanation or extra text.
7. If the answer is not present in the provided information, output: "NOT FOUND".

REASONING STEPS:
(Work through your reasoning process here, connecting information across different passages)

FINAL_ANSWER_TEXT:
"""
        
        # Call the LLM for generation
        logger.info(f"Generating response using {len(retrieved_chunks)} chunks")
        llm_raw_output, _ = self.complete_func(prompt, self.generation_llm_model)

        # Extract final answer
        final_answer_marker = "FINAL_ANSWER_TEXT:"
        if final_answer_marker in llm_raw_output:
            llm_answer = llm_raw_output.split(final_answer_marker, 1)[-1].strip()
        else:
            logger.warning(f"Final answer marker '{final_answer_marker}' not found in LLM output. Using fallback.")
            # Fallback: use last non-empty line
            lines = [line.strip() for line in llm_raw_output.strip().split('\n') if line.strip()]
            llm_answer = lines[-1] if lines else llm_raw_output.strip()

        # Prepare source attribution
        cited_sources = []
        seen_doc_ids = set()
        
        # Get unique source documents from the retrieved chunks
        for chunk in retrieved_chunks:
            doc_id = chunk.get('source_doc_id')
            chunk_id = chunk.get('chunk_id')
            
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                source_doc = {
                    "id": doc_id,
                    "title": chunk.get('title', 'Unknown'),
                    "chunk_id": chunk_id
                }
                cited_sources.append(source_doc)
        
        # Update timing stats
        self.timing_stats["generation_time"] += time.time() - start_time
        
        return llm_answer, cited_sources, llm_raw_output
    
    def query(self, query_text: str, top_k_final_traces: int = 5) -> Tuple[str, List[Dict]]:
        """
        Process a query from start to finish.
        
        Args:
            query_text: The query text
            top_k_final_traces: Number of top traces to use for final response generation
            
        Returns:
            Tuple of (answer text, list of source documents)
        """
        # Track total time
        start_time = time.time()
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve_relevant_chunks(query_text, top_k=top_k_final_traces)
        
        # Generate response using retrieved chunks
        answer, sources = self.generate_response(query_text, retrieved_chunks)
        
        # Update total time
        self.timing_stats["total_time"] += time.time() - start_time
        
        return answer, sources
    
    def save(self) -> None:
        """Save all data to disk."""
        try:
            # Save chunk store
            self.chunk_store.save_store(self.chunks_idx_file, self.chunks_meta_file)
            
            # Save manifest
            with open(self.manifest_file, 'w') as f:
                json.dump(self.indexed_doc_manifest, f, indent=2)
            
            logger.info(f"Saved BaselineRAG data to {self.workspace_path}")
        except Exception as e:
            logger.error(f"Error saving BaselineRAG data: {str(e)}")
    
    def reset_chunk_store(self) -> None:
        """Reset the chunk store."""
        self.chunk_store = FAISSChunkStore(self.embedding_dimension)
        self.indexed_doc_manifest = {}
        self.save()
        logger.info("Reset chunk store and manifest")