# ecphory_rag.py

import os
import json
import numpy as np
import uuid
import networkx as nx
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Union
from collections import deque
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from functools import lru_cache
import threading

# 修改为绝对导入
from ecphoryrag.src.ollama_clients import get_ollama_embedding, get_ollama_completion, get_token_usage, reset_token_counters
from ecphoryrag.src.entity_extractor import extract_entities_llm
from ecphoryrag.src.faiss_store import FAISSMemoryTraceStore, FAISSChunkStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EcphoryRAG')

class EmbeddingCache:
    """
    Embedding cache class using LRU cache and thread-safe mechanism.
    """
    def __init__(self, maxsize=1000):
        self.cache = {}
        self.lock = threading.Lock()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[np.ndarray]:
        with self.lock:
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def set(self, key: str, value: np.ndarray):
        with self.lock:
            if len(self.cache) >= self.maxsize:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = value
    
    def get_stats(self) -> Dict[str, int]:
        with self.lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "size": len(self.cache)
            }

class EcphoryRAG:
    """
    Main class implementing the EcphoryRAG methodology, inspired by neurocognitive memory models.
    
    Ecphory is the process by which retrieval cues interact with stored memory traces
    to produce conscious remembering. This class implements this concept for RAG systems,
    using entities as memory traces and sophisticated retrieval mechanisms.
    """
    
    def __init__(
        self,
        workspace_path: str = "workspace",
        embedding_model_name: str = "bge-m3",
        extraction_llm_model: str = "phi4",
        generation_llm_model: str = "phi4",
        ollama_host: str = "http://localhost:11434",
        embedding_dimension: int = 1024,
        enable_chunking: bool = True,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        enable_hybrid_retrieval: bool = False
    ):
        """
        Initialize the EcphoryRAG system.
        
        Args:
            workspace_path: Path to the workspace directory for storing all persistent data
            embedding_model_name: Name of the Ollama embedding model (default: "bge-small-en")
            extraction_llm_model: Name of the Ollama LLM for entity extraction (default: "phi3:mini")
            generation_llm_model: Name of the Ollama LLM for response generation (default: "phi3:mini")
            ollama_host: URL of the Ollama server (default: "http://localhost:11434")
            embedding_dimension: Dimension of the embeddings (default: 1024 for bge-m3)
            enable_chunking: Whether to enable text chunking (default: True)
            chunk_size: Size of text chunks (default: 512)
            chunk_overlap: Overlap between chunks (default: 128)
        """
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Define standard file paths within workspace
        self.entity_traces_idx_file = os.path.join(workspace_path, "entity_traces.faiss")
        self.entity_traces_meta_file = os.path.join(workspace_path, "entity_traces_meta.json")
        self.chunks_idx_file = os.path.join(workspace_path, "chunks.faiss")
        self.chunks_meta_file = os.path.join(workspace_path, "chunks_meta.json")
        self.manifest_file = os.path.join(workspace_path, "workspace_manifest.json")
        self.graph_file = os.path.join(workspace_path, "knowledge_graph.graphml")
        
        # Store Ollama configuration
        self.ollama_host = ollama_host
        os.environ["OLLAMA_HOST"] = ollama_host
        
        # Store model names
        self.embedding_model_name = embedding_model_name
        self.extraction_llm_model = extraction_llm_model
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
        self.enable_hybrid_retrieval = enable_hybrid_retrieval
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize knowledge graph
        self.graph = nx.DiGraph()
        
        # Initialize indexed document manifest
        self.indexed_doc_manifest = {}
        
        # Initialize search parameters
        self.top_k_initial = 20
        self.retrieval_depth = 1
        
        # Initialize available metrics
        self.available_metrics = ["exact_match", "f1_score", "rouge1", "rouge2", "rougeL"]
        
        # Initialize a simple working memory (Short-Term Memory) for active traces
        self.stm_capacity = 20  # Number of traces that can be active in STM
        self.short_term_memory = deque(maxlen=self.stm_capacity)
        
        # Initialize token usage and timing stats
        self.reset_usage_stats()
        
        # Add embedding cache
        self.embedding_cache = EmbeddingCache(maxsize=1000)
        
        # Add batch processing queue
        self.embedding_batch_size = 10
        self.embedding_batch_queue = []
        self.embedding_batch_lock = threading.Lock()
        
        # Try to load existing data
        try:
            # Load entity trace store
            if os.path.exists(self.entity_traces_idx_file) and os.path.exists(self.entity_traces_meta_file):
                logger.info(f"Loading existing entity trace store from {self.entity_traces_idx_file}")
                self.trace_store = FAISSMemoryTraceStore.load_store(
                    self.entity_traces_idx_file,
                    self.entity_traces_meta_file,
                    embedding_dimension
                )
            else:
                logger.info("Creating new entity trace store")
                self.trace_store = FAISSMemoryTraceStore(embedding_dimension)
                
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
                
            # Load knowledge graph
            if os.path.exists(self.graph_file):
                logger.info(f"Loading knowledge graph from {self.graph_file}")
                self.graph = nx.read_graphml(self.graph_file)
            else:
                logger.info("Creating new knowledge graph")
                self.graph = nx.DiGraph()
                
        except Exception as e:
            logger.error(f"Error loading workspace data: {str(e)}")
            # Initialize components as new if loading fails
            self.trace_store = FAISSMemoryTraceStore(embedding_dimension)
            self.chunk_store = FAISSChunkStore(embedding_dimension)
            self.indexed_doc_manifest = {}
            self.graph = nx.DiGraph()
    
    def reset_usage_stats(self):
        """Reset token usage and timing statistics."""
        # Reset Ollama token counters
        reset_token_counters()
        
        # Initialize timing stats
        self.timing_stats = {
            "embedding_time": 0,
            "extraction_time": 0,
            "retrieval_time": 0,
            "generation_time": 0,
            "total_time": 0
        }
        
        # Initialize operation counts
        self.operation_counts = {
            "embedding_calls": 0,
            "extraction_calls": 0,
            "retrieval_calls": 0,
            "generation_calls": 0
        }
    
    def get_usage_stats(self):
        """Get current usage statistics."""
        # Get Ollama token usage
        token_usage = get_token_usage()
        
        # Get embedding cache statistics
        cache_stats = self.embedding_cache.get_stats()
        
        # Combine statistics
        stats = {
            "token_usage": token_usage,
            "timing": self.timing_stats,
            "operations": self.operation_counts,
            "embedding_cache": cache_stats
        }
        
        return stats
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get the embedding vector of the text, using cache and batch processing to optimize performance.
        
        Args:
            text: The text to embed
            
        Returns:
            A NumPy array of shape (1, embedding_dimension)
        """
        try:
            # Check cache
            cached_embedding = self.embedding_cache.get(text)
            if cached_embedding is not None:
                return cached_embedding
            
            # Track time and count
            start_time = time.time()
            self.operation_counts["embedding_calls"] += 1
            
            # Get embedding
            embedding = self.embed_func(text, self.embedding_model_name)
            
            # Update timing statistics
            self.timing_stats["embedding_time"] += time.time() - start_time
            
            if not embedding or not isinstance(embedding, (list, np.ndarray)):
                logger.warning("Embedding is empty or not a list/array, returning zeros.")
                return np.zeros((1, self.embedding_dimension), dtype=np.float32)
                
            embedding_np = np.array(embedding, dtype=np.float32).reshape(1, -1)
            
            if embedding_np.shape[1] != self.embedding_dimension:
                logger.warning(f"Embedding dimension mismatch: got {embedding_np.shape[1]}, expected {self.embedding_dimension}. Returning zeros.")
                return np.zeros((1, self.embedding_dimension), dtype=np.float32)
            
            # Cache result
            self.embedding_cache.set(text, embedding_np)
            
            return embedding_np
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {str(e)}")
            return np.zeros((1, self.embedding_dimension), dtype=np.float32)

    def _batch_get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Batch get the embedding vectors of multiple texts.
        
        Args:
            texts: The list of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        # Check cache
        embeddings = []
        texts_to_fetch = []
        
        for text in texts:
            cached = self.embedding_cache.get(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                texts_to_fetch.append(text)
        
        if texts_to_fetch:
            # Batch get uncached embeddings
            try:
                start_time = time.time()
                batch_embeddings = self.embed_func(texts_to_fetch, self.embedding_model_name)
                self.timing_stats["embedding_time"] += time.time() - start_time
                
                # Process results
                for text, embedding in zip(texts_to_fetch, batch_embeddings):
                    if embedding is not None:
                        embedding_np = np.array(embedding, dtype=np.float32).reshape(1, -1)
                        self.embedding_cache.set(text, embedding_np)
                        embeddings.append(embedding_np)
                    else:
                        embeddings.append(np.zeros((1, self.embedding_dimension), dtype=np.float32))
                        
            except Exception as e:
                logger.error(f"Failed to get batch embeddings: {str(e)}")
                # Add zero vectors for failed requests
                embeddings.extend([np.zeros((1, self.embedding_dimension), dtype=np.float32)] * len(texts_to_fetch))
        
        return embeddings
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks of specified size with overlap.
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Ensure chunk_size is reasonable
        if self.chunk_size <= 0:
            self.chunk_size = 256
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = self.chunk_size // 4
            
        # Simple character-based chunking
        chunks = []
        start = 0
        
        # Safety counter to prevent infinite loops
        max_iterations = len(text) * 2  # Should be more than enough
        iteration = 0
        
        while start < len(text) and iteration < max_iterations:
            iteration += 1
            
            # Take a chunk of size chunk_size
            end = min(start + self.chunk_size, len(text))
            
            # If we're not at the beginning, try to find a good break point
            if start > 0 and end < len(text):
                # Try to find a period, newline, or space to break on
                found_break = False
                for break_char in ['. ', '\n', ' ']:
                    pos = text[start:end].rfind(break_char)
                    if pos != -1:
                        end = start + pos + len(break_char)
                        found_break = True
                        break
            
            # Add the chunk to our list
            chunks.append(text[start:end])
            
            # Move to next chunk, ensuring we make progress
            new_start = end - self.chunk_overlap
            
            # Ensure we're making forward progress to avoid infinite loops
            if new_start <= start:
                new_start = start + 1
                
            start = new_start
            
            # Log progress for very large texts
            if iteration % 100 == 0:
                logger.info(f"Processed {iteration} chunks, current position: {start}/{len(text)}")
        
        return chunks
    
    def index_documents(
        self, 
        documents: List[Dict[str, str]], 
        enable_chunking: Optional[bool] = None,
        chunk_size: Optional[int] = None, 
        chunk_overlap: Optional[int] = None,
        force_reindex_all: bool = False
    ) -> Tuple[int, int]:
        """
        Process documents to extract entities and store them as memory traces.
        Supports incremental indexing with document versioning.
        
        Args:
            documents: List of document dictionaries, each with 'id' and 'text' keys
            enable_chunking: Whether to split documents into chunks (default: use instance setting)
            chunk_size: Size of text chunks to create (default: use instance setting)
            chunk_overlap: Overlap between consecutive chunks (default: use instance setting)
            force_reindex_all: If True, clear all existing data and reindex everything
            
        Returns:
            Tuple of (number of memory traces created, number of chunks created)
        """
        # Handle force reindex
        if force_reindex_all:
            logger.info("Force reindex requested. Clearing all existing data.")
            self.trace_store = FAISSMemoryTraceStore(self.embedding_dimension)
            self.chunk_store = FAISSChunkStore(self.embedding_dimension)
            self.indexed_doc_manifest = {}
            self.graph = nx.DiGraph()
        
        # Update chunking parameters if provided
        if enable_chunking is not None:
            self.enable_chunking = enable_chunking
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
            
        # Validate chunking parameters
        self.chunk_size = max(50, self.chunk_size)  # Ensure minimum chunk size
        self.chunk_overlap = min(self.chunk_overlap, self.chunk_size // 2)  # Ensure overlap isn't too large
        
        logger.info(f"Chunking settings: enabled={self.enable_chunking}, size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        total_traces_added = 0
        total_chunks_added = 0
        
        # Prepare storage for all embeddings and metadata
        all_chunk_embeddings_to_add = []
        all_chunk_metadata_to_add = []
        all_entity_trace_embeddings_to_add = []
        all_entity_trace_metadata_to_add = []
        
        # Process each document
        for doc_idx, doc in enumerate(documents):
            doc_id = doc.get('id')
            doc_text = doc.get('text', '')
            
            if not doc_id or not doc_text.strip():
                logger.warning(f"Skipping document with missing ID or empty text: {doc_id}")
                continue
            
            # Calculate document content hash
            doc_content_hash = hashlib.md5(doc_text.encode('utf-8')).hexdigest()
            
            # Check if document is unchanged
            if doc_id in self.indexed_doc_manifest and self.indexed_doc_manifest[doc_id] == doc_content_hash:
                logger.info(f"Document {doc_id} unchanged, skipping processing")
                continue
            
            logger.info(f"Processing document {doc_id} ({doc_idx+1}/{len(documents)})")
            
            # Determine whether to process the whole document or split into chunks
            if self.enable_chunking:
                # 1. Split document into chunks
                text_chunks = self._split_text(doc_text)
                logger.info(f"Split document {doc_id} into {len(text_chunks)} chunks")
            else:
                # Process the document as a whole
                text_chunks = [doc_text]
                logger.info(f"Processing document {doc_id} as a whole (chunking disabled)")
            
            # Process each chunk or the whole document
            for chunk_idx, chunk_text in enumerate(text_chunks):
                # Skip empty chunks
                if not chunk_text.strip():
                    continue
                
                chunk_id = uuid.uuid4().hex
                timestamp = datetime.now().isoformat()
                
                # 2. Store Chunk
                chunk_embedding = self._get_embedding(chunk_text)
                all_chunk_embeddings_to_add.append(chunk_embedding[0])  # Flatten to 1D
                all_chunk_metadata_to_add.append({
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text,
                    "source_doc_id": doc_id,
                    "created_at": timestamp
                })
                
                # Add chunk to graph
                self.graph.add_node(
                    chunk_id,
                    type='chunk',
                    text=chunk_text,
                    doc_id=doc_id
                )
                
                # 3. Extract Rich Entities from this chunk
                try:
                    rich_entities = extract_entities_llm(
                        chunk_text,
                        ollama_completion_func=self.complete_func,
                        extraction_llm_model=self.extraction_llm_model
                    )
                except Exception as e:
                    logger.error(f"Error extracting entities from chunk {chunk_idx} of document {doc_id}: {str(e)}")
                    rich_entities = []
                
                if not rich_entities:
                    logger.warning(f"No entities extracted from chunk {chunk_idx} of document {doc_id}")
                    continue
                
                logger.info(f"Extracted {len(rich_entities)} entities from chunk {chunk_idx} of document {doc_id}")
                
                # Process each entity
                for entity in rich_entities:
                    # Get the entity text and description
                    entity_text = entity['text']
                    entity_description = entity.get('description', '')
                    
                    # Combine entity text and description for embedding
                    combined_text = f"{entity_text}"
                    if entity_description:
                        combined_text += f": {entity_description}"
                    
                    # Get embedding for the combined text
                    entity_embedding = self._get_embedding(combined_text)
                    
                    # Create entity trace metadata
                    entity_trace_id = uuid.uuid4().hex
                    entity_trace_metadata = {
                        "id": entity_trace_id,
                        "entity_text": entity_text,
                        "entity_type": entity.get('type', 'MISC'),
                        "entity_description": entity_description,
                        "entity_importance": entity.get('importance_score', 3),
                        "source_chunk_id": chunk_id,
                        "source_doc_id": doc_id,
                        "status": "dormant",
                        "created_at": timestamp,
                        "last_accessed_at": timestamp,
                        "access_count": 0
                    }
                    
                    # Add to batch
                    all_entity_trace_embeddings_to_add.append(entity_embedding[0])  # Flatten to 1D
                    all_entity_trace_metadata_to_add.append(entity_trace_metadata)
                    
                    # Add entity to graph and create edge to chunk
                    self.graph.add_node(
                        entity_trace_id,
                        type='entity',
                        text=entity_text,
                        entity_type=entity.get('type', 'MISC'),
                        description=entity_description
                    )
                    self.graph.add_edge(
                        entity_trace_id,
                        chunk_id,
                        type='EXTRACTED_FROM'
                    )
            
            # Process in batches to avoid memory issues
            if len(all_chunk_embeddings_to_add) >= 100 or doc_idx == len(documents) - 1:
                # Add chunks to chunk store if not empty
                if all_chunk_embeddings_to_add:
                    chunk_embeddings_array = np.vstack(all_chunk_embeddings_to_add)
                    chunk_ids = self.chunk_store.add_chunks(chunk_embeddings_array, all_chunk_metadata_to_add)
                    total_chunks_added += len(chunk_ids)
                    logger.info(f"Added {len(chunk_ids)} chunks to store")
                    
                    # Clear the lists for the next batch
                    all_chunk_embeddings_to_add = []
                    all_chunk_metadata_to_add = []
            
            if len(all_entity_trace_embeddings_to_add) >= 100 or doc_idx == len(documents) - 1:
                # Add entity traces to trace store if not empty
                if all_entity_trace_embeddings_to_add:
                    entity_embeddings_array = np.vstack(all_entity_trace_embeddings_to_add)
                    trace_ids = self.trace_store.add_traces(entity_embeddings_array, all_entity_trace_metadata_to_add)
                    total_traces_added += len(trace_ids)
                    logger.info(f"Added {len(trace_ids)} entity traces to store")
                    
                    # Clear the lists for the next batch
                    all_entity_trace_embeddings_to_add = []
                    all_entity_trace_metadata_to_add = []
            
            # Update manifest with document hash
            self.indexed_doc_manifest[doc_id] = doc_content_hash
        
        # Save all changes
        self.save()
        
        logger.info(f"Indexing complete. Added {total_traces_added} entity traces and {total_chunks_added} chunks.")
        return total_traces_added, total_chunks_added
    
    def _extract_query_cues(self, query_text: str) -> List[str]:
        """
        Extract entity cues from the query text.
        
        Args:
            query_text: The query text
            
        Returns:
            A list of extracted entity strings that will serve as retrieval cues
        """
        logger.info(f"Extracting cues from query: {query_text[:50]}...")
        
        # Use the entity extractor to find initial entity cues
        entities = extract_entities_llm(
            query_text,
            self.complete_func,
            self.extraction_llm_model
        )
        
        if not entities:
            logger.warning("No explicit cues extracted from query")
            return []
        
        # Extract just the text from each entity dictionary
        cues = [entity['text'] for entity in entities if isinstance(entity, dict) and 'text' in entity]
        logger.info(f"Extracted {len(cues)} cues from query: {', '.join(cues[:3])}{'...' if len(cues) > 3 else ''}")
            
        return cues
    
    def retrieve_relevant_traces(
        self, 
        query_text: str, 
        top_k_initial: int = 20, 
        top_k_final: int = 5, 
        retrieval_depth: int = 1
    ) -> Dict[str, List[Dict]]:
        """
        Retrieve and activate relevant memory traces using ecphory-inspired process.
        
        Args:
            query_text: The query text
            top_k_initial: Number of initial traces to retrieve in primary ecphory
            top_k_final: Number of final traces to return after all processing
            retrieval_depth: Depth of associative retrieval (0 = no associative retrieval)
            
        Returns:
            Dictionary containing:
            - "entities": List of memory trace metadata dictionaries, sorted by relevance
            - "chunks": List of relevant document chunks
        """
        # Track timing
        start_time = time.time()
        self.operation_counts["retrieval_calls"] += 1
        
        logger.info(f"Retrieving traces for query: {query_text[:50]}...")
        
        # Get query embedding
        query_embedding = self._get_embedding(query_text)
        
        # Step 1: Initial Entity Retrieval
        initial_traces = self.trace_store.search_traces(
            query_embedding, 
            k=top_k_initial,
            allowed_statuses=['dormant', 'active_in_stm']
        )
        
        logger.info(f"Retrieved {len(initial_traces)} initial traces")
        
        # Step 2: Get associated chunks from retrieved entities 
        associated_chunks = []
        seen_chunk_ids = set()
        
        for trace in initial_traces:
            chunk_id = trace.get('source_chunk_id')
            if chunk_id and chunk_id not in seen_chunk_ids:
                chunk = self.chunk_store.get_chunk_by_id(chunk_id)
                if chunk:
                    # Add similarity information
                    chunk['similarity'] = trace.get('similarity', 0.0)
                    associated_chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)
        
        logger.info(f"Retrieved {len(associated_chunks)} associated chunks")
            
        # Step 3: Direct chunk retrieval - reduce number, improve quality
        direct_chunks = self.chunk_store.search_chunks(
            query_embedding,
            k=min(10, top_k_initial)  # Reduce number of direct chunks
        )
        
        logger.info(f"Retrieved {len(direct_chunks)} direct chunks")
        
        # Combine and deduplicate chunks 
        all_chunks = associated_chunks + direct_chunks
        unique_chunks = []
        seen_chunk_ids = set()
        
        for chunk in all_chunks:
            chunk_id = chunk.get('id') or chunk.get('chunk_id')
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                unique_chunks.append(chunk)
        
        # Sort chunks by relevance and take only top chunks
        unique_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        final_chunks = unique_chunks[:min(5, len(unique_chunks))]  
        
        # Process entities through associative retrieval if depth > 0
        all_traces = initial_traces.copy()
        
        if retrieval_depth > 0:
            for depth in range(retrieval_depth):
                logger.info(f"Starting associative retrieval at depth {depth+1}")
                
                # Take top traces so far 
                seed_traces = all_traces[:min(10, len(all_traces))]
                logger.info(f"Selected {len(seed_traces)} seed traces for depth {depth+1}")
                
                # Batch get embeddings of seed entities
                seed_texts = []
                for trace in seed_traces:
                    entity_text = trace.get('entity_text', trace.get('text', ''))
                    entity_description = trace.get('entity_description', '')
                    if entity_text:
                        combined_text = f"{entity_text}"
                        if entity_description:
                            combined_text += f": {entity_description}"
                        seed_texts.append(combined_text)
                
                # Batch get embeddings
                seed_embeddings = self._batch_get_embeddings(seed_texts)
                
                if not seed_embeddings:
                    logger.warning(f"No valid seed embeddings found at depth {depth+1}")
                    continue
                
                # Use vectorized operation to calculate weighted average
                weights = np.array([trace.get('similarity', 0.5) for trace in seed_traces])
                weights = weights / np.sum(weights)
                
                # Use matrix operation to calculate weighted average
                seed_embeddings_array = np.vstack(seed_embeddings)
                seed_embedding = np.average(seed_embeddings_array, axis=0, weights=weights, keepdims=True)
                
                # Search for associated traces, excluding ones we already have
                existing_ids = {trace['id'] for trace in all_traces}
                # Increase secondary retrieval range to 3 times top_k_initial
                secondary_k = top_k_initial * 3
                
                logger.info(f"Searching for {secondary_k} associated traces at depth {depth+1}")
                
                secondary_traces = self.trace_store.search_traces(
                    seed_embedding, 
                    k=secondary_k,
                    allowed_statuses=['dormant', 'active_in_stm']
                )
                
                # Filter out traces we already have
                new_traces = [t for t in secondary_traces if t['id'] not in existing_ids]
                
                if not new_traces:
                    logger.info(f"No new traces found at depth {depth+1}")
                    break
                
                logger.info(f"Found {len(new_traces)} new associated traces at depth {depth+1}")
                
                # Add new traces to our collection
                all_traces.extend(new_traces)
                
                # Update status of new traces
                for trace in new_traces:
                    self.trace_store.update_trace_status(trace['id'], 'active_in_stm')
                    if trace['id'] not in self.short_term_memory:
                        self.short_term_memory.append(trace['id'])
                
                logger.info(f"Total traces after depth {depth+1}: {len(all_traces)}")
        
        # Re-rank traces based on relevance to original query
        for trace in all_traces:
            # Get the entity text and description
            entity_text = trace.get('entity_text', trace.get('text', ''))
            entity_description = trace.get('entity_description', '')
            
            if not entity_text:
                logger.warning(f"Trace {trace.get('id', 'unknown')} has no text content, skipping scoring")
                trace['similarity'] = 0.0
                continue
                
            # Combine text and description for embedding
            combined_text = f"{entity_text}"
            if entity_description:
                combined_text += f": {entity_description}"
            
            # Get embedding for the combined text
            trace_embedding = self._get_embedding(combined_text)
            
            # Calculate cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            trace_norm = np.linalg.norm(trace_embedding)
            
            if query_norm > 0 and trace_norm > 0:
                similarity = np.dot(query_embedding.flatten(), trace_embedding.flatten()) / (query_norm * trace_norm)
                trace['similarity'] = float(similarity)
            else:
                trace['similarity'] = 0.0
        
        # Record detailed information of newly discovered entities
        for trace in all_traces[:3]:  # Only record the first 3, to avoid too long log
            entity_text = trace.get('entity_text', trace.get('text', ''))
            similarity = trace.get('similarity', 0.0)
            logger.info(f"New trace at depth {depth+1}: {entity_text[:50]}... (similarity: {similarity:.3f})")
                
        # Sort by similarity (descending)
        ranked_traces = sorted(all_traces, key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Select top_k_final traces
        final_traces = ranked_traces[:top_k_final]
        
        # Update timing stats
        self.timing_stats["retrieval_time"] += time.time() - start_time
        
        logger.info(f"Returning {len(final_traces)} final traces and {len(final_chunks)} chunks for generation")
        return {
            "entities": final_traces,
            "chunks": final_chunks
        }
    
    def _compress_chunk(self, chunk_text: str, query_text: str) -> str:
        """
        Compress a long text chunk while preserving information relevant to the query.
        
        Args:
            chunk_text: The text chunk to compress
            query_text: The original query text
            
        Returns:
            Compressed text chunk
        """
        if len(chunk_text.split()) <= 200:  # If chunk is not too long, return as is
            return chunk_text
            
        try:
            compression_prompt = f"""Please compress the following text while preserving key information relevant to the question: "{query_text}"

Text to compress:
{chunk_text}

Compressed version (max 100 words):
"""
            compressed_text, _ = self.complete_func(compression_prompt, self.generation_llm_model)
            compressed_text = compressed_text.strip()
            logger.info(f"Compressed chunk from {len(chunk_text.split())} to {len(compressed_text.split())} words")
            return compressed_text
        except Exception as e:
            logger.warning(f"Failed to compress chunk: {str(e)}")
            return chunk_text

    def _get_chunk_count_for_top_k(self, top_k: int) -> int:
        """
        Dynamically determine the number of text chunks to use based on top_k value.
        
        Args:
            top_k: The number of entities selected in the end
            
        Returns:
            The number of text chunks to use
        """
        # Define mapping: top_k -> number of text chunks
        chunk_mapping = {
            1: 2,    
            3: 3,     
            5: 4,      
            10: 5,     
            20: 6,    
            40: 8,     
            80: 10,   
            100: 12,   
            120: 15    
        }
        
        # If top_k is in the mapping, return the corresponding value
        if top_k in chunk_mapping:
            return chunk_mapping[top_k]
        
        # If top_k is not in the mapping, use linear interpolation
        sorted_keys = sorted(chunk_mapping.keys())
        
        # If top_k is less than the minimum, use the minimum
        if top_k < sorted_keys[0]:
            return chunk_mapping[sorted_keys[0]]
        
        # If top_k is greater than the maximum, use the maximum
        if top_k > sorted_keys[-1]:
            return chunk_mapping[sorted_keys[-1]]
        
        # Find the appropriate interval for linear interpolation
        for i in range(len(sorted_keys) - 1):
            if sorted_keys[i] <= top_k <= sorted_keys[i + 1]:
                k1, k2 = sorted_keys[i], sorted_keys[i + 1]
                c1, c2 = chunk_mapping[k1], chunk_mapping[k2]
                
                # Linear interpolation
                ratio = (top_k - k1) / (k2 - k1)
                chunk_count = int(c1 + ratio * (c2 - c1))
                return max(1, chunk_count)  # Ensure at least 1 is returned
        
        # Default case
        return 5

    def generate_response(self, query_text: str, retrieved_results: Union[List[Dict], Dict], top_k: int = 5) -> Tuple[str, List[Dict]]:
        """
        Generate a response to the query using the retrieved entities and chunks.
        
        Args:
            query_text: The original query text
            retrieved_results: Either a list of trace dictionaries (old format) or a dictionary 
                containing retrieved entities and chunks (new format): {"entities": [...], "chunks": [...]}
            top_k: Number of top entities used (for determining chunk count)
            
        Returns:
            Tuple of (generated response string, list of source documents)
        """
        # Track timing
        start_time = time.time()
        self.operation_counts["generation_calls"] += 1
        
        # Handle both old format (list of traces) and new format (dict with entities and chunks)
        if isinstance(retrieved_results, list):
            # Old format - convert to new format
            entities = retrieved_results
            chunks = []
        else:
            # New format
            entities = retrieved_results.get("entities", [])
            chunks = retrieved_results.get("chunks", [])
        
        if not entities and not chunks:
            logger.warning("No entities or chunks retrieved, generating response with limited context")
            # Simple fallback prompt
            prompt = f"""
            Answer the following question to the best of your ability:
            
            Question: {query_text}
            
            Answer:
            """
            llm_raw_output, token_info = self.complete_func(prompt, self.generation_llm_model)
            return llm_raw_output, []
        
        # Sort entities by similarity for better reasoning flow
        sorted_entities = sorted(entities, key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Format entities for the prompt - 优先使用实体信息
        entity_texts = []
        for idx, entity in enumerate(sorted_entities, 1):
            entity_text = entity.get('entity_text', 'Unknown entity')
            entity_type = entity.get('entity_type', 'MISC')
            entity_desc = entity.get('entity_description', '')
            
            # Format with entity type and description
            if entity_desc:
                formatted_text = f"{idx}. {entity_text} (Type: {entity_type}): {entity_desc}"
            else:
                formatted_text = f"{idx}. {entity_text} (Type: {entity_type})"
                
            entity_texts.append(formatted_text)
        
        entity_context = "\n".join(entity_texts)
        
        chunk_context = ""
        if chunks:
            # Dynamically determine the number of text chunks to use based on top_k value
            chunk_count = self._get_chunk_count_for_top_k(top_k)
            logger.info(f"Using {chunk_count} chunks for top_k={top_k}")
            
            # Select the most relevant text chunks
            top_chunks = sorted(chunks, key=lambda x: x.get('similarity', 0), reverse=True)[:chunk_count]
            
            chunk_texts = []
            for idx, chunk in enumerate(top_chunks, 1):
                chunk_text = chunk.get('chunk_text', '')
                similarity = chunk.get('similarity', 0.0)
                
                # Compress very long text chunks
                if len(chunk_text.split()) > 300:
                    chunk_text = self._compress_chunk(chunk_text, query_text)
                
                # Simplify format
                formatted_text = f"Passage {idx} (Relevance: {similarity:.2f}):\n{chunk_text}"
                chunk_texts.append(formatted_text)
            
            chunk_context = "\n\n".join(chunk_texts)
        
        # Construct simplified prompt, prioritize entity information
        if entity_context and chunk_context:
            prompt = f"""You are an expert research assistant. Answer the following question using the provided information.

QUESTION: {query_text}

KEY ENTITIES:
{entity_context}

SUPPORTING PASSAGES:
{chunk_context}

INSTRUCTIONS:
1. Focus on the key entities first, then use passages for additional context.
2. Your answer MUST be derived from the provided information only.
3. After your reasoning, provide the final answer on a new line, prefixed with 'FINAL_ANSWER_TEXT:'.
4. Your answer MUST be a single word or short phrase that directly answers the question.
5. If the answer is not present in the provided information, output: "NOT FOUND".

REASONING:
(Work through your reasoning process here)

FINAL_ANSWER_TEXT:
"""
        elif entity_context:
            
            prompt = f"""You are an expert research assistant. Answer the following question using the provided entities.

QUESTION: {query_text}

KEY ENTITIES:
{entity_context}

INSTRUCTIONS:
1. Use the provided entities to answer the question.
2. Your answer MUST be derived from the provided information only.
3. After your reasoning, provide the final answer on a new line, prefixed with 'FINAL_ANSWER_TEXT:'.
4. Your answer MUST be a single word or short phrase that directly answers the question.
5. If the answer is not present in the provided information, output: "NOT FOUND".

REASONING:
(Work through your reasoning process here)

FINAL_ANSWER_TEXT:
"""
        else:
            prompt = f"""You are an expert research assistant. Answer the following question using the provided passages.

QUESTION: {query_text}

SUPPORTING PASSAGES:
{chunk_context}

INSTRUCTIONS:
1. Use the provided passages to answer the question.
2. Your answer MUST be derived from the provided information only.
3. After your reasoning, provide the final answer on a new line, prefixed with 'FINAL_ANSWER_TEXT:'.
4. Your answer MUST be a single word or short phrase that directly answers the question.
5. If the answer is not present in the provided information, output: "NOT FOUND".

REASONING:
(Work through your reasoning process here)

FINAL_ANSWER_TEXT:
"""
        
        # Call the LLM for generation
        logger.info(f"Generating response using {len(entities)} entities and {len(chunks)} chunks")
        llm_raw_output, token_info = self.complete_func(prompt, self.generation_llm_model)

        # --- POST-PROCESSING LOGIC TO EXTRACT FINAL ANSWER ---
        final_answer_marker = "FINAL_ANSWER_TEXT:"
        if final_answer_marker in llm_raw_output:
            llm_answer = llm_raw_output.split(final_answer_marker, 1)[-1].strip()
        else:
            logger.warning(f"Final answer marker '{final_answer_marker}' not found in LLM output. Using fallback.")
            # Fallback: use last non-empty line
            lines = [line.strip() for line in llm_raw_output.strip().split('\n') if line.strip()]
            llm_answer = lines[-1] if lines else llm_raw_output.strip()
            
        # Ensure llm_answer is a string with additional logging
        logger.info(f"Raw LLM answer type: {type(llm_answer)}")
        logger.info(f"Raw LLM answer: {llm_answer}")
        
        if isinstance(llm_answer, list):
            logger.warning(f"LLM answer is a list: {llm_answer}")
            llm_answer = llm_answer[0] if llm_answer else "NOT FOUND"
        elif not isinstance(llm_answer, str):
            logger.warning(f"LLM answer is not a string: {llm_answer}")
            llm_answer = str(llm_answer)
            
        # Final check to ensure we have a string
        if not isinstance(llm_answer, str):
            logger.error(f"Failed to convert answer to string, using fallback")
            llm_answer = "NOT FOUND"
            
        logger.info(f"Final processed answer: {llm_answer}")
        # --- END POST-PROCESSING LOGIC ---

        # Prepare source attribution
        cited_sources = []
        seen_doc_ids = set()
        
        # Get unique source documents from the retrieved chunks
        for chunk in chunks:
            doc_id = chunk.get('source_doc_id')
            chunk_id = chunk.get('chunk_id')
            
            if doc_id and doc_id not in seen_doc_ids:
                # Get the chunk text
                chunk_text = chunk.get('chunk_text', '')
                title = chunk.get('title', '')
                
                cited_sources.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "title": title,
                    "text_preview": chunk_text[:150] + "..." if chunk_text else "No text preview available"
                })
                seen_doc_ids.add(doc_id)
        
        # Update timing stats
        self.timing_stats["generation_time"] += time.time() - start_time
        
        return llm_answer, cited_sources
    
    def generate_response_with_raw(self, query_text: str, retrieved_results: Union[List[Dict], Dict], top_k: int = 5):
        """
        Generate a response and return (answer, sources, llm_raw_output) for detailed evaluation.
        
        Args:
            query_text: The query text
            retrieved_results: Either a list of trace dictionaries (old format) or a dictionary 
                containing retrieved entities and chunks (new format): {"entities": [...], "chunks": [...]}
            top_k: Number of top entities used (for determining chunk count)
        """
        # Track timing
        start_time = time.time()
        self.operation_counts["generation_calls"] += 1
        
        # 复用原有逻辑
        if isinstance(retrieved_results, list):
            entities = retrieved_results
            chunks = []
        else:
            entities = retrieved_results.get("entities", [])
            chunks = retrieved_results.get("chunks", [])
            
        if not entities and not chunks:
            prompt = f"""
            Answer the following question to the best of your ability:
            
            Question: {query_text}
            
            Answer:
            """
            llm_raw_output, token_info = self.complete_func(prompt, self.generation_llm_model)
            
            # Update timing stats
            self.timing_stats["generation_time"] += time.time() - start_time
            
            return llm_raw_output, [], llm_raw_output
            
        # 构造prompt
        sorted_entities = sorted(entities, key=lambda x: x.get('similarity', 0), reverse=True)
        entity_texts = []
        for idx, entity in enumerate(sorted_entities, 1):
            entity_text = entity.get('entity_text', 'Unknown entity')
            entity_type = entity.get('entity_type', 'MISC')
            entity_desc = entity.get('entity_description', '')
            if entity_desc:
                formatted_text = f"{idx}. {entity_text} (Type: {entity_type}): {entity_desc}"
            else:
                formatted_text = f"{idx}. {entity_text} (Type: {entity_type})"
            entity_texts.append(formatted_text)
        entity_context = "\n".join(entity_texts)
        
        # Dynamically determine the number of text chunks to use based on top_k value
        chunk_count = self._get_chunk_count_for_top_k(top_k)
        logger.info(f"Using {chunk_count} chunks for top_k={top_k}")
        
        # Select the most relevant text chunks
        top_chunks = sorted(chunks, key=lambda x: x.get('similarity', 0), reverse=True)[:chunk_count]
        
        chunk_texts = []
        for idx, chunk in enumerate(top_chunks, 1):
            chunk_id = chunk.get('chunk_id', 'unknown_id')
            source_doc_id = chunk.get('source_doc_id', 'unknown_doc')
            title = chunk.get('title', 'Untitled')
            chunk_text = chunk.get('chunk_text', '')
            
            # Compress very long text chunks
            if len(chunk_text.split()) > 300:
                chunk_text = self._compress_chunk(chunk_text, query_text)
                
            formatted_text = f"Passage {idx} (ID: {chunk_id}, From Document: {source_doc_id}, Title: {title}):\n{chunk_text}"
            chunk_texts.append(formatted_text)
        chunk_context = "\n\n".join(chunk_texts)
        
        prompt = f"""You are an expert research assistant specializing in multi-hop reasoning and connecting information across different sources. Your task is to answer a complex question by carefully analyzing the provided information.

QUESTION: {query_text}

AVAILABLE INFORMATION:

KEY ENTITIES IDENTIFIED:
{entity_context}

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
        llm_raw_output, token_info = self.complete_func(prompt, self.generation_llm_model)
        
        # Extract final answer
        final_answer_marker = "FINAL_ANSWER_TEXT:"
        if final_answer_marker in llm_raw_output:
            llm_answer = llm_raw_output.split(final_answer_marker, 1)[-1].strip()
        else:
            lines = [line.strip() for line in llm_raw_output.strip().split('\n') if line.strip()]
            llm_answer = lines[-1] if lines else llm_raw_output.strip()
            
        # Prepare source document information
        cited_sources = []
        seen_doc_ids = set()
        
        # Get source document information from retrieved chunks
        for chunk in top_chunks:  # Use top_chunks instead of all chunks
            doc_id = chunk.get('source_doc_id')
            chunk_id = chunk.get('chunk_id')
            
            if doc_id and doc_id not in seen_doc_ids:
                chunk_text = chunk.get('chunk_text', '')
                title = chunk.get('title', '')
                
                cited_sources.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "title": title,
                    "text": chunk_text,  # Save full text content
                    "text_preview": chunk_text[:150] + "..." if chunk_text else "No text preview available"
                })
                seen_doc_ids.add(doc_id)
                
        # Get additional source document information from retrieved entities
        for entity in entities:
            doc_id = entity.get('source_doc_id')
            chunk_id = entity.get('source_chunk_id')
            
            if doc_id and doc_id not in seen_doc_ids:
                entity_text = entity.get('entity_text', '')
                entity_desc = entity.get('entity_description', '')
                
                cited_sources.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "title": f"Entity: {entity_text}",
                    "text": entity_desc if entity_desc else entity_text,
                    "text_preview": (entity_desc or entity_text)+ "..."
                })
                seen_doc_ids.add(doc_id)
        
        # Update timing stats
        self.timing_stats["generation_time"] += time.time() - start_time
                
        return llm_answer, cited_sources, llm_raw_output
    
    def query(self, query_text: str, top_k_final_traces: int = 5) -> Tuple[str, List[Dict]]:
        """
        Main method to query the EcphoryRAG system.
        
        Args:
            query_text: The query text
            top_k_final_traces: Number of top traces to use for generation
            
        Returns:
            Tuple of (generated response, list of source documents)
        """
        # Track total query time
        total_start_time = time.time()
        
        logger.info(f"Processing query: {query_text}")
        
        if self.enable_hybrid_retrieval:
            # Use hybrid retrieval approach
            logger.info("Using hybrid retrieval approach")
            retrieved_chunks = self.retrieve_hybrid_results(query_text, k=top_k_final_traces)
            
            # Convert chunks to the format expected by generate_response
            relevant_traces = {
                "traces": [
                    {
                        "text": chunk.get("text", ""),
                        "metadata": {
                            "source": chunk.get("source", ""),
                            "chunk_id": chunk.get("id", ""),
                            "title": chunk.get("title", "")
                        }
                    }
                    for chunk in retrieved_chunks
                ]
            }
        else:
            # Use entity-based retrieval (default)
            logger.info("Using entity-based retrieval approach")
            relevant_traces = self.retrieve_relevant_traces(
                query_text, 
                top_k_final=top_k_final_traces
            )
        
        # Step 2: Generate response using retrieved traces
        answer, sources = self.generate_response(query_text, relevant_traces, top_k_final_traces)
        
        # Additional logic for trace management could go here
        # e.g., decay of STM activations, learning from the query, etc.
        
        # Update total timing
        self.timing_stats["total_time"] += time.time() - total_start_time
        
        logger.info(f"Query processing complete. Found {len(sources)} source documents.")
        return answer, sources
    
    def save(self) -> None:
        """
        Save all components to their predefined paths within the workspace.
        This includes:
        - Entity trace store (FAISS index and metadata)
        - Chunk store (FAISS index and metadata)
        - Workspace manifest (document indexing information)
        - Knowledge graph (if implemented)
        """
        try:
            # Save entity trace store
            logger.info(f"Saving entity trace store to {self.entity_traces_idx_file}")
            self.trace_store.save_store(
                self.entity_traces_idx_file,
                self.entity_traces_meta_file
            )
            
            # Save chunk store
            logger.info(f"Saving chunk store to {self.chunks_idx_file}")
            self.chunk_store.save_store(
                self.chunks_idx_file,
                self.chunks_meta_file
            )
            
            # Save workspace manifest
            logger.info(f"Saving workspace manifest to {self.manifest_file}")
            with open(self.manifest_file, 'w') as f:
                json.dump(self.indexed_doc_manifest, f, indent=2)
            
            # Save knowledge graph
            logger.info(f"Saving knowledge graph to {self.graph_file}")
            nx.write_graphml(self.graph, self.graph_file)
            
            # Save knowledge graph JSON
            self.save_knowledge_graph_json()
            
            logger.info("Successfully saved all workspace components")
            
        except Exception as e:
            logger.error(f"Error saving workspace data: {str(e)}")
            raise

    def save_knowledge_graph_json(self):
        """Save knowledge graph as JSON format"""
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        # Add nodes
        for node_id, node_data in self.graph.nodes(data=True):
            node = {
                "id": node_id,
                "type": node_data.get("type", "unknown"),
                "text": node_data.get("text", ""),
                "metadata": {k: v for k, v in node_data.items() if k not in ["type", "text"]}
            }
            graph_data["nodes"].append(node)
            
        # Add edges
        for source, target, edge_data in self.graph.edges(data=True):
            edge = {
                "source": source,
                "target": target,
                "type": edge_data.get("type", "unknown"),
                "metadata": {k: v for k, v in edge_data.items() if k != "type"}
            }
            graph_data["edges"].append(edge)
            
        # Save JSON file
        with open(self.workspace_path / "knowledge_graph.json", "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

    def reset_trace_store(self) -> None:
        """
        Reset the trace store and related components to their initial state.
        This is useful when processing new documents or contexts that should not
        be influenced by previously indexed data.
        """
        logger.info("Resetting trace store and related components")
        
        # Reset trace store
        self.trace_store = FAISSMemoryTraceStore(self.embedding_dimension)
        
        # Reset chunk store
        self.chunk_store = FAISSChunkStore(self.embedding_dimension)
        
        # Reset knowledge graph
        self.graph = nx.DiGraph()
        
        # Reset indexed document manifest
        self.indexed_doc_manifest = {}
        
        # Reset short-term memory
        self.short_term_memory.clear()
        
        logger.info("Successfully reset all components")

    def retrieve_hybrid_results(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve results using hybrid approach."""
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            print(f"\nDEBUG: Starting hybrid retrieval for query: '{query}'")
            print(f"DEBUG: Query embedding shape: {query_embedding.shape}")
            
            # Step 1: Entity Retrieval
            print("\nDEBUG: Step 1 - Entity Retrieval")
            retrieved_entities_metadata = self.trace_store.search_traces(
                query_embedding, k=k
            )
            print(f"DEBUG: Retrieved {len(retrieved_entities_metadata)} entities")
            if retrieved_entities_metadata:
                print(f"DEBUG: First entity metadata: {retrieved_entities_metadata[0]}")
            
            # Step 2a: Direct Chunk Retrieval
            print("\nDEBUG: Step 2a - Direct Chunk Retrieval")
            retrieved_chunks = self.chunk_store.search_chunks(
                query_embedding, k=k
            )
            print(f"DEBUG: Retrieved {len(retrieved_chunks)} chunks directly")
            if retrieved_chunks:
                print(f"DEBUG: First chunk: {retrieved_chunks[0]}")
            
            # Step 2b: Chunk Retrieval via Entities
            print("\nDEBUG: Step 2b - Chunk Retrieval via Entities")
            entity_chunks = []
            if retrieved_entities_metadata:
                for entity_metadata in retrieved_entities_metadata:
                    chunk_id = entity_metadata.get("source_chunk_id")
                    if chunk_id:
                        chunk = self.chunk_store.get_chunk_by_id(chunk_id)
                        if chunk:
                            entity_chunks.append(chunk)
                            print(f"DEBUG: Found chunk {chunk_id} for entity")
                        else:
                            print(f"DEBUG: No chunk found for ID {chunk_id}")
            
            print(f"DEBUG: Retrieved {len(entity_chunks)} chunks via entities")
            
            # Combine and deduplicate results
            all_chunks = list(set(retrieved_chunks + entity_chunks))
            print(f"\nDEBUG: Final combined unique chunks: {len(all_chunks)}")
            
            return all_chunks
            
        except Exception as e:
            print(f"ERROR in retrieve_hybrid_results: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize the EcphoryRAG system
    rag = EcphoryRAG(
        workspace_path="workspace",
        embedding_model_name="bge-small-en",
        extraction_llm_model="phi3:mini",
        generation_llm_model="phi3:mini",
        enable_chunking=True,  # Default: process documents as a whole
        chunk_size=512,        # Default chunk size for when chunking is enabled
        chunk_overlap=128       # Default chunk overlap for when chunking is enabled
    )
    
    # Example documents for indexing
    example_docs = [
        {
            'id': 'doc1',
            'text': """
            Python is a high-level, interpreted programming language known for its readability.
            It was created by Guido van Rossum and released in 1991.
            Python's design philosophy emphasizes code readability with its notable use of whitespace.
            """
        },
        {
            'id': 'doc2',
            'text': """
            PyTorch is an open source machine learning framework developed by Facebook's AI Research lab.
            It provides a flexible platform for deep learning and artificial intelligence research.
            TensorFlow is another popular machine learning framework, developed by Google.
            """
        }
    ]
    
    # Index the documents with hybrid retrieval
    num_entities, num_chunks = rag.index_documents(example_docs)
    print(f"Indexed {num_entities} entities and {num_chunks} chunks")
    
    # Example query
    query = "When was Python created and by whom?"
    
    # Get answer
    answer, sources = rag.query(query)
    
    print(f"Query: {query}")
    print(f"Answer: {answer}")
    
    # Save the system state with both stores
    rag.save() 