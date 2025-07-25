# faiss_store.py

import faiss
import numpy as np
import json
import uuid
from typing import List, Dict, Optional, Callable, ClassVar, Any, Tuple, Union
import os
from datetime import datetime

class FAISSMemoryTraceStore:
    """
    A class to manage storage and retrieval of entity 'memory traces' using FAISS.
    
    Memory traces represent entities extracted from documents, and their embeddings
    enable semantic search. Each trace has associated metadata including its 
    text content, unique ID, source document, status, and access metrics.
    """
    
    def __init__(self, embedding_dimension: int):
        """
        Initialize an empty FAISS memory trace store.
        
        Args:
            embedding_dimension: Dimension of the embeddings (e.g., 1024 for bge-m3)
        """
        # Create a FAISS index that allows for custom IDs
        # We use IndexIDMap2 wrapped around IndexFlatL2 for L2 distance search with custom IDs
        self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(embedding_dimension))
        self.embedding_dimension = embedding_dimension
        
        # Dictionary to store metadata for each trace, keyed by trace_id
        self.trace_metadata = {}
        
        # Keep track of the next available internal ID
        self.next_id = 0
        
        # Map between trace_id (string) and internal FAISS ID (int)
        self.trace_id_to_faiss_id = {}
        self.faiss_id_to_trace_id = {}
        
        # print(f"DEBUG: Initialized FAISSMemoryTraceStore with dimension {embedding_dimension}")
    
    def add_traces(self, trace_embeddings: np.ndarray, trace_metadata_list: List[Dict]) -> List[str]:
        """
        Add memory traces to the FAISS index and store their metadata.
        
        Args:
            trace_embeddings: NumPy array of shape (N, embedding_dimension) containing trace embeddings
            trace_metadata_list: List of N dictionaries, each containing metadata for a trace:
                - entity_text: The string of the entity itself (e.g., "Albert Einstein")
                - entity_type: Type of entity (e.g., "PERSON", "CONCEPT")
                - entity_description: LLM-generated description of the entity
                - entity_importance: Score or category indicating importance
                - source_chunk_id: ID of the text chunk this entity was extracted from
                - source_doc_id: ID of the document from which it was extracted
                - id: Unique ID for this trace (if not provided, will be generated)
                - status: Initial status (e.g., "dormant")
                - Optional: created_at, last_accessed_at, access_count
        
        Returns:
            List of trace IDs that were added
        
        Raises:
            ValueError: If dimensions don't match or metadata is missing required fields
        """
        if trace_embeddings.shape[0] != len(trace_metadata_list):
            raise ValueError(f"Number of embeddings ({trace_embeddings.shape[0]}) must match number of metadata entries ({len(trace_metadata_list)})")
        
        if trace_embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {trace_embeddings.shape[1]}")
        
        # Create FAISS IDs for the new traces
        n_traces = trace_embeddings.shape[0]
        faiss_ids = np.arange(self.next_id, self.next_id + n_traces, dtype=np.int64)
        self.next_id += n_traces
        
        # Process metadata and update mappings
        added_trace_ids = []
        
        for i, metadata in enumerate(trace_metadata_list):
            # Validate and prepare metadata
            if 'entity_text' not in metadata:
                # For backward compatibility, check if 'text' exists and use it
                if 'text' in metadata:
                    metadata['entity_text'] = metadata['text']
                else:
                    raise ValueError(f"Metadata entry {i} missing 'entity_text' field")
            
            if 'source_doc_id' not in metadata:
                raise ValueError(f"Metadata entry {i} missing 'source_doc_id' field")
            
            # Set default values for new fields if not provided
            metadata.setdefault('entity_type', 'MISC')
            metadata.setdefault('entity_description', '')
            metadata.setdefault('entity_importance', 3)
            metadata.setdefault('source_chunk_id', '')
            
            # Generate or use provided ID
            trace_id = metadata.get('id', uuid.uuid4().hex)
            metadata['id'] = trace_id
            
            # Set default values for optional fields
            now = datetime.now().isoformat()
            metadata.setdefault('status', 'dormant')
            metadata.setdefault('created_at', now)
            metadata.setdefault('last_accessed_at', now)
            metadata.setdefault('access_count', 0)
            
            # Store metadata and update mappings
            self.trace_metadata[trace_id] = metadata
            faiss_id = int(faiss_ids[i])
            self.trace_id_to_faiss_id[trace_id] = faiss_id
            self.faiss_id_to_trace_id[faiss_id] = trace_id
            
            added_trace_ids.append(trace_id)
        
        # Add embeddings to FAISS index with their IDs
        self.index.add_with_ids(trace_embeddings, faiss_ids)
        
        # print(f"\nDEBUG: Adding {len(trace_metadata_list)} traces to FAISS store")
        # print(f"DEBUG: First trace: {trace_metadata_list[0]}")
        # print(f"DEBUG: First embedding shape: {trace_embeddings[0].shape}")
        
        # print(f"DEBUG: FAISS index total vectors after adding: {self.index.ntotal}")
        
        # print(f"DEBUG: Stored {len(trace_metadata_list)} trace metadata entries")
        
        return added_trace_ids
    
    def search_traces(self, query_embedding: np.ndarray, k: int = 10, 
                      allowed_statuses: Optional[List[str]] = None) -> List[Dict]:
        """
        Search for memory traces similar to the query embedding.
        
        Args:
            query_embedding: NumPy array of shape (1, embedding_dimension) containing the query embedding
            k: Number of nearest traces to retrieve
            allowed_statuses: Optional list of statuses to filter results by (e.g., ['dormant', 'active_in_stm'])
                              If None, all statuses are allowed
        
        Returns:
            List of trace metadata dictionaries for the relevant traces, sorted by similarity
        
        Raises:
            ValueError: If query_embedding dimensions are incorrect
        """
        if query_embedding.shape[1] != self.embedding_dimension:
            raise ValueError(f"Query embedding dimension mismatch: expected {self.embedding_dimension}, got {query_embedding.shape[1]}")
        
        # If we're filtering by status, we need to retrieve more results initially
        # since some might be filtered out
        initial_k = k * 3 if allowed_statuses else k
        initial_k = min(initial_k, self.index.ntotal)  # Can't retrieve more than exist
        
        if initial_k == 0:  # Empty index
            print("WARNING: FAISS index is empty in search_traces")
            return []
        
        # Search the FAISS index
        distances, faiss_ids = self.index.search(query_embedding, initial_k)
        
        # print(f"\nDEBUG: Searching traces with k={k}")
        # print(f"DEBUG: Query embedding shape: {query_embedding.shape}")
        # print(f"DEBUG: FAISS index total vectors: {self.index.ntotal}")
        
        # print(f"DEBUG: FAISS search returned distances: {distances[0]}")
        # print(f"DEBUG: FAISS search returned indices: {faiss_ids[0]}")
        
        # Convert to list of metadata dictionaries
        results = []
        for i in range(len(faiss_ids[0])):  # faiss_ids is a 2D array: [[id1, id2, ...]]
            faiss_id = int(faiss_ids[0][i])
            if faiss_id == -1:  # FAISS returns -1 for padded results when fewer than k exist
                continue
                
            trace_id = self.faiss_id_to_trace_id.get(faiss_id)
            if not trace_id:
                continue  # Skip if mapping is missing
                
            metadata = self.trace_metadata.get(trace_id)
            if not metadata:
                continue  # Skip if metadata is missing
            
            # Filter by status if needed
            if allowed_statuses and metadata['status'] not in allowed_statuses:
                continue
            
            # Add distance information to metadata (useful for ranking)
            result = metadata.copy()
            result['distance'] = float(distances[0][i])
            
            # Update access metrics
            self._update_access_metrics(trace_id)
            
            results.append(result)
            
            # Stop once we have enough results after filtering
            if len(results) >= k:
                break
        
        # print(f"DEBUG: Retrieved {len(results)} trace results")
        return results
    
    def get_trace_by_id(self, trace_id: str) -> Optional[Dict]:
        """
        Retrieve a memory trace by its unique ID.
        
        Args:
            trace_id: The unique ID of the trace to retrieve
        
        Returns:
            The trace metadata dictionary, or None if not found
        """
        metadata = self.trace_metadata.get(trace_id)
        if metadata:
            # Update access metrics
            self._update_access_metrics(trace_id)
            return metadata.copy()  # Return a copy to prevent accidental modification
        return None
    
    def update_trace_status(self, trace_id: str, new_status: str) -> bool:
        """
        Update the status of a specific memory trace.
        
        Args:
            trace_id: The unique ID of the trace to update
            new_status: The new status value (e.g., 'dormant', 'active_in_stm')
        
        Returns:
            True if the update was successful, False if the trace was not found
        """
        if trace_id not in self.trace_metadata:
            return False
        
        self.trace_metadata[trace_id]['status'] = new_status
        return True
    
    def _update_access_metrics(self, trace_id: str) -> None:
        """
        Update access metrics for a trace (last_accessed_at and access_count).
        
        Args:
            trace_id: The unique ID of the trace to update
        """
        if trace_id in self.trace_metadata:
            self.trace_metadata[trace_id]['last_accessed_at'] = datetime.now().isoformat()
            self.trace_metadata[trace_id]['access_count'] += 1
    
    def save_store(self, index_file_path: str, metadata_file_path: str) -> None:
        """
        Save the FAISS index and trace metadata to disk.
        
        Args:
            index_file_path: Path to save the FAISS index
            metadata_file_path: Path to save the trace metadata
        """
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(index_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_file_path)
        
        # Save metadata and mappings
        save_data = {
            'trace_metadata': self.trace_metadata,
            'trace_id_to_faiss_id': self.trace_id_to_faiss_id,
            'faiss_id_to_trace_id': {str(k): v for k, v in self.faiss_id_to_trace_id.items()},  # Convert int keys to strings for JSON
            'next_id': self.next_id,
            'embedding_dimension': self.embedding_dimension
        }
        
        with open(metadata_file_path, 'w') as f:
            json.dump(save_data, f)
    
    @classmethod
    def load_store(cls, index_file_path: str, metadata_file_path: str, 
                  embedding_dimension: int) -> 'FAISSMemoryTraceStore':
        """
        Load a FAISSMemoryTraceStore from disk.
        
        Args:
            index_file_path: Path to the saved FAISS index
            metadata_file_path: Path to the saved trace metadata
            embedding_dimension: Dimension of the embeddings
        
        Returns:
            A loaded FAISSMemoryTraceStore instance
        
        Raises:
            FileNotFoundError: If index or metadata files don't exist
            ValueError: If embedding dimensions don't match
        """
        # Create a new instance
        store = cls(embedding_dimension)
        
        # Load FAISS index
        store.index = faiss.read_index(index_file_path)
        
        # Check embedding dimension
        if store.index.d != embedding_dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {embedding_dimension}, got {store.index.d}")
        
        # Load metadata and mappings
        with open(metadata_file_path, 'r') as f:
            save_data = json.load(f)
        
        store.trace_metadata = save_data['trace_metadata']
        store.trace_id_to_faiss_id = save_data['trace_id_to_faiss_id']
        # Convert string keys back to integers for faiss_id_to_trace_id
        store.faiss_id_to_trace_id = {int(k): v for k, v in save_data['faiss_id_to_trace_id'].items()}
        store.next_id = save_data['next_id']
        
        return store

class FAISSChunkStore:
    """
    A class to manage storage and retrieval of text chunks using FAISS.
    
    This store maintains embeddings of original text chunks for hybrid retrieval,
    complementing the entity-based retrieval in FAISSMemoryTraceStore.
    Each chunk has associated metadata including its text content, unique ID,
    source document, and creation timestamp.
    """
    
    def __init__(self, embedding_dimension: int = 1024):
        """
        Initialize an empty FAISS chunk store.
        
        Args:
            embedding_dimension: Dimension of the embeddings (e.g., 1024 for bge-m3)
        """
        self.embedding_dimension = embedding_dimension
        # Use IndexIDMap2 wrapped around IndexFlatL2 for L2 distance search with custom IDs
        self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(embedding_dimension))
        self.chunk_metadata = {}  # Add metadata storage
        self.chunk_id_to_faiss_id = {}
        self.faiss_id_to_chunk_id = {}
        self.next_id = 0  # Add next_id counter
        # print(f"DEBUG: Initialized FAISSChunkStore with dimension {embedding_dimension}")
    
    def add_chunks(self, chunk_embeddings: np.ndarray, chunk_metadata_list: List[Dict]) -> List[str]:
        """
        Add text chunks to the FAISS index and store their metadata.
        
        Args:
            chunk_embeddings: NumPy array of shape (N, embedding_dimension) containing chunk embeddings
            chunk_metadata_list: List of N dictionaries, each containing metadata for a chunk:
                - chunk_text: The original text of the chunk
                - source_doc_id: ID of the document it belongs to
                - chunk_id: Unique ID for this chunk (if not provided, will be generated)
                - created_at: Timestamp (if not provided, current time will be used)
        
        Returns:
            List of chunk IDs that were added
        
        Raises:
            ValueError: If dimensions don't match or metadata is missing required fields
        """
        if len(chunk_embeddings) != len(chunk_metadata_list):
            raise ValueError("Number of embeddings must match number of metadata entries")
            
        # print(f"\nDEBUG: Adding {len(chunk_metadata_list)} chunks to FAISS store")
        # print(f"DEBUG: First chunk: {chunk_metadata_list[0]}")
        # print(f"DEBUG: First embedding shape: {chunk_embeddings[0].shape}")
        
        # Generate FAISS IDs
        faiss_ids = np.arange(self.next_id, self.next_id + len(chunk_embeddings), dtype=np.int64)
        self.next_id += len(chunk_embeddings)
        
        # Add embeddings to FAISS index with their IDs
        self.index.add_with_ids(chunk_embeddings, faiss_ids)
        # print(f"DEBUG: FAISS index total vectors after adding: {self.index.ntotal}")
        
        # Store metadata and maintain ID mappings
        added_chunk_ids = []
        for faiss_id, chunk_metadata in zip(faiss_ids, chunk_metadata_list):
            # Generate or use provided chunk ID
            chunk_id = chunk_metadata.get('chunk_id', str(uuid.uuid4()))
            
            # Set default values for optional fields
            now = datetime.now().isoformat()
            chunk_metadata.setdefault('created_at', now)
            
            # Store metadata and update mappings
            self.chunk_metadata[chunk_id] = chunk_metadata
            self.chunk_id_to_faiss_id[chunk_id] = int(faiss_id)
            self.faiss_id_to_chunk_id[int(faiss_id)] = chunk_id
            added_chunk_ids.append(chunk_id)
            
        # print(f"DEBUG: Stored {len(chunk_metadata_list)} chunk metadata entries")
        return added_chunk_ids
    
    def search_chunks(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for text chunks similar to the query embedding.
        
        Args:
            query_embedding: NumPy array of shape (1, embedding_dimension) containing the query embedding
            k: Number of nearest chunks to retrieve
        
        Returns:
            List of chunk metadata dictionaries for the relevant chunks, sorted by similarity
        
        Raises:
            ValueError: If query_embedding dimensions are incorrect
        """
        initial_k = min(k, self.index.ntotal)
        if initial_k == 0:  # Empty index
            print("WARNING: FAISS index is empty in search_chunks")
            return []
            
        # print(f"\nDEBUG: Searching chunks with k={k}")
        # print(f"DEBUG: Query embedding shape: {query_embedding.shape}")
        # print(f"DEBUG: FAISS index total vectors: {self.index.ntotal}")
        
        # Search the FAISS index
        distances, faiss_ids = self.index.search(query_embedding, initial_k)
        # print(f"DEBUG: FAISS search returned distances: {distances[0]}")
        # print(f"DEBUG: FAISS search returned indices: {faiss_ids[0]}")
        
        # Convert to list of metadata dictionaries
        results = []
        for i, faiss_id in enumerate(faiss_ids[0]):
            if faiss_id in self.faiss_id_to_chunk_id:
                chunk_id = self.faiss_id_to_chunk_id[faiss_id]
                metadata = self.chunk_metadata.get(chunk_id, {})
                result = metadata.copy()
                result['chunk_id'] = chunk_id
                result['distance'] = float(distances[0][i])
                results.append(result)
                
        # print(f"DEBUG: Retrieved {len(results)} chunk results")
        return results
        
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve a text chunk by its unique ID.
        
        Args:
            chunk_id: The unique ID of the chunk to retrieve
        
        Returns:
            The chunk metadata dictionary, or None if not found
        """
        if chunk_id not in self.chunk_metadata:
            print(f"WARNING: Chunk ID {chunk_id} not found in metadata")
            return None

        metadata = self.chunk_metadata[chunk_id].copy()
        metadata['chunk_id'] = chunk_id
        return metadata

    
    def save_store(self, index_file_path: str, metadata_file_path: str) -> None:
        """
        Save the FAISS index and chunk metadata to disk.
        
        Args:
            index_file_path: Path to save the FAISS index
            metadata_file_path: Path to save the chunk metadata
        """
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(index_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_file_path)
        
        # Save metadata and mappings
        save_data = {
            'chunk_metadata': self.chunk_metadata,
            'chunk_id_to_faiss_id': self.chunk_id_to_faiss_id,
            'faiss_id_to_chunk_id': {str(k): v for k, v in self.faiss_id_to_chunk_id.items()},  # Convert int keys to strings for JSON
            'next_id': self.next_id,
            'embedding_dimension': self.embedding_dimension
        }
        
        with open(metadata_file_path, 'w') as f:
            json.dump(save_data, f)
    
    @classmethod
    def load_store(cls, index_file_path: str, metadata_file_path: str, 
                  embedding_dimension: int) -> 'FAISSChunkStore':
        """
        Load a FAISSChunkStore from disk.
        
        Args:
            index_file_path: Path to the saved FAISS index
            metadata_file_path: Path to the saved chunk metadata
            embedding_dimension: Dimension of the embeddings
        
        Returns:
            A loaded FAISSChunkStore instance
        
        Raises:
            FileNotFoundError: If index or metadata files don't exist
            ValueError: If embedding dimensions don't match
        """
        # Create a new instance
        store = cls(embedding_dimension)
        
        # Load FAISS index
        store.index = faiss.read_index(index_file_path)
        
        # Check embedding dimension
        if store.index.d != embedding_dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {embedding_dimension}, got {store.index.d}")
        
        # Load metadata and mappings
        with open(metadata_file_path, 'r') as f:
            save_data = json.load(f)
        
        store.chunk_metadata = save_data['chunk_metadata']
        store.chunk_id_to_faiss_id = save_data['chunk_id_to_faiss_id']
        # Convert string keys back to integers for faiss_id_to_chunk_id
        store.faiss_id_to_chunk_id = {int(k): v for k, v in save_data['faiss_id_to_chunk_id'].items()}
        store.next_id = save_data['next_id']
        
        return store


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Example 1: Using the entity trace store with enhanced metadata
    print("\n=== ENTITY TRACE STORE EXAMPLE ===")
    embedding_dim = 1024
    trace_store = FAISSMemoryTraceStore(embedding_dim)
    
    # Generate some example embeddings and metadata
    num_traces = 3
    trace_embeddings = np.random.random((num_traces, embedding_dim)).astype(np.float32)
    
    trace_metadata = [
        {
            'entity_text': "Albert Einstein",
            'entity_type': "PERSON",
            'entity_description': "German-born theoretical physicist who developed the theory of relativity.",
            'entity_importance': 5,
            'source_chunk_id': f"chunk_{i}",
            'source_doc_id': "doc_physics",
            'status': 'dormant'
        }
        for i in range(num_traces)
    ]
    
    # Add traces to the store
    trace_ids = trace_store.add_traces(trace_embeddings, trace_metadata)
    print(f"Added {len(trace_ids)} traces with IDs: {trace_ids}")
    
    # Retrieve a trace by ID
    retrieved_trace = trace_store.get_trace_by_id(trace_ids[0])
    print(f"\nRetrieved trace by ID: {retrieved_trace['entity_text']} ({retrieved_trace['entity_type']})")
    print(f"Description: {retrieved_trace['entity_description']}")
    print(f"Importance: {retrieved_trace['entity_importance']}")
    
    # Search for similar traces
    query_embedding = np.random.random((1, embedding_dim)).astype(np.float32)
    results = trace_store.search_traces(query_embedding, k=2)
    print(f"\nSearch results: {len(results)} traces found")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['entity_text']} ({result['entity_type']}) - Distance: {result['distance']:.4f}")
    
    # Example 2: Using the chunk store
    print("\n=== CHUNK STORE EXAMPLE ===")
    chunk_store = FAISSChunkStore(embedding_dim)
    
    # Generate some example embeddings and metadata
    num_chunks = 3
    chunk_embeddings = np.random.random((num_chunks, embedding_dim)).astype(np.float32)
    
    chunk_metadata = [
        {
            'chunk_text': f"This is chunk {i} containing information about physics and relativity.",
            'source_doc_id': "doc_physics"
        }
        for i in range(num_chunks)
    ]
    
    # Add chunks to the store
    chunk_ids = chunk_store.add_chunks(chunk_embeddings, chunk_metadata)
    print(f"Added {len(chunk_ids)} chunks with IDs: {chunk_ids}")
    
    # Retrieve a chunk by ID
    retrieved_chunk = chunk_store.get_chunk_by_id(chunk_ids[0])
    print(f"\nRetrieved chunk by ID: {retrieved_chunk['chunk_text'][:50]}...")
    
    # Search for similar chunks
    results = chunk_store.search_chunks(query_embedding, k=2)
    print(f"\nSearch results: {len(results)} chunks found")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['chunk_text'][:50]}... (Distance: {result['distance']:.4f})")
    
    # Save and load the stores
    print("\n=== SAVING AND LOADING STORES ===")
    trace_store.save_store("example_trace_index.faiss", "example_trace_metadata.json")
    chunk_store.save_store("example_chunk_index.faiss", "example_chunk_metadata.json")
    
    loaded_trace_store = FAISSMemoryTraceStore.load_store(
        "example_trace_index.faiss", 
        "example_trace_metadata.json", 
        embedding_dim
    )
    
    loaded_chunk_store = FAISSChunkStore.load_store(
        "example_chunk_index.faiss", 
        "example_chunk_metadata.json", 
        embedding_dim
    )
    
    print(f"Loaded trace store has {loaded_trace_store.index.ntotal} traces")
    print(f"Loaded chunk store has {loaded_chunk_store.index.ntotal} chunks") 