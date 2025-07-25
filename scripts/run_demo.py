#!/usr/bin/env python3
# run_demo.py

import os
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from ecphoryrag.src.ecphory_rag import EcphoryRAG

# Define workspace directory
WORKSPACE_DIR = "./my_ecphory_test_workspace"

def main():
    # Sample documents for demonstration
    sample_docs = [
        {
            'id': 'doc1',
            'text': """
            Albert Einstein was born in Germany in 1879. He developed the theory of relativity,
            one of the two pillars of modern physics. His work is also known for its influence
            on the philosophy of science. Einstein is best known for developing the theory of
            relativity, but he also made important contributions to the development of the theory
            of quantum mechanics.
            """
        },
        {
            'id': 'doc2',
            'text': """
            The theory of relativity usually encompasses two interrelated theories by Albert Einstein:
            special relativity and general relativity. Special relativity applies to all physical
            phenomena in the absence of gravity. General relativity explains the law of gravitation
            and its relation to other forces of nature.
            """
        },
        {
            'id': 'doc3',
            'text': """
            Quantum mechanics is a fundamental theory in physics that provides a description of the
            physical properties of nature at the scale of atoms and subatomic particles. It is the
            foundation of all quantum physics including quantum chemistry, quantum field theory,
            quantum technology, and quantum information science.
            """
        }
    ]

    # Initialize EcphoryRAG with workspace
    print(f"Initializing EcphoryRAG with workspace: {WORKSPACE_DIR}")
    rag_instance = EcphoryRAG(
        workspace_path=WORKSPACE_DIR,
        embedding_model_name="bge-m3",
        extraction_llm_model="phi4",
        generation_llm_model="phi4",
        enable_chunking=True,
        chunk_size=256,
        chunk_overlap=32
    )

    # Check if workspace needs indexing
    if not rag_instance.indexed_doc_manifest or not os.path.exists(rag_instance.entity_traces_idx_file):
        print(f"\nWorkspace at {WORKSPACE_DIR} seems empty or incomplete.")
        print("Indexing sample documents...")
        num_entities, num_chunks = rag_instance.index_documents(
            sample_docs,
            force_reindex_all=True
        )
        print(f"Indexed {num_entities} entities and {num_chunks} chunks")
    else:
        print(f"\nLoaded existing workspace from {WORKSPACE_DIR}")
        print(f"Found {len(rag_instance.indexed_doc_manifest)} indexed documents")
        print(f"Knowledge graph has {rag_instance.graph.number_of_nodes()} nodes and {rag_instance.graph.number_of_edges()} edges")

    # Example queries
    example_queries = [
        "What did Einstein work on?",
        "What is the theory of relativity?",
        "What is quantum mechanics?"
    ]

    # Process each query
    for query_text in example_queries:
        print("\n" + "="*80)
        print(f"Query: {query_text}")
        
        # Get answer and sources
        answer, sources = rag_instance.query(query_text)
        
        # Print answer
        print("\nAnswer:")
        print(answer)
        
        # Print sources
        if sources:
            print("\nSources:")
            for src in sources:
                print(f"\n  - Document ID: {src['doc_id']}")
                print(f"    Chunk ID: {src['chunk_id']}")
                print(f"    Preview: {src['text_preview']}")
        else:
            print("\nNo sources found")

    # Save the final state
    print("\nSaving workspace state...")
    rag_instance.save()
    print("Done!")

if __name__ == "__main__":
    main() 