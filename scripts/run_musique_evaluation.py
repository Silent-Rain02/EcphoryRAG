#!/usr/bin/env python
"""
MuSiQue Evaluation Script for EcphoryRAG

This script demonstrates how to use the Evaluator class to evaluate 
EcphoryRAG performance on the MuSiQue multi-step question answering dataset,
using the processed dataset format.
"""

import argparse
import logging
import os
import sys
import time
import json
from pathlib import Path
import uuid
from typing import Union, List, Tuple, Dict, Optional
import numpy as np


current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ecphoryrag.src.ecphory_rag import EcphoryRAG
from ecphoryrag.evaluation.processed_datasets import ProcessedMusiqueDatasetLoader
from ecphoryrag.evaluation.evaluator import Evaluator
from ecphoryrag.evaluation.utils import print_results_table



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def create_test_subset(data_path: str, subset_size: int = 10) -> str:
    """
    Create a small test subset from the full dataset.
    
    Args:
        data_path: Path to the processed dataset
        subset_size: Subset size
        
    Returns:
        Path to the subset file
    """
    # Create subset directory
    subset_dir = os.path.join(os.path.dirname(data_path), "test_subset")
    os.makedirs(subset_dir, exist_ok=True)
    
    # Subset file path
    subset_path = os.path.join(subset_dir, "musique_subset.jsonl")
    
    # If subset already exists and is the correct size, return it
    if os.path.exists(subset_path):
        with open(subset_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        if line_count == subset_size:
            logger.info(f"Test subset already exists at {subset_path} with {subset_size} samples")
            return subset_path
        else:
            logger.info(f"Existing subset has {line_count} samples, creating new subset with {subset_size} samples")
    
    # Read processed dataset
    logger.info(f"Creating test subset of size {subset_size} from {data_path}")
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= subset_size:
                break
            samples.append(json.loads(line))
    
    # Save subset
    with open(subset_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"Created test subset at {subset_path} with {len(samples)} samples")
    return subset_path


def get_processed_musique_data(processed_dir: str = "data/processed_musique") -> str:
    """
    Get the path to the processed MuSiQue dataset.
    
    Args:
        processed_dir: Directory containing the processed dataset
    
    Returns:
        Path to the processed development dataset file
    """
    # Get the absolute path of the current script
    current_file = os.path.abspath(__file__)

    # Calculate the project root directory path
    project_root = os.path.dirname(os.path.dirname(current_file))

    # Define path to processed data
    dev_path = os.path.join(project_root, processed_dir, "dev_processed.jsonl")
    
    # Check if file exists
    if os.path.exists(dev_path):
        logger.info(f"Processed MuSiQue dataset found at {dev_path}")
        return dev_path
    else:
        logger.error(f"Processed MuSiQue dataset not found at {dev_path}")
        logger.info(
            "Please process the raw MuSiQue dataset first using the process_musique_data.py script:"
            "\npython scripts/process_musique_data.py --raw_data_file data/musique/musique_ans_v1.0_dev.jsonl "
            "--output_dir data/processed_musique --split_name dev"
        )
        raise FileNotFoundError(f"Processed dataset not found: {dev_path}")
        
    return dev_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate EcphoryRAG on processed MuSiQue dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset options
    parser.add_argument("--data-path", type=str, 
                       help="Path to processed MuSiQue dataset file. If not provided, will use default location.")
    parser.add_argument("--processed-dir", type=str, default="data/processed_musique",
                       help="Directory containing processed dataset")
    parser.add_argument("--test-subset", default=True, action="store_true",
                       help="Use a small test subset instead of the full dataset")
    parser.add_argument("--subset-size", type=int, default=500,
                       help="Size of the test subset when --test-subset is used")
    
    # Workspace options
    parser.add_argument("--workspace-dir", type=str, default="musique_evaluation_workspace",
                       help="Directory for storing EcphoryRAG workspace data")
    parser.add_argument("--force-reindex", action="store_true",
                       help="Force reindexing of all documents")
    parser.add_argument("--skip-index", action="store_true",
                       help="Skip indexing and use existing index for evaluation")
    
    # Model configuration
    parser.add_argument("--embedding-model", type=str, default="bge-m3",
                       help="Embedding model name")
    parser.add_argument("--extraction-model", type=str, default="phi4",
                       help="Entity extraction LLM model")
    parser.add_argument("--generation-model", type=str, default="phi4",
                       help="Answer generation LLM model")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434",
                       help="Ollama API host")
    
    # Retrieval parameters
    parser.add_argument("--top-k-initial", type=int, default=10,
                       help="Top-k value for initial trace retrieval")
    parser.add_argument("--top-k-final-values", type=str, default="80",
                       help="Comma-separated list of top-k values for final trace selection")
    parser.add_argument("--retrieval-depth", type=int, default=2,
                       help="Depth for secondary ecphory retrieval")
    
    # Chunking parameters
    parser.add_argument("--enable-chunking", action="store_true", default=False,
                       help="Enable document chunking")
    parser.add_argument("--chunk-size", type=int, default=1200,
                       help="Size of text chunks when chunking is enabled")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                       help="Overlap between consecutive chunks")
    
    # Retrieval parameters
    parser.add_argument("--enable-hybrid-retrieval", action="store_true", default=False,
                       help="Enable hybrid retrieval (combines entity and chunk retrieval)")
    
    # Evaluation options
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to evaluate. Default is all.")
    parser.add_argument("--output-dir", type=str, default="musique500depth2",
                       help="Directory to save evaluation results")
    parser.add_argument("--output-file", type=str, default="musique_evaluation.json",
                       help="Filename for evaluation results")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode for testing individual questions")
    
    return parser.parse_args()


def print_results_table(results: Dict[str, Dict[str, float]], title: Optional[str] = None) -> None:
    """
    Print evaluation results in a formatted table.
    
    Args:
        results: Dictionary of results for each top_k value
        title: Optional title for the table
    """
    try:
        from tabulate import tabulate
    except ImportError:
        logger.warning("tabulate not installed, using simple printing")
        if title:
            print(f"\n{title}")
        for top_k, metrics in results.items():
            print(f"\n{top_k}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        return
    
    # Prepare table data
    headers = ["top_k", "exact_match", "f1_score", "avg_processing_time"]
    if "rouge1" in next(iter(results.values())):
        headers.extend(["rouge1", "rouge2", "rougeL"])
    
    table_data = []
    for top_k, metrics in results.items():
        row = [top_k]
        for metric in headers[1:]:
            value = metrics.get(metric, "N/A")
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(value)
        table_data.append(row)
    
    # Print table
    if title:
        print(f"\n{title}")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def print_token_usage_table(token_usage: Dict[str, Dict[str, int]], title: Optional[str] = None) -> None:
    """
    Print token usage statistics in a formatted table.
    
    Args:
        token_usage: Dictionary of token usage for each top_k value
        title: Optional title for the table
    """
    try:
        from tabulate import tabulate
    except ImportError:
        logger.warning("tabulate not installed, using simple printing")
        if title:
            print(f"\n{title}")
        for top_k, stats in token_usage.items():
            print(f"\n{top_k}:")
            for metric, value in stats.items():
                print(f"  {metric}: {value}")
        return
    
    # Prepare table data
    headers = ["top_k", "embedding_tokens", "completion_tokens", "total_tokens", "avg_tokens_per_query"]
    
    table_data = []
    for top_k, stats in token_usage.items():
        row = [top_k]
        for metric in headers[1:]:
            value = stats.get(metric, "N/A")
            if isinstance(value, float):
                row.append(f"{value:.2f}")
            else:
                row.append(value)
        table_data.append(row)
    
    # Print table
    if title:
        print(f"\n{title}")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def main():
    """Run MuSiQue evaluation on EcphoryRAG using processed dataset."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set data set path
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = get_processed_musique_data(args.processed_dir)
    
    # If using test subset, create subset
    if args.test_subset:
        data_path = create_test_subset(data_path, args.subset_size)
        logger.info(f"Using full MuSiQue dataset at: {data_path}")
    
    # Initialize processed MuSiQue dataset loader
    logger.info(f"Loading processed MuSiQue dataset from {data_path}")
    dataset_loader = ProcessedMusiqueDatasetLoader(data_path)
    
    # Parse top-k values
    top_k_final_values = [int(k) for k in args.top_k_final_values.split(',')]
    
    # Initialize EcphoryRAG with optimized parameters
    logger.info(f"Initializing EcphoryRAG with models: "
               f"{args.embedding_model} (embedding), "
               f"{args.extraction_model} (extraction), "
               f"{args.generation_model} (generation)")
    
    try:
        # Create workspace directory if it doesn't exist
        os.makedirs(args.workspace_dir, exist_ok=True)
        
        # Set workspace directory based on whether using subset
        workspace_dir = args.workspace_dir
        if args.test_subset:
            workspace_dir = os.path.join(args.workspace_dir, "test_subset")
            os.makedirs(workspace_dir, exist_ok=True)
            logger.info(f"Using subset-specific workspace at: {workspace_dir}")
        
        rag_system = EcphoryRAG(
            workspace_path=workspace_dir,
            embedding_model_name=args.embedding_model,
            extraction_llm_model=args.extraction_model,
            generation_llm_model=args.generation_model,
            ollama_host=args.ollama_host,
            enable_chunking=args.enable_chunking,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            enable_hybrid_retrieval=args.enable_hybrid_retrieval
        )
        
        # Customize retrieval parameters
        rag_system.top_k_initial = args.top_k_initial
        rag_system.retrieval_depth = args.retrieval_depth
        
        # Track indexing time and token usage
        indexing_start_time = time.time()
        
        # If not skip indexing mode, then index
        if not args.skip_index:
            # Load dataset
            data = dataset_loader.load_data()
            logger.info(f"Loaded {len(data)} datapoints for indexing")
            
            # Get documents
            documents = dataset_loader.get_documents_for_indexing()

            # Index documents
            logger.info("Indexing documents...")
            num_entities, num_chunks = rag_system.index_documents(
                documents,
                force_reindex_all=args.force_reindex
            )
            logger.info(f"Indexed {num_entities} entities and {num_chunks} chunks")
            
            # Get indexing token usage and timing
            indexing_time = time.time() - indexing_start_time
            indexing_token_usage = rag_system.get_usage_stats()
            
            # Save indexing stats
            indexing_stats = {
                "indexing_time": indexing_time,
                "num_entities": num_entities,
                "num_chunks": num_chunks,
                "token_usage": indexing_token_usage
            }
            with open(os.path.join(args.output_dir, "indexing_stats.json"), "w") as f:
                json.dump(indexing_stats, f, indent=2)
            
            logger.info(f"Indexing completed in {indexing_time:.2f} seconds")
            logger.info(f"Indexing token usage: {indexing_token_usage['token_usage']}")
        
        # If interactive mode, run interactive question answering
        if args.interactive:
            print("\n" + "="*50)
            print("Interactive question answering mode (enter 'q' to exit)")
            print("="*50)
            
            while True:
                question = input("\nPlease enter a question: ").strip()
                if question.lower() == 'q':
                    break
                    
                if not question:
                    continue
                    
                try:
                    # Reset usage stats for this query
                    rag_system.reset_usage_stats()
                    query_start_time = time.time()
                    
                    # Get answer
                    answer, sources = rag_system.query(question)
                    
                    # Get query timing and token usage
                    query_time = time.time() - query_start_time
                    query_stats = rag_system.get_usage_stats()
                    
                    # Print answer
                    print("\nAnswer:")
                    print("-"*30)
                    print(answer)
                    
                    # Print sources
                    if sources:
                        print("\nSources:")
                        print("-"*30)
                        for i, source in enumerate(sources, 1):
                            print(f"{i}. {source}")
                        
                    # Print usage stats
                    print("\nQuery statistics:")
                    print("-"*30)
                    print(f"Query time: {query_time:.2f} seconds")
                    print(f"Embedding tokens: {query_stats['token_usage']['embedding_tokens']}")
                    print(f"Completion tokens: {query_stats['token_usage']['completion_tokens']}")
                    print(f"Total tokens: {query_stats['token_usage']['total_tokens']}")
                                
                except Exception as e:
                    logger.error(f"Error processing question: {e}")
                    continue
                    
            return
        
        # Otherwise run evaluation
        evaluator = Evaluator(
            rag_system=rag_system,
            dataset_loader=dataset_loader,
            output_dir=args.output_dir
        )
        
        # Run evaluation
        start_time = time.time()
        logger.info(f"Starting evaluation on {args.num_samples or 'all'} samples...")
        
        # Reset usage stats for evaluation
        rag_system.reset_usage_stats()
        
        # Add parameter to track token usage and timing
        results = evaluator.run_evaluation(
            num_samples=args.num_samples,
            output_file=args.output_file,
            top_k_values=top_k_final_values,
            skip_indexing=args.skip_index
        )
        
        total_time = time.time() - start_time
        logger.info(f"Evaluation completed in {total_time:.2f} seconds")
        
        # Get final usage stats
        usage_stats = rag_system.get_usage_stats()
        
        # Save usage statistics
        usage_output = {
            "total_evaluation_time": total_time,
            "token_usage": usage_stats["token_usage"],
            "timing": usage_stats["timing"],
            "operations": usage_stats["operations"],
            "average_query_time": usage_stats["timing"]["total_time"] / max(1, usage_stats["operations"]["retrieval_calls"]),
            "average_tokens_per_query": usage_stats["token_usage"]["total_tokens"] / max(1, usage_stats["operations"]["retrieval_calls"])
        }
        
        with open(os.path.join(args.output_dir, "usage_stats.json"), "w") as f:
            json.dump(usage_output, f, indent=2)
        
        # Print results summary
        print("\n" + "="*50)
        print("EVALUATION RESULTS SUMMARY")
        print("="*50)
        
        try:
            # Use tabulate for nicer display if available
            print_results_table(results, title="MuSiQue Evaluation Results")
        except ImportError:
            # Fallback to simple printing
            for top_k, metrics in results.items():
                print(f"\n{top_k}:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
        
        # Print token usage statistics
        print("\n" + "="*50)
        print("TOKEN USAGE STATISTICS")
        print("="*50)
        print_token_usage_table(usage_stats["token_usage"], title="Token Usage by top_k")
        
        # Print workspace and knowledge graph statistics
        print("\n" + "="*50)
        print("WORKSPACE STATISTICS")
        print("="*50)
        print(f"Workspace directory: {args.workspace_dir}")
        print(f"Indexed documents: {len(rag_system.indexed_doc_manifest)}")
        print(f"Knowledge graph nodes: {rag_system.graph.number_of_nodes()}")
        print(f"Knowledge graph edges: {rag_system.graph.number_of_edges()}")
        
        # Print token usage and timing statistics
        print("\n" + "="*50)
        print("TOKEN USAGE AND TIMING STATISTICS")
        print("="*50)
        print(f"Total evaluation time: {total_time:.2f} seconds")
        print(f"Embedding tokens: {usage_stats['token_usage']['embedding_tokens']}")
        print(f"Completion tokens: {usage_stats['token_usage']['completion_tokens']}")
        print(f"Total tokens: {usage_stats['token_usage']['total_tokens']}")
        print(f"Average query time: {usage_output['average_query_time']:.2f} seconds")
        print(f"Average tokens per query: {usage_output['average_tokens_per_query']:.2f}")
        
        # Inform user where results are saved
        output_path = os.path.join(args.output_dir, args.output_file)
        print("\nDetailed evaluation results saved to:")
        print(f"  - Summary: {output_path}")
        print(f"  - Token usage: {os.path.join(args.output_dir, 'usage_stats.json')}")
        for top_k in top_k_final_values:
            detailed_path = os.path.join(args.output_dir, 
                                       f"{args.output_file.replace('.json', '')}_top_k_{top_k}.json")
            print(f"  - top_k={top_k} details: {detailed_path}")
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please make sure the EcphoryRAG package is properly installed.")
        logger.info("You can install it with: pip install -e .")
        return
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return


if __name__ == "__main__":
    main() 