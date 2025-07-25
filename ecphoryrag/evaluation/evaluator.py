"""
EcphoryRAG Evaluator Module

This module provides the Evaluator class for evaluating EcphoryRAG 
system performance on various datasets.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ecphoryrag import EcphoryRAG
from ecphoryrag.evaluation.datasets import DatasetLoader
from ecphoryrag.evaluation.metrics import (
    calculate_exact_match,
    calculate_f1_score,
    calculate_rouge_scores
)

# Configure logging
logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator class for evaluating EcphoryRAG system performance.
    
    This class coordinates the evaluation process by running a RAG system
    on datasets and calculating various performance metrics.
    """
    
    def __init__(
        self,
        rag_system: EcphoryRAG,
        dataset_loader: DatasetLoader,
        metrics_calculators: Optional[List[Callable]] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the Evaluator.
        
        Args:
            rag_system: An instance of EcphoryRAG to evaluate
            dataset_loader: Loader for evaluation dataset
            metrics_calculators: List of metric calculation functions
            output_dir: Directory to save evaluation results
        """
        self.rag_system = rag_system
        self.dataset_loader = dataset_loader
        
        # Set default metrics if none provided
        if metrics_calculators is None:
            self.metrics_calculators = [
                lambda pred, truth: {"exact_match": calculate_exact_match(pred, truth)},
                lambda pred, truth: {"f1_score": calculate_f1_score(pred, truth)},
                lambda pred, truth: calculate_rouge_scores(pred, truth) if "rouge" in self.rag_system.available_metrics else {}
            ]
        else:
            self.metrics_calculators = metrics_calculators
        
        # Set up output directory
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _prepare_rag_for_datapoint(self, datapoint: Dict) -> None:
        """
        Prepare the RAG system for a specific datapoint.
        
        For datasets like MuSiQue where each question has its own specific context,
        this method ensures we only use the relevant contexts for the current question.
        
        Args:
            datapoint: The current data point with question and contexts
        """
        # No need to reindex documents, just ensure we're using the right contexts
        contexts = datapoint.get('contexts', [])
        if not contexts:
            logger.warning(f"No contexts found for datapoint {datapoint.get('id', 'unknown')}")
            return
        
        logger.debug(f"Using {len(contexts)} contexts for datapoint {datapoint.get('id', 'unknown')}")
    
    def run_evaluation(
        self,
        num_samples: Optional[int] = None,
        output_file: Optional[str] = None,
        top_k_values: Optional[List[int]] = None,
        skip_indexing: bool = False,
        save_full_responses: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Run evaluation on the dataset.
        
        Args:
            num_samples: Number of samples to evaluate (None for all)
            output_file: Name of output file to save results
            top_k_values: List of top_k values to evaluate
            skip_indexing: Whether to skip document indexing
            save_full_responses: Whether to save full response details
            
        Returns:
            Dictionary of evaluation results for each top_k value
        """
        # Load dataset
        data = self.dataset_loader.load_data()
        if not data:
            logger.error("No data loaded for evaluation")
            return {}
            
        # If num_samples is specified, limit the data
        if num_samples is not None:
            data = data[:num_samples]
            
        # Set default top_k values if none provided
        if top_k_values is None:
            top_k_values = [5]
            
        # Initialize results storage
        all_results = {}
        all_token_usage = {}
        
        # Process each top_k value
        for top_k in top_k_values:
            logger.info(f"Evaluating with top_k={top_k}")
            
            # Reset usage stats for this top_k evaluation
            self.rag_system.reset_usage_stats()
            
            # Initialize metrics sums for averaging
            metric_sums = {}
            total_processing_time = 0
            results_data = []
            
            # Process each question
            for item in data:
                question = item["question"]
                ground_truth = item["ground_truth_answer"]
                ground_truth_aliases = item.get("ground_truth_aliases", [])
                
                # Add ground truth to aliases if not already present
                if ground_truth not in ground_truth_aliases:
                    ground_truth_aliases.append(ground_truth)
                
                # Time the query
                start_time = time.time()
                
                # Get answer from RAG system
                predicted_answer, sources = self.rag_system.query(
                    question,
                    top_k_final_traces=top_k
                )
                
                query_time = time.time() - start_time
                total_processing_time += query_time
                
                # Initialize result dictionary
                result = {
                    "id": item.get("id", ""),
                    "question": question,
                    "predicted_answer": predicted_answer,
                    "ground_truth_answer": ground_truth,
                    "ground_truth_aliases": ground_truth_aliases,
                    "processing_time": query_time,
                    "sources": sources
                }
                
                # Calculate metrics
                for metric_fn in self.metrics_calculators:
                    metric_result = metric_fn(predicted_answer, ground_truth_aliases)
                    if isinstance(metric_result, dict):
                        result.update(metric_result)
                        
                        # Update sums for average calculation
                        for metric_name, metric_value in metric_result.items():
                            if metric_name not in metric_sums:
                                metric_sums[metric_name] = 0.0
                            metric_sums[metric_name] += metric_value
                
                # Add result to collection
                results_data.append(result)
                
                # Log detailed result for debugging
                logger.debug(f"Question: {question}")
                logger.debug(f"Predicted: {predicted_answer}")
                logger.debug(f"Ground truth: {ground_truth}")
                metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in result.items() 
                                         if k not in ['id', 'question', 'predicted_answer', 
                                                      'ground_truth_answer', 'ground_truth_aliases',
                                                      'processing_time', 'sources']])
                logger.info(f"Metrics: {metrics_str} (processed in {query_time:.2f}s)")
            
            # Calculate averages
            avg_metrics = {}
            for metric_name, metric_sum in metric_sums.items():
                avg_metrics[metric_name] = metric_sum / len(data) if data else 0
            
            # Add avg processing time
            avg_metrics["avg_processing_time"] = total_processing_time / len(data) if data else 0
            
            # Get token usage stats for this top_k
            usage_stats = self.rag_system.get_usage_stats()
            token_usage = {
                "embedding_tokens": usage_stats["token_usage"]["embedding_tokens"],
                "completion_tokens": usage_stats["token_usage"]["completion_tokens"],
                "total_tokens": usage_stats["token_usage"]["total_tokens"],
                "avg_tokens_per_query": usage_stats["token_usage"]["total_tokens"] / len(data) if data else 0
            }
            
            # Store results for this top_k
            all_results[f"top_k_{top_k}"] = avg_metrics
            all_token_usage[f"top_k_{top_k}"] = token_usage
            
            # Save detailed results for this top_k if requested
            if output_file:
                detailed_output = {
                    "results": results_data,
                    "metrics": avg_metrics,
                    "token_usage": token_usage
                }
                
                detailed_path = self.output_dir / f"{output_file.replace('.json', '')}_top_k_{top_k}.json"
                with open(detailed_path, 'w', encoding='utf-8') as f:
                    json.dump(detailed_output, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved detailed results for top_k={top_k} to {detailed_path}")
        
        # Save summary results
        if output_file:
            summary_output = {
                "results": all_results,
                "token_usage": all_token_usage
            }
            
            summary_path = self.output_dir / output_file
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_output, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved summary results to {summary_path}")
        
        return all_results


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example paths
    dataset_path = "data/musique/musique-dev.jsonl"
    aliases_path = "data/musique/answer_aliases.json"
    
    try:
        # Initialize dataset loader
        loader = MusiqueDatasetLoader(dataset_path, aliases_path)
        
        # Initialize EcphoryRAG system
        from ecphoryrag import EcphoryRAG
        rag_system = EcphoryRAG(
            embedding_model_name="bge-small-en",
            extraction_llm_model="phi3:mini",
            generation_llm_model="phi3:mini"
        )
        
        # Initialize evaluator
        evaluator = Evaluator(
            rag_system=rag_system,
            dataset_loader=loader,
            output_dir="evaluation_results"
        )
        
        # Run evaluation
        results = evaluator.run_evaluation(
            num_samples=5,  # Limited for demo
            output_file="musique_dev_evaluation.json",
            top_k_values=[1, 3],
            save_full_responses=True
        )
        
        # Print summary results
        print("\nEvaluation Results Summary:")
        for top_k, metrics in results.items():
            print(f"\n{top_k}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"Error running example: {e}", exc_info=True) 