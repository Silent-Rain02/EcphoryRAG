"""
Evaluation metrics for RAG systems

This module provides functions to calculate common evaluation metrics for RAG systems:
- Exact Match: checks if predicted answer exactly matches ground truth (after normalization)
- F1 Score: token-level F1 between predicted and ground truth
- ROUGE Scores: calculates ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
- Retrieval Metrics: precision and recall for context retrieval
"""

import re
import string
from collections import Counter
from typing import Dict, List, Set, Tuple, Optional, Union

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison:
    - Convert to lowercase
    - Remove punctuation
    - Remove articles and other stopwords
    - Remove extra whitespace

    Args:
        text: The text to normalize (can be string or list)

    Returns:
        Normalized text string
    """
    # Handle list input
    if isinstance(text, list):
        if len(text) > 0:
            text = text[0]  # Take first element if list is not empty
        else:
            text = ""  # Use empty string if list is empty
    
    # Handle non-string input
    if not isinstance(text, str):
        text = str(text)
    
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove articles and common stopwords
    stopwords = {"a", "an", "the", "and", "but", "or", "is", "are", "was", "were"}
    tokens = text.split()
    tokens = [token for token in tokens if token not in stopwords]
    
    # Rejoin and normalize whitespace
    text = " ".join(tokens)
    
    return text


def get_tokens(text: str) -> List[str]:
    """
    Tokenize and normalize text for metric calculation.

    Args:
        text: Text to tokenize

    Returns:
        List of normalized tokens
    """
    return normalize_text(text).split()


def calculate_exact_match(predicted_answer: str, ground_truth_answers: List[str]) -> bool:
    """
    Check if the predicted answer exactly matches any of the ground truth answers
    after normalization.

    Args:
        predicted_answer: Model-generated answer
        ground_truth_answers: List of acceptable ground truth answers

    Returns:
        True if match found, False otherwise
    """
    if not predicted_answer or not ground_truth_answers:
        return False
    
    # Normalize predicted answer
    norm_pred = normalize_text(predicted_answer)
    
    # Check if normalized predicted answer matches any normalized ground truth
    for ground_truth in ground_truth_answers:
        norm_truth = normalize_text(ground_truth)
        if norm_pred == norm_truth:
            return True
    
    return False


def calculate_f1_score(predicted_answer: str, ground_truth_answers: List[str]) -> float:
    """
    Calculate token-level F1 score between predicted answer and ground truth answers.
    Returns the maximum F1 achieved against any ground truth answer.

    Args:
        predicted_answer: Model-generated answer
        ground_truth_answers: List of acceptable ground truth answers

    Returns:
        F1 score (0.0 to 1.0)
    """
    if not predicted_answer or not ground_truth_answers:
        return 0.0
    
    # Get tokens from predicted answer
    pred_tokens = get_tokens(predicted_answer)
    if not pred_tokens:
        return 0.0
    
    # Calculate F1 against each ground truth and return the maximum
    max_f1 = 0.0
    
    for ground_truth in ground_truth_answers:
        truth_tokens = get_tokens(ground_truth)
        
        # Skip empty ground truths
        if not truth_tokens:
            continue
        
        # Calculate precision and recall using token counts
        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_common = sum(common.values())
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(truth_tokens)
        
        # Calculate F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        # Update max F1
        max_f1 = max(max_f1, f1)
    
    return max_f1


def calculate_rouge_scores(predicted_answer: str, ground_truth_answers: List[str]) -> Dict[str, float]:
    """
    Calculate ROUGE-1, ROUGE-2, and ROUGE-L F1 scores between predicted answer
    and ground truth answers. Returns scores for the ground truth that yields 
    the highest ROUGE-L F1.

    Args:
        predicted_answer: Model-generated answer
        ground_truth_answers: List of acceptable ground truth answers

    Returns:
        Dictionary of ROUGE scores: {'rouge1': score1, 'rouge2': score2, 'rougeL': scoreL}
    """
    if not ROUGE_AVAILABLE:
        raise ImportError(
            "Rouge metrics require the rouge-score package. "
            "Install it with: pip install rouge-score"
        )
    
    if not predicted_answer or not ground_truth_answers:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    # Initialize rouge scorer with the metrics we want
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Score against each ground truth and track the best result (by ROUGE-L)
    best_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    for ground_truth in ground_truth_answers:
        if not ground_truth.strip():
            continue
            
        # Calculate scores
        scores = scorer.score(ground_truth, predicted_answer)
        
        # Extract F1 scores
        rouge1_f1 = scores['rouge1'].fmeasure
        rouge2_f1 = scores['rouge2'].fmeasure
        rougeL_f1 = scores['rougeL'].fmeasure
        
        # Update best scores if this ground truth gives better ROUGE-L
        if rougeL_f1 > best_scores['rougeL']:
            best_scores = {
                "rouge1": rouge1_f1,
                "rouge2": rouge2_f1,
                "rougeL": rougeL_f1
            }
    
    return best_scores


def calculate_retrieval_metrics(retrieved_context_ids: List[str], 
                              relevant_context_ids: List[str], 
                              k: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate retrieval metrics (precision, recall) for context retrieval.

    Args:
        retrieved_context_ids: IDs of contexts retrieved by the system
        relevant_context_ids: IDs of actually relevant contexts (ground truth)
        k: Optional cutoff to only consider top-k retrieved contexts

    Returns:
        Dictionary of metrics: {'precision': p_score, 'recall': r_score, 'f1': f1_score}
    """
    if not retrieved_context_ids or not relevant_context_ids:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Apply k cutoff if specified
    if k is not None and k > 0:
        retrieved_context_ids = retrieved_context_ids[:k]
    
    # Convert to sets for intersection
    retrieved_set = set(retrieved_context_ids)
    relevant_set = set(relevant_context_ids)
    
    # Calculate intersection
    intersection = retrieved_set & relevant_set
    num_intersect = len(intersection)
    
    # Calculate metrics
    precision = num_intersect / len(retrieved_set) if retrieved_set else 0.0
    recall = num_intersect / len(relevant_set) if relevant_set else 0.0
    
    # Calculate F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_answer(predicted_answer: str, ground_truth_answers: List[str]) -> Dict[str, float]:
    """
    Evaluate a predicted answer using multiple metrics.

    Args:
        predicted_answer: Model-generated answer
        ground_truth_answers: List of acceptable ground truth answers

    Returns:
        Dictionary of evaluation metrics
    """
    results = {
        "exact_match": float(calculate_exact_match(predicted_answer, ground_truth_answers)),
        "f1_score": calculate_f1_score(predicted_answer, ground_truth_answers)
    }
    
    if ROUGE_AVAILABLE:
        rouge_scores = calculate_rouge_scores(predicted_answer, ground_truth_answers)
        results.update(rouge_scores)
    
    return results


# Example usage
if __name__ == "__main__":
    # Example answers
    predicted = "The Eiffel Tower is located in Paris, France."
    ground_truths = [
        "The Eiffel Tower is in Paris.",
        "Paris, France is home to the Eiffel Tower."
    ]
    
    # Calculate metrics
    print(f"Exact Match: {calculate_exact_match(predicted, ground_truths)}")
    print(f"F1 Score: {calculate_f1_score(predicted, ground_truths):.4f}")
    
    if ROUGE_AVAILABLE:
        rouge = calculate_rouge_scores(predicted, ground_truths)
        print(f"ROUGE-1 F1: {rouge['rouge1']:.4f}")
        print(f"ROUGE-2 F1: {rouge['rouge2']:.4f}")
        print(f"ROUGE-L F1: {rouge['rougeL']:.4f}")
    
    # Example for retrieval metrics
    retrieved_ids = ["doc1", "doc2", "doc3", "doc4"]
    relevant_ids = ["doc1", "doc3", "doc5"]
    
    metrics = calculate_retrieval_metrics(retrieved_ids, relevant_ids, k=3)
    print(f"\nRetrieval Metrics (top-3):")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}") 