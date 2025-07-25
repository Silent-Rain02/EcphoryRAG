"""
EcphoryRAG evaluation module

Provides tools for evaluating EcphoryRAG system performance:
- datasets: Dataset loading and preprocessing
- metrics: Calculate evaluation metrics (EM, F1, ROUGE, etc.)
- evaluator: Main evaluator class
"""

from ecphoryrag.evaluation.datasets import DatasetLoader
from ecphoryrag.evaluation.metrics import calculate_exact_match, calculate_f1_score, calculate_rouge_scores
from ecphoryrag.evaluation.evaluator import Evaluator

__all__ = [
    'DatasetLoader',
    'MusiqueDatasetLoader',
    'calculate_exact_match',
    'calculate_f1_score',
    'calculate_rouge_scores',
    'Evaluator',
] 