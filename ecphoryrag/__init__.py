"""
EcphoryRAG: A neurocognitive-inspired RAG system

This package implements the EcphoryRAG methodology, which is inspired by
neurocognitive models of human memory, particularly ecphory - the process
by which retrieval cues interact with stored memory traces.
"""

__version__ = "0.1.0"

# 直接导出主类，便于从包根目录导入
from ecphoryrag.src.ecphory_rag import EcphoryRAG

__all__ = ["EcphoryRAG"]