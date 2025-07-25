"""
Processed dataset loaders for EcphoryRAG evaluation.

This module contains dataset loaders for processed datasets in the standardized format.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .datasets import DatasetLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProcessedMusiqueDatasetLoader(DatasetLoader):
    """
    Loader for processed MuSiQue dataset in the standardized format.
    
    This loader works with the processed JSONL files created by the MusiqueProcessor.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the ProcessedMusiqueDatasetLoader.
        
        Args:
            file_path: Path to the processed MuSiQue dataset file (JSONL format)
        """
        self.file_path = Path(file_path)
        self.data = None
        
        # Validate file exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"Processed MuSiQue dataset file not found: {self.file_path}")
    
    def load_data(self) -> List[Dict]:
        """
        Load the processed MuSiQue dataset.
        
        Returns:
            List[Dict]: List of question-answer data points
        """
        if self.data is not None:
            return self.data
        
        data = []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    
                    # Extract necessary fields
                    question_id = item.get('id', '')
                    question = item.get('question', '')
                    answer = item.get('answer', '')
                    chunks = item.get('chunks', [])
                    answer_aliases = item.get('answer_aliases', [])
                    
                    # Create standardized data point
                    data_point = {
                        "id": question_id,
                        "question": question,
                        "contexts": chunks,  # Already in the right format with id, title, chunk
                        "ground_truth_answer": answer,
                        "ground_truth_aliases": answer_aliases
                    }
                    
                    data.append(data_point)
            
            logger.info(f"Loaded {len(data)} question-answer data points")
            self.data = data
            return data
            
        except Exception as e:
            logger.error(f"Error loading processed MuSiQue dataset: {e}")
            raise
    
    def get_documents_for_indexing(self) -> List[Dict[str, Any]]:
        """
        Return a list of documents suitable for indexing in EcphoryRAG.
        
        For multi-hop reasoning, we index each chunk as a separate document.
        
        Returns:
            List[Dict[str, Any]]: List of documents, each with 'id', 'title', and 'chunk' keys.
        """
        if self.data is None:
            self.load_data()
        
        documents = []
        for item in self.data:
            for chunk in item['contexts']:
                documents.append({
                    "id": chunk["id"],
                    "title": chunk.get("title", ""),
                    "text": chunk["chunk"]  # The chunk text is in the "chunk" field
                })
        
        logger.info(f"Prepared {len(documents)} documents for indexing")
        return documents
    
    def get_question_document_mapping(self) -> Dict[str, List[str]]:
        """
        Return a mapping from question IDs to document IDs.
        
        This is useful for evaluation to track which documents are relevant to each question.
        
        Returns:
            Dict[str, List[str]]: Mapping from question IDs to lists of document IDs
        """
        if self.data is None:
            self.load_data()
        
        mapping = {}
        for item in self.data:
            question_id = item["id"]
            doc_ids = [chunk["id"] for chunk in item["contexts"]]
            mapping[question_id] = doc_ids
        
        return mapping
    
    def get_questions(self) -> List[Dict[str, Any]]:
        """
        Return a list of just questions and answers for evaluation.
        
        Returns:
            List[Dict[str, Any]]: List of questions, each with id, question text, and answer
        """
        if self.data is None:
            self.load_data()
        
        questions = []
        for item in self.data:
            questions.append({
                "id": item["id"],
                "question": item["question"],
                "ground_truth_answer": item["ground_truth_answer"],
                "ground_truth_aliases": item["ground_truth_aliases"]
            })
        
        return questions


class ProcessedHotpotQADatasetLoader(DatasetLoader):
    """
    Loader for processed HotpotQA dataset in the standardized format.
    
    This loader works with the processed JSONL files created by the HotpotQAProcessor.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the ProcessedHotpotQADatasetLoader.
        
        Args:
            file_path: Path to the processed HotpotQA dataset file (JSONL format)
        """
        self.file_path = Path(file_path)
        self.data = None
        
        # Validate file exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"Processed HotpotQA dataset file not found: {self.file_path}")
    
    def load_data(self) -> List[Dict]:
        """
        Load the processed HotpotQA dataset.
        
        Returns:
            List[Dict]: List of question-answer data points
        """
        if self.data is not None:
            return self.data
        
        data = []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    
                    # Extract necessary fields
                    question_id = item.get('id', '')
                    question = item.get('question', '')
                    answer = item.get('answer', '')
                    supporting_facts = item.get('supporting_facts', [])
                    answer_aliases = item.get('answer_aliases', [])
                    
                    # Create standardized data point
                    data_point = {
                        "id": question_id,
                        "question": question,
                        "contexts": supporting_facts,  # Already in the right format with id, title, chunk
                        "ground_truth_answer": answer,
                        "ground_truth_aliases": answer_aliases
                    }
                    
                    data.append(data_point)
            
            logger.info(f"Loaded {len(data)} question-answer data points")
            self.data = data
            return data
            
        except Exception as e:
            logger.error(f"Error loading processed HotpotQA dataset: {e}")
            raise
    
    def get_documents_for_indexing(self) -> List[Dict[str, Any]]:
        """
        Return a list of documents suitable for indexing in EcphoryRAG.
        
        For multi-hop reasoning, we index each supporting fact as a separate document.
        
        Returns:
            List[Dict[str, Any]]: List of documents, each with 'id', 'title', and 'text' keys.
        """
        if self.data is None:
            self.load_data()
        
        documents = []
        for item in self.data:
            for fact in item['contexts']:
                documents.append({
                    "id": fact["id"],
                    "title": fact.get("title", ""),
                    "text": fact["chunk"]  # Use the chunk field as text
                })
        
        logger.info(f"Prepared {len(documents)} documents for indexing")
        return documents
    
    def get_question_document_mapping(self) -> Dict[str, List[str]]:
        """
        Return a mapping from question IDs to document IDs.
        
        This is useful for evaluation to track which documents are relevant to each question.
        
        Returns:
            Dict[str, List[str]]: Mapping from question IDs to lists of document IDs
        """
        if self.data is None:
            self.load_data()
        
        mapping = {}
        for item in self.data:
            question_id = item["id"]
            doc_ids = [fact["id"] for fact in item["contexts"]]
            mapping[question_id] = doc_ids
        
        return mapping
    
    def get_questions(self) -> List[Dict[str, Any]]:
        """
        Return a list of just questions and answers for evaluation.
        
        Returns:
            List[Dict[str, Any]]: List of questions, each with id, question text, and answer
        """
        if self.data is None:
            self.load_data()
        
        questions = []
        for item in self.data:
            questions.append({
                "id": item["id"],
                "question": item["question"],
                "ground_truth_answer": item["ground_truth_answer"],
                "ground_truth_aliases": item["ground_truth_aliases"]
            })
        
        return questions


class Processed2WikiDatasetLoader(DatasetLoader):
    """
    Dataset loader for processed 2Wiki dataset.
    
    The processed 2Wiki dataset in this system has the following format:
    {
        "id": "2wiki_ID",
        "hop": 2,
        "type": "unknown",
        "question": "str",
        "answer": "str",
        "chunks": [
            {
                "id": "2wiki_ID_N",
                "title": "Document Title",
                "chunk": "Document Content"
            },
            ...
        ],
        "supporting_facts": [
            {
                "id": "2wiki_sf_ID_N",
                "title": "Document Title",
                "chunk": "Supporting Fact Content"
            },
            ...
        ],
        "evidences": [
            {
                "subject": "Entity1",
                "relation": "Relation",
                "object": "Entity2"
            },
            ...
        ]
    }
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the 2Wiki dataset loader.
        
        Args:
            file_path: Path to processed 2Wiki dataset file
        """
        super().__init__()
        self.file_path = Path(file_path)
        self.data = None
        
        # Validate file exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"Processed 2Wiki dataset file not found: {self.file_path}")
    
    def load_data(self) -> List[Dict]:
        """
        Load the processed 2Wiki dataset.
        
        Returns:
            List[Dict]: List of question-answer data points
        """
        if self.data is not None:
            return self.data
        
        data = []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    
                    # Extract necessary fields
                    question_id = item.get('id', '')
                    question = item.get('question', '')
                    answer = item.get('answer', '')
                    
                    # 获取chunks，这些是已经处理好的文档片段
                    chunks = item.get('chunks', [])
                    
                    # 标准化格式，确保每个chunk都有id, title和chunk字段
                    contexts = []
                    for chunk in chunks:
                        if not chunk.get('id') or not chunk.get('chunk'):
                            continue
                            
                        contexts.append({
                            "id": chunk["id"],
                            "title": chunk.get("title", ""),
                            "chunk": chunk["chunk"]
                        })
                    
                    # 创建标准格式的数据点
                    data_point = {
                        "id": question_id,
                        "question": question,
                        "contexts": contexts,
                        "ground_truth_answer": answer,
                        "ground_truth_aliases": []  # 2Wiki没有别名
                    }
                    
                    data.append(data_point)
            
            logger.info(f"Loaded {len(data)} question-answer data points from 2Wiki dataset")
            self.data = data
            return data
            
        except Exception as e:
            logger.error(f"Error loading processed 2Wiki dataset: {e}")
            raise
    
    def get_documents_for_indexing(self) -> List[Dict[str, Any]]:
        """
        Return a list of documents suitable for indexing in EcphoryRAG.
        
        Returns:
            List[Dict[str, Any]]: List of documents, each with 'id', 'title', and 'text' keys.
        """
        if self.data is None:
            self.load_data()
        
        documents = []
        for item in self.data:
            for context in item['contexts']:
                # 创建索引文档
                documents.append({
                    "id": context["id"],
                    "title": context.get("title", ""),
                    "text": context["chunk"]  # 使用chunk字段作为text
                })
        
        logger.info(f"Prepared {len(documents)} documents for indexing from 2Wiki dataset")
        return documents
    
    def get_question_document_mapping(self) -> Dict[str, List[str]]:
        """
        Return a mapping from question IDs to document IDs.
        
        Returns:
            Dict[str, List[str]]: Mapping from question IDs to lists of document IDs
        """
        if self.data is None:
            self.load_data()
        
        mapping = {}
        for item in self.data:
            question_id = item["id"]
            doc_ids = [context["id"] for context in item["contexts"]]
            mapping[question_id] = doc_ids
        
        return mapping
    
    def get_questions(self) -> List[Dict[str, Any]]:
        """
        Return a list of just questions and answers for evaluation.
        
        Returns:
            List[Dict[str, Any]]: List of questions, each with id, question text, and answer
        """
        if self.data is None:
            self.load_data()
        
        questions = []
        for item in self.data:
            questions.append({
                "id": item["id"],
                "question": item["question"],
                "ground_truth_answer": item["ground_truth_answer"],
                "ground_truth_aliases": item.get("ground_truth_aliases", [])
            })
        
        return questions
