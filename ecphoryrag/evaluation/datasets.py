"""
Dataset loading and preprocessing module

This module provides classes and functions for loading and preprocessing various evaluation datasets.
Currently supported datasets:
- MuSiQue: Multi-step reasoning question answering dataset
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any

# Configure logging
logger = logging.getLogger(__name__)

class DatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    
    All specific dataset loaders should inherit this class and implement its methods.
    """
    
    @abstractmethod
    def load_data(self) -> List[Dict]:
        """
        Load the dataset and return a list of standardized data points.
        
        Returns:
            List[Dict]: List of data points, each data point is a dictionary containing id, question, context, and answer.
        """
        pass
    
    @abstractmethod
    def get_documents_for_indexing(self) -> List[Dict[str, str]]:
        """
        Return a list of documents suitable for indexing with EcphoryRAG.
        
        Returns:
            List[Dict[str, str]]: List of documents, each document contains 'id' and 'text' keys.
        """
        pass

