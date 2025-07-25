import json
import re
from typing import List, Dict, Any, Iterator, Optional
import uuid  # For generating unique chunk IDs

from .base_processor import BaseDatasetProcessor


class MusiqueProcessor(BaseDatasetProcessor):
    def load_raw_data(self) -> Iterator[Dict[str, Any]]:
        """
        Loads raw MuSiQue data from the JSONL file, yielding one sample at a time.
        """
        with open(self.raw_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)

    def _extract_hop_from_id(self, sample_id: str) -> int:
        """
        Extracts the hop count from a MuSiQue sample ID (e.g., "2hop_..." -> 2).
        
        Args:
            sample_id: The ID string from the MuSiQue dataset
            
        Returns:
            int: The number of hops, or 0 if not found
        """
        match = re.match(r"(\d+)hop_", sample_id)
        if match:
            return int(match.group(1))
        return 0  # Default or raise error

    def transform_sample(self, raw_sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transforms a raw MuSiQue sample into the standardized target format.
        
        Args:
            raw_sample: A dictionary containing a raw MuSiQue sample
            
        Returns:
            Optional[Dict[str, Any]]: The transformed sample, or None if the sample should be skipped
        """
        if not raw_sample.get("answerable", True):  # Skip unanswerable questions if desired
            return None

        sample_id = raw_sample.get("id", str(uuid.uuid4()))
        hop = self._extract_hop_from_id(sample_id)

        # Chunks to be indexed by EcphoryRAG
        # THESE ARE THE CORPUS PARAGRAPHS ONLY
        processed_chunks = []
        for para_idx, para_data in enumerate(raw_sample.get("paragraphs", [])):
            # Generate a unique ID for each chunk to be indexed
            chunk_internal_id = f"{sample_id}_chunk_{para_data.get('idx', para_idx)}"
            processed_chunks.append({
                "id": chunk_internal_id,
                "title": para_data.get("title", ""),
                "chunk": para_data.get("paragraph_text", "")
            })

        # Supporting facts (optional to keep, but good for analysis)
        # These are also from raw_sample["paragraphs"] but filtered by "is_supporting"
        supporting_facts_processed = []
        for para_idx, para_data in enumerate(raw_sample.get("paragraphs", [])):
            if para_data.get("is_supporting"):
                # Use the same ID as in processed_chunks for consistency if desired
                # Or generate a new one if they are treated completely separately
                sf_internal_id = f"{sample_id}_sf_chunk_{para_data.get('idx', para_idx)}"
                supporting_facts_processed.append({
                    "id": sf_internal_id,  # Or the same ID as in processed_chunks
                    "title": para_data.get("title", ""),
                    "chunk": para_data.get("paragraph_text", "")
                })
        
        # Question decomposition
        # Store the list of question dicts directly, or just the sub-questions
        decomposition_processed = []
        for decomp_item in raw_sample.get("question_decomposition", []):
            decomposition_processed.append({
                "id": decomp_item.get("id"),
                "question": decomp_item.get("question"),
                "answer": decomp_item.get("answer")  # Keep sub-answers for analysis
                # "paragraph_support_idx": decomp_item.get("paragraph_support_idx")  # Optional
            })

        return {
            "id": sample_id,
            "hop": hop,
            "type": None,  # MuSiQue doesn't explicitly provide a high-level 'type' like 'compose'
            "question": raw_sample.get("question", ""),
            "answer": raw_sample.get("answer", ""),  # Keep original answer
            "answer_aliases": raw_sample.get("answer_aliases", []),  # Keep aliases
            "chunks": processed_chunks,
            "supporting_facts": supporting_facts_processed,
            "decomposition": decomposition_processed
        }
