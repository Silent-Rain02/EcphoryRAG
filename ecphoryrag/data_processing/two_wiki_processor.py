import json
from typing import List, Dict, Any, Iterator, Optional
import uuid  # For generating unique chunk IDs

from .base_processor import BaseDatasetProcessor


class TwoWikiProcessor(BaseDatasetProcessor):
    def load_raw_data(self) -> Iterator[Dict[str, Any]]:
        """
        Loads raw 2Wiki data from the JSON file, yielding one sample at a time.
        """
        with open(self.raw_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # Load the entire JSON list
            for sample in data:
                yield sample

    def _build_chunk_text_from_sentences(self, title: str, sentences: List[str]) -> str:
        """Helper to combine title and sentences into a single chunk string."""
        return f"{title}: {' '.join(sentences)}"

    def transform_sample(self, raw_sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transforms a raw 2Wiki sample into the standardized target format.
        
        Args:
            raw_sample: A dictionary containing a raw 2Wiki sample
            
        Returns:
            Optional[Dict[str, Any]]: The transformed sample, or None if the sample should be skipped
        """
        sample_id = raw_sample.get("_id", str(uuid.uuid4()))
        question_type = raw_sample.get("type", None)  # "bridge" or "comparison"

        # 1. Process all context paragraphs into indexable chunks for EcphoryRAG
        processed_chunks = []
        # Create a mapping from title to full chunk text for easier lookup later
        title_to_full_chunk_text = {}
        title_to_chunk_id = {}

        for para_idx, context_item in enumerate(raw_sample.get("context", [])):
            title = context_item[0]
            sentences = context_item[1]
            full_chunk_text = self._build_chunk_text_from_sentences(title, sentences)
            
            # Generate a unique ID for each chunk to be indexed
            # Ensure it's unique across the entire dataset if multiple samples share titles
            chunk_internal_id = f"{sample_id}_ctxchunk_{para_idx}"  # Link to sample_id and its paragraph index
            
            processed_chunks.append({
                "id": chunk_internal_id,
                "title": title,
                "chunk": full_chunk_text
            })
            title_to_full_chunk_text[title] = full_chunk_text
            title_to_chunk_id[title] = chunk_internal_id

        # 2. Process supporting facts
        # These will reference the chunks created above.
        supporting_facts_processed = []
        # Use a set to avoid duplicate supporting chunks if multiple sentences from the same paragraph are SFs
        processed_supporting_chunk_ids = set()

        for sf_title, sf_sent_id in raw_sample.get("supporting_facts", []):
            # sf_sent_id is not directly used here as we treat the whole paragraph as a chunk.
            # We just need to identify which of our `processed_chunks` is the supporting one.
            if sf_title in title_to_full_chunk_text:
                chunk_id_for_sf = title_to_chunk_id[sf_title]
                if chunk_id_for_sf not in processed_supporting_chunk_ids:
                    supporting_facts_processed.append({
                        "id": chunk_id_for_sf,  # Use the same ID as the one in `processed_chunks`
                        "title": sf_title,
                        "chunk": title_to_full_chunk_text[sf_title]
                    })
                    processed_supporting_chunk_ids.add(chunk_id_for_sf)
            else:
                # This case should ideally not happen if data is consistent
                print(f"Warning: Supporting fact title '{sf_title}' not found in context for sample {sample_id}")

        # 3. Process evidences (knowledge graph triples)
        evidences_processed = []
        for evidence in raw_sample.get("evidences", []):
            if len(evidence) >= 3:
                evidences_processed.append({
                    "subject": evidence[0],
                    "relation": evidence[1],
                    "object": evidence[2]
                })

        return {
            "id": sample_id,
            "hop": 2,  # 2Wiki is generally considered 2-hop
            "type": question_type,
            "question": raw_sample.get("question", ""),
            "answer": raw_sample.get("answer", ""),
            "answer_aliases": [],  # 2Wiki format doesn't typically have aliases
            "chunks": processed_chunks,  # All context paragraphs
            "supporting_facts": supporting_facts_processed,
            "decomposition": [],  # 2Wiki doesn't have explicit decomposition
            "evidences": evidences_processed  # Additional field for 2Wiki evidences
        } 