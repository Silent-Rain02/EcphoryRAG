from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator, Optional
import json
import os


class BaseDatasetProcessor(ABC):
    """
    Abstract base class for processing different raw RAG datasets
    into a standardized format.
    """

    def __init__(self, raw_data_path: str, output_dir: str, split_name: str):
        """
        Args:
            raw_data_path: Path to the raw dataset file (e.g., a JSONL file).
            output_dir: Directory where the processed file will be saved.
            split_name: Name of the split (e.g., "train", "dev", "test").
        """
        self.raw_data_path = raw_data_path
        self.output_dir = output_dir
        self.split_name = split_name
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file_path = os.path.join(self.output_dir, f"{self.split_name}_processed.jsonl")

    @abstractmethod
    def load_raw_data(self) -> Iterator[Dict[str, Any]]:
        """Loads raw data from the source file, yielding one raw sample at a time."""
        pass

    @abstractmethod
    def transform_sample(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transforms a single raw sample into the standardized target format.
        Target format:
        {
            "id": str,
            "hop": int,
            "type": Optional[str], // Can be None if not applicable/derivable
            "question": str,
            "answer": str,
            "chunks": List[Dict[str, Any]], // {"id": str, "title": str, "chunk": str}
            "supporting_facts": List[Dict[str, Any]], // Same format as chunks
            "decomposition": List[Any] // Structure depends on source dataset
        }
        """
        pass

    def process_and_save(self, limit: Optional[int] = None):
        """
        Processes all raw samples and saves them to the output file in JSONL format.
        Args:
            limit: Optional an integer to limit the number of samples to process (for testing).
        """
        count = 0
        with open(self.output_file_path, 'w', encoding='utf-8') as outfile:
            for raw_sample in self.load_raw_data():
                if limit and count >= limit:
                    print(f"Processed {count} samples (limit reached).")
                    break
                try:
                    transformed_sample = self.transform_sample(raw_sample)
                    if transformed_sample:  # Ensure transformation was successful
                        outfile.write(json.dumps(transformed_sample) + '\n')
                        count += 1
                except Exception as e:
                    sample_id = raw_sample.get("id", "UNKNOWN_ID")
                    print(f"Error processing sample {sample_id}: {e}")
                    continue  # Skip problematic samples
        print(f"Successfully processed {count} samples and saved to {self.output_file_path}")
