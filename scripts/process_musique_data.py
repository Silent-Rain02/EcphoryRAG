#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ecphoryrag.data_processing.musique_processor import MusiqueProcessor


def main():
    parser = argparse.ArgumentParser(description="Process raw MuSiQue dataset.")
    parser.add_argument(
        "--raw_data_file",
        type=str,
        default='/home/lzr24/EcphoryRAG/data/musique/data/musique_full_v1.0_dev.jsonl',
        help="Path to the raw MuSiQue JSONL file (e.g., data/musique/musique_ans_v1.0_dev.jsonl)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='/home/lzr24/EcphoryRAG/data/processed_musique',
        help="Directory to save the processed data (e.g., processed_data/musique/)."
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="dev",
        help="Name of the dataset split (e.g., dev, train, test)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on the number of samples to process (for testing)."
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Starting processing for: {args.raw_data_file}")
    processor = MusiqueProcessor(
        raw_data_path=args.raw_data_file,
        output_dir=args.output_dir,
        split_name=args.split_name
    )
    processor.process_and_save(limit=args.limit)
    print("Processing complete.")


if __name__ == "__main__":
    main()
