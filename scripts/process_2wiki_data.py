#!/usr/bin/env python3
import argparse
import os
import sys

# Adjust path to import from ecphoryrag package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ecphoryrag.data_processing.two_wiki_processor import TwoWikiProcessor


def main():
    parser = argparse.ArgumentParser(description="Process raw 2Wiki dataset.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/home/lzr24/EcphoryRAG/data/2wikimultihop/data/dev.json",
        help="Path to the raw 2Wiki JSON file (e.g., data/2wiki/raw_data.json)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/lzr24/EcphoryRAG/data/processed_2wiki",
        help="Directory to save the processed data (e.g., processed_data/2wiki/)."
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

    print(f"Starting processing for 2Wiki: {args.input_file}")
    processor = TwoWikiProcessor(
        raw_data_path=args.input_file,
        output_dir=args.output_dir,
        split_name=args.split_name
    )
    processor.process_and_save(limit=args.limit)
    print("2Wiki processing complete.")


if __name__ == "__main__":
    main() 