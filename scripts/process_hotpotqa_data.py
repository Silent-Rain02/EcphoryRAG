import argparse
import os
import sys

# Adjust path to import from ecphoryrag package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ecphoryrag.data_processing.hotpotqa_processor import HotpotQAProcessor

def main():
    parser = argparse.ArgumentParser(description="Process raw HotpotQA dataset.")
    parser.add_argument(
        "--raw_data_file",
        type=str,
        default="/home/lzr24/EcphoryRAG/data/hotpot/hotpot_train_v1.1.json",
        help="Path to the raw HotpotQA JSON file (e.g., data/hotpotqa/hotpot_train_v1.1.json)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/lzr24/EcphoryRAG/data/processed_hotpotqa",
        help="Directory to save the processed data (e.g., processed_data/hotpotqa/)."
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="train",
        help="Name of the dataset split (e.g., train, dev, test)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of samples to process (for testing)."
    )
    args = parser.parse_args()

    print(f"Starting processing for HotpotQA: {args.raw_data_file}")
    processor = HotpotQAProcessor(
        raw_data_path=args.raw_data_file,
        output_dir=args.output_dir,
        split_name=args.split_name
    )
    processor.process_and_save(limit=args.limit)
    print("HotpotQA processing complete.")

if __name__ == "__main__":
    main() 