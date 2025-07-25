#!/usr/bin/env python
"""
MuSiQue Evaluation Script for EcphoryRAG

This script demonstrates how to use the Evaluator class to evaluate 
EcphoryRAG performance on the MuSiQue multi-step question answering dataset.
"""

import argparse
import logging
import os
import sys
import time
import json
from pathlib import Path
import uuid

# 修复路径引用问题
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 确保使用正确的绝对引用路径
from ecphoryrag import EcphoryRAG  # 从主包直接引入
from ecphoryrag.evaluation.datasets import MusiqueDatasetLoader
from ecphoryrag.evaluation.evaluator import Evaluator
from ecphoryrag.evaluation.utils import print_results_table


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def create_test_subset(data_path: str, subset_size: int = 10) -> str:
    """
    从完整数据集中创建一个小的测试子集。
    
    Args:
        data_path: 原始数据集路径
        subset_size: 子集大小
        
    Returns:
        子集文件路径
    """
    # 创建子集目录
    subset_dir = os.path.join(os.path.dirname(data_path), "test_subset")
    os.makedirs(subset_dir, exist_ok=True)
    
    # 子集文件路径
    subset_path = os.path.join(subset_dir, "musique_ans_v1.0_dev_subset.jsonl")
    
    # 如果子集已存在且大小正确，直接返回
    if os.path.exists(subset_path):
        with open(subset_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        if line_count == subset_size:
            logger.info(f"Test subset already exists at {subset_path} with {subset_size} samples")
            return subset_path
        else:
            logger.info(f"Existing subset has {line_count} samples, creating new subset with {subset_size} samples")
    
    # 读取原始数据集
    logger.info(f"Creating test subset of size {subset_size} from {data_path}")
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= subset_size:
                break
            samples.append(json.loads(line))
    
    # 保存子集
    with open(subset_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"Created test subset at {subset_path} with {len(samples)} samples")
    return subset_path


def download_musique_data(download_dir: str = "data/musique") -> str:
    """
    Download MuSiQue dataset if it's not already present.
    
    Args:
        download_dir: Directory to save the dataset
    
    Returns:
        Path to the development dataset file
    """
    os.makedirs(download_dir, exist_ok=True)
    # 获取当前脚本的绝对路径
    current_file = os.path.abspath(__file__)

    # 计算上两级目录路径
    grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

    # Define paths
    dev_path = os.path.join(grandparent_dir+'/'+download_dir, "data/musique_ans_v1.0_dev.jsonl")
    
    # Check if files already exist
    if os.path.exists(dev_path):
        logger.info(f"MuSiQue dataset already exists at {dev_path}")
        return dev_path
    
    # If not, download the files
    import urllib.request
    
    logger.info("Downloading MuSiQue dataset...")
    
    # URLs for the dataset files
    dev_url = "https://github.com/StonyBrookNLP/musique/raw/main/data/musique_ans_v1.0_dev.jsonl"
    
    # Download files
    try:
        urllib.request.urlretrieve(dev_url, dev_path)
        logger.info(f"Downloaded development dataset to {dev_path}")
        return dev_path
        
    except Exception as e:
        logger.error(f"Failed to download MuSiQue dataset: {e}")
        logger.info(
            "Please manually download the dataset from "
            "https://github.com/StonyBrookNLP/musique/tree/main/data "
            f"and place it in {download_dir}"
        )
        return dev_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate EcphoryRAG on MuSiQue dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset options
    parser.add_argument("--data-path", type=str, 
                       help="Path to MuSiQue dataset file. If not provided, will download or use default location.")
    parser.add_argument("--download-dir", type=str, default="data/musique",
                       help="Directory to download dataset if needed")
    parser.add_argument("--test-subset", default=True, action="store_true",
                       help="Use a small test subset instead of the full dataset")
    parser.add_argument("--subset-size", type=int, default=10,
                       help="Size of the test subset when --test-subset is used")
    
    # Workspace options
    parser.add_argument("--workspace-dir", type=str, default="evaluation_workspace",
                       help="Directory for storing EcphoryRAG workspace data")
    parser.add_argument("--force-reindex", action="store_true",
                       help="Force reindexing of all documents")
    parser.add_argument("--skip-index", action="store_true",
                       help="Skip indexing and use existing index for evaluation")
    
    # Model configuration
    parser.add_argument("--embedding-model", type=str, default="bge-m3",
                       help="Embedding model name")
    parser.add_argument("--extraction-model", type=str, default="phi4",
                       help="Entity extraction LLM model")
    parser.add_argument("--generation-model", type=str, default="phi4",
                       help="Answer generation LLM model")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434",
                       help="Ollama API host")
    
    # Retrieval parameters
    parser.add_argument("--top-k-initial", type=int, default=10,
                       help="Top-k value for initial trace retrieval")
    parser.add_argument("--top-k-final-values", type=str, default="1,3,5,10",
                       help="Comma-separated list of top-k values for final trace selection")
    parser.add_argument("--retrieval-depth", type=int, default=2,
                       help="Depth for secondary ecphory retrieval")
    
    # Chunking parameters
    parser.add_argument("--enable-chunking", action="store_true", default=False,
                       help="Enable document chunking")
    parser.add_argument("--chunk-size", type=int, default=1200,
                       help="Size of text chunks when chunking is enabled")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                       help="Overlap between consecutive chunks")
    
    # Evaluation options
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to evaluate. Default is all.")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--output-file", type=str, default="musique_evaluation.json",
                       help="Filename for evaluation results")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode for testing individual questions")
    
    return parser.parse_args()


def main():
    """Run MuSiQue evaluation on EcphoryRAG."""
    args = parse_args()
    
    # Get dataset path (download if needed)
    data_path = args.data_path
    if not data_path or not os.path.exists(data_path):
        data_path = download_musique_data(args.download_dir)
    
    if not os.path.exists(data_path):
        logger.error(f"Dataset file not found at {data_path}")
        return
    
    # 如果使用测试子集，创建或加载子集
    if args.test_subset:
        data_path = create_test_subset(data_path, args.subset_size)
        logger.info(f"Using test subset at: {data_path}")
    else:
        logger.info(f"Using full MuSiQue dataset at: {data_path}")
    
    # Initialize MuSiQue dataset loader
    dataset_loader = MusiqueDatasetLoader(data_path)
    
    # Parse top-k values
    top_k_final_values = [int(k) for k in args.top_k_final_values.split(',')]
    
    # Initialize EcphoryRAG with optimized parameters
    logger.info(f"Initializing EcphoryRAG with models: "
               f"{args.embedding_model} (embedding), "
               f"{args.extraction_model} (extraction), "
               f"{args.generation_model} (generation)")
    
    try:
        # Create workspace directory if it doesn't exist
        os.makedirs(args.workspace_dir, exist_ok=True)
        
        # 根据是否使用子集来设置工作区目录
        workspace_dir = args.workspace_dir
        if args.test_subset:
            workspace_dir = os.path.join(args.workspace_dir, "test_subset")
            os.makedirs(workspace_dir, exist_ok=True)
            logger.info(f"Using subset-specific workspace at: {workspace_dir}")
        
        rag_system = EcphoryRAG(
            workspace_path=workspace_dir,
            embedding_model_name=args.embedding_model,
            extraction_llm_model=args.extraction_model,
            generation_llm_model=args.generation_model,
            ollama_host=args.ollama_host,
            enable_chunking=args.enable_chunking,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        # Customize retrieval parameters
        rag_system.top_k_initial = args.top_k_initial
        rag_system.retrieval_depth = args.retrieval_depth
        
        # 如果不是跳过索引模式，则进行索引
        if not args.skip_index:
            # 加载数据集
            data = dataset_loader.load_data()
            logger.info(f"Loaded {len(data)} datapoints for indexing")
            
            # 准备文档
            documents = []
            for item in data:
                # 将问题和答案组合成文档
                doc_text = f"Question: {item['question']}\nAnswer: {item['ground_truth_answer']}"
                if 'ground_truth_aliases' in item:
                    doc_text += "\nAlternative answers: " + ", ".join(item['ground_truth_aliases'])
                documents.append({
                    'id': item.get('id', str(uuid.uuid4())),
                    'text': doc_text
                })
            
            # 索引文档
            logger.info("Indexing documents...")
            num_entities, num_chunks = rag_system.index_documents(
                documents,
                force_reindex_all=args.force_reindex
            )
            logger.info(f"Indexed {num_entities} entities and {num_chunks} chunks")
        
        # 如果是交互模式，运行交互式问答
        if args.interactive:
            print("\n" + "="*50)
            print("交互式问答模式 (输入 'q' 退出)")
            print("="*50)
            
            while True:
                question = input("\n请输入问题: ").strip()
                if question.lower() == 'q':
                    break
                    
                if not question:
                    continue
                    
                try:
                    # 获取答案
                    answer, sources = rag_system.query(question)
                    
                    # 打印答案
                    print("\n答案:")
                    print("-"*30)
                    print(answer)
                    
                    # 打印来源
                    if sources:
                        print("\n来源:")
                        print("-"*30)
                        for i, source in enumerate(sources, 1):
                            print(f"{i}. {source}")
                            
                except Exception as e:
                    logger.error(f"处理问题时出错: {e}")
                    continue
                    
            return
        
        # 否则运行评估
        evaluator = Evaluator(
            rag_system=rag_system,
            dataset_loader=dataset_loader,
            output_dir=args.output_dir
        )
        
        # Run evaluation
        start_time = time.time()
        logger.info(f"Starting evaluation on {args.num_samples or 'all'} samples...")
        
        results = evaluator.run_evaluation(
            num_samples=args.num_samples,
            output_file=args.output_file,
            top_k_values=top_k_final_values,
            skip_indexing=args.skip_index
        )
        
        total_time = time.time() - start_time
        logger.info(f"Evaluation completed in {total_time:.2f} seconds")
        
        # Print results summary
        print("\n" + "="*50)
        print("EVALUATION RESULTS SUMMARY")
        print("="*50)
        
        try:
            # Use tabulate for nicer display if available
            print_results_table(results, title="MuSiQue Evaluation Results")
        except ImportError:
            # Fallback to simple printing
            for top_k, metrics in results.items():
                print(f"\n{top_k}:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
        
        # Print workspace and knowledge graph statistics
        print("\n" + "="*50)
        print("WORKSPACE STATISTICS")
        print("="*50)
        print(f"Workspace directory: {args.workspace_dir}")
        print(f"Indexed documents: {len(rag_system.indexed_doc_manifest)}")
        print(f"Knowledge graph nodes: {rag_system.graph.number_of_nodes()}")
        print(f"Knowledge graph edges: {rag_system.graph.number_of_edges()}")
        
        # Inform user where results are saved
        output_path = os.path.join(args.output_dir, args.output_file)
        print("\nDetailed evaluation results saved to:")
        print(f"  - Summary: {output_path}")
        for top_k in top_k_final_values:
            detailed_path = os.path.join(args.output_dir, 
                                       f"{args.output_file.replace('.json', '')}_top_k_{top_k}.json")
            print(f"  - top_k={top_k} details: {detailed_path}")
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please make sure the EcphoryRAG package is properly installed.")
        logger.info("You can install it with: pip install -e .")
        return
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return


if __name__ == "__main__":
    main() 