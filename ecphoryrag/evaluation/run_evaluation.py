#!/usr/bin/env python
"""
EcphoryRAG评估运行脚本

提供命令行接口，用于运行EcphoryRAG系统的评估测试。
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from ecphoryrag import EcphoryRAG
from ecphoryrag.evaluation.datasets import MusiqueDatasetLoader
from ecphoryrag.evaluation.evaluator import Evaluator

# 配置日志
logger = logging.getLogger(__name__)


def setup_logging(log_level: str, log_file: Optional[str] = None) -> None:
    """
    配置日志记录。
    
    Args:
        log_level: 日志级别 ('debug', 'info', 'warning', 'error', 'critical')
        log_file: 日志文件路径（可选）
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'无效的日志级别: {log_level}')
    
    handlers = []
    # 添加控制台处理器
    handlers.append(logging.StreamHandler())
    
    # 如果提供了日志文件，添加文件处理器
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    # 配置根日志记录器
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数。
    
    Returns:
        argparse.Namespace: 包含解析后参数的命名空间对象
    """
    parser = argparse.ArgumentParser(
        description='运行EcphoryRAG系统评估',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据集参数
    parser.add_argument('--dataset', required=True, help='数据集文件路径 (JSONL格式)')
    parser.add_argument('--dataset-type', default='musique', choices=['musique'], 
                       help='数据集类型')
    parser.add_argument('--aliases', help='答案别名文件路径 (JSON格式)')
    
    # 系统配置参数
    parser.add_argument('--embedding-model', default='bge-m3', 
                       help='嵌入模型名称')
    parser.add_argument('--extraction-model', default='phi3', 
                       help='实体提取LLM模型名称')
    parser.add_argument('--generation-model', default='phi3', 
                       help='生成LLM模型名称')
    parser.add_argument('--ollama-host', default='http://localhost:11434', 
                       help='Ollama服务器地址')
    
    # 评估参数
    parser.add_argument('--top-k', type=lambda s: [int(item) for item in s.split(',')],
                       default=[1, 3, 5, 10], 
                       help='要评估的top-k值列表，用逗号分隔')
    parser.add_argument('--output-dir', default='evaluation_results', 
                       help='评估结果输出目录')
    parser.add_argument('--experiment-name', 
                       help='实验名称 (默认为自动生成)')
    
    # 执行参数
    parser.add_argument('--force-reindex', action='store_true', 
                       help='强制重新索引文档')
    parser.add_argument('--skip-indexing', action='store_true', 
                       help='跳过文档索引阶段')
    parser.add_argument('--log-level', default='info', 
                       choices=['debug', 'info', 'warning', 'error', 'critical'],
                       help='日志级别')
    parser.add_argument('--log-file', help='日志文件路径')
    
    return parser.parse_args()


def run_evaluation(args: argparse.Namespace) -> Dict:
    """
    使用指定参数运行评估。
    
    Args:
        args: 命令行参数
    
    Returns:
        Dict: 评估结果
    """
    logger.info("初始化评估...")
    
    # 设置数据集加载器
    if args.dataset_type == 'musique':
        dataset_loader = MusiqueDatasetLoader(args.dataset, args.aliases)
    else:
        raise ValueError(f"不支持的数据集类型: {args.dataset_type}")
    
    # 初始化EcphoryRAG系统
    logger.info(f"使用模型: {args.embedding_model} (嵌入), "
               f"{args.extraction_model} (提取), {args.generation_model} (生成)")
    
    rag_system = EcphoryRAG(
        embedding_model_name=args.embedding_model,
        extraction_llm_model=args.extraction_model,
        generation_llm_model=args.generation_model,
        ollama_host=args.ollama_host
    )
    
    # 初始化评估器
    evaluator = Evaluator(
        rag_system=rag_system,
        dataset_loader=dataset_loader,
        output_dir=args.output_dir
    )
    
    # 索引文档
    if not args.skip_indexing:
        logger.info("开始索引文档...")
        start_time = time.time()
        evaluator.index_documents(force_reindex=args.force_reindex)
        elapsed_time = time.time() - start_time
        logger.info(f"文档索引完成，耗时: {elapsed_time:.2f}秒")
    else:
        logger.info("跳过文档索引阶段")
    
    # 运行评估
    logger.info(f"开始评估 (top-k值: {args.top_k})...")
    start_time = time.time()
    
    results = evaluator.run_evaluation(
        experiment_name=args.experiment_name,
        top_k_values=args.top_k
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"评估完成，耗时: {elapsed_time:.2f}秒")
    
    # 输出摘要
    print("\n评估结果摘要:")
    for top_k, metrics in results.items():
        print(f"\n{top_k}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    return results


def main() -> None:
    """主函数"""
    args = parse_arguments()
    
    # 设置日志
    setup_logging(args.log_level, args.log_file)
    
    try:
        # 运行评估
        run_evaluation(args)
    except KeyboardInterrupt:
        logger.info("评估已被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"评估过程中出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 