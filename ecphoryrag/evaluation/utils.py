"""
EcphoryRAG评估工具模块

提供用于可视化评估结果和比较实验的辅助函数。
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

# 配置日志
logger = logging.getLogger(__name__)


def load_results(result_path: Union[str, Path]) -> Dict[str, Any]:
    """
    从文件加载评估结果。
    
    Args:
        result_path: 结果文件路径
        
    Returns:
        Dict[str, Any]: 加载的评估结果
    """
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载结果文件时出错: {e}")
        return {}


def results_to_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """
    将评估结果转换为Pandas DataFrame。
    
    Args:
        results: 评估结果字典
        
    Returns:
        pd.DataFrame: 包含评估结果的DataFrame
    """
    data = []
    
    for top_k, metrics in results.items():
        row = {"top_k": top_k}
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                row[metric] = value
        data.append(row)
    
    return pd.DataFrame(data)


def compare_results(
    results_dict: Dict[str, Dict[str, Any]],
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    比较多个实验的结果。
    
    Args:
        results_dict: 包含多个实验结果的字典，格式为 {"experiment_name": results, ...}
        metrics: 要比较的评估指标列表，默认为 ["exact_match", "f1", "rouge_1"]
        
    Returns:
        pd.DataFrame: 包含比较结果的DataFrame
    """
    if not metrics:
        metrics = ["exact_match", "f1", "rouge_1"]
    
    data = []
    
    for exp_name, exp_results in results_dict.items():
        for top_k, top_k_metrics in exp_results.items():
            row = {"experiment": exp_name, "top_k": top_k}
            
            for metric in metrics:
                if metric in top_k_metrics:
                    row[metric] = top_k_metrics[metric]
            
            data.append(row)
    
    return pd.DataFrame(data)


def print_results_table(
    results: Dict[str, Any],
    metrics: Optional[List[str]] = None,
    title: Optional[str] = None
) -> None:
    """
    以表格形式打印评估结果。
    
    Args:
        results: 评估结果字典
        metrics: 要显示的评估指标列表
        title: 表格标题
    """
    if not metrics:
        metrics = ["exact_match", "f1", "rouge_1", "rouge_2", "rouge_l"]
    
    # 准备表格数据
    headers = ["top_k"] + metrics
    table_data = []
    
    for top_k, top_k_metrics in results.items():
        row = [top_k]
        for metric in metrics:
            if metric in top_k_metrics and isinstance(top_k_metrics[metric], (int, float)):
                row.append(f"{top_k_metrics[metric]:.4f}")
            else:
                row.append("N/A")
        table_data.append(row)
    
    # 打印表格
    if title:
        print(f"\n{title}")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def plot_metrics(
    results: Dict[str, Any],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    绘制评估指标图表。
    
    Args:
        results: 评估结果字典
        metrics: 要显示的评估指标列表
        figsize: 图表尺寸
        title: 图表标题
        save_path: 保存图表的文件路径
    """
    if not metrics:
        metrics = ["exact_match", "f1", "rouge_1"]
    
    df = results_to_dataframe(results)
    
    # 提取top_k值
    top_k_values = [k.replace("top_k_", "") for k in results.keys()]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    for metric in metrics:
        if metric in df.columns:
            values = [results[k][metric] for k in results.keys() if metric in results[k]]
            ax.plot(top_k_values, values, marker='o', label=metric)
    
    # 设置图表属性
    ax.set_xlabel('Top K值')
    ax.set_ylabel('分数')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('评估指标对比')
    ax.legend()
    ax.grid(True)
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_experiments_plot(
    results_dict: Dict[str, Dict[str, Any]],
    metric: str = "exact_match",
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    绘制多个实验结果的对比图表。
    
    Args:
        results_dict: 包含多个实验结果的字典
        metric: 要比较的评估指标
        figsize: 图表尺寸
        title: 图表标题
        save_path: 保存图表的文件路径
    """
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    for exp_name, results in results_dict.items():
        # 提取top_k值和对应的指标值
        top_k_values = [k.replace("top_k_", "") for k in results.keys()]
        metric_values = [results[k][metric] for k in results.keys() if metric in results[k]]
        
        # 绘制线图
        ax.plot(top_k_values, metric_values, marker='o', label=exp_name)
    
    # 设置图表属性
    ax.set_xlabel('Top K值')
    ax.set_ylabel(f'{metric}分数')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'不同实验的{metric}对比')
    ax.legend()
    ax.grid(True)
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def analyze_predictions(
    predictions_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    分析预测结果，生成详细报告。
    
    Args:
        predictions_path: 预测结果文件路径
        output_dir: 保存分析报告的目录
        
    Returns:
        Dict[str, Any]: 分析结果
    """
    try:
        # 加载预测
        with open(predictions_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # 提取实验名称和top_k
        pred_file = Path(predictions_path).stem
        parts = pred_file.split('_')
        top_k = parts[-1] if 'top_k' in pred_file else 'unknown'
        experiment = '_'.join(parts[0:-1]) if 'top_k' in pred_file else pred_file
        
        # 分析结果
        analysis = {
            "experiment": experiment,
            "top_k": top_k,
            "total_questions": len(predictions),
            "exact_match_count": sum(1 for p in predictions if p.get("exact_match", False)),
            "avg_f1": np.mean([p.get("f1", 0) for p in predictions]),
        }
        
        # 按F1分数对预测排序
        sorted_by_f1 = sorted(predictions, key=lambda x: x.get("f1", 0))
        
        # 找出最好和最差的预测
        analysis["worst_predictions"] = sorted_by_f1[:5]
        analysis["best_predictions"] = sorted_by_f1[-5:]
        
        # 将分析结果保存到文件
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            analysis_path = output_dir / f"{experiment}_{top_k}_analysis.json"
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            logger.info(f"分析结果已保存至 {analysis_path}")
        
        return analysis
    
    except Exception as e:
        logger.error(f"分析预测结果时出错: {e}")
        return {}


# 示例用法
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 示例：加载和可视化实验结果
    try:
        # 模拟评估结果
        example_results = {
            "top_k_1": {"exact_match": 0.42, "f1": 0.65, "rouge_1": 0.58},
            "top_k_3": {"exact_match": 0.48, "f1": 0.71, "rouge_1": 0.63},
            "top_k_5": {"exact_match": 0.51, "f1": 0.75, "rouge_1": 0.67},
            "top_k_10": {"exact_match": 0.53, "f1": 0.77, "rouge_1": 0.69}
        }
        
        # 打印表格形式的结果
        print_results_table(example_results, title="示例评估结果")
        
        # 绘制图表
        plot_metrics(example_results, title="示例评估结果图表")
        
        # 比较多个实验的结果
        example_experiments = {
            "实验A": example_results,
            "实验B": {
                "top_k_1": {"exact_match": 0.45, "f1": 0.68, "rouge_1": 0.60},
                "top_k_3": {"exact_match": 0.50, "f1": 0.73, "rouge_1": 0.65},
                "top_k_5": {"exact_match": 0.54, "f1": 0.78, "rouge_1": 0.70},
                "top_k_10": {"exact_match": 0.56, "f1": 0.80, "rouge_1": 0.72}
            }
        }
        
        # 对比图表
        compare_experiments_plot(example_experiments, metric="f1", title="实验F1分数对比")
        
    except Exception as e:
        logger.error(f"运行示例时出错: {e}") 