# Code for AAAI 2026 Submission ID 10766

This repository contains the implementation of EcphoryRAG, a novel retrieval-augmented generation system that leverages entity-based memory traces inspired by neurocognitive memory models. The system demonstrates improved performance on multi-hop question answering tasks through sophisticated entity extraction and retrieval mechanisms.

## Features

- **Entity-based Retrieval**: Extracts and stores entities as memory traces for enhanced retrieval
- **Hybrid Retrieval**: Combines entity-based and chunk-based retrieval approaches
- **Multi-hop QA Support**: Optimized for complex reasoning tasks requiring multiple information sources
- **Knowledge Graph Integration**: Maintains entity relationships and connections
- **Configurable Retrieval**: Supports both pure entity retrieval and hybrid retrieval modes

## Installation

### Prerequisites

- Python 3.10
- Ollama (for LLM inference)
- Required Python packages (see requirements.txt)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd EcphoryRAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama and download required models:
```bash
# Install Ollama (follow instructions at https://ollama.ai)
# Download required models
ollama pull bge-m3
ollama pull phi4
```

## Dataset Preparation

### Download Datasets

1. **HotpotQA**:
```bash
mkdir -p data/hotpotqa
wget https://hotpotqa.github.io/hotpotqa_dev_v1.1.json -O data/hotpotqa/hotpotqa_dev.json
```

2. **MuSiQue**:
```bash
mkdir -p data/musique
wget https://github.com/microsoft/MuSiQue/raw/main/data/musique_ans_v1.0_dev.jsonl -O data/musique/musique_ans_v1.0_dev.jsonl
```

3. **2WikiMultiHop**:
```bash
mkdir -p data/2wiki
# Download from https://github.com/yao8839836/2wikimultihop
# Place raw_data.json in data/2wiki/
```

### Process Datasets

Run the data processing scripts to convert raw datasets into the required format:

```bash
# Process HotpotQA
python scripts/process_hotpotqa_data.py \
    --raw_data_file data/hotpotqa/hotpotqa_dev.json \
    --output_dir data/processed_hotpotqa \
    --split_name dev

# Process MuSiQue
python scripts/process_musique_data.py \
    --raw_data_file data/musique/musique_ans_v1.0_dev.jsonl \
    --output_dir data/processed_musique \
    --split_name dev

# Process 2WikiMultiHop
python scripts/process_2wiki_data.py \
    --input_file data/2wiki/raw_data.json \
    --output_dir data/processed_2wiki
```

## Usage

### Basic Evaluation

Run evaluations on the processed datasets:

```bash
# HotpotQA evaluation (entity-based retrieval)
python scripts/run_hotpotqa_evaluation.py \
    --data-path data/processed_hotpotqa/dev_processed.jsonl \
    --output-dir results/hotpotqa_entity \
    --num-samples 100

# HotpotQA evaluation (hybrid retrieval)
python scripts/run_hotpotqa_evaluation.py \
    --data-path data/processed_hotpotqa/dev_processed.jsonl \
    --output-dir results/hotpotqa_hybrid \
    --enable-hybrid-retrieval \
    --num-samples 100

# MuSiQue evaluation
python scripts/run_musique_evaluation.py \
    --data-path data/processed_musique/dev_processed.jsonl \
    --output-dir results/musique \
    --num-samples 100

# 2WikiMultiHop evaluation
python scripts/run_2wiki_evaluation.py \
    --data-path data/processed_2wiki/2wiki_processed.jsonl \
    --output-dir results/2wiki \
    --num-samples 100
```

### Interactive Mode

Test individual questions interactively:

```bash
python scripts/run_hotpotqa_evaluation.py \
    --data-path data/processed_hotpotqa/dev_processed.jsonl \
    --interactive
```

### Key Parameters

- `--enable-hybrid-retrieval`: Enable hybrid retrieval (combines entity and chunk retrieval)
- `--top-k-final-values`: Comma-separated list of top-k values for evaluation
- `--retrieval-depth`: Depth for secondary ecphory retrieval
- `--enable-chunking`: Enable document chunking
- `--num-samples`: Number of samples to evaluate

## Model Configuration

The system supports various model configurations:

- **Embedding Model**: bge-m3 (default)
- **Entity Extraction**: phi4 (default)
- **Answer Generation**: phi4 (default)

You can modify these in the evaluation scripts or by passing command-line arguments.

## Results

Evaluation results are saved in the specified output directory with the following structure:

```
results/
├── hotpotqa_entity/
│   ├── hotpotqa_evaluation.json
│   ├── usage_stats.json
│   └── indexing_stats.json
├── hotpotqa_hybrid/
│   ├── hotpotqa_evaluation.json
│   ├── usage_stats.json
│   └── indexing_stats.json
└── ...
```

## Reproducibility

To ensure reproducible results:

1. Set random seeds in the evaluation scripts
2. Use the same model versions and configurations
3. Run multiple experiments with different seeds for statistical significance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

We thank the authors of HotpotQA, MuSiQue, and 2WikiMultiHop for providing the datasets used in this work. 
