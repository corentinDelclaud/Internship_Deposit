# GH-Retriever - Graph-Hybrid Retrieval System

## Overview

GH-Retriever is an innovative hybrid retrieval system that combines traditional textual representations with knowledge graph structures. This model uses an adapted transformer architecture to simultaneously process textual and graph information.

## Features

### Hybrid Architecture
- **Textual Retrieval** - Processing of classic textual documents
- **Graph Retrieval** - Exploitation of knowledge graph relationships
- **Adaptive Fusion** - Optimal combination of both modalities

### Supported Models
- **Llama-based architectures** - Llama family models
- **Graph Neural Networks** - Neural networks for graphs
- **Hybrid Transformers** - Multi-modal adapted architecture

## Installation

```bash
pip install torch transformers datasets
pip install torch-geometric  # For graphs
pip install wandb  # For experiment tracking
```

### Environment Variables Configuration

Create a `.env` file with:
```bash
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
```

## Usage

### Model Training

```bash
python train.py --config configs/default.yaml
```

### Inference

```bash
python inference.py --model_path path/to/trained/model --query "your query here"
```

### Quick Launch Script

```bash
bash run.sh
```

## Project Structure

```
GH-retriever/
├── src/
│   ├── model.py              # Model architecture
│   ├── dataset.py            # Data management
│   ├── config.py             # Configuration and arguments
│   └── utils/
│       ├── evaluate.py       # Evaluation functions
│       ├── ckpt.py          # Checkpoint management
│       ├── collate.py       # Batch functions
│       ├── seed.py          # Reproducibility management
│       └── lr_schedule.py   # Learning rate scheduling
├── dataset/                  # Training data
├── train.py                  # Main training script
├── inference.py              # Inference script
├── run.sh                    # Launch script
├── .env                      # Environment variables
└── README.md                 # This documentation
```

## Configuration

### Main Training Arguments

```python
--project: WandB project name
--seed: Seed for reproducibility
--num_epochs: Number of training epochs
--learning_rate: Learning rate
--batch_size: Batch size
--max_txt_len: Maximum text length
--max_new_tokens: Maximum number of new tokens
--gnn_model_name: GNN model name
--llm_model_name: LLM model name
--patience: Patience for early stopping
```

### Model Architecture

The GH-Retriever model combines:
1. **Text Encoder** - Transformers for text
2. **Graph Encoder** - GNN for structures
3. **Fusion Module** - Cross-attention between modalities
4. **Unified Decoder** - Final response generation

## Supported Datasets

The system is designed to work with:
- **Knowledge Graphs** (KG) in RDF/JSON format
- **Structured text corpora**
- **Question-answer data** with graph context

### Data Format

```json
{
    "text": "Textual document...",
    "graph": {
        "nodes": [...],
        "edges": [...],
        "features": [...]
    },
    "question": "Asked question",
    "answer": "Expected answer"
}
```

## Training

### Training Process

1. **Data Preparation** - Loading and preprocessing
2. **Model Initialization** - Hybrid configuration
3. **Training Loop** - Multi-modal optimization
4. **Evaluation** - Performance metrics
5. **Saving** - Checkpoints and final model

### Evaluation Metrics

- **Accuracy** - Response accuracy
- **F1 Score** - Average F1 score
- **BLEU/ROUGE** - Generation quality
- **Graph-aware metrics** - Graph-specific metrics

## Advanced Features

### Optimization Techniques
- **Gradient clipping** - Training stability
- **Learning rate scheduling** - Dynamic adaptation
- **Early stopping** - Overfitting prevention
- **Mixed precision** - Memory optimization

### Checkpoint Management
- Automatic best model saving
- Training resumption from checkpoint
- Model versioning with WandB

## Monitoring and Logging

### WandB Integration
- Real-time metric tracking
- Learning curve visualization
- Experiment comparison
- Artifact saving

### Local Logs
- Detailed training logs
- Evaluation metrics
- Performance diagnostics

## Performance Optimization

### GPU Recommendations
- **Minimum** : RTX 3080 (10GB VRAM)
- **Recommended** : RTX 4090 (24GB VRAM)
- **Optimal** : A100 (40GB VRAM)

### Memory Optimizations
```python
# Gradient checkpointing
model.gradient_checkpointing_enable()

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
```

## Utility Scripts

### run.sh
Launch script with default configuration:
```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train.py \
    --project GH-Retriever \
    --num_epochs 10 \
    --learning_rate 2e-5 \
    --batch_size 4
```

## Troubleshooting

### Common Issues

1. **OutOfMemoryError**
   - Reduce `batch_size`
   - Use `gradient_checkpointing`
   - Reduce `max_txt_len`

2. **Convergence errors**
   - Adjust `learning_rate`
   - Check data quality
   - Increase `patience`

3. **Reproducibility issues**
   - Fix `seed` in all scripts
   - Use `torch.backends.cudnn.deterministic = True`

## Contributing

To contribute to the project:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

## References

- Architecture based on hybrid transformers
- Graph Neural Network techniques
- Multi-modal retrieval methods
