# Evaluate System Performance

## Overview

This module provides a comprehensive performance evaluation system for RAG (Retrieval-Augmented Generation) systems. It implements various automatic metrics and offers comparative evaluation capabilities using LLMs.

## Features

### Automatic Metrics
- **Exact Match (EM)** - Exact answer matching
- **F1 Score** - Token-based F1 score
- **BLEU Score** - Translation quality evaluation
- **ROUGE Score** - Automatic summarization evaluation
- **BERTScore** - BERT-based semantic similarity

### Comparative Evaluation
- Comparison between two RAG systems
- LLM-based evaluation (Llama, etc.)
- Qualitative response analysis

## Installation

```bash
pip install -r requirements.txt
```

### Main Dependencies
- `transformers` - For language models
- `bert-score` - For BERTScore evaluation
- `nltk` - For BLEU metrics
- `rouge-score` - For ROUGE metrics
- `torch` - Deep learning framework

## Usage

### Basic Automatic Evaluation

```bash
python generalRAGevaluator.py --predictions example/predictions.json --references example/references.json
```

### Evaluation with Limited Examples

```bash
python generalRAGevaluator.py --predictions example/fakepredictionsquestions_2WikiMultihopQA_structured.json --references example/questions2WikiMultihopQA_structured.json --max_eval 500
```

### Comparative Evaluation Between Two RAG Systems

```bash
python generalRAGevaluator.py --predictions example/predictions.json --references example/references.json --comparative --other_predictions example/other_predictions.json --model_name meta-llama/Llama-3.2-1B-Instruct
```

### Example Commands for Different Datasets

#### 2WikiMultihopQA
```bash
python generalRAGevaluator.py --predictions example/fakepredictionsquestions_2WikiMultihopQA_structured.json --references example/questions2WikiMultihopQA_structured.json --max_eval 500
```

#### IIRC Dataset
```bash
python generalRAGevaluator.py --predictions example/fakepredictionsquestions_questionsIIRC_structured.json --references example/questionsIIRC_structured.json --max_eval 500
```

#### StrategyQA Dataset
```bash
python generalRAGevaluator.py --predictions example/fakepredictionsquestions_questionsStrategyQA_structured.json --references example/questionsStrategyQA_structured.json --max_eval 500
```

## File Structure

```
evaluate_system_performance/
├── generalRAGevaluator.py      # Main evaluation script
├── cleaning_log.py             # Log cleaning
├── extract.py                  # Data extraction
├── fake_answer.py              # Fake answer generation
├── evaluation_results.json     # Evaluation results
├── evaluation_run.log          # Execution logs
├── requirements.txt            # Python dependencies
├── example/                    # Example files
├── logs/                       # Logs directory
└── README.md                   # This documentation
```

## Data Format

### Predictions File
```json
[
    {
        "question": "What is the capital of France?",
        "predicted_answer": "Paris",
        "context": "France is a country in Europe..."
    }
]
```

### References File
```json
[
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "context": "France is a country in Europe..."
    }
]
```

## Output Metrics

The system generates a comprehensive report including:
- **Global scores** for each metric
- **Detailed analysis** per question
- **Distribution statistics** of scores
- **Detailed logs** for debugging

## Configuration

### Main Parameters
- `--max_eval` : Maximum number of examples to evaluate
- `--model_name` : LLM model for comparative evaluation
- `--comparative` : Enable comparative evaluation
- `--other_predictions` : Second system predictions file

### Logging
The system automatically generates:
- `evaluation_run.log` : Detailed execution logs
- `evaluation_results.json` : Results in JSON format

## Utility Scripts

- **cleaning_log.py** : Cleans and formats evaluation logs
- **extract.py** : Extracts specific data from results
- **fake_answer.py** : Generates fake answers for testing

## Important Notes

- Ensure predictions and references files have the same number of entries
- LLM models require appropriate GPU configuration
- Some metrics may be computationally expensive

## Troubleshooting

1. **Memory errors** : Reduce `--max_eval`
2. **Model not found** : Check model name and internet connection
3. **Invalid JSON format** : Validate your input files

