import argparse
import json
import numpy as np
import re
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm
import logging
from datetime import datetime
import os
import warnings

import transformers
transformers.logging.set_verbosity_error()  # <--- SUPPRESS ALL HF LOGS

logfile = "evaluation_run.log"
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("transformers").propagate = False  # <--- DO NOT PROPAGATE TO ROOT LOGGER
logging.getLogger("tokenizer").propagate = False  # <--- DO NOT PROPAGATE TO ROOT LOGGER

warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized*")

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_em(
    prediction: str,
    ground_truth: str,
    ignore_case: bool = True,
    ignore_punctuation: bool = True,
    ignore_numbers: bool = False,
    regexes_to_ignore: Optional[List[str]] = None
) -> float:
    def process(text: str) -> str:
        if ignore_case:
            text = text.lower()
        if ignore_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        if ignore_numbers:
            text = re.sub(r'\d+', '', text)
        if regexes_to_ignore:
            for pattern in regexes_to_ignore:
                text = re.sub(pattern, '', text)
        text = ' '.join(text.split())
        return text
    return float(process(prediction) == process(ground_truth))

def compute_f1(prediction: str, ground_truth: str) -> float:
    from collections import Counter
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    common = pred_counter & gt_counter
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def compute_bleu(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.strip().split()
    gt_tokens = ground_truth.strip().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    smoothie = SmoothingFunction().method4
    return sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoothie)

def compute_rouge(prediction: str, ground_truth: str) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    return {k: v.fmeasure for k, v in scores.items()}

def compute_bertscore(prediction: str, ground_truth: str, lang: str = 'en') -> float:
    P, R, F1 = bert_score([prediction], [ground_truth], lang=lang, rescale_with_baseline=True, model_type='roberta-base')
    return float(F1[0])

def evaluate_automatic(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    ems, f1s, bleus, rouges, bertscores = [], [], [], [], []
    for pred, gt in tqdm(zip(predictions, ground_truths), total=len(predictions), desc="Evaluation"):
        ems.append(compute_em(pred, gt))
        f1s.append(compute_f1(pred, gt))
        bleus.append(compute_bleu(pred, gt))
        rouge_dict = compute_rouge(pred, gt)
        rouges.append(rouge_dict)
        bertscores.append(compute_bertscore(pred, gt))
    avg_rouge = {k: np.mean([r[k] for r in rouges]) for k in rouges[0]} if rouges else {}
    # Filter out BERTScore values that are exactly 0
    filtered_bertscores = [b for b in bertscores if b != 0]
    return {
        "EM": np.mean(ems),
        "F1": np.mean(f1s),
        "BLEU": np.mean(bleus),
        **{f"ROUGE-{k.upper()}": v for k, v in avg_rouge.items()},
        "BERTScore": np.mean(filtered_bertscores) if filtered_bertscores else 0.0
    }

def extract_first_json(text):
    matches = list(re.finditer(r'\{.*?\}', text, re.DOTALL))
    for match in matches:
        try:
            return json.loads(match.group())
        except Exception:
            continue
    raise ValueError("No valid JSON object found in model output.")

def evaluate_llm_comparative(
    questions: List[str],
    contexts: List[str],
    answers_list: List[List[Dict[str, str]]],  # Each inner list: [{"name":..., "answer":...}, ...]
    model_name: str,
    device: str = "cuda"
) -> List[Dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    def build_prompt(question, context, answers):
        prompt = f"""Compare the following answers for the given question and context.\n"""
        prompt += "Evaluation Criteria:\n"
        prompt += "1. Internalization: Does the answer integrate knowledge, not just repeat context?\n"
        prompt += "2. Fluency: Is the answer well-structured and readable?\n"
        prompt += "3. Relevance: Is the answer on-topic and deep?\n"
        prompt += "4. EM: Does the answer match the ground truth?\n"
        prompt += "5. F1: Does the answer contain relevant information from the context?\n\n"
        prompt += f"Question: {question}\nContext: {context}\n"
        for idx, ans in enumerate(answers):
            prompt += f"Answer {ans['name']}: {ans['answer']}\n"
        prompt += ("\nRespond ONLY with a single JSON object in the following format, "
                   "listing the winner (by name) and the reason for your choice. "
                   "It's possible to have a tie. Also, provide EM and F1 scores for each answer as a dictionary.\n"
                   "{\n  'win model': <name or list of names>,\n  'reason': <reason>,\n  'EM': {<name>: <score>, ...},\n  'F1': {<name>: <score>, ...}\n}\n")
        return prompt

    results = []
    for q, c, answers in zip(questions, contexts, answers_list):
        prompt = build_prompt(q, c, answers)
        response = generator(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]
        try:
            result = extract_first_json(response)
        except Exception as e:
            result = {"win model": "Error", "reason": str(e)}
        results.append({
            "question": q,
            "context": c,
            "answers": answers,
            "evaluation": result
        })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True, help="Path to RAG predictions JSONL or JSON")
    parser.add_argument("--references", type=str, required=True, help="Path to ground truth answers JSONL or JSON")
    parser.add_argument("--comparative", action="store_true", help="If set, run LLM-based comparative evaluation")
    parser.add_argument("--other_predictions", type=str, help="Path to second RAG predictions for comparison")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model for LLM-based eval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    parser.add_argument("--max_eval", type=int, default=None, help="Maximum number of examples to evaluate")
    args = parser.parse_args()

    # Logging setup
    logfile = "evaluation_run.log"
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.getLogger("transformers").propagate = False
    log_msg = (
        f"predictions={os.path.abspath(args.predictions)} | "
        f"references={os.path.abspath(args.references)} | "
        f"max_eval={args.max_eval if args.max_eval is not None else 'all'}"
    )
    logging.info(log_msg)

    # Load predictions and references
    with open(args.predictions, encoding="utf-8") as f:
        preds = json.load(f)
    with open(args.references, encoding="utf-8") as f:
        refs = json.load(f)

    # Limit number of examples if requested
    if args.max_eval is not None:
        preds = preds[:args.max_eval]
        refs = refs[:args.max_eval]

    # Automatic metrics
    predictions = [p["answer"] if isinstance(p, dict) else p for p in preds]
    ground_truths = [r["answer"] if isinstance(r, dict) else r for r in refs]
    auto_metrics = evaluate_automatic(predictions, ground_truths)
    print("Automatic Metrics:", auto_metrics)
    logging.info(f"Automatic Metrics: {auto_metrics}")

    # LLM-based comparative evaluation
    if args.comparative and args.other_predictions:
        with open(args.other_predictions, encoding="utf-8") as f:
            other_preds = json.load(f)
        questions = [p["question"] if isinstance(p, dict) else "" for p in preds]
        contexts = [p.get("context", "") if isinstance(p, dict) else "" for p in preds]
        answers_list = [
            [{"name": "Model A", "answer": p["prediction"] if isinstance(p, dict) else p},
             {"name": "Model B", "answer": op["prediction"] if isinstance(op, dict) else op}]
            for p, op in zip(preds, other_preds)
        ]
        results = evaluate_llm_comparative(
            questions, contexts, answers_list, args.model_name, args.device
        )
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"LLM-based comparative evaluation saved to {args.output}")

if __name__ == "__main__":
    main()