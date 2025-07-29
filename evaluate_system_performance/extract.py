import json

# -------- 2WikiMultihopQA Extraction --------
input_path = "data/2WikiMultihopQA/dev.json"
questions_structured_path = "4(Evaluation-KPIs)/example/questions2WikiMultihopQA_structured.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

questions_structured = []
for idx, item in enumerate(data):
    # Concatenate all context sentences from all passages
    context = " ".join(
        sentence
        for passage in item.get("context", [])
        for sentence in passage[1]
    )
    questions_structured.append({
        "question": item.get("question", ""),
        "context": context,
        "answer": item.get("answer", ""),
        # "prediction": predictions[idx] if idx < len(predictions) else ""
    })

with open(questions_structured_path, "w", encoding="utf-8") as f:
    json.dump(questions_structured, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(questions_structured)} structured questions to {questions_structured_path}")

# -------- StrategyQA Extraction --------
input_path = "data/strategyqa/strategyqa_train.json"
output_path = "4(Evaluation-KPIs)/example/questionsStrategyQA_structured.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

def flatten(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

questions_structured = []
for item in data:
    evidence = item.get("evidence", [])
    flat_evidence = list(flatten(evidence))
    context = " ".join(flat_evidence)
    questions_structured.append({
        "question": item.get("question", ""),
        "context": context,
        "answer": str(item.get("answer", "")),  # Convert boolean to string if needed
        # "prediction": ""  # Add this if you have predictions
    })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(questions_structured, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(questions_structured)} structured questions to {output_path}")

# -------- IIRC Extraction --------
input_path = "data/iirc/iirc_train_dev/dev.json"
output_path = "4(Evaluation-KPIs)/example/questionsIIRC_structured.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

questions_structured = []
for passage in data:
    context = passage.get("text", "")
    for q in passage.get("questions", []):
        questions_structured.append({
            "question": q.get("question", ""),
            "context": context,
            "answer": q.get("answer", ""),
            # "prediction": ""  # Add this if you have predictions
        })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(questions_structured, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(questions_structured)} structured questions to {output_path}")