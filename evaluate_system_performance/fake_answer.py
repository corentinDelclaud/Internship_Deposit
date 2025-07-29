import json

# Path to your structured questions file (e.g., questions2WikiMultihopQA_structured.json)
questions_path = "4(Evaluation-KPIs)/example/questionsStrategyQA_structured.json"
fake_predictions_path = "4(Evaluation-KPIs)/example/fakepredictionsquestions_questionsStrategyQA_structured.json"

with open(questions_path, "r", encoding="utf-8") as f:
    questions = json.load(f)

# Each prediction is a dict: {"answer": ...}
fake_predictions = [{"answer": q.get("answer", "")} for q in questions]

with open(fake_predictions_path, "w", encoding="utf-8") as f:
    json.dump(fake_predictions, f, indent=2, ensure_ascii=False)

print(f"Fake predictions written to {fake_predictions_path}")