import json

input_path = "4(Evaluation-KPIs)/example/questionsIIRC_structured.json"

def extract_answer(ans):
    if not isinstance(ans, dict):
        return str(ans)
    t = ans.get("type")
    if t == "none":
        return ""
    elif t == "span":
        spans = ans.get("answer_spans", [])
        return " ".join(span.get("text", "") for span in spans)
    elif t == "value":
        val = str(ans.get("answer_value", ""))
        unit = ans.get("answer_unit", "")
        return f"{val} {unit}".strip() if unit else val
    elif t == "binary":
        return str(ans.get("answer_value", ""))
    else:
        return str(ans)

with open(input_path, "r", encoding="utf-8") as f:
    questions = json.load(f)

for q in questions:
    q["answer"] = extract_answer(q.get("answer", ""))

with open(input_path, "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)

print(f"Answers flattened and file overwritten: {input_path}")