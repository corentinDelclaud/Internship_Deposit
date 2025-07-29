import json

# Input and output file paths
json_path = "questions2WikiMultihopQA_structured.json"
txt_path = "questions2WikiMultihopQA_contexts.txt"

# Read the JSON file
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract contexts and write to txt, one per line
with open(txt_path, "w", encoding="utf-8") as f:
    for item in data:
        context = item.get("context", "").replace("\n", " ").strip()
        if context:
            f.write(context + "\n")
