input_log = "evaluation_run.log"
output_log = "logs/evaluation_run_cleaned.log"

# Liste des motifs à supprimer
patterns_to_remove = [
    "Using default tokenizer."
    # Ajoute ici d'autres motifs si besoin
]

with open(input_log, "r", encoding="utf-8") as fin, open(output_log, "w", encoding="utf-8") as fout:
    for line in fin:
        if not any(pattern in line for pattern in patterns_to_remove):
            fout.write(line)

print(f"Log nettoyé écrit dans {output_log}")