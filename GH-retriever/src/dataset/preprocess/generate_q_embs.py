import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PATH = 'dataset/expla_graphs'
OUTDIR = f'{PATH}/q_embs'
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(f'{PATH}/train_dev.tsv', sep='\t')
questions = [
    f"Argument 1: {row.arg1}\nArgument 2: {row.arg2}"
    for _, row in df.iterrows()
]

model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')  # ou ton modèle SBERT

for idx, question in tqdm(enumerate(questions), total=len(questions)):
    emb = model.encode(question, convert_to_tensor=True)
    torch.save(emb, f'{OUTDIR}/{idx}.pt')
