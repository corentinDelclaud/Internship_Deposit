import os
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from src.dataset.utils.retrieval import hybrid_khop_pcst_subgraph, select_target_nodes_by_similarity
from src.utils.lm_modeling import load_sbert, sber_text2embedding

PATH = 'dataset/expla_graphs'
PATH_NODES = f'{PATH}/nodes'
PATH_EDGES = f'{PATH}/edges'
PATH_GRAPHS = f'{PATH}/graphs'
CACHED_GRAPH = f'{PATH}/cached_graphs'
CACHED_DESC = f'{PATH}/cached_desc'

def preprocess():
    os.makedirs(CACHED_GRAPH, exist_ok=True)
    os.makedirs(CACHED_DESC, exist_ok=True)

    dataset = pd.read_csv(f'{PATH}/train_dev.tsv', sep='\t')
    model, tokenizer, device = load_sbert()

    for index in tqdm(range(len(dataset))):
        if os.path.exists(f'{CACHED_GRAPH}/{index}.pt'):
            continue
        graph = torch.load(f'{PATH_GRAPHS}/{index}.pt')
        nodes = pd.read_csv(f'{PATH_NODES}/{index}.csv')
        edges = pd.read_csv(f'{PATH_EDGES}/{index}.csv')
        # Génère la question à la volée
        row = dataset.iloc[index]
        question = f"Argument 1: {row.arg1}\nArgument 2: {row.arg2}"
        q_emb = sber_text2embedding(model, tokenizer, device, [question])[0]
        # Sélectionne les nœuds cibles par similarité (ex : top-1)
        target_nodes = select_target_nodes_by_similarity(graph, q_emb, topk=1)
        # Extraction hybride K-hop + PCST
        subg, desc = hybrid_khop_pcst_subgraph(
            graph, q_emb, nodes, edges,
            target_nodes=target_nodes, k=2, topk=10, topk_e=5, cost_e=0.5
        )
        torch.save(subg, f'{CACHED_GRAPH}/{index}.pt')
        with open(f'{CACHED_DESC}/{index}.txt', 'w') as f:
            f.write(desc)

class ExplaGraphsKhopDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.text = pd.read_csv(f'{PATH}/train_dev.tsv', sep='\t')
        self.prompt = 'Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:'
        self.graph_type = 'Explanation Graph'

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text.iloc[index]
        graph = torch.load(f'{CACHED_GRAPH}/{index}.pt')
        desc = open(f'{CACHED_DESC}/{index}.txt', 'r').read()
        question = f'Argument 1: {text.arg1}\nArgument 2: {text.arg2}\n{self.prompt}'
        return {
            'id': index,
            'label': text['label'],
            'desc': desc,
            'graph': graph,
            'question': question,
        }

    def get_idx_split(self):
        with open(f'{PATH}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{PATH}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{PATH}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]
        return {'train': train_indices, 'val': val_indices, 'test': test_indices}

if __name__ == '__main__':
    preprocess()
    dataset = ExplaGraphsKhopDataset()
    data = dataset[0]
    for k, v in data.items():
        print(f'{k}: {v}')
    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')