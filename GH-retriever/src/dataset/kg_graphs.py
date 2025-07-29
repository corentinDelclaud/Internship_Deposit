import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data.data import Data
from torch.utils.data import Dataset
from src.utils.lm_modeling import load_model, load_text2embedding
from sklearn.model_selection import train_test_split
from src.dataset.utils.retrieval import hybrid_khop_pcst_subgraph, select_target_nodes_by_similarity
from src.dataset.utils.hybrid_retrieve_on_graphs_pcst import hybrid_retrieve_on_graphs_pcst
from src.config import parse_args_llama


model_name = 'sbert'
path = 'dataset/kg'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'
path_desc = f'{path}/cached_desc'

def textualize_graph(rel):
    # rel: dict with 'node_1', 'node_2', 'relationship'
    nodes = [
        {'node_id': 0, 'node_attr': rel['node_1']},
        {'node_id': 1, 'node_attr': rel['node_2']}
    ]
    edges = [
        {'src': 0, 'edge_attr': rel['relationship'], 'dst': 1}
    ]
    return nodes, edges

def step_one():
    with open(f'{path}/kg_llm_relationships_all.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    for idx, rel in tqdm(enumerate(data), total=len(data)):
        # If rel['node_1'] and rel['node_2'] are dicts, extract a string (adapt as needed)
        def node_repr(node):
            if isinstance(node, dict):
                for key in ['name', 'label', 'link']:
                    if key in node:
                        return node[key]
                return str(node)
            return str(node)
        rel_proc = {
            'node_1': node_repr(rel.get('node_1', 'node_1')),
            'node_2': node_repr(rel.get('node_2', 'node_2')),
            'relationship': rel.get('relationship', 'related')
        }
        node_attr, edge_attr = textualize_graph(rel_proc)
        pd.DataFrame(node_attr, columns=['node_id', 'node_attr']).to_csv(f'{path_nodes}/{idx}.csv', index=False)
        pd.DataFrame(edge_attr, columns=['src', 'edge_attr', 'dst']).to_csv(f'{path_edges}/{idx}.csv', index=False)

def step_two():
    os.makedirs(path_graphs, exist_ok=True)
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]
    
    questions_path = f'{path}/questions_all.tsv'
    questions_df = pd.read_csv(questions_path, sep='\t')
    q_embs = text2embedding(model, tokenizer, device, questions_df['Question'].tolist())
    torch.save(q_embs, f'{path}/q_embs.pt')
    

    file_indices = [
        int(f.split('.')[0]) for f in os.listdir(path_nodes)
        if f.endswith('.csv') and f.split('.')[0].isdigit()
    ]
    for idx in tqdm(file_indices):
        nodes = pd.read_csv(f'{path_nodes}/{idx}.csv')
        edges = pd.read_csv(f'{path_edges}/{idx}.csv')
        if len(nodes) == 0:
            print(f'Empty graph, skipping idx {idx}')
            continue
        node_attr_list = [str(x) if pd.notnull(x) else "" for x in nodes.node_attr.tolist()]
        edge_attr_list = [str(x) if pd.notnull(x) else "" for x in edges.edge_attr.tolist()]
        node_attr = text2embedding(model, tokenizer, device, node_attr_list)
        edge_attr = text2embedding(model, tokenizer, device, edge_attr_list)
        edge_index = torch.tensor([edges.src, edges.dst]).long()
        pyg_graph = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
        torch.save(pyg_graph, f'{path_graphs}/{idx}.pt')

def generate_split():
    # Use sep='\t' for TSV files
    questions = pd.read_csv(f"{path}/questions_all.tsv", sep='\t')
    # If your questions file does not have a 'graph_id' column, you can use the index or another unique identifier
    if 'graph_id' in questions.columns:
        unique_ids = questions['graph_id'].unique()
    else:
        unique_ids = questions.index.values
        questions['graph_id'] = questions.index  # Add this for consistency

    np.random.seed(42)
    shuffled_ids = np.random.permutation(unique_ids)
    train_ids, temp_ids = train_test_split(shuffled_ids, test_size=0.4, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    id_to_set = {i: 'train' for i in train_ids}
    id_to_set.update({i: 'val' for i in val_ids})
    id_to_set.update({i: 'test' for i in test_ids})
    questions['set'] = questions['graph_id'].map(id_to_set)
    train_df = questions[questions['set'] == 'train']
    val_df = questions[questions['set'] == 'val']
    test_df = questions[questions['set'] == 'test']
    os.makedirs(f'{path}/split', exist_ok=True)
    train_df.index.to_series().to_csv(f'{path}/split/train_indices.txt', index=False, header=False)
    val_df.index.to_series().to_csv(f'{path}/split/val_indices.txt', index=False, header=False)
    test_df.index.to_series().to_csv(f'{path}/split/test_indices.txt', index=False, header=False)

def preprocess():
    os.makedirs(path_desc, exist_ok=True)
    cached_graph = f'{path}/cached_graphs'
    os.makedirs(cached_graph, exist_ok=True)

    questions = pd.read_csv(f'{path}/questions_all.tsv', sep='\t')
    q_embs = torch.load(f'{path}/q_embs.pt',weights_only=False)
    for index in tqdm(range(len(questions))):
        if os.path.exists(f'{cached_graph}/{index}.pt'):
            continue
        # For KG, assume each question corresponds to a graph with the same index
        graph = torch.load(f'{path_graphs}/{index}.pt')
        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        q_emb = q_embs[index]
        target_nodes = select_target_nodes_by_similarity(graph, q_emb, topk=1)
        subg, desc = hybrid_khop_pcst_subgraph(graph, q_emb, nodes, edges, target_nodes=target_nodes,  k=args.khop_k, topk=args.khop_topk, topk_e=args.khop_topk_e, cost_e=args.khop_cost_e)
        #subg, desc = hybrid_retrieve_on_graphs_pcst(graph, q_emb, nodes, edges,k=args.khop_k, topk=args.khop_topk, topk_e=args.khop_topk_e, cost_e=0, augment="none")
        
        torch.save(subg, f'{cached_graph}/{index}.pt')
        with open(f'{path_desc}/{index}.txt', 'w', encoding='utf-8') as f:
            f.write(desc)
            
class KGGraphsDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = None
        self.graph = None
        self.graph_type = 'KG'
        self.path = 'dataset/kg'
        self.questions = pd.read_csv(f'{self.path}/questions_all.tsv', sep='\t')
        self.path_graphs = f'{self.path}/graphs'
        self.path_desc = f'{self.path}/cached_desc'
        self.indices = list(range(len(self.questions)))  # indices de 0 à N-1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        row = self.questions.iloc[real_idx]
        graph_path = os.path.join(self.path_graphs, f'{real_idx}.pt')
        desc_path = os.path.join(self.path_desc, f'{real_idx}.txt')
        graph = torch.load(graph_path, weights_only=False)
        desc = open(desc_path).read() if os.path.exists(desc_path) else ""
        return {
            'id': real_idx,
            'label': str(row['Label']),
            'question': row['Question'],
            'graph': graph,
            'desc': desc,
        }

    def get_idx_split(self):
        # Ici, tu fais le split sur la longueur du dataset
        N = len(self)
        idxs = list(range(N))
        train, test = train_test_split(idxs, test_size=0.2, random_state=42)
        train, val = train_test_split(train, test_size=0.1, random_state=42)
        return {'train': train, 'val': val, 'test': test}
    
if __name__ == '__main__':
    args = parse_args_llama()

    #step_one()
    #step_two()
    #generate_split()
    preprocess()
    dataset = KGGraphsDataset()

    data = dataset[0]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')