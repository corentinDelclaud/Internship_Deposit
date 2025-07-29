import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from src.utils.lm_modeling import load_model, load_text2embedding
from src.dataset.utils.retrieval_v2 import hybrid_khop_pcst_subgraph, select_target_nodes_by_similarity
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

def generate_split(questions_path, out_dir):
    questions = pd.read_csv(questions_path, sep='\t')
    N = len(questions)
    idxs = list(range(N))
    train, test = train_test_split(idxs, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.1, random_state=42)
    os.makedirs(out_dir, exist_ok=True)
    pd.Series(train).to_csv(os.path.join(out_dir, 'train_indices.txt'), index=False, header=False)
    pd.Series(val).to_csv(os.path.join(out_dir, 'val_indices.txt'), index=False, header=False)
    pd.Series(test).to_csv(os.path.join(out_dir, 'test_indices.txt'), index=False, header=False)
    print(f"Splits saved in {out_dir}")

def batched_text2embedding(texts, model, tokenizer, device, text2embedding, batch_size=4, desc="Embedding"):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i:i+batch_size]
        embs = text2embedding(model, tokenizer, device, batch)
        all_embs.append(embs.cpu())
        torch.cuda.empty_cache()
    return torch.cat(all_embs, dim=0)

def build_global_graph_from_json(json_path, model, tokenizer, device, text2embedding):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    node_dict = {}  # (name, label) -> global_id
    all_nodes = []
    all_edges = []
    node_attr_texts = []
    edge_attr_texts = []
    node_idx = 0

    for rel in data:
        # Helper to extract info from node dicts
        def node_repr(node):
            if isinstance(node, dict):
                name = node.get('name', '')
                label = node.get('label', '')
                source = node.get('source', '')
                key = (name, label)
                return key, name or label, source
            return (str(node), ''), str(node), ''

        # Node 1
        key1, n1, n1_source = node_repr(rel.get('node_1', 'node_1'))
        if key1 not in node_dict:
            node_dict[key1] = node_idx
            all_nodes.append({
                'node_id': node_idx,
                'node_attr': n1,
                'source': n1_source
            })
            node_attr_texts.append(" | ".join([n1, n1_source]))
            node_idx += 1
        id1 = node_dict[key1]

        # Node 2
        key2, n2, n2_source = node_repr(rel.get('node_2', 'node_2'))
        if key2 not in node_dict:
            node_dict[key2] = node_idx
            all_nodes.append({
                'node_id': node_idx,
                'node_attr': n2,
                'source': n2_source
            })
            node_attr_texts.append(" | ".join([n2, n2_source]))
            node_idx += 1
        id2 = node_dict[key2]

        # Edge
        relationship = rel.get('relationship', 'related')
        edge_source = rel.get('source', '')
        edge_desc = rel.get('description', '')
        all_edges.append({
            'src': id1,
            'dst': id2,
            'edge_attr': relationship,
            'source': edge_source,
            'description': edge_desc
        })
        edge_attr_texts.append(" | ".join([relationship, edge_source, edge_desc]))

    nodes_df = pd.DataFrame(all_nodes)
    edges_df = pd.DataFrame(all_edges)

    # Embeddings (en batch pour éviter OOM)
    node_attr = batched_text2embedding(node_attr_texts, model, tokenizer, device, text2embedding, batch_size=4, desc="Embedding nodes")
    edge_attr = batched_text2embedding(edge_attr_texts, model, tokenizer, device, text2embedding, batch_size=4, desc="Embedding edges")
    edge_index = torch.tensor([edges_df['src'], edges_df['dst']]).long()

    pyg_graph = Data(
        x=node_attr,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(nodes_df)
    )
    return pyg_graph, nodes_df, edges_df

def retrieve_subgraph_for_question(global_graph, nodes_df, edges_df, q_emb, args):
    target_nodes = select_target_nodes_by_similarity(global_graph, q_emb, topk=1)
    subg, desc = hybrid_khop_pcst_subgraph(
        global_graph, q_emb, nodes_df, edges_df,
        target_nodes=target_nodes,
        k=args.khop_k, topk=args.khop_topk, topk_e=args.khop_topk_e, cost_e=args.khop_cost_e
    )
    return subg, desc

class KGGraphsDataset(Dataset):
    def __init__(self, path='dataset/kg2', split=None):
        """
        split: None (tout), 'train', 'val', 'test' pour n'utiliser qu'un split précis
        """
        self.path = path
        self.questions = pd.read_csv(f'{self.path}/questions_all.tsv', sep='\t')
        self.path_graphs = f'{self.path}/cached_graphs'
        self.path_desc = f'{self.path}/cached_desc'
        self.split = split
        self.prompt = 'Please answer the given question.'
        self.graph = None
        self.graph_type = 'Knowledge Graph'

        # Gestion des splits
        if split in ['train', 'val', 'test']:
            split_path = os.path.join(self.path, 'split', f'{split}_indices.txt')
            with open(split_path) as f:
                self.indices = [int(line.strip()) for line in f if line.strip()]
        else:
            self.indices = list(range(len(self.questions)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        row = self.questions.iloc[real_idx]
        graph_path = os.path.join(self.path_graphs, f'{real_idx}.pt')
        desc_path = os.path.join(self.path_desc, f'{real_idx}.txt')
        graph = torch.load(graph_path)
        desc = open(desc_path, encoding='utf-8').read() if os.path.exists(desc_path) else ""
        return {
            'id': real_idx,
            'label': str(row['Label']),
            'question': row['Question'],
            'graph': graph,
            'desc': desc,
        }

    def get_idx_split(self):
        """Retourne un dict {'train': [...], 'val': [...], 'test': [...]} d'indices pour chaque split"""
        split_dir = os.path.join(self.path, 'split')
        splits = {}
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(split_dir, f'{split}_indices.txt')
            if os.path.exists(split_path):
                with open(split_path) as f:
                    splits[split] = [int(line.strip()) for line in f if line.strip()]
        return splits

if __name__ == "__main__":
    # Chemins
    path = 'dataset/kg2'
    json_path = f'{path}/kg_llm_relationships_all.json'
    questions_path = f'{path}/questions_all.tsv'
    os.makedirs(f'{path}/cached_graphs', exist_ok=True)
    os.makedirs(f'{path}/cached_desc', exist_ok=True)

    # Chargement du modèle d'embedding
    model_name = 'sbert'
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    # 1. Construire le graphe global à partir du JSON
    print("Construction du graphe global...")
    global_graph, nodes_df, edges_df = build_global_graph_from_json(json_path, model, tokenizer, device, text2embedding)

    # 2. Charger les questions et leurs embeddings (en batch pour éviter OOM)
    questions_df = pd.read_csv(questions_path, sep='\t')
    def batched_questions_embedding(questions, model, tokenizer, device, text2embedding, batch_size=4):
        all_embs = []
        for i in tqdm(range(0, len(questions), batch_size), desc="Embedding questions"):
            batch = questions[i:i+batch_size]
            embs = text2embedding(model, tokenizer, device, batch)
            all_embs.append(embs.cpu())
            torch.cuda.empty_cache()
        return torch.cat(all_embs, dim=0)
    q_embs = batched_questions_embedding(questions_df['Question'].tolist(), model, tokenizer, device, text2embedding, batch_size=4)

    # 3. Pour chaque question, extraire le sous-graphe pertinent
    from types import SimpleNamespace
    args = SimpleNamespace(khop_k=2, khop_topk=10, khop_topk_e=5, khop_cost_e=0.5)  # adapte selon tes besoins

    for idx, q_emb in tqdm(enumerate(q_embs), total=len(q_embs)):
        subg, desc = retrieve_subgraph_for_question(global_graph, nodes_df, edges_df, q_emb, args)
        torch.save(subg, f'{path}/cached_graphs/{idx}.pt')
        with open(f'{path}/cached_desc/{idx}.txt', 'w', encoding='utf-8') as f:
            f.write(desc)
    # 4. Générer les splits
    print("Génération des splits...")
    generate_split(questions_path, f'{path}/split')