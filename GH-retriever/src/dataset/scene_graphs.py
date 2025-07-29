import os
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from src.dataset.utils.retrieval import hybrid_khop_pcst_subgraph, select_target_nodes_by_similarity
from src.dataset.utils.hybrid_retrieve_on_graphs_pcst import hybrid_retrieve_on_graphs_pcst
from src.config import parse_args_llama



model_name = 'sbert'
path = 'dataset/scene_graphs'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'


class SceneGraphsDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = None
        self.graph = None
        self.graph_type = 'Scene Graph'
        self.questions = pd.read_csv(f'{path}/questions.csv')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.questions)

    def __getitem__(self, index):
        data = self.questions.iloc[index]
        question = f'Question: {data["question"]}\n\nAnswer:'
        graph = torch.load(f'{cached_graph}/{index}.pt')
        desc = open(f'{cached_desc}/{index}.txt', 'r').read()

        return {
            'id': index,
            'image_id': data['image_id'],
            'question': question,
            'label': data['answer'],
            'full_label': data['full_answer'],
            'graph': graph,
            'desc': desc,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


def preprocess():
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)

    questions = pd.read_csv(f'{path}/questions.csv')
    q_embs = torch.load(f'{path}/q_embs.pt')
    for index in tqdm(range(len(questions))):
        if os.path.exists(f'{cached_graph}/{index}.pt'):
            continue
        image_id = questions.iloc[index]['image_id']
        graph = torch.load(f'{path_graphs}/{image_id}.pt')
        nodes = pd.read_csv(f'{path_nodes}/{image_id}.csv')
        edges = pd.read_csv(f'{path_edges}/{image_id}.csv')
        q_emb = q_embs[index]
        #target_nodes = select_target_nodes_by_similarity(graph, q_emb, topk=1)
        #subg, desc = hybrid_khop_pcst_subgraph(graph, q_emb, nodes, edges, target_nodes=target_nodes,  k=args.khop_k, topk=args.khop_topk, topk_e=args.khop_topk_e, cost_e=args.khop_cost_e)
        subg, desc = hybrid_retrieve_on_graphs_pcst(graph, q_emb, nodes, edges,k=args.khop_k, topk=args.khop_topk, topk_e=args.khop_topk_e, cost_e=args.khop_cost_e, augment="none")
        torch.save(subg, f'{cached_graph}/{index}.pt')
        open(f'{cached_desc}/{index}.txt', 'w').write(desc)


if __name__ == '__main__':
    args = parse_args_llama()

    preprocess()

    dataset = SceneGraphsDataset()

    data = dataset[0]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
