import torch
from src.dataset.utils.graph_retrieval import retrive_on_graphs
from src.dataset.utils.retrieval import hybrid_khop_pcst_subgraph

def hybrid_retrieve_on_graphs_pcst(
    graph, q_emb, textual_nodes, textual_edges,
    k=2, topk=10, topk_entity=5, topk_pcst=10, topk_e=5, cost_e=0.5, augment="none"
):
    """
    1. Utilise retrive_on_graphs pour scorer chaque nœud (via son K-hop) par similarité avec la requête.
    2. Sélectionne les top-k nœuds les plus pertinents comme cibles.
    3. Applique hybrid_khop_pcst_subgraph à partir de ces nœuds cibles.
    """
    # 1. Similarité sur K-hop autour de chaque nœud
    sims, _ = retrive_on_graphs(
        graph, q_emb, textual_nodes, textual_edges,
        topk=graph.num_nodes, k=k, topk_entity=topk_entity, augment=augment
    )
    # 2. Sélection des top-k nœuds cibles
    if isinstance(sims, torch.Tensor):
        topk = min(topk, sims.size(0)) 
        topk_indices = torch.topk(sims, topk, largest=True).indices.tolist()
    else:
        topk_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:topk]
    # Correction : toujours convertir en tensor long pour PyG
    if not torch.is_tensor(topk_indices):
        topk_indices = torch.tensor(topk_indices, dtype=torch.long)
    # 3. Extraction hybride K-hop + PCST
    subgraph, desc = hybrid_khop_pcst_subgraph(
        graph, q_emb, textual_nodes, textual_edges,
        target_nodes=topk_indices, k=k, topk=topk_pcst, topk_e=topk_e, cost_e=cost_e
    )
    return subgraph, desc

# Exemple d'utilisation :
if __name__ == "__main__":
    # Charger vos données ici (exemple)
    import pandas as pd
    graph = torch.load('dataset/kg/graphs/0.pt')
    textual_nodes = pd.read_csv('dataset/kg/nodes/0.csv')
    textual_edges = pd.read_csv('dataset/kg/edges/0.csv')
    q_embs = torch.load('dataset/kg/q_embs.pt')
    q_emb = q_embs[0]

    subgraph, desc = hybrid_retrieve_on_graphs_pcst(
        graph, q_emb, textual_nodes, textual_edges,
        k=2, topk=3, topk_entity=5, topk_pcst=3, topk_e=3, cost_e=0.5, augment="none"
    )    
    print(subgraph)
    print(desc[:500])