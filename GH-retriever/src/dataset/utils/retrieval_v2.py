from pcst_fast import pcst_fast
from torch_geometric.data.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch.nn.functional import cosine_similarity
import pandas as pd
import torch

def subgraph_to_text(nodes, edges):
    node_lines = [
        f"Node {row['node_id']}: {row['node_attr']} | Source: {row.get('source','')} | Description: {row.get('description','')}"
        for _, row in nodes.iterrows()
    ]
    edge_lines = [
        f"Edge {row['src']} -[{row['edge_attr']}]-> {row['dst']} | Source: {row.get('source','')} | Description: {row.get('description','')}"
        for _, row in edges.iterrows()
    ]
    return "NODES:\n" + "\n".join(node_lines) + "\nEDGES:\n" + "\n".join(edge_lines)

def hybrid_khop_pcst_subgraph(
    graph, q_emb, nodes, edges, target_nodes, k=3, topk=10, topk_e=5, cost_e=0.5
):
    # 1. Extraire le sous-graphe k-hop autour des target_nodes
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        target_nodes, k, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes
    )
    # 2. Sélectionner les topk nœuds les plus similaires à la question
    node_emb = graph.x[subset]
    sims = cosine_similarity(node_emb, q_emb.unsqueeze(0))
    topk_idx = torch.topk(sims, min(topk, sims.size(0)), largest=True).indices
    selected_nodes = subset[topk_idx]

    # 3. Extraire les arêtes correspondantes dans le sous-graphe
    # edge_mask: booléen sur les arêtes du graphe global, edge_index: arêtes du sous-graphe (relabelées)
    # On veut les indices des arêtes du sous-graphe dans le graphe global
    subgraph_edge_indices = edge_mask.nonzero(as_tuple=True)[0]
    srcs = graph.edge_index[0, subgraph_edge_indices]
    dsts = graph.edge_index[1, subgraph_edge_indices]
    keep = torch.isin(srcs, selected_nodes) & torch.isin(dsts, selected_nodes)
    final_edge_indices = subgraph_edge_indices[keep]
    selected_edges = torch.stack([srcs[keep], dsts[keep]], dim=0)

    # 4. Préparer la textualisation enrichie
    nodes_df = nodes[nodes['node_id'].isin(selected_nodes.cpu().numpy())].reset_index(drop=True)
    edges_df = edges[
        edges['src'].isin(selected_nodes.cpu().numpy()) & edges['dst'].isin(selected_nodes.cpu().numpy())
    ].reset_index(drop=True)
    desc = subgraph_to_text(nodes_df, edges_df)

    # 5. Retourner le sous-graphe PyG et la description textuelle
    subgraph = Data(
        x=graph.x[selected_nodes],
        edge_index=selected_edges,
        edge_attr=graph.edge_attr[final_edge_indices],
        num_nodes=len(selected_nodes)
    )
    return subgraph, desc

def select_target_nodes_by_similarity(graph, q_emb, topk=1):
    # Calcule la similarité cosinus entre chaque nœud et la question
    sims = cosine_similarity(graph.x, q_emb.unsqueeze(0))
    topk_idx = torch.topk(sims, min(topk, sims.size(0)), largest=True).indices
    return topk_idx