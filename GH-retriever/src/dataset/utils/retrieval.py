import torch
import numpy as np
from pcst_fast import pcst_fast
from torch_geometric.data.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch.nn.functional import cosine_similarity


def retrieval_via_pcst(graph, q_emb, textual_nodes, textual_edges, topk=3, topk_e=3, cost_e=0.5):
    c = 0.01
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst']) #don"t edit this line to change name
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    # Ensure q_emb is 2D and on the same device as graph.x
    if q_emb.dim() == 1:
        q_emb = q_emb.unsqueeze(0)
    q_emb = q_emb.to(graph.x.device)

    root = -1  # unrooted
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0
    if topk > 0:
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
        topk = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)

        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float().to(n_prizes.device)
    else:
        n_prizes = torch.zeros(graph.num_nodes, device=graph.x.device)

    if topk_e > 0:
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.edge_attr)
        topk_e = min(topk_e, e_prizes.unique().size(0))

        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            value = min((topk_e-k)/sum(indices), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value*(1-c)
        # reduce the cost of the edges such that at least one edge is selected
        cost_e = min(cost_e, e_prizes.max().item()*(1-c/2))
    else:
        e_prizes = torch.zeros(graph.num_edges, device=graph.x.device)

    costs = []
    edges = []
    vritual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    # Build edges using subgraph's node indices (0..N-1)
    num_nodes = graph.num_nodes
    for i in range(graph.edge_index.shape[1]):
        src = int(graph.edge_index[0, i])
        dst = int(graph.edge_index[1, i])
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = num_nodes + len(vritual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            vritual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes, np.array(vritual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        edges += virtual_edges
        costs += virtual_costs
    edges = np.array(edges)
    costs = np.array(costs)

    # Use subgraph's edge indices directly (already contiguous 0..N-1)
    pcst_edges = [tuple(edge) for edge in edges]

    # --- Robustness check: ensure all edge endpoints are valid ---
    max_node_idx = len(prizes) - 1
    for src, dst in pcst_edges:
        if src > max_node_idx or dst > max_node_idx:
            raise ValueError(f"Edge endpoint out of range: ({src}, {dst}) for {len(prizes)} nodes")

    # Now call pcst_fast with contiguous subgraph edge indices
    vertices, edges = pcst_fast(pcst_edges, prizes, costs, root, num_clusters, pruning, verbosity_level)

    selected_nodes = vertices[vertices < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= graph.num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges+virtual_edges)

    edge_index = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))

    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]
    desc = n.to_csv(index=False)+'\n'+e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
    
    # Map selected_nodes (node IDs) to positions in the subgraph DataFrame
    if hasattr(graph, 'n_id') and graph.n_id is not None:
        # PyG subgraphs may have n_id attribute for mapping
        node_id_to_pos = {nid: i for i, nid in enumerate(graph.n_id.cpu().numpy())}
        selected_nodes_pos = [node_id_to_pos[n] for n in selected_nodes]
    else:
        # Assume selected_nodes are already positions (0..N-1)
        selected_nodes_pos = selected_nodes

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]
    edge_index = torch.LongTensor([src, dst])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    return data, desc

def hybrid_khop_pcst_subgraph(graph, q_emb, textual_nodes, textual_edges, target_nodes, k=2, topk=10, topk_e=5, cost_e=0.5):
    """
    Extraction hybride : K-hop autour des nœuds cibles puis PCST sur ce sous-graphe.
    - graph : Data PyG complet
    - q_emb : embedding de la requête
    - textual_nodes, textual_edges : DataFrames textuels
    - target_nodes : liste des nœuds cibles (ex : [user, item])
    - k : profondeur K-hop
    - topk, topk_e, cost_e : paramètres PCST
    """
    # 1. Extraction du sous-graphe K-hop autour des nœuds cibles
    nodes, edge_index, _, edge_mask = k_hop_subgraph(
        node_idx=target_nodes,
        num_hops=k,
        edge_index=graph.edge_index,
        relabel_nodes=True,  # Ensure subgraph node indices are contiguous 0..N-1
        num_nodes=graph.num_nodes
    )
    subgraph = Data(
        x=graph.x[nodes],
        edge_index=edge_index,
        edge_attr=graph.edge_attr[edge_mask] if graph.edge_attr is not None else None,
        num_nodes=len(nodes)
    )
    # Adapter les DataFrames textuels au sous-graphe
    textual_nodes_sub = textual_nodes.iloc[nodes.cpu().numpy()].reset_index(drop=True)
    textual_edges_sub = textual_edges.iloc[edge_mask.cpu().numpy()].reset_index(drop=True)
    
    # 2. Application de PCST sur le sous-graphe K-hop
    data_pcst, desc_pcst = retrieval_via_pcst(
        subgraph, q_emb, textual_nodes_sub, textual_edges_sub,
        topk=topk, topk_e=topk_e, cost_e=cost_e
    )
    # Remapping des indices pour correspondre aux indices globaux si besoin
    return data_pcst, desc_pcst

def select_target_nodes_by_similarity(graph, q_emb, topk=1):
    """
    Select indices of the top-k nodes most similar to q_emb using cosine similarity.
    Handles device and shape mismatches robustly.
    """
    # Ensure q_emb is 2D: [1, d]
    if q_emb.dim() == 1:
        q_emb = q_emb.unsqueeze(0)
    # Ensure q_emb and graph.x are on the same device
    q_emb = q_emb.to(graph.x.device)
    # Compute cosine similarity between q_emb and each node embedding
    sims = torch.nn.functional.cosine_similarity(q_emb, graph.x, dim=1)  # [N]
    topk = min(topk, graph.x.shape[0])
    topk_indices = torch.topk(sims, topk, largest=True).indices.tolist()
    return topk_indices