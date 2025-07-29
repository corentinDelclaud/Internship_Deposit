from torch_geometric.data import Batch

def collate_fn(batch):
    # Filtrer les exemples où le graphe est None ou vide
    batch = [
        d for d in batch
        if d['graph'] is not None
        and hasattr(d['graph'], 'x')
        and d['graph'].x is not None
        and d['graph'].x.shape[0] > 0
    ]
    if len(batch) == 0:
        return None  # ou raise une erreur
    batch_dict = {k: [d[k] for d in batch] for k in batch[0]}
    if 'graph' in batch_dict:
        batch_dict['graph'] = Batch.from_data_list(batch_dict['graph'])
    return batch_dict