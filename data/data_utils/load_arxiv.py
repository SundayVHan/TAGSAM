from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import os.path as osp

def get_raw_text_arxiv(use_text=False, seed=0):
    data = torch.load(f"./data/arxiv/arxiv.pt", map_location='cpu')
    data.edge_index = torch.stack(data.edge_index.coo()[:2], dim=0)
    data.num_nodes = data.y.shape[0]

    nodeidx2paperid = pd.read_csv(
        f'./data/arxiv/nodeidx2paperid.csv.gz', compression='gzip')

    raw_text = pd.read_csv(f'./data/arxiv/titleabs.tsv',
                           sep='\t', header=None, names=['paper id', 'title', 'abs'], skiprows=[0])
    raw_text = raw_text.dropna()

    # nodeidx2paperid['paper id'] = nodeidx2paperid['paper id'].astype('int64')
    raw_text['paper id'] = raw_text['paper id'].astype('int64')
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')

    text = []
    for ti, ab in zip(df['title'], df['abs']):
        t = 'Title: ' + ti + '\n' + 'Abstract: ' + ab
        text.append(t)
    return data, text