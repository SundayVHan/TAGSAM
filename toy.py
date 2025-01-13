import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import k_hop_subgraph

from data.dataset import GraphDataset
from epoch import epoch_test
from model import CLIP


def main(args):
    graph_dataset = GraphDataset(args)
    test_loader = DataLoader(graph_dataset, batch_size=args.batch_size_test_eval, shuffle=False)

    buffer_save_dir = os.path.join(args.buffer_save_dir, args.dataset_name, args.graph_encoder, args.text_encoder)
    expert_model = CLIP(args).to(args.device)
    expert_model.load_state_dict(torch.load(os.path.join(str(buffer_save_dir), f"expert_state_{args.use_text_emb}.pt")))
    expert_model.eval()

    expert_acc = epoch_test(model=expert_model, test_loader=test_loader, args=args)
    print(expert_acc)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # base
    parser.add_argument("--dataset_name", type=str, default="art")
    parser.add_argument("--buffer_save_dir", type=str, default="./buffer")
    parser.add_argument("--gpu", type=int, default=1)

    # distill
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--num_syn", type=int, default=200)
    parser.add_argument("--init_pair", type=str, default="random")
    parser.add_argument("--syn_graph_lr", type=float, default=100)
    parser.add_argument("--lr_lr", type=float, default=1e-6)
    parser.add_argument("--syn_steps", type=int, default=15)
    parser.add_argument("--mini_batch_size", type=int, default=20)
    parser.add_argument("--num_target", type=int, default=2000)
    parser.add_argument("--loss_type", type=str, default="WBCE")
    parser.add_argument("--cl_strategy", type=str, default="none")
    parser.add_argument("--cl_init_proportion", type=float, default=0.4)
    parser.add_argument("--cl_iterations", type=int, default=1000)
    parser.add_argument("--augment_rate", type=float, default=0.0)

    # eval
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--num_eval", type=int, default=5)
    parser.add_argument("--batch_size_train_eval", type=int, default=32)
    parser.add_argument("--train_epochs_eval", type=int, default=15)
    parser.add_argument("--batch_size_test_eval", type=int, default=512)

    # text type
    parser.add_argument("--use_text_emb", type=bool, default=True)

    # graph encoder
    parser.add_argument("--graph_encoder", type=str, default="gcn")
    parser.add_argument("--lr_graph_encoder", type=float, default=2e-3)

    # text encoder
    parser.add_argument("--text_encoder", type=str, default="bert")
    parser.add_argument("--lr_text_encoder", type=float, default=2e-5)
    parser.add_argument("--text_emb_dim", type=int, default=768)

    # task
    parser.add_argument('--k_spt', type=int, default=5)
    parser.add_argument('--k_val', type=int, default=5)
    parser.add_argument('--k_qry', type=int, default=50)
    parser.add_argument('--n_way', type=int, default=5)

    # ablation
    parser.add_argument("--ablation_no_expert", type=bool, default=False)
    parser.add_argument("--ablation_cl_reverse", type=bool, default=True)

    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if args.dataset_name == "cora":
        args.gnn_input_dim = 128
        args.gnn_hidden_dim = 128
        args.gnn_output_dim = 128
    else:
        args.gnn_input_dim = 384
        args.gnn_hidden_dim = 384
        args.gnn_output_dim = 384
    main(args)
