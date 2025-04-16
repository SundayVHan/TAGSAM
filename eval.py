import os
import random

import numpy as np
import torch
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm

from dataset import SynGraphDataset, GraphDataset
from epoch import epoch_train, epoch_test
from model import CLIP

def eval_syn(
    graph_dataset: GraphDataset,
    syn_dataset: SynGraphDataset,
    args,
    is_distill=False
):
    syn_dataset.set_eval_model()

    sampler = NeighborSampler(
        syn_dataset.edge_index,
        sizes=[-1, -1],
        batch_size=args.batch_size_train,
        shuffle=True,
    )

    best_val_list = []
    best_acc_list = []
    for _ in tqdm(range(args.eval_time), desc="eval", position=1, leave=False):
        eval_model = CLIP(args)

        optimizer = torch.optim.SGD([
            {"params": eval_model.graph_encoder.parameters(), "lr": syn_dataset.graph_encoder_lr.item(),  "momentum": 0.9, "weight_decay": 5e-4},
            {"params": eval_model.text_encoder.parameters(), "lr": syn_dataset.text_encoder_lr.item(), "momentum": 0.9, "weight_decay": 5e-4},
        ])
        optimizer.zero_grad()

        best_val = 0
        best_acc = 0
        for epoch in range(args.num_epoch_train):
            epoch_train(model=eval_model, optimizer=optimizer, dataset=syn_dataset, sampler=sampler, args=args, is_distill=is_distill)
            val_acc, test_acc = epoch_test(model=eval_model, dataset=graph_dataset, args=args, is_distill=is_distill)

            if not is_distill:
                print(f"Epoch {epoch}: Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")   

            if val_acc > best_val:
                best_val = val_acc
                best_acc = test_acc

        best_val_list.append(best_val)
        best_acc_list.append(best_acc)

    mean_val = np.mean(best_val_list)
    mean_acc = np.mean(best_acc_list)
    tqdm.write(f"Mean Val Acc: {mean_val}, Mean Test Acc: {mean_acc}")
    return mean_val, mean_acc

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # base
    parser.add_argument("--dataset_name", type=str, default="computer")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--it", type=int, default=5000)
    parser.add_argument("--syn_size", type=int, default=100)
    parser.add_argument("--syn_num_summary", type=int, default=4)
    parser.add_argument("--syn_ratio_summary", type=float, default=60.0)
    parser.add_argument("--syn_lr", type=float, default=100)
    parser.add_argument("--syn_lr_lr", type=float, default=2e-6)
    parser.add_argument("--is_distill", type=bool, default=False)
    parser.add_argument("--run_name", type=str, default="")

    # eval
    parser.add_argument("--batch_size_train", type=int, default=20)
    parser.add_argument("--batch_size_test", type=int, default=2048)
    parser.add_argument("--num_epoch_train", type=int, default=15)
    parser.add_argument("--eval_time", type=int, default=1)

    # graph encoder
    parser.add_argument("--graph_encoder", type=str, default="gcn")
    parser.add_argument("--graph_encoder_lr", type=float, default=5e-3)
    parser.add_argument("--gnn_input_dim", type=int, default=384)
    parser.add_argument("--gnn_hidden_dim", type=int, default=384)
    parser.add_argument("--gnn_output_dim", type=int, default=384)

    # text encoder
    parser.add_argument("--text_encoder", type=str, default="bert")
    parser.add_argument("--text_encoder_lr", type=float, default=5e-3)
    parser.add_argument("--lm_output_dim", type=int, default=768)

    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    args.buffer_save_dir = os.path.join("./buffer", args.dataset_name, args.graph_encoder, args.text_encoder)

    if args.run_name:
        args.name = args.run_name
    else:
        args.name = f"{args.dataset_name}-{args.syn_size}-{args.seed}-{args.syn_num_summary}-{args.syn_ratio_summary}"

    if args.dataset_name == "art":
        args.sample_size = [10, 10]
    elif args.dataset_name == "products":
        args.sample_size = [10, 5]
    else:
        args.sample_size = [-1, -1]

    def seed_everything(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed_everything(args.seed)
    
    syn_dataset = SynGraphDataset(args)
    syn_dataset.load(args.it)
    graph_dataset = GraphDataset(args)
    eval_syn(graph_dataset, syn_dataset, args)