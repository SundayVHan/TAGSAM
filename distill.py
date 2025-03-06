import os
import random

import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm

import wandb
from dataset import GraphDataset, SynGraphDataset
from epoch import epoch_test, epoch_train_manual
from model import CLIP, wBCELoss
from reparam import ReparamModule
from selection import select_text


def main(args):
    wandb.init(
        project="TAGSAM-distill",
        name=args.name,
        config=args,
    )

    graph_dataset = GraphDataset(args)

    expert_model = CLIP(args).to(args.device)
    expert_state = torch.load(os.path.join(str(args.buffer_save_dir), f"expert_state.pt"), map_location=args.device, weights_only=True)
    expert_model.load_state_dict(expert_state)
    expert_model.eval()

    expert_acc = epoch_test(model=expert_model, test_dataset=graph_dataset, args=args)
    print(f"Expert Acc: {expert_acc}")

    save_it_pool = np.arange(0, args.syn_iteration, args.save_interval).tolist()
    match_loss = wBCELoss()
    match_sampler = NeighborSampler(graph_dataset.edge_index, node_idx=torch.arange(len(graph_dataset)),
                                    sizes=args.sample_size, batch_size=args.syn_match,
                                    shuffle=True, num_workers=8)

    graph_syn, text_syn, selected_text = select_text(graph_dataset, args)
    syn_dataset = SynGraphDataset(graph_syn, text_syn, args)

    for it in tqdm(range(args.syn_iteration), desc="distill", position=0, leave=True):
        if it in save_it_pool:
            save_data = {
                "node_f": syn_dataset.node_f,
                "text_embeds": syn_dataset.text_embeds,
                "graph_encoder_lr": syn_dataset.graph_encoder_lr,
                "text_encoder_lr": syn_dataset.text_encoder_lr,
                "selected_text": selected_text,
            }
            torch.save(save_data, os.path.join(str(args.buffer_save_dir), f"{args.name}_{it}.pt"))

        torch.cuda.empty_cache()
        syn_dataset.set_train_model()
        syn_dataset.zero_grad()

        student_model = ReparamModule(CLIP(args)).to(args.device)
        student_param = torch.cat([p.detach().reshape(-1) for p in student_model.parameters()], dim=0).requires_grad_(True)

        for step in range(args.syn_loop):
            loss, student_param = epoch_train_manual(model=student_model,
                                      param=student_param,
                                      graph_encoder_lr=syn_dataset.graph_encoder_lr,
                                      text_encoder_lr=syn_dataset.text_encoder_lr,
                                      train_dataset=syn_dataset,
                                      args=args)

        match_size, match_idx, match_adjs = next(iter(match_sampler))
        match_node_f = graph_dataset.node_f[match_idx].to(args.device)
        match_adjs = [adj.to(args.device) for adj in match_adjs]
        match_text_embeds = graph_dataset.text_embeds[match_idx[:match_size]].to(args.device)
        with torch.no_grad():
            expert_logits = expert_model(match_size, match_node_f, match_adjs, match_text_embeds, is_eval=True)

        student_logits = student_model(match_size, match_node_f, match_adjs, match_text_embeds, is_eval=True, flat_param=student_param)

        syn_loss = match_loss(student_logits, expert_logits)

        syn_dataset.compute_grad(syn_loss)
        syn_dataset.step()

        wandb.log({
            "syn_loss": syn_loss.item(),
            "graph_lr": syn_dataset.graph_encoder_lr.item(),
            "text_lr": syn_dataset.text_encoder_lr.item(),
        }, step=it)

    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # base
    parser.add_argument("--dataset_name", type=str, default="computer")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--save_interval", type=int, default=500)

    # distill
    parser.add_argument("--syn_iteration", type=int, default=5000)
    parser.add_argument("--syn_size", type=int, default=100)
    parser.add_argument("--syn_lr", type=float, default=100)
    parser.add_argument("--syn_lr_lr", type=float, default=2e-6)
    parser.add_argument("--syn_loop", type=int, default=15)
    parser.add_argument("--syn_batch_size_train", type=int, default=20)
    parser.add_argument("--syn_match", type=int, default=2000)
    parser.add_argument("--syn_num_summary", type=int, default=4)
    parser.add_argument("--syn_ratio_summary", type=float, default=60.0)

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
    args.name = f"{args.dataset_name}-{args.syn_size}-{args.seed}-{args.syn_num_summary}-{args.syn_ratio_summary}"

    if args.dataset_name == "art":
        args.sample_size = [10, 10]
    else:
        args.sample_size = [10, 10]

    seed_everything(args.seed)
    main(args)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

