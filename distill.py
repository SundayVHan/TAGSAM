import os
import random

import numpy as np
import torch
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm

import wandb
from dataset import GraphDataset, SynGraphDataset
from epoch import epoch_test, epoch_train_manual
from model import CLIP, wBCELoss, LinkPredictor
from reparam import ReparamModule
from selection import select_text
from eval import eval_syn

def main(args):
    wandb.init(
        project="TAGSAM",
        group="distill",
        name=args.name,
        config=args,
    )
    graph_dataset = GraphDataset(args)

    expert_model = CLIP(args).to(args.device)
    expert_state = torch.load(os.path.join(str(args.buffer_save_dir), f"expert_state.pt"), map_location=args.device, weights_only=True)
    expert_model.load_state_dict(expert_state)
    expert_model.eval()

    save_it_pool = np.arange(0, args.syn_iteration+1, args.save_interval).tolist()

    match_loss = wBCELoss()
    match_sampler = NeighborSampler(
        graph_dataset.edge_index,
        node_idx=torch.arange(len(graph_dataset)),
        sizes=args.sample_size, 
        batch_size=args.syn_match,
        shuffle=True, 
        num_workers=0
    )

    _, expert_acc = epoch_test(model=expert_model, dataset=graph_dataset, args=args)
    tqdm.write(f"Expert Acc: {expert_acc}")
    wandb.summary["expert_acc"] = expert_acc

    graph_syn, text_syn = select_text(graph_dataset, args)

    link_model = LinkPredictor(args.gnn_hidden_dim).to(args.device)
    link_model.load_state_dict(torch.load(os.path.join(str(args.buffer_save_dir), f"link_model.pt"), map_location=args.device))
    link_model.eval()
    
    with torch.no_grad():
        syn_text_embeds = expert_model.text_encoder(text_syn)
    
        num_nodes = len(text_syn)
        indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=args.device)
        src_nodes, dst_nodes = indices[0], indices[1]
    
        pred = link_model(syn_text_embeds[src_nodes], syn_text_embeds[dst_nodes])
    
        mask = pred > 1 - 3e-8
        edge_index_upper = torch.stack([src_nodes[mask], dst_nodes[mask]], dim=0)
        edge_index_lower = torch.stack([edge_index_upper[1], edge_index_upper[0]], dim=0)
        self_loops = torch.arange(num_nodes, device=args.device)
        self_loops = torch.stack([self_loops, self_loops], dim=0)
    
        syn_edge_index = torch.cat([edge_index_upper, edge_index_lower, self_loops], dim=1)
        tqdm.write(f"Generated {syn_edge_index.size(1)} edges for {num_nodes} nodes")
        
    syn_dataset = SynGraphDataset(args)
    syn_dataset.init(graph_syn, text_syn, syn_edge_index)

    inner_sampler = NeighborSampler(
        syn_dataset.edge_index,
        sizes=[-1, -1],
        num_nodes=len(syn_dataset),
        num_workers=0,
        shuffle=True,
        batch_size=args.syn_batch_size_train,
    )

    best_val = 0
    best_acc = 0
    for it in tqdm(range(args.syn_iteration+1), desc="distill", position=0, leave=True):
        if it in save_it_pool:
            syn_dataset.save(it)
            mean_val, mean_acc = eval_syn(graph_dataset, syn_dataset, args, is_distill=True)
            if mean_val > best_val:
                best_val = mean_val
                best_acc = mean_acc
            wandb.log({
                "val": mean_val,
                "acc": mean_acc,
            })

        # torch.cuda.empty_cache()
        syn_dataset.set_train_model()
        syn_dataset.zero_grad()

        student_model = ReparamModule(CLIP(args)).to(args.device)
        student_param = torch.cat([p.detach().reshape(-1) for p in student_model.parameters()], dim=0).requires_grad_(True)

        for step in range(args.syn_loop):
            _, student_param = epoch_train_manual(
                model=student_model,
                param=student_param,
                dataset=syn_dataset,
                sampler=inner_sampler,
                graph_encoder_lr=syn_dataset.graph_encoder_lr,
                text_encoder_lr=syn_dataset.text_encoder_lr,
                args=args
            )

        match_size, match_idx, match_adjs = next(iter(match_sampler))
        match_node_f = graph_dataset.node_f[match_idx].to(args.device)
        match_adjs = [adj.to(args.device) for adj in match_adjs]
        match_text_embeds = graph_dataset.text_embeds[match_idx[:match_size]].to(args.device)
        with torch.no_grad():
            expert_logits = expert_model(match_node_f, match_adjs, match_text_embeds, is_eval=True)

        student_logits = student_model(match_node_f, match_adjs, match_text_embeds, is_eval=True, flat_param=student_param)

        syn_loss = match_loss(student_logits, expert_logits)

        syn_dataset.compute_grad(syn_loss)
        syn_dataset.step()

        if it % 10 == 0:
            tqdm.write(f"syn_loss: {syn_loss.item()}")

        wandb.log({
            "loss": syn_loss.item(),
        })

    tqdm.write(f"Best Val: {best_val}, Best Acc: {best_acc}")
    wandb.summary["best_val"] = best_val
    wandb.summary["best_acc"] = best_acc
    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # base
    parser.add_argument("--dataset_name", type=str, default="computer")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=44)

    # distill
    parser.add_argument("--syn_iteration", type=int, default=5000)
    parser.add_argument("--syn_size", type=int, default=200)
    parser.add_argument("--syn_lr", type=float, default=2000)
    parser.add_argument("--syn_lr_lr", type=float, default=2e-5)
    parser.add_argument("--syn_loop", type=int, default=15)
    parser.add_argument("--syn_batch_size_train", type=int, default=20)
    parser.add_argument("--syn_match", type=int, default=2000)
    parser.add_argument("--syn_num_summary", type=int, default=4)
    parser.add_argument("--syn_ratio_summary", type=float, default=60.0)
    parser.add_argument("--save_interval", type=int, default=500)

    # eval
    parser.add_argument("--batch_size_train", type=int, default=20)
    parser.add_argument("--batch_size_test", type=int, default=2048)
    parser.add_argument("--num_epoch_train", type=int, default=15)
    parser.add_argument("--eval_time", type=int, default=3)

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
    args.name = f"{args.dataset_name}-{args.syn_size}-{args.seed}-{args.syn_num_summary}-{args.syn_ratio_summary}"
    args.buffer_save_dir = os.path.join("./buffer", args.dataset_name, args.graph_encoder, args.text_encoder)
    os.makedirs(os.path.join(str(args.buffer_save_dir), args.name), exist_ok=True)

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
    main(args)

