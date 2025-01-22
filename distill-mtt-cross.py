import copy
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric import seed_everything
from tqdm import tqdm

import wandb
from data.dataset import GraphDataset
from epoch import epoch_train, epoch_test, epoch_train_manual, epoch_ft, epoch_ft_test
from model import CLIP, CoOp
from reparam_module import ReparamModule
from scheduler import Scheduler
from similarity_mining import MultilabelContrastiveLoss, calculate_cmi_emp
from data.dataset import SynTAGDataset
from utils import select_balanced_labels, select_cluster_center_labels, select_cluster_boundary_labels, \
    select_subgraph_labels, create_few_shot_index, aggregate_text, summary_text


def init_synthetic_data(dataset, args):
    subgraph_labels = None
    if args.init_pair == "balance":
        selected_idx = select_balanced_labels(dataset, args.num_syn)
        idx_shuffle = np.random.permutation(selected_idx)
    elif args.init_pair == "random" or "aggregate" or "summary":
        idx_shuffle = np.random.permutation(len(dataset))[:args.num_syn]
    elif args.init_pair == "text_kmeans_center":
        selected_idx = select_cluster_center_labels(dataset, args.num_syn)
        idx_shuffle = np.random.permutation(selected_idx)
    elif args.init_pair == "text_kmeans_bound":
        selected_idx = select_cluster_boundary_labels(dataset, args.num_syn)
        idx_shuffle = np.random.permutation(selected_idx)
    elif args.init_pair == "subgraph":
        selected_idx, subgraph_labels = select_subgraph_labels(dataset, args.num_syn//4, 4)
        idx_shuffle = selected_idx
        subgraph_labels = np.array(subgraph_labels)
    else:
        raise ValueError("Unknown init pair type")

    node_f = dataset.node_f
    graph_syn = torch.stack([node_f[dataset[i][0]] for i in idx_shuffle])
    if args.use_text_emb:
        text_syn = torch.stack([dataset[i][1] for i in idx_shuffle])
    else:
        text_syn = [dataset[i][1] for i in idx_shuffle]

    text_summary = []
    if args.init_pair == "aggregate":
        graph_syn, text_syn = aggregate_text(dataset, idx_shuffle, args)
    elif args.init_pair == "summary":
        graph_syn, text_syn, text_summary = summary_text(dataset, idx_shuffle, args)

    return graph_syn, text_syn, subgraph_labels, text_summary


def main(args):
    wandb.init(
        project="TAGC-distill-mtt-cross",
        name=args.name,
        config=args,
    )

    graph_dataset = GraphDataset(args)
    test_loader = DataLoader(graph_dataset, batch_size=args.batch_size_test_eval, shuffle=False)

    buffer_save_dir = args.buffer_save_dir
    n = 0
    param_trajectories = []
    while os.path.exists(os.path.join(str(buffer_save_dir), "replay_buffer_{}.pt".format(n))):
        temp_trajectory = torch.load(os.path.join(str(buffer_save_dir), f"replay_buffer_{n}.pt"))
        param_trajectories.append(temp_trajectory)
        n += 1

    eval_it_pool = np.arange(0, args.iterations, args.eval_interval).tolist()

    graph_syn, text_syn, subgraph_labels, text_summary = init_synthetic_data(graph_dataset, args)
    syn_dataset = SynTAGDataset(graph_syn, text_syn, args)

    expert_idx = 0
    gnn_model = ["mlp", "sage", "sgc", "gcn", "appnp", "cheby"]

    for it in tqdm(range(args.iterations), desc="distill", position=0, leave=True):
        if it in eval_it_pool:
            for gnn_type in gnn_model:
                acc_list = []
                for _ in tqdm(range(args.num_eval), desc="eval", position=1, leave=False):
                    args_copy = copy.copy(args)
                    args_copy.graph_encoder = gnn_type
                    eval_model = CLIP(args_copy).to(args.device)

                    syn_dataset.set_eval_model()

                    student_lr_graph = syn_dataset.student_lr_graph.item()
                    student_lr_text = syn_dataset.student_lr_text.item()

                    eval_train_loader = DataLoader(syn_dataset, batch_size=args.batch_size_train_eval, shuffle=True)
                    eval_optimizer = torch.optim.SGD([
                        {"params": eval_model.graph_encoder.parameters(), "lr": student_lr_graph, "momentum": 0.9,
                         "weight_decay": 5e-4},
                        {"params": eval_model.text_encoder.parameters(), "lr": student_lr_text, "momentum": 0.9,
                         "weight_decay": 5e-4},
                    ])

                    best_acc = 0
                    for epoch in range(args.train_epochs_eval):
                        epoch_train(model=eval_model, optimizer=eval_optimizer, train_loader=eval_train_loader,
                                    args=args, is_distill=True)
                        acc = epoch_test(model=eval_model, test_loader=test_loader, args=args, is_distill=True)
                        if acc > best_acc:
                            best_acc = acc
                    acc_list.append(best_acc)

                print(f"{gnn_type}_test_acc", np.mean(acc_list))
                wandb.log({f"{gnn_type}_test_acc": np.mean(acc_list)}, step=it)

        param_trajectory = param_trajectories[expert_idx]
        expert_idx += 1
        if expert_idx == len(param_trajectories):
            expert_idx = 0
            np.random.shuffle(param_trajectories)

        syn_dataset.set_train_model()
        syn_dataset.zero_grad()

        start_epoch = np.random.randint(0, args.max_start_epoch)

        student_model = ReparamModule(CLIP(args)).to(args.device)
        student_param_start = torch.cat([p.detach().reshape(-1) for p in param_trajectory[start_epoch]], dim=0).to(args.device)
        student_param_target = torch.cat([p.detach().reshape(-1) for p in param_trajectory[start_epoch+args.match_epoch]], dim=0).to(args.device)
        student_param = student_param_start.clone().requires_grad_(True).to(args.device)
        syn_loader = DataLoader(syn_dataset, batch_size=args.mini_batch_size, shuffle=True)

        for step in range(args.syn_steps):
            loss, student_param = epoch_train_manual(model=student_model,
                                                     param=student_param,
                                                     lr_graph=syn_dataset.student_lr_graph,
                                                     lr_text=syn_dataset.student_lr_text,
                                                     train_loader=syn_loader,
                                                     args=args)

        graph_mask = student_model.get_params_mask_for_module("graph_encoder", student_param)
        text_mask = student_model.get_params_mask_for_module("text_encoder", student_param)
        graph_param = student_param[graph_mask]
        text_param = student_param[text_mask]
        graph_param_start = student_param_start[graph_mask]
        text_param_start = student_param_start[text_mask]
        graph_param_target = student_param_target[graph_mask]
        text_param_target = student_param_target[text_mask]

        graph_param_loss = torch.nn.functional.mse_loss(graph_param, graph_param_target, reduction="sum")
        graph_param_dist = torch.nn.functional.mse_loss(graph_param_start, graph_param_target, reduction="sum")
        text_param_loss = torch.nn.functional.mse_loss(text_param, text_param_target, reduction="sum")
        text_param_dist = torch.nn.functional.mse_loss(text_param_start, text_param_target, reduction="sum")

        syn_loss = (graph_param_loss / graph_param_dist + text_param_loss / text_param_dist) / 2

        syn_dataset.compute_grad(syn_loss)
        syn_dataset.update()

        print(f"Syn Loss: {syn_loss.item()}， Graph LR: {syn_dataset.student_lr_graph.item()}, Text LR: {syn_dataset.student_lr_text.item()}")
        print(f"Graph Grad: {syn_dataset.node_f.grad.norm(p=1).item()}, Graph LR Grad: {syn_dataset.student_lr_graph.grad.item()}, Text LR Grad: {syn_dataset.student_lr_text.grad.item()}")
        wandb.log({
            "syn_loss": syn_loss.item(),
            "graph_lr": syn_dataset.student_lr_graph.item(),
            "text_lr": syn_dataset.student_lr_text.item(),
        }, step=it)

    torch.save(syn_dataset.node_f, os.path.join(str(args.buffer_save_dir), f"{args.name}_node_f.pt"))
    torch.save(syn_dataset.text_embeds, os.path.join(str(args.buffer_save_dir), f"{args.name}_text_embeds.pt"))
    torch.save(syn_dataset.student_lr_graph,
               os.path.join(str(args.buffer_save_dir), f"{args.name}_student_lr_graph.pt"))
    torch.save(syn_dataset.student_lr_text, os.path.join(str(args.buffer_save_dir), f"{args.name}_student_lr_text.pt"))
    with open(os.path.join(str(args.buffer_save_dir), f"{args.name}_raw_text.json"), "w") as file:
        json.dump(text_summary, file)
    wandb.finish()
    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # base
    parser.add_argument("--dataset_name", type=str, default="photo")
    parser.add_argument("--buffer_save_dir", type=str, default="./buffer")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=44)

    # distill
    parser.add_argument("--iterations", type=int, default=5001)
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
    parser.add_argument("--max_start_epoch", type=int, default=10)
    parser.add_argument("--match_epoch", type=int, default=3)

    # eval
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--num_eval", type=int, default=5)
    parser.add_argument("--batch_size_train_eval", type=int, default=32)
    parser.add_argument("--train_epochs_eval", type=int, default=15)
    parser.add_argument("--batch_size_test_eval", type=int, default=512)
    parser.add_argument("--ft_epochs_eval", type=int, default=50)
    parser.add_argument('--coop_n_ctx', type=int, default=10)
    parser.add_argument('--prompt_lr', type=float, default=0.01)
    parser.add_argument('--context_length', type=int, default=128)

    # text type
    parser.add_argument("--use_text_emb", type=bool, default=True)

    # graph encoder
    parser.add_argument("--graph_encoder", type=str, default="gcn")
    parser.add_argument("--lr_graph_encoder", type=float, default=5e-3)

    # text encoder
    parser.add_argument("--text_encoder", type=str, default="bert")
    parser.add_argument("--lr_text_encoder", type=float, default=5e-3)
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
    args.device = f"cuda:{args.gpu}"
    args.buffer_save_dir = os.path.join(args.buffer_save_dir, args.dataset_name, args.graph_encoder, args.text_encoder)
    args.name = f"{args.dataset_name}-{args.num_syn}-{args.init_pair}-{args.seed}-{args.num_summary}-{args.ratio_summary}"
    if args.dataset_name == "cora":
        args.gnn_input_dim = 128
        args.gnn_hidden_dim = 128
        args.gnn_output_dim = 128
    else:
        args.gnn_input_dim = 384
        args.gnn_hidden_dim = 384
        args.gnn_output_dim = 384
    seed_everything(args.seed)
    main(args)


def seed_everything(seed=42):
    """
    Set the random seed for reproducibility.

    Parameters:
    seed (int): The seed value to set for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # For deterministic behavior in CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False