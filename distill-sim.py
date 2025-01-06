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

    if args.init_pair == "aggregate":
        graph_syn, text_syn = aggregate_text(dataset, idx_shuffle, args)
    elif args.init_pair == "summary":
        graph_syn, text_syn = summary_text(dataset, idx_shuffle, args)

    return graph_syn, text_syn, subgraph_labels


def main(args):
    wandb.init(
        project="TAGC",
        name=f"{args.dataset_name}-{args.num_syn}-{args.init_pair}-{args.seed}",
        config=args,
    )

    graph_dataset = GraphDataset(args)
    test_loader = DataLoader(graph_dataset, batch_size=args.batch_size_test_eval, shuffle=False)

    origin_node_f = graph_dataset.node_f.to(args.device)
    origin_edge_index = graph_dataset.edge_index.to(args.device)
    if args.use_text_emb:
        origin_texts = graph_dataset.text_embeds.to(args.device)
    else:
        origin_texts = graph_dataset.text_list

    buffer_save_dir = os.path.join(args.buffer_save_dir, args.dataset_name, args.graph_encoder, args.text_encoder)
    expert_model = CLIP(args).to(args.device)
    expert_model.load_state_dict(torch.load(os.path.join(str(buffer_save_dir), f"expert_state_{args.use_text_emb}.pt")))
    expert_model.eval()

    eval_it_pool = np.arange(0, args.iterations, args.eval_interval).tolist()

    criterion = MultilabelContrastiveLoss(args.loss_type).to(args.device)

    graph_syn, text_syn, subgraph_labels = init_synthetic_data(graph_dataset, args)
    syn_dataset = SynTAGDataset(graph_syn, text_syn, args)

    scheduler = Scheduler(dataset=graph_dataset,
                          batch_size=args.num_target,
                          max_iteration=args.cl_iterations,
                          strategy=args.cl_strategy,
                          init_proportion=args.cl_init_proportion)
    # scheduler.difficulty_measure(expert_logits=expert_logits)
    scheduler.update_schedule()

    for it in tqdm(range(args.iterations), desc="distill", position=0, leave=True):
        if it in eval_it_pool:
            acc_list = []
            ft_acc_list = []
            for _ in tqdm(range(args.num_eval), desc="eval", position=1, leave=False):
                eval_model = CLIP(args).to(args.device)

                syn_dataset.set_eval_model()
                student_lr_graph = syn_dataset.student_lr_graph.item()
                student_lr_text = syn_dataset.student_lr_text.item()

                eval_train_loader = DataLoader(syn_dataset, batch_size=args.batch_size_train_eval, shuffle=True)
                eval_optimizer = torch.optim.SGD([
                    {"params": eval_model.graph_encoder.parameters(), "lr": student_lr_graph, "momentum": 0.9, "weight_decay": 5e-4},
                    {"params": eval_model.text_encoder.parameters(), "lr": student_lr_text, "momentum": 0.9, "weight_decay": 5e-4},
                ])

                best_acc = 0
                for epoch in range(args.train_epochs_eval):
                    epoch_train(model=eval_model, optimizer=eval_optimizer, train_loader=eval_train_loader, args=args, is_distill=True)
                    acc = epoch_test(model=eval_model, test_loader=test_loader, args=args, is_distill=True)
                    if acc > best_acc:
                        best_acc = acc
                acc_list.append(best_acc)
                # all_labels = test_loader.dataset.all_labels
                # labels_desc = test_loader.dataset.labels_desc
                # coop = CoOp(args, all_labels, eval_model, labels_desc)
                # ft_index, ft_test_index = create_few_shot_index(test_loader.dataset)
                #
                # best_acc = 0
                # for epoch in range(args.ft_epochs_eval):
                #     epoch_ft(coop, test_loader, ft_index, args)
                #     acc = epoch_ft_test(coop, test_loader, ft_test_index, args)
                #     if acc > best_acc:
                #         best_acc = acc
                # ft_acc_list.append(best_acc)

            print(f"Test Acc: {np.mean(acc_list)}")
            wandb.log({"test_acc": np.mean(acc_list)}, step=it)
            # wandb.log({"ft_test_acc": np.mean(ft_acc_list)}, step=it)

        syn_dataset.set_train_model()
        syn_dataset.zero_grad()

        student_model = ReparamModule(CLIP(args)).to(args.device)
        student_param = torch.cat([p.detach().reshape(-1) for p in student_model.parameters()], dim=0).requires_grad_(True)
        syn_loader = DataLoader(syn_dataset, batch_size=args.mini_batch_size, shuffle=True)

        for step in range(args.syn_steps):
            loss, student_param = epoch_train_manual(model=student_model,
                                                     param=student_param,
                                                     lr_graph=syn_dataset.student_lr_graph,
                                                     lr_text=syn_dataset.student_lr_text,
                                                     train_loader=syn_loader,
                                                     args=args)

        target_idx = next(scheduler)
        with torch.no_grad():
            expert_logits = expert_model(origin_node_f, origin_edge_index, target_idx, origin_texts[target_idx], is_eval=True)

        student_logits = student_model(origin_node_f, origin_edge_index, target_idx, origin_texts[target_idx], is_eval=True, flat_param=student_param)

        syn_loss = criterion(student_logits, expert_logits)

        syn_dataset.compute_grad(syn_loss)
        syn_dataset.update()

        print(f"Syn Loss: {syn_loss.item()}， Graph LR: {syn_dataset.student_lr_graph.item()}, Text LR: {syn_dataset.student_lr_text.item()}")
        print(f"Graph Grad: {syn_dataset.node_f.grad.norm(p=1).item()}, Graph LR Grad: {syn_dataset.student_lr_graph.grad.item()}, Text LR Grad: {syn_dataset.student_lr_text.grad.item()}")
        wandb.log({
            "syn_loss": syn_loss.item(),
            "graph_lr": syn_dataset.student_lr_graph.item(),
            "text_lr": syn_dataset.student_lr_text.item(),
        }, step=it)

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
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if args.dataset_name == "cora" or args.dataset_name == "arxiv" or args.dataset_name == "art":
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