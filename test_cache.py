import copy
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SynTAGDataset, GraphDataset
from epoch import epoch_train, epoch_test
from model import TextEncoder, CLIP


def main(args):
    node_f_path = os.path.join(str(args.buffer_save_dir), f"{args.name}_node_f.pt")
    text_embeds_path = os.path.join(str(args.buffer_save_dir), f"{args.name}_text_embeds.pt")
    student_lr_graph_path = os.path.join(str(args.buffer_save_dir), f"{args.name}_student_lr_graph.pt")
    student_lr_text_path = os.path.join(str(args.buffer_save_dir), f"{args.name}_student_lr_text.pt")
    raw_text_path = os.path.join(str(args.buffer_save_dir), f"{args.name}_raw_text.json")

    # 读取 .pt 文件
    node_f = torch.load(node_f_path, map_location=args.device)
    text_embeds = torch.load(text_embeds_path, map_location=args.device)
    student_lr_graph = torch.load(student_lr_graph_path, map_location=args.device)
    student_lr_text = torch.load(student_lr_text_path, map_location=args.device)

    # 读取 JSON 文件
    with open(raw_text_path, "r") as file:
        text_summary = json.load(file)

    graph_dataset = GraphDataset(args)

    syn_dataset = SynTAGDataset(node_f, text_embeds, args)
    syn_dataset.student_lr_graph = student_lr_graph
    syn_dataset.student_lr_text = student_lr_text

    args_copy = copy.copy(args)
    args_copy.text_encoder = "roberta"
    roberta_encoder = TextEncoder(args_copy).to(args.device)
    roberta_embeds = roberta_encoder(text_summary)

    test_dataset = copy.deepcopy(graph_dataset)
    labels_with_desc = [f"{label} {desc}" for label, desc in zip(test_dataset.all_labels, test_dataset.labels_desc)]
    test_dataset.all_labels_embeds = roberta_encoder(labels_with_desc)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_test_eval, shuffle=False)

    gnn_model = ["mlp", "sage", "sgc", "gcn", "appnp", "cheby"]
    for gnn_type in gnn_model:
        roberta_acc_list = []
        for _ in tqdm(range(args.num_eval), desc="eval", position=1, leave=False):
            args_copy = copy.copy(args)
            args_copy.graph_encoder = gnn_type
            eval_model = CLIP(args_copy).to(args.device)

            eval_dataset = copy.deepcopy(syn_dataset)
            eval_dataset.set_eval_model()
            eval_dataset.text_embeds = roberta_embeds
            student_lr_graph = eval_dataset.student_lr_graph.item()
            student_lr_text = eval_dataset.student_lr_text.item()

            eval_train_loader = DataLoader(eval_dataset, batch_size=args.batch_size_train_eval, shuffle=True)
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
            roberta_acc_list.append(best_acc)

        print(f"{gnn_type} test acc: {np.mean(roberta_acc_list)}")


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # base
    parser.add_argument("--dataset_name", type=str, default="computer")
    parser.add_argument("--buffer_save_dir", type=str, default="./buffer")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=44)

    # distill
    parser.add_argument("--iterations", type=int, default=5001)
    parser.add_argument("--num_syn", type=int, default=200)
    parser.add_argument("--init_pair", type=str, default="summary")
    parser.add_argument("--syn_graph_lr", type=float, default=100)
    parser.add_argument("--lr_lr", type=float, default=2e-6)
    parser.add_argument("--syn_steps", type=int, default=15)
    parser.add_argument("--mini_batch_size", type=int, default=20)
    parser.add_argument("--num_target", type=int, default=2000)
    parser.add_argument("--loss_type", type=str, default="WBCE")

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
    parser.add_argument("--num_summary", type=int, default=4)
    parser.add_argument("--ratio_summary", type=float, default=60)

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


