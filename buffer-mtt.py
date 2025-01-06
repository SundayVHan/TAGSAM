import os.path
from datetime import datetime

import torch
import wandb
from torch_geometric.data import DataLoader

from data.dataset import GraphDataset
from epoch import epoch_train, epoch_test
from model import CLIP


def main(args):
    wandb.init(
        project="TAGC-buffer",
        name=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        config=args,
    )

    buffer_save_dir = os.path.join(args.buffer_save_dir, args.dataset_name, args.graph_encoder, args.text_encoder)
    if not os.path.exists(buffer_save_dir):
        os.makedirs(buffer_save_dir)

    graph_dataset = GraphDataset(args)
    train_loader = DataLoader(graph_dataset, batch_size=args.batch_size_train, shuffle=True)
    test_loader = DataLoader(graph_dataset, batch_size=args.batch_size_test, shuffle=False)

    param_trajectory = []

    for it in range(0, args.num_expert):
        expert_model = CLIP(args).to(args.device)

        optimizer = torch.optim.Adam([
            {"params": expert_model.graph_encoder.parameters(), "lr": args.lr_graph_encoder},
            {"params": expert_model.text_encoder.parameters(), "lr": args.lr_text_encoder},
        ])
        optimizer.zero_grad()

        param_timestamp = [[p.detach().cpu() for p in expert_model.parameters()]]

        for e in range(args.num_epochs):
            epoch_train(model=expert_model, optimizer=optimizer, train_loader=train_loader, args=args)
            acc = epoch_test(model=expert_model, test_loader=test_loader, args=args)

            param_timestamp.append([p.detach().cpu() for p in expert_model.graph_encoder.parameters()])
            wandb.log({f"acc": acc}, step=e)
            print(f"Epoch {e} Acc: {acc}")

        param_trajectory.append(param_timestamp)

        n = 0
        while os.path.exists(os.path.join(str(buffer_save_dir), "replay_buffer_{}.pt".format(n))):
            n += 1
        torch.save(param_trajectory, os.path.join(str(buffer_save_dir), "replay_buffer_{}.pt".format(n)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # base
    parser.add_argument("--dataset_name", type=str, default="computer")
    parser.add_argument("--buffer_save_dir", type=str, default="./buffer")
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size_train", type=int, default=1024)
    parser.add_argument("--batch_size_test", type=int, default=1024)
    parser.add_argument("--num_expert", type=int, default=10)

    # text type
    parser.add_argument("--use_text_emb", type=bool, default=True)

    # graph encoder
    parser.add_argument("--graph_encoder", type=str, default="gcn")
    parser.add_argument("--lr_graph_encoder", type=float, default=2e-5)
    parser.add_argument("--gnn_input_dim", type=int, default=128)
    parser.add_argument("--gnn_hidden_dim", type=int, default=128)
    parser.add_argument("--gnn_output_dim", type=int, default=128)

    # text encoder
    parser.add_argument("--text_encoder", type=str, default="bert")
    parser.add_argument("--lr_text_encoder", type=float, default=2e-5)
    parser.add_argument("--text_emb_dim", type=int, default=768)

    # task
    parser.add_argument('--k_spt', type=int, default=5)
    parser.add_argument('--k_val', type=int, default=5)
    parser.add_argument('--k_qry', type=int, default=50)
    parser.add_argument('--n_way', type=int, default=5)
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

    main(args)