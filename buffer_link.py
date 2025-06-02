import os
import random
import numpy as np
import torch
from model import CLIP, LinkPredictor

from epoch import epoch_train_link
from dataset import GraphDataset



def main(args):
    graph_dataset = GraphDataset(args)

    expert_model = CLIP(args).to(args.device)
    expert_state = torch.load(os.path.join(str(args.buffer_save_dir), f"expert_state.pt"), map_location=args.device, weights_only=True)
    expert_model.load_state_dict(expert_state)
    expert_model.eval()

    link_model = LinkPredictor(args.gnn_output_dim).to(args.device)
    link_model.train()

    optimizer = torch.optim.Adam(link_model.parameters(), lr=args.link_lr, weight_decay=5e-4)
    for epoch in range(args.num_epoch_train):
        loss, train_auc = epoch_train_link(model=expert_model, decoder=link_model, dataset=graph_dataset, optimizer=optimizer, args=args)
        print(f"Link Train Epoch: {epoch}, Loss: {loss:.4f}, AUC: {train_auc:.4f}")

    torch.save(link_model.state_dict(), os.path.join(str(args.buffer_save_dir), f"link_model.pt"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # base
    parser.add_argument("--link_lr", type=float, default=5e-3)

    # base
    parser.add_argument("--dataset_name", type=str, default="photo")
    parser.add_argument("--num_epoch_train", type=int, default=500)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size_train", type=int, default=102400)
    parser.add_argument("--batch_size_test", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=44)

    # graph encoder
    parser.add_argument("--graph_encoder", type=str, default="gcn")
    parser.add_argument("--graph_encoder_lr", type=float, default=2e-5)
    parser.add_argument("--gnn_input_dim", type=int, default=384)
    parser.add_argument("--gnn_hidden_dim", type=int, default=384)
    parser.add_argument("--gnn_output_dim", type=int, default=384)

    # text encoder
    parser.add_argument("--text_encoder", type=str, default="bert")
    parser.add_argument("--text_encoder_lr", type=float, default=2e-5)
    parser.add_argument("--lm_output_dim", type=int, default=768)

    args = parser.parse_args()
    args.buffer_save_dir = os.path.join("./buffer", args.dataset_name, args.graph_encoder, args.text_encoder)
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    if args.dataset_name == "art":
        args.sample_size = [10, 10]
    elif args.dataset_name == "products":
        args.sample_size = [10, 5]
    else:
        args.sample_size = [-1, -1]

    os.makedirs(args.buffer_save_dir, exist_ok=True)

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