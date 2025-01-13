import os
import random
from datetime import datetime
from pyexpat import features

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from torch_geometric import seed_everything
from tqdm import tqdm

from data.dataset import GraphDataset, SynTAGDataset
from epoch import epoch_train, epoch_test
from model import CLIP

@torch.no_grad()
def extract_features(model, dataloader, args):
    buffer_save_dir = os.path.join(args.buffer_save_dir, args.dataset_name, args.graph_encoder, args.text_encoder)
    if os.path.exists(os.path.join(buffer_save_dir, f"features.pt")):
        features = torch.load(os.path.join(buffer_save_dir, f"features.pt"))
        return features
    graph_feature = []
    text_feature = []

    for i, (node_idx, texts) in enumerate(tqdm(dataloader, disable=False)):
        node_idx = node_idx.to(args.device)
        texts = texts.to(args.device)

        node_f = dataloader.dataset.node_f.to(args.device)
        edge_index = dataloader.dataset.edge_index.to(args.device)

        _, graph_f, text_f = model(node_f, edge_index, node_idx, texts, is_eval=True, is_distill=True)
        graph_feature.append(graph_f)
        text_feature.append(text_f)

    graph_feature = torch.cat(graph_feature, dim=0)
    text_feature = torch.cat(text_feature, dim=0)

    feature = torch.cat([graph_feature, text_feature], dim=1)
    torch.save(feature, os.path.join(buffer_save_dir, f"features.pt"))

    return feature


def herding_algorithm(model, dataloader, coreset_size, args):
    # 提取特征
    features = extract_features(model, dataloader, args)
    features = features.to(args.device)

    # 计算数据集中心
    dataset_center = torch.mean(features, dim=0)

    # 初始化核心集
    coreset_idx = []
    coreset_features = []
    selected_mask = torch.zeros(features.size(0), dtype=torch.bool, device=args.device)

    # 迭代选择样本
    for _ in tqdm(range(coreset_size), desc="herding", position=1, leave=False):
        if len(coreset_features) == 0:
            coreset_center = torch.zeros_like(dataset_center)
        else:
            coreset_center = torch.mean(torch.stack(coreset_features), dim=0)

        # 计算每个样本到核心集中心的距离
        batch_size = 1024  # 根据 GPU 内存情况调整这个值
        distances = []
        for i in range(0, features.size(0), batch_size):
            batch_features = features[i:i + batch_size]
            batch_distances = torch.norm(batch_features - coreset_center, dim=1)
            distances.append(batch_distances)
        distances = torch.cat(distances)
        distances[selected_mask] = float('inf')

        # 找到距离最近的样本
        closest_index = torch.argmin(distances).item()

        # 添加该样本到核心集
        coreset_idx.append(closest_index)
        coreset_features.append(features[closest_index])

        selected_mask[closest_index] = True

    return coreset_idx


def k_center_algorithm(model, dataloader, coreset_size, args):
    features = extract_features(model, dataloader, args)
    features = features.to(args.device)

    # 随机选择一个初始中心
    num_samples = features.shape[0]
    initial_index = torch.randint(0, num_samples, (1,), device=args.device).item()
    centers = [features[initial_index]]
    selected_indices = [initial_index]

    # 迭代选择中心点
    for _ in tqdm(range(1, coreset_size), desc="k_center", position=1, leave=False):
        # 计算每个样本到最近中心的最小距离
        min_distances = torch.full((num_samples,), float('inf'), device=args.device)

        # 分批计算距离
        batch_size = 1024  # 根据 GPU 内存情况调整这个值
        for i in range(0, num_samples, batch_size):
            batch_features = features[i:i + batch_size]
            batch_distances = torch.stack([torch.norm(batch_features - center, dim=1) for center in centers])
            batch_min_distances = torch.min(batch_distances, dim=0).values

            # 更新全局最小距离
            min_distances[i:i + batch_size] = torch.min(min_distances[i:i + batch_size], batch_min_distances)

        # 找到距离最近中心最远的样本
        next_center_index = torch.argmax(min_distances).item()
        centers.append(features[next_center_index])
        selected_indices.append(next_center_index)

    return selected_indices


def main(args):
    wandb.init(
        project="TAGC-coreset",
        name=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        config=args,
    )

    dataset = GraphDataset(args)
    test_loader = DataLoader(dataset, batch_size=args.batch_size_test_eval, shuffle=False)

    buffer_save_dir = os.path.join(args.buffer_save_dir, args.dataset_name, args.graph_encoder, args.text_encoder)
    expert_model = CLIP(args).to(args.device)
    expert_model.load_state_dict(torch.load(os.path.join(str(buffer_save_dir), f"expert_state_{args.use_text_emb}.pt")))
    expert_model.eval()

    if args.init_pair == "herding":
        selected_idx = herding_algorithm(expert_model, test_loader, args.num_syn, args)
    elif args.init_pair == "k_center":
        selected_idx = k_center_algorithm(expert_model, test_loader, args.num_syn, args)
    elif args.init_pair == "random":
        selected_idx = random.sample(range(len(dataset)), args.num_syn)
    selected_idx = torch.tensor(selected_idx)
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_idx.tolist())}
    node_f = dataset.node_f
    graph_syn = torch.stack([node_f[dataset[i][0]] for i in selected_idx])
    text_syn = torch.stack([dataset[i][1] for i in selected_idx])
    mask = (torch.isin(dataset.edge_index[0], selected_idx) & torch.isin(dataset.edge_index[1], selected_idx))
    adj = dataset.edge_index[:, mask]
    adj_mapped = torch.stack([torch.tensor([index_map[idx.item()] for idx in adj_row]) for adj_row in adj])
    eval_dataset = SynTAGDataset(graph_syn, text_syn, args)
    # eval_dataset.edge_index = adj_mapped

    acc_list = []
    for _ in tqdm(range(args.num_eval), desc="eval", position=1, leave=False):
        eval_model = CLIP(args).to(args.device)

        eval_dataset.set_eval_model()
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
            epoch_train(model=eval_model, optimizer=eval_optimizer, train_loader=eval_train_loader, args=args,
                        is_distill=True)
            acc = epoch_test(model=eval_model, test_loader=test_loader, args=args, is_distill=True)
            if acc > best_acc:
                best_acc = acc
        acc_list.append(best_acc)

    print(f"Test Acc: {np.mean(acc_list)}")
    wandb.log({"Test Acc": np.mean(acc_list)})


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # base
    parser.add_argument("--dataset_name", type=str, default="photo")
    parser.add_argument("--buffer_save_dir", type=str, default="./buffer")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=44)

    # distill
    parser.add_argument("--iterations", type=int, default=5001)
    parser.add_argument("--num_syn", type=int, default=100)
    parser.add_argument("--init_pair", type=str, default="herding")
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
    if args.dataset_name == "cora" or args.dataset_name == "art":
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