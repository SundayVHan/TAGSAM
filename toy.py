import argparse
import os

import numpy as np
import pandas as pd
import torch
from spacy.lang.ja.syntax_iterators import labels
from torch.utils.data import DataLoader
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm

from data.dataset import GraphDataset
from epoch import epoch_test
from model import CLIP
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_features(graph_feature, text_feature, labels):
    # 将特征移动到 CPU（如果它们在 GPU 上）
    graph_feature = graph_feature.cpu().detach().numpy()
    text_feature = text_feature.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()  # 假设 labels 是一个 torch 张量

    # 使用 t-SNE 将特征降维到 2D
    tsne = TSNE(n_components=2, random_state=42)
    all_features = np.vstack((graph_feature, text_feature))
    all_features_2d = tsne.fit_transform(all_features)

    # 分离降维后的特征
    graph_feature_2d = all_features_2d[:len(graph_feature)]
    text_feature_2d = all_features_2d[len(graph_feature):]

    # 创建图形
    plt.figure(figsize=(8, 8))

    # 绘制 graph_feature，使用圆形标记
    scatter1 = plt.scatter(graph_feature_2d[:, 0], graph_feature_2d[:, 1], c=labels[:len(graph_feature)],
                           cmap='viridis', marker='o', label='Graph Features', alpha=0.6)

    # 绘制 text_feature，使用三角形标记
    scatter2 = plt.scatter(text_feature_2d[:, 0], text_feature_2d[:, 1], c=labels[len(graph_feature):],
                           cmap='viridis', marker='^', label='Text Features', alpha=0.6)

    # 添加图例和标题
    plt.legend(loc='best')
    plt.title('t-SNE Visualization of Graph and Text Features')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # 添加颜色条
    plt.colorbar(scatter1, label='Label')

    plt.savefig("./vis", bbox_inches='tight')


def main(args):
    graph_dataset = GraphDataset(args)
    test_loader = DataLoader(graph_dataset, batch_size=args.batch_size_test_eval, shuffle=False)

    buffer_save_dir = os.path.join(args.buffer_save_dir, args.dataset_name, args.graph_encoder, args.text_encoder)
    expert_model = CLIP(args).to(args.device)
    expert_model.load_state_dict(torch.load(os.path.join(str(buffer_save_dir), f"expert_state_{args.use_text_emb}.pt")))
    expert_model.eval()

    node_f = test_loader.dataset.node_f.to(args.device)
    edge_index = test_loader.dataset.edge_index.to(args.device)

    label_to_idx = {label: idx for idx, label in enumerate(test_loader.dataset.all_labels)}
    label = [label_to_idx[test_loader.dataset.label_list[i]] for i in range(len(test_loader.dataset.label_list))]
    label = torch.tensor(label).to(args.device)

    graph_feature = []
    text_feature = []
    for i, (node_idx, texts) in enumerate(tqdm(test_loader, disable=False)):
        node_idx = node_idx.to(args.device)
        texts = texts.to(args.device)

        _, graph_f, text_f = expert_model(node_f, edge_index, node_idx, texts, is_eval=True, is_distill=True)
        graph_feature.append(graph_f)
        text_feature.append(text_f)

    graph_feature = torch.cat(graph_feature, dim=0)
    text_feature = torch.cat(text_feature, dim=0)
    visualize_features(graph_feature, text_feature, label)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # base
    parser.add_argument("--dataset_name", type=str, default="photo")
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
