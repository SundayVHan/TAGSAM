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
from model import CLIP, GCN, GraphEncoder
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_features(graph_feature, text_feature, labels, save_path=None):
    # 将特征移动到 CPU（如果它们在 GPU 上）
    shuffle_idx = np.random.permutation(len(graph_feature))
    shuffle_idx = shuffle_idx[:1000]
    graph_feature = graph_feature.cpu().detach().numpy()
    text_feature = text_feature.cpu().detach().numpy()
    graph_feature = graph_feature[shuffle_idx]
    text_feature = text_feature[shuffle_idx]
    labels = labels.cpu().detach().numpy()  # 假设 labels 是一个 torch 张量

    # 使用 t-SNE 将特征降维到 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    all_features = np.vstack((graph_feature, text_feature))
    all_features_2d = tsne.fit_transform(all_features)

    # 分离降维后的特征
    graph_feature_2d = all_features_2d[:len(graph_feature)]
    text_feature_2d = all_features_2d[len(graph_feature):]

    # 创建图形
    plt.figure(figsize=(8, 8))

    # 绘制 graph_feature，使用圆形标记
    scatter1 = plt.scatter(graph_feature_2d[:, 0], graph_feature_2d[:, 1], c=labels,
                           cmap='viridis', marker='o', label='Graph Features', alpha=0.6, s=20)

    # 绘制 text_feature，使用三角形标记
    scatter2 = plt.scatter(text_feature_2d[:, 0], text_feature_2d[:, 1], c=labels,
                           cmap='viridis', marker='^', label='Text Features', alpha=0.6, s=20)

    # 添加图例和标题
    plt.legend(loc='best')
    plt.title('t-SNE Visualization of Graph and Text Features')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # 添加颜色条
    plt.colorbar(scatter1, label='Label')

    plt.savefig("./vis", bbox_inches='tight')

@torch.no_grad()
def similarity(args):
    graph_dataset = GraphDataset(args)
    test_loader = DataLoader(graph_dataset, batch_size=args.batch_size_test_eval, shuffle=False)
    expert_model = CLIP(args).to(args.device)

    node_f = test_loader.dataset.node_f.to(args.device)
    edge_index = test_loader.dataset.edge_index.to(args.device)

    label_to_idx = {label: idx for idx, label in enumerate(test_loader.dataset.all_labels)}
    label = [label_to_idx[test_loader.dataset.label_list[i]] for i in range(len(test_loader.dataset.label_list))]
    label = torch.tensor(label).to(args.device)

    buffer_save_dir = os.path.join(args.buffer_save_dir, args.dataset_name, args.graph_encoder, args.text_encoder)

    def load_parameters_to_model(model, parameters):
        for model_param, saved_param in zip(model.parameters(), parameters):
            model_param.data.copy_(saved_param)

    torch.manual_seed(42)
    sample_idx = torch.randint(0, len(test_loader.dataset), (7,))
    print([label[i] for i in sample_idx])
    num_batch = sample_idx.shape[0]

    sims = []

    for x in range(4):
        temp_trajectory = torch.load(os.path.join(str(buffer_save_dir), f"replay_buffer_{x}.pt"))
        state = temp_trajectory[-1]
        load_parameters_to_model(expert_model, state)

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
        graph_feature = graph_feature[sample_idx]
        text_feature = text_feature[sample_idx]
        sim = graph_feature @ text_feature.T
        sims.append(sim)

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    # 在第一个子图上绘制 matrix1 的热力图
    axes[0].imshow(sims[0].cpu(), cmap='hot', interpolation='nearest')

    # 在第二个子图上绘制 matrix2 的热力图
    axes[1].imshow(sims[1].cpu(), cmap='hot', interpolation='nearest')

    # 在第三个子图上绘制 matrix3 的热力图
    axes[2].imshow(sims[2].cpu(), cmap='hot', interpolation='nearest')

    axes[3].imshow(sims[3].cpu(), cmap='hot', interpolation='nearest')

    # 显示图像
    plt.tight_layout()
    plt.savefig("./vis/similarity", bbox_inches='tight')

def train(args):
    graph_dataset = GraphDataset(args)
    node_f = graph_dataset.node_f.to(args.device)
    edge_index = graph_dataset.edge_index.to(args.device)
    train_loader = DataLoader(graph_dataset, batch_size=args.batch_size_train_eval, shuffle=True)
    test_loader = DataLoader(graph_dataset, batch_size=args.batch_size_test_eval, shuffle=False)

    label_to_idx = {label: idx for idx, label in enumerate(test_loader.dataset.all_labels)}
    label = [label_to_idx[test_loader.dataset.label_list[i]] for i in range(len(test_loader.dataset.label_list))]
    label = torch.tensor(label).to(args.device)

    plt.figure(figsize=(8, 6))
    torch.manual_seed(42)
    sample_idx = torch.randint(0, len(test_loader.dataset), (7,))
    num_batch = sample_idx.shape[0]
    g_f = []
    for i in range(10):
        gnn = GraphEncoder(args).to(args.device)
        classifier = torch.nn.Linear(args.gnn_output_dim, len(graph_dataset.all_labels)).to(args.device)
        optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr_graph_encoder)
        for epoch in range(3):
            for i, (node_idx, texts) in enumerate(tqdm(train_loader, disable=False)):
                node_idx = node_idx.to(args.device)

                optimizer.zero_grad()
                output = gnn(node_f, edge_index, node_idx)
                output = classifier(output)
                loss = torch.nn.functional.cross_entropy(output, label[node_idx])
                loss.backward()
                optimizer.step()

        graph_f = []
        for i, (node_idx, texts) in enumerate(tqdm(test_loader, disable=False)):
            node_idx = node_idx.to(args.device)
            output = gnn(node_f, edge_index, node_idx)
            graph_f.append(output)

        graph_f = torch.cat(graph_f, dim=0)

        graph_feature = graph_f[sample_idx]
        g_f.append(graph_feature.detach().clone())
        del gnn, classifier, optimizer
        torch.cuda.empty_cache()  # 清理显存

    graph_feature = torch.cat(g_f, dim=0)

    reducer = TSNE(n_components=2, perplexity=5, random_state=42)
    reduce_g = reducer.fit_transform(graph_feature.cpu().detach().numpy())
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w',  # 原有颜色
                'orange', 'purple', 'pink', 'lime', 'teal', 'brown', 'gray', 'olive', 'navy']
    markers = ['o', 's', '^', 'p', '*', 'h', '+']

    for i in range(10):
        for j in range(num_batch):
            index = i * num_batch + j
            g = reduce_g[index]
            plt.scatter(g[0], g[1], c=colors[i], marker=markers[j], alpha=0.6, label=f'sample {j}')
    plt.savefig("./vis/gnn", bbox_inches='tight')

def param(args):
    graph_dataset = GraphDataset(args)
    node_f = graph_dataset.node_f.to(args.device)
    edge_index = graph_dataset.edge_index.to(args.device)
    train_loader = DataLoader(graph_dataset, batch_size=args.batch_size_train_eval, shuffle=True)
    test_loader = DataLoader(graph_dataset, batch_size=args.batch_size_test_eval, shuffle=False)

    label_to_idx = {label: idx for idx, label in enumerate(test_loader.dataset.all_labels)}
    label = [label_to_idx[test_loader.dataset.label_list[i]] for i in range(len(test_loader.dataset.label_list))]
    label = torch.tensor(label).to(args.device)

    plt.figure(figsize=(8, 6))
    torch.manual_seed(42)
    sample_idx = torch.randint(0, len(test_loader.dataset), (7,))
    num_batch = sample_idx.shape[0]
    param1 = []
    param2 = []
    for i in range(10):
        gnn = GraphEncoder(args).to(args.device)
        classifier = torch.nn.Linear(args.gnn_output_dim, len(graph_dataset.all_labels)).to(args.device)
        optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr_graph_encoder)
        for epoch in range(3):
            for i, (node_idx, texts) in enumerate(tqdm(train_loader, disable=False)):
                node_idx = node_idx.to(args.device)

                optimizer.zero_grad()
                output = gnn(node_f, edge_index, node_idx)
                output = classifier(output)
                loss = torch.nn.functional.cross_entropy(output, label[node_idx])
                loss.backward()
                optimizer.step()
        flattened_params_1 = []
        flattened_params_2 = []
        for param in gnn.model.conv1.parameters():
            flattened_params_1.append(param.detach().cpu().numpy().flatten())
        for param in gnn.model.conv2.parameters():
            flattened_params_2.append(param.detach().cpu().numpy().flatten())

        param1.append(np.concatenate(flattened_params_1))
        param2.append(np.concatenate(flattened_params_2))

    param1 = np.array(param1)
    param2 = np.array(param2)

    reducer = TSNE(n_components=2, perplexity=5, random_state=42)
    reduce_g1 = reducer.fit_transform(param1)
    reduce_g2 = reducer.fit_transform(param2)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w',  # 原有颜色
                'orange', 'purple', 'pink', 'lime', 'teal', 'brown', 'gray', 'olive', 'navy']
    markers = ['o', 's', '^', 'p', '*', 'h', '+']

    for i in range(10):
        g1 = reduce_g1[i]
        g2 = reduce_g2[i]
        plt.scatter(g1[0], g1[1], c=colors[i], marker=markers[0], alpha=0.6)
        plt.scatter(g2[0], g2[1], c=colors[i], marker=markers[1], alpha=0.6)

    plt.savefig("./vis/gnn_param", bbox_inches='tight')

@torch.no_grad()
def main(args):
    graph_dataset = GraphDataset(args)
    test_loader = DataLoader(graph_dataset, batch_size=args.batch_size_test_eval, shuffle=False)
    expert_model = CLIP(args).to(args.device)

    node_f = test_loader.dataset.node_f.to(args.device)
    edge_index = test_loader.dataset.edge_index.to(args.device)

    label_to_idx = {label: idx for idx, label in enumerate(test_loader.dataset.all_labels)}
    label = [label_to_idx[test_loader.dataset.label_list[i]] for i in range(len(test_loader.dataset.label_list))]
    label = torch.tensor(label).to(args.device)

    buffer_save_dir = os.path.join(args.buffer_save_dir, args.dataset_name, args.graph_encoder, args.text_encoder)
    # state = torch.load(os.path.join(str(buffer_save_dir), f"expert_state_{args.use_text_emb}.pt"))
    # expert_model.load_state_dict(state)
    # expert_model.eval()

    def load_parameters_to_model(model, parameters):
        for model_param, saved_param in zip(model.parameters(), parameters):
            model_param.data.copy_(saved_param)

    g_f = []
    t_f = []

    # sample_idx = torch.tensor([1, 10, 100, 1000, 5000, 10000, 20000])
    torch.manual_seed(42)
    sample_idx = torch.randint(0, len(test_loader.dataset), (7,))
    print([label[i] for i in sample_idx])
    num_batch = sample_idx.shape[0]
    reducer = TSNE(n_components=2, perplexity=5, random_state=42)

    for x in range(4):
        temp_trajectory = torch.load(os.path.join(str(buffer_save_dir), f"replay_buffer_{x}.pt"))
        state = temp_trajectory[-1]
        load_parameters_to_model(expert_model, state)

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
        graph_feature = graph_feature[sample_idx]
        text_feature = text_feature[sample_idx]

        g_f.append(graph_feature)
        t_f.append(text_feature)

    graph_feature = torch.cat(g_f, dim=0)
    text_feature = torch.cat(t_f, dim=0)

    reduce_g = reducer.fit_transform(graph_feature.cpu().detach().numpy())
    reduce_t = reducer.fit_transform(text_feature.cpu().detach().numpy())

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    markers = ['o', 's', '^', 'p', '*', 'h', '+']

    plt.figure(figsize=(8, 6))
    for i  in range(4):
        for j in range(num_batch):
            index = i * num_batch + j
            g = reduce_g[index]
            t = reduce_t[index]
            plt.scatter(g[0], g[1], c=colors[i], marker=markers[j], alpha=0.6, label=f'sample {j}')
            plt.scatter(t[0], t[1], c=colors[i], marker=markers[j], alpha=0.6, label=f'sample {j}',edgecolors='k')
            plt.plot([g[0], t[0]], [g[1], t[1]], c=colors[i], linestyle='--', alpha=0.6)

    plt.grid(True)
    plt.savefig("./trained", bbox_inches='tight')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # base
    parser.add_argument("--dataset_name", type=str, default="photo")
    parser.add_argument("--buffer_save_dir", type=str, default="./buffer")
    parser.add_argument("--gpu", type=int, default=0)

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
    parser.add_argument("--batch_size_train_eval", type=int, default=512)
    parser.add_argument("--train_epochs_eval", type=int, default=15)
    parser.add_argument("--batch_size_test_eval", type=int, default=1024)

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
    similarity(args)
