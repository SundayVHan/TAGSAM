import math
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling

from dataset import GraphDataset, SynGraphDataset
from model import CLIP
from reparam import ReparamModule


def epoch_train(
        model: CLIP,
        dataset: GraphDataset,
        sampler: NeighborSampler,
        optimizer: torch.optim.Optimizer,
        args,
        is_distill=False
):
    model.train()
    model=model.to(args.device)

    loss_sum, num_samples = 0, 0
    for i, (batch_size, n_id, adjs) in enumerate(tqdm(sampler, disable=is_distill)):
        node_f = dataset.node_f[n_id].to(args.device)
        adjs = [adj.to(args.device) for adj in adjs]
        text_embeds = dataset.text_embeds[n_id[:batch_size]].to(args.device)

        loss, logits = model(node_f, adjs, text_embeds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        num_samples += batch_size

        if i % 10 == 0 and not is_distill:
            tqdm.write(f'Training Loss: {loss_sum / num_samples:.4f}')

    return round(loss_sum / num_samples, 4)


@torch.no_grad()
def epoch_test(
        model: CLIP,
        dataset: GraphDataset,
        args,
        is_distill=False
):
    model.eval()
    model = model.to(args.device)

    label_list = dataset.label_list
    all_labels = dataset.all_labels
    all_labels_embeds = dataset.all_labels_embeds.to(args.device)

    def evaluate_tasks(tasks):
        acc_list = []
        for task in tqdm(tasks, disable=is_distill):
            acc_sum = 0.0
            num_samples = 0
            for labels_idx, batch_size, n_id, adjs in task:
                ground_truth = label_list[n_id[:batch_size]]
                if isinstance(ground_truth, str):
                    ground_truth = np.array([ground_truth])
                text_input = all_labels_embeds[labels_idx]
                labels = all_labels[labels_idx]

                node_f = dataset.node_f[n_id].to(args.device)
                adjs = [adj.to(args.device) for adj in adjs]

                logits = model(node_f, adjs, text_input, is_eval=True)
                pred = logits.argmax(dim=-1).cpu().numpy().reshape(-1)
                y_pred = labels[pred]
                acc = accuracy_score(ground_truth, y_pred)
                acc_sum += acc * batch_size
                num_samples += batch_size

            acc = acc_sum / num_samples
            acc_list.append(acc)
        return round(np.mean(acc_list), 4)

    val_acc = evaluate_tasks(dataset.val_tasks)
    test_acc = evaluate_tasks(dataset.test_tasks)

    return val_acc, test_acc


def epoch_train_manual(
        model: ReparamModule,
        param: torch.Tensor,
        dataset: SynGraphDataset,
        sampler: NeighborSampler,
        graph_encoder_lr: torch.Tensor,
        text_encoder_lr: torch.Tensor,
        args,
):
    model.train()
    model = model.to(args.device)

    loss_sum, num_batches = 0.0, 0

    for batch_size, n_id, adjs in sampler:
        # 获取当前批次的数据
        node_f = dataset.node_f[n_id].to(args.device)
        text_embeds = dataset.text_embeds[n_id[:batch_size]].to(args.device)
        adjs = [adj.to(args.device) for adj in adjs]

        # 计算损失
        loss, logits = model(node_f, adjs, text_embeds, flat_param=param)
        
        # 手动更新参数
        graph_mask = model.get_params_mask_for_module("graph_encoder", param)
        text_mask = model.get_params_mask_for_module("text_encoder", param)
        grad = torch.autograd.grad(loss, param, create_graph=True)[0]
        
        graph_grad = torch.where(graph_mask, grad, torch.tensor(0.0, device=args.device))
        text_grad = torch.where(text_mask, grad, torch.tensor(0.0, device=args.device))
        step = graph_encoder_lr * graph_grad + text_encoder_lr * text_grad
        param = param - step
        
        loss_sum += loss.item()
        num_batches += 1

    avg_loss = round(loss_sum / num_batches, 4)
    return avg_loss, param


def epoch_train_link(
        model: CLIP,
        decoder,
        dataset,
        optimizer,
        args
):
    model.eval()
    decoder.train()

    pos_edge_index = dataset.edge_index.to(args.device)
    num_edges = pos_edge_index.size(1)
    batch_size = 204800
    num_batches = math.ceil(num_edges / batch_size)
    
    total_loss = 0
    total_auc = 0
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_edges)
        
        # 获取当前批次的正样本边
        batch_pos_edge = pos_edge_index[:, start_idx:end_idx]
        
        # 为当前批次生成负样本边
        batch_neg_edge = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=dataset.node_f.size(0),
            num_neg_samples=(end_idx - start_idx) * 3,
            method='sparse'
        ).to(args.device)

        # 合并当前批次的边和标签
        edge_label_index = torch.cat([batch_pos_edge, batch_neg_edge], dim=1)
        edge_label = torch.cat([
            torch.ones(batch_pos_edge.size(1)),
            torch.zeros(batch_neg_edge.size(1))
        ]).to(args.device)

        # 获取节点特征
        with torch.no_grad():
            text_embeddings = model.encode_text(dataset.text_embeds.to(args.device))

        # 预测边的概率
        pred = decoder(
            text_embeddings[edge_label_index[0]],
            text_embeddings[edge_label_index[1]]
        )

        # 计算损失
        criterion = torch.nn.BCELoss()
        loss = criterion(pred, edge_label)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算当前批次的AUC
        batch_auc = roc_auc_score(
            edge_label.cpu().numpy(),
            pred.detach().cpu().numpy()
        )

        total_loss += loss.item()
        total_auc += batch_auc

    avg_loss = total_loss / num_batches
    avg_auc = total_auc / num_batches

    return avg_loss, avg_auc