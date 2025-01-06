import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import CLIP, CoOp
from utils import multitask_data_generator
from reparam_module import ReparamModule


def epoch_train(
        model: CLIP,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        args,
        is_distill=False
):
    model.train()
    model.to(args.device)

    loss_sum, num_samples = 0, 0
    for i, (node_idx, texts) in enumerate(tqdm(train_loader, disable=is_distill)):
        node_idx = node_idx.to(args.device)
        if args.use_text_emb:
            texts = texts.to(args.device)
        node_f = train_loader.dataset.node_f.to(args.device)
        edge_index = train_loader.dataset.edge_index.to(args.device)

        loss, logits = model(node_f, edge_index, node_idx, texts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        num_samples += node_idx.size(0)

        if i % 10 == 0 and not is_distill:
            print(f'Training Loss: {loss_sum / num_samples:.4f}')

    return loss_sum / num_samples

@torch.no_grad()
def epoch_test(
        model: CLIP,
        test_loader: DataLoader,
        args,
        is_distill=False
):
    model.eval()
    model.to(args.device)

    label_list = test_loader.dataset.label_list
    labeled_ids = test_loader.dataset.labeled_ids
    all_labels = test_loader.dataset.all_labels
    if args.use_text_emb:
        all_labels_embeds = test_loader.dataset.all_labels_embeds.to(args.device)
    task_list, train_idx, val_idx, test_idx = multitask_data_generator(label_list, labeled_ids, all_labels, args.k_spt,
                                                                       args.k_val, args.k_qry, args.n_way)

    node_f = test_loader.dataset.node_f.to(args.device)
    edge_index = test_loader.dataset.edge_index.to(args.device)

    acc_list = []
    for j in tqdm(range(len(task_list)), disable=is_distill):
        ground_truth = label_list[test_idx[j]]
        task_all_labels = all_labels[task_list[j]]

        if args.use_text_emb:
            text_input = all_labels_embeds[task_list[j]]
        else:
            text_input = task_all_labels.tolist()

        logits = model(node_f, edge_index, test_idx[j], text_input, is_eval=True)
        pred = logits.argmax(dim=-1).cpu().numpy().reshape(-1)
        y_pred = task_all_labels[pred]
        acc = accuracy_score(ground_truth, y_pred)
        acc_list.append(acc)
    acc = np.mean(acc_list)

    # node_f = test_loader.dataset.node_f.to(args.device)
    # edge_index = test_loader.dataset.edge_index.to(args.device)
    #
    # ground_truth = label_list
    #
    # if args.use_text_emb:
    #     text_input = all_labels_embeds
    # else:
    #     text_input = all_labels.tolist()
    #
    # target_idx = np.arange(len(label_list))
    # logits = model(node_f, edge_index, target_idx, text_input, is_eval=True)
    # pred = logits.argmax(dim=-1).cpu().numpy().reshape(-1)
    # y_pred = all_labels[pred]
    # acc = accuracy_score(ground_truth, y_pred)

    return round(acc, 4)


def epoch_train_manual(
        model: ReparamModule,
        param: torch.Tensor,
        train_loader: DataLoader,
        lr_graph: torch.Tensor,
        lr_text: torch.Tensor,
        args,
):
    model.train()
    model.to(args.device)

    loss_sum, num_samples = 0, 0
    for i, (node_idx, texts) in enumerate(tqdm(train_loader, disable=True)):
        node_idx = node_idx.to(args.device)
        if args.use_text_emb:
            texts = texts.to(args.device)
        node_f = train_loader.dataset.node_f.to(args.device)
        edge_index = train_loader.dataset.edge_index.to(args.device)

        loss, logits = model(node_f, edge_index, node_idx, texts, flat_param=param)

        graph_mask = model.get_params_mask_for_module("graph_encoder", param)
        text_mask = model.get_params_mask_for_module("text_encoder", param)
        grad = torch.autograd.grad(loss, param, create_graph=True)[0]
        graph_grad = torch.where(graph_mask, grad, torch.tensor(0.0))
        text_grad = torch.where(text_mask, grad, torch.tensor(0.0))
        step = lr_graph * graph_grad + lr_text * text_grad
        param = param - step

        loss_sum += loss.item()
        num_samples += node_idx.size(0)

    return loss_sum / num_samples, param


def epoch_ft(
        model: CoOp,
        train_loader: DataLoader,
        train_idx,
        args,
        is_distill=False
):
    model.train()
    model.to(args.device)

    node_f = train_loader.dataset.node_f.to(args.device)
    edge_index = train_loader.dataset.edge_index.to(args.device)
    label_list = train_loader.dataset.label_list
    all_labels = train_loader.dataset.all_labels

    labels = label_list[train_idx]
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    labels_idx = np.vectorize(label_to_idx.get)(labels)
    labels_idx = torch.tensor(labels_idx, dtype=torch.long).to(args.device)

    logits = model(node_f, edge_index, train_idx, labels_idx, training=True)
    return


def epoch_ft_test(
        model: CoOp,
        test_loader: DataLoader,
        test_idx,
        args,
        is_distill=False
):
    model.eval()
    model.to(args.device)

    node_f = test_loader.dataset.node_f.to(args.device)
    edge_index = test_loader.dataset.edge_index.to(args.device)
    label_list = test_loader.dataset.label_list
    all_labels = test_loader.dataset.all_labels

    labels = label_list[test_idx]
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    labels_idx = np.vectorize(label_to_idx.get)(labels)
    labels_idx = torch.tensor(labels_idx, dtype=torch.long).to(args.device)

    logits = model(node_f, edge_index, test_idx, labels_idx, training=False)
    pred = logits.argmax(dim=-1)
    acc = accuracy_score(labels_idx.cpu(), pred.cpu())
    return round(acc, 4)