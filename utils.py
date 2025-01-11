import math
import pickle
import random
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import spacy
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from torch.utils.data import Subset, DataLoader
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm

import pmi
from model import TextEncoder, GPT2
from pmi import get_npmi_matrix


def select_balanced_labels(dataset, num_select=100):
    label_list = dataset.label_list
    label_count = defaultdict(int)
    for label in label_list:
        label_count[label] += 1

    unique_labels = list(label_count.keys())
    num_labels = len(unique_labels)

    selected_indices = []
    selected_count = defaultdict(int)
    indices_by_label = defaultdict(list)

    for idx, label in enumerate(label_list):
        indices_by_label[label].append(idx)

    for label in unique_labels:
        if indices_by_label[label]:
            selected_idx = indices_by_label[label].pop(0)
            selected_indices.append(selected_idx)
            selected_count[label] += 1

    remaining_slots = num_select - len(selected_indices)
    target_per_label = remaining_slots // num_labels

    for label in unique_labels:
        while selected_count[label] < target_per_label + 1 and indices_by_label[label]:
            selected_idx = indices_by_label[label].pop(0)
            selected_indices.append(selected_idx)
            selected_count[label] += 1
            if len(selected_indices) >= num_select:
                break

    if len(selected_indices) < num_select:
        remaining_indices = [i for i in range(len(label_list)) if i not in selected_indices]
        additional_needed = num_select - len(selected_indices)
        if additional_needed > len(remaining_indices):
            additional_needed = len(remaining_indices)
        selected_indices.extend(random.sample(remaining_indices, additional_needed))

    return selected_indices


def multitask_data_generator(label_list, labeled_ids, all_labels, k_spt, k_val, k_qry, n_way):
    labels_local = label_list

    class_idx_list = []
    train_class_list = []
    val_class_list = []
    test_class_list = []
    for i in range(len(all_labels)):
        class_idx_list.append([])
        train_class_list.append([])
        val_class_list.append([])
        test_class_list.append([])

    # for j in labeled_ids:
    #     for i in range(len(all_labels)):
    #         if (labels_local[j] == all_labels[i]):
    #             class_idx_list[i].append(j)
    class_idx_list = [[] for _ in range(len(all_labels))]
    label_to_index = {label: index for index, label in enumerate(all_labels)}

    for j in labeled_ids:
        label = labels_local[j]
        if label in label_to_index:
            index = label_to_index[label]
            class_idx_list[index].append(j)

    usable_labels = []
    for i in range(len(class_idx_list)):
        if len(class_idx_list[i]) >= 30:
            usable_labels.append(i)

    random.shuffle(usable_labels)
    task_list = []
    for i in range(len(usable_labels) // n_way):
        task_idx = usable_labels[i * n_way:(i + 1) * n_way]
        task_list.append(task_idx)

    for i in range(len(all_labels)):
        if i not in set(usable_labels):
            continue
        train_class_list[i] = np.random.choice(class_idx_list[i], k_spt, replace=False).tolist()
        val_class_temp = [n1 for n1 in class_idx_list[i] if n1 not in train_class_list[i]]
        val_class_list[i] = np.random.choice(val_class_temp, k_val, replace=False).tolist()
        test_class_temp = [n1 for n1 in class_idx_list[i] if
                           (n1 not in train_class_list[i]) and (n1 not in val_class_list[i])]
        test_class_list[i] = test_class_temp

    train_idx = []
    test_idx = []
    val_idx = []

    for i in range(len(task_list)):
        train_idx.append([])
        test_idx.append([])
        val_idx.append([])
        # print(task_list[i])
        for j in task_list[i]:
            train_idx[i] += train_class_list[j]
            val_idx[i] += val_class_list[j]
            test_idx[i] += test_class_list[j]

    return task_list, train_idx, val_idx, test_idx


def select_cluster_center_labels(dataset, num_select=100):
    text_embeds = dataset.text_embeds

    kmeans = KMeans(n_clusters=num_select // 10, random_state=0)
    kmeans.fit(text_embeds.detach().cpu().numpy())
    labels = torch.tensor(kmeans.labels_)

    closest_indices = []
    all_indices = torch.arange(text_embeds.size(0))

    for k in range(kmeans.n_clusters):
        cluster_indices = (labels == k).nonzero(as_tuple=True)[0]
        if len(cluster_indices) > num_select:
            random_selection = torch.randperm(len(cluster_indices))[:num_select]
            selected_indices = cluster_indices[random_selection].tolist()
            closest_indices.extend(selected_indices)

    return closest_indices


def select_cluster_boundary_labels(dataset, num_select=100):
    text_embeds = dataset.text_embeds

    kmeans = KMeans(n_clusters=num_select, random_state=0)
    kmeans.fit(text_embeds.detach().cpu().numpy())
    labels = torch.tensor(kmeans.labels_)
    centroids = torch.tensor(kmeans.cluster_centers_)

    boundary_indices = []
    for k in range(centroids.size(0)):
        cluster_indices = (labels == k).nonzero(as_tuple=True)[0]
        cluster_points = text_embeds[cluster_indices]
        distances = torch.norm(cluster_points - centroids[k], dim=1)
        max_distance_index = torch.argmax(distances)

        boundary_index = cluster_indices[max_distance_index].item()
        boundary_indices.append(boundary_index)

    return boundary_indices


def select_class_sub_center_labels(dataset, num_select=100):
    label_list = dataset.label_list
    text_embeds = dataset.text_embeds

    label_count = defaultdict(int)
    label_embeds = defaultdict(list)
    for idx, label in enumerate(label_list):
        label_count[label] += 1
        label_embeds[label].append(text_embeds[idx])

    selected_indices = []
    for label in label_count.keys():
        label_embeds[label] = torch.stack(label_embeds[label])
        kmeans = KMeans(n_clusters=1, random_state=0)
        kmeans.fit(label_embeds[label].detach().cpu().numpy())
        centroid = torch.tensor(kmeans.cluster_centers_)

        distances = torch.norm(label_embeds[label] - centroid, dim=1)
        min_distance_index = torch.argmin(distances)

        selected_indices.append(min_distance_index.item())

    return selected_indices


def select_target_idx(dataset):
    text_embeds = dataset.text_embeds
    node_f = dataset.node_f
    num_samples = len(dataset)
    select_idx = []

    kmeans = KMeans(n_clusters=int(math.sqrt(num_samples) / 2), random_state=0)
    kmeans.fit(text_embeds.detach().cpu().numpy())
    label = torch.tensor(kmeans.labels_)
    centroids = torch.tensor(kmeans.cluster_centers_)

    for k in range(centroids.size(0)):
        cluster_indices = (label == k).nonzero(as_tuple=True)[0]
        cluster_points = text_embeds[cluster_indices]
        distances = torch.norm(cluster_points - centroids[k], dim=1)
        min_distance_index = torch.argsort(distances)[:10]
        select_idx.extend(cluster_indices[min_distance_index])

    kmeans = KMeans(n_clusters=int(math.sqrt(num_samples) / 2), random_state=0)
    kmeans.fit(node_f.detach().cpu().numpy())
    label = torch.tensor(kmeans.labels_)
    centroids = torch.tensor(kmeans.cluster_centers_)

    for k in range(centroids.size(0)):
        cluster_indices = (label == k).nonzero(as_tuple=True)[0]
        cluster_points = node_f[cluster_indices]
        distances = torch.norm(cluster_points - centroids[k], dim=1)
        min_distance_index = torch.argsort(distances)[:10]
        select_idx.extend(cluster_indices[min_distance_index])

    return select_idx


def select_subgraph_labels(dataset, num_subgraphs=50, subgraph_size=4):
    all_nodes = len(dataset)
    label_list = dataset.label_list
    select_idx = []
    subgraph_labels = []

    same_label = 0

    for subgraph_id in range(num_subgraphs):
        while True:
            center_node = random.randint(0, all_nodes - 1)

            subset, _, _, _ = k_hop_subgraph(
                center_node, num_hops=1, edge_index=dataset.edge_index, relabel_nodes=False
            )

            if len(subset) >= subgraph_size:
                selected_nodes = random.sample(subset.tolist(), subgraph_size)
                select_idx.extend(selected_nodes)
                labels = label_list[selected_nodes]
                counter = Counter(labels)
                core_label = counter.most_common(1)[0][0]
                same_label += sum([1 for label in labels if label == core_label])

                subgraph_labels.extend([subgraph_id] * subgraph_size)
                break
    print("Same label ratio: ", same_label / (num_subgraphs * subgraph_size))

    select_labels = label_list[select_idx]
    original_label_counts = Counter(label_list)

    # 计算选中标签的分布
    selected_label_counts = Counter(select_labels)

    # 获取所有可能的标签
    all_labels = set(original_label_counts.keys()).union(set(selected_label_counts.keys()))

    # 构建原始数据集的概率分布
    original_distribution = np.array([original_label_counts.get(label, 0) for label in all_labels])
    original_distribution = original_distribution / original_distribution.sum()

    # 构建选中标签的概率分布
    selected_distribution = np.array([selected_label_counts.get(label, 0) for label in all_labels])
    selected_distribution = selected_distribution / selected_distribution.sum()

    # 计算 Jensen-Shannon 散度
    js_divergence = jensenshannon(original_distribution, selected_distribution)
    print(f"Jensen-Shannon divergence: {js_divergence}")
    return select_idx, subgraph_labels


def create_few_shot_index(dataset, percent=0.01):
    labels = dataset.label_list
    unique_labels = set(labels)

    label_to_indices = {label: [] for label in unique_labels}

    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    few_shot_indices = []
    for label, indices in label_to_indices.items():
        sample_size = max(1, int(len(indices) * percent))
        sampled_indices = random.sample(indices, sample_size)
        few_shot_indices.extend(sampled_indices)

    all_indices = set(range(len(dataset)))
    test_indices = list(all_indices - set(few_shot_indices))

    return few_shot_indices, test_indices


def aggregate_text(dataset, select_idx, args):
    text_encoder = TextEncoder(args).to(args.device)
    text_aggregate = []
    feature_aggregate =[]
    text_list = np.array(dataset.text_list)

    for center_node in tqdm(select_idx):
        center_node = torch.tensor([center_node])
        subset, _, _, _ = k_hop_subgraph(
            center_node, num_hops=1, edge_index=dataset.edge_index, relabel_nodes=False
        )

        if len(subset) > 4:
            subset = np.array(random.sample(subset.tolist(), 4))
        sub_texts = text_list[subset]
        sub_text = " ".join(sub_texts)
        text_aggregate.append(sub_text)

        sub_features = dataset.node_f[subset]
        sub_feature = sub_features.mean(dim=0)
        feature_aggregate.append(sub_feature)

    text_embeds = text_encoder(text_aggregate)
    node_embeds = torch.stack(feature_aggregate)
    return node_embeds, text_embeds


def summary_text(dataset, select_idx, args):
    text_encoder = TextEncoder(args).to(args.device)
    text_aggregate = []
    feature_aggregate =[]
    text_list = np.array(dataset.text_list)
    nlp = spacy.load("en_core_web_sm")

    def remove_unicode(text):
        return ''.join([i if ord(i) < 128 else ' ' for i in text])

    for center_node in tqdm(select_idx):
        center_node = torch.tensor([center_node])
        subset, _, _, _ = k_hop_subgraph(
            center_node, num_hops=1, edge_index=dataset.edge_index, relabel_nodes=False
        )

        if len(subset) > 4:
            subset = np.array(random.sample(subset.tolist(), 4))

        sub_texts = text_list[subset]

        num_sentences = 0
        for sub_text in sub_texts:
            doc = nlp(str(sub_text))
            sentences = [remove_unicode(sentence.text) for sentence in doc.sents]
            num_sentences += len(sentences)

        sentence_ranges = []
        start_idx = 0
        combine = 5
        for sub_text in sub_texts:
            doc = nlp(str(sub_text))
            sentences = [remove_unicode(sentence.text) for sentence in doc.sents]

            merged_sentences = [' '.join(sentences[i:i + combine]) for i in range(0, len(sentences), combine)]

            sentence_ranges.append((start_idx, start_idx + len(merged_sentences)))
            start_idx += len(merged_sentences)

        all_sentences = []
        for sub_text in sub_texts:
            doc = nlp(str(sub_text))
            sentences = [remove_unicode(sentence.text) for sentence in doc.sents]
            merged_sentences = [' '.join(sentences[i:i + combine]) for i in range(0, len(sentences), combine)]
            all_sentences.extend(merged_sentences)

        pmi.model = GPT2(device=args.device)
        normalised, matrix, surprise = get_npmi_matrix(all_sentences, batch_size=5)
        matrix[matrix < 0] = 0

        relevance = [sum(matrix[idx]) for idx in range(len(all_sentences))]
        penalty = [0 for _ in range(len(all_sentences))]
        selected = []

        num_selected_sentences = int(len(all_sentences) * 0.6)
        for k in range(num_selected_sentences):
            maxIdx = -1
            maxVal = -float('inf')

            for i in range(len(all_sentences)):
                temp = 1 * relevance[i] + (-1) * penalty[i]
                if temp > maxVal and i not in selected:
                    maxIdx = i
                    maxVal = temp

            for i in range(len(all_sentences)):
                penalty[i] += matrix[i][maxIdx]

            selected.append(maxIdx)

        # 生成摘要
        summary = " ".join(all_sentences[i] for i in sorted(selected))
        text_aggregate.append(summary)

        # 计算每个文档的抽取比例并聚合特征
        weighted_features = []
        for (start, end), idx in zip(sentence_ranges, subset):
            doc_selected = [i for i in selected if start <= i < end]
            extraction_ratio = len(doc_selected) / (end - start)
            sub_feature = dataset.node_f[idx]
            weighted_features.append(sub_feature * extraction_ratio)

        sub_feature = torch.stack(weighted_features).sum(dim=0)

        feature_aggregate.append(sub_feature)

    text_embeds = text_encoder(text_aggregate)
    node_embeds = torch.stack(feature_aggregate)
    return node_embeds, text_embeds