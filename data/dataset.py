import json
import os
import pickle

import numpy as np
import torch
from networkx.classes import nodes
from sklearn import preprocessing
from torch.utils.data import Dataset
from tqdm import tqdm

from data.loader import load_data
from model import TextEncoder


def read_edge_index(dataset_name):
    if dataset_name == "cora":
        raw_edge_index = [[], []]
        with open(f'./data/{dataset_name}/mapped_edges.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                raw_edge_index[0].append(int(line[0]))
                raw_edge_index[1].append(int(line[1]))

        edge_index = [raw_edge_index[0] + raw_edge_index[1], raw_edge_index[1] + raw_edge_index[0]]
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    else:
        edge_index = np.load(f'./data/{dataset_name}/edge.npy')
        edge_index = torch.from_numpy(edge_index)
    return edge_index

def read_node_features(dataset_name):
    node_f = torch.load(f'./data/{dataset_name}/node_f.pt')
    return node_f

def read_text_list(dataset_name):
    if dataset_name == "cora":
        text_list = []
        with open(f'./data/{dataset_name}/train_text.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                text_list.append(line[2])

        text_list = np.array(text_list)
    else:
        text_dict = json.load(open(f'./data/{dataset_name}/text.json'))
        text_list = []
        for i in range(len(text_dict)):
            text_list.append(text_dict[str(i)])
    return text_list

def read_label(dataset_name):
    if dataset_name == "cora":
        label_list = []
        labeled_ids = []
        with open(f'./data/{dataset_name}/train_text.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                label_list.append(line[3])

        for i in range(len(label_list)):
            if label_list[i] != 'nan':
                labeled_ids.append(i)

        with open(f'./data/{dataset_name}/lab_list.txt', 'r') as f:
            all_label_line = f.readline().strip().split('\t')

        all_labels = []
        for i in all_label_line:
            if i != 'nan':
                all_labels.append(i)

        label_list = np.array(label_list)
        labeled_ids = np.array(labeled_ids)
        all_labels = np.array(all_labels)
    else:
        id_label_dict = json.load(open(f"./data/{dataset_name}/id_labels.json"))
        id_label_list = sorted(id_label_dict.items(), key=lambda x: int(x[0]))
        label_list = ["nan"] * (int(id_label_list[-1][0])+1)
        labeled_ids = []
        for item in id_label_list:
            if item[1] != "nan" or item[1] != "" or item[1] != " ":
                label_list[int(item[0])] = item[1]
                labeled_ids.append(int(item[0]))

        all_labels = sorted(list(set(label_list)))
        label_list = np.array(label_list)
        labeled_ids = np.array(labeled_ids)
        all_labels = np.array(all_labels)

    return label_list, labeled_ids, all_labels


def generate_random_edge_index(num_nodes, num_edges):
    row = torch.randint(0, num_nodes, (num_edges,))
    col = torch.randint(0, num_nodes, (num_edges,))
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


class GraphDataset(Dataset):
    def __init__(self, args):
        super(GraphDataset, self).__init__()
        self.args = args

        if args.dataset_name == "cora" or args.dataset_name == "art":
            self.edge_index = read_edge_index(args.dataset_name)
            self.node_f = read_node_features(args.dataset_name)
            self.text_list = read_text_list(args.dataset_name)
            self.label_list, self.labeled_ids, self.all_labels = read_label(args.dataset_name)
        else:
            data, text, all_labels, labels_desc = load_data(args.dataset_name)
            self.edge_index = data.edge_index
            self.node_f = data.x
            self.text_list = text
            self.label_list = np.array([all_labels[i.item()] for i in data.y])
            self.labeled_ids = np.array(range(len(self)))
            self.all_labels = np.array(all_labels)
            self.labels_desc = np.array(labels_desc)

        if args.use_text_emb:
            self.text_embeds = None
            self.all_labels_embeds = None
            self.process_text2emb()

        self.test_labels = None
        self.test_samples = None
        self.test_split()


    def __len__(self):
        return len(self.node_f)

    def __getitem__(self, idx):
        if self.args.use_text_emb:
            text_embed = self.text_embeds[idx]
            return idx, text_embed
        else:
            text = self.text_list[idx]
            return idx, text

    @torch.no_grad()
    def process_text2emb(self):
        cache_file = f'./data/{self.args.dataset_name}/cache.pt'
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.text_embeds = cache_data.get('text_embeds')
                self.all_labels_embeds = cache_data.get('all_labels_embeds')
                return

        print("Start to process text to embeddings...")
        all_text_embeds = []
        all_labels_embeds = []
        batch_size = 512
        text_encoder = TextEncoder(self.args).to(self.args.device)

        for i in tqdm(range(0, len(self.text_list), batch_size)):
            batch_texts = self.text_list[i:i + batch_size]
            batch_embeds = text_encoder(batch_texts).detach().cpu()

            all_text_embeds.append(batch_embeds)
            del batch_embeds
            torch.cuda.empty_cache()

        labels_with_desc = [f"{label} {desc}" for label, desc in zip(self.all_labels, self.labels_desc)]
        for i in tqdm(range(0, len(self.all_labels), batch_size)):
            batch_texts = labels_with_desc[i:i + batch_size]
            batch_embeds = text_encoder(batch_texts).detach().cpu()

            all_labels_embeds.append(batch_embeds)
            del batch_embeds
            torch.cuda.empty_cache()

        self.text_embeds = torch.cat(all_text_embeds, dim=0)
        self.all_labels_embeds = torch.cat(all_labels_embeds, dim=0)

        cache_data = {
            'text_embeds': self.text_embeds,
            'all_labels_embeds': self.all_labels_embeds
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

    def test_split(self):
        filtered_labels = [label for label in self.all_labels if label != "nan"]
        filtered_labels = np.array(filtered_labels)
        if self.args.dataset_name == "art":
            np.random.seed(0)
            filtered_labels = filtered_labels[np.random.permutation(len(filtered_labels))]
            np.random.seed(self.args.seed)
        num_labels = len(filtered_labels)
        n_way = self.args.n_way
        num_groups = num_labels // n_way

        label_to_all_index = {label: idx for idx, label in enumerate(self.all_labels)}

        label_to_list_index = {}
        for idx, label in enumerate(self.label_list):
            if label not in label_to_list_index:
                label_to_list_index[label] = []
            label_to_list_index[label].append(idx)

        label_to_list_index = {label: np.array(indices) for label, indices in label_to_list_index.items()}

        labels = []
        samples = []
        for i in range(num_groups):
            test_labels = filtered_labels[i * n_way: (i + 1) * n_way]
            test_labels_idx = [label_to_all_index[label] for label in test_labels]
            labels.append(test_labels_idx)

            test_samples_idx = np.concatenate([label_to_list_index[label] for label in test_labels])
            samples.append(test_samples_idx)

        self.test_labels = labels
        self.test_samples = samples

    def get_test_labels_samples(self):
        return self.test_labels, self.test_samples


class SynTAGDataset(Dataset):
    def __init__(self, syn_graph, syn_text, args):
        self.args = args

        self.node_f = syn_graph.detach().to(args.device).requires_grad_(True)
        self.edge_index = torch.empty((2, 0), dtype=torch.long, device=args.device)
        self.student_lr_graph = torch.tensor(args.lr_graph_encoder).to(args.device).requires_grad_(True)
        self.student_lr_text = torch.tensor(args.lr_text_encoder).to(args.device).requires_grad_(True)

        self.optimizer = torch.optim.SGD([
            {'params': self.node_f, 'lr': args.syn_graph_lr, "momentum": 0.5},
            {'params': self.student_lr_graph, 'lr': args.lr_lr, "momentum": 0.5},
            {'params': self.student_lr_text, 'lr': args.lr_lr, "momentum": 0.5}
        ])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.95)

        self.text_embeds = syn_text
        self.text_list = []

    def __len__(self):
        return len(self.node_f)

    def __getitem__(self, idx):
        if self.args.use_text_emb:
            text_embed = self.text_embeds[idx]
            return idx, text_embed
        else:
            text = self.text_list[idx]
            return idx, text

    def set_eval_model(self):
        self.node_f.requires_grad_(False)
        self.student_lr_graph.requires_grad_(False)
        self.student_lr_text.requires_grad_(False)

    def set_train_model(self):
        self.node_f.requires_grad_(True)
        self.student_lr_graph.requires_grad_(True)
        self.student_lr_text.requires_grad_(True)

    def compute_grad(self, loss):
        grad = torch.autograd.grad(loss, [self.node_f, self.student_lr_graph, self.student_lr_text])
        self.node_f.grad = grad[0]
        self.student_lr_graph.grad = grad[1]
        self.student_lr_text.grad = grad[2]
        torch.nn.utils.clip_grad_norm_(self.node_f, 5)

    def update(self):
        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
