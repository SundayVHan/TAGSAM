import os
import pickle

import numpy as np
import torch
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm

from model import TextEncoder


class GraphDataset:
    def __init__(self, args):
        self.args = args

        data = torch.load(f'./data/{args.dataset_name}.pt', weights_only=False, map_location="cpu")
        self.node_f = data['node_f']
        self.edge_index = data['edge_index']
        self.text_list = data['text_list']
        self.label_list = data['label_list']
        self.all_labels = data['all_labels']
        self.labels_desc = data['labels_desc']

        self.text_embeds: torch.Tensor = torch.empty(0)
        self.all_labels_embeds: torch.Tensor = torch.empty(0)
        self.process_text()

        self.val_tasks: list[list[tuple]] = []
        self.test_tasks: list[list[tuple]] = []
        self.split_dataset()

    def __len__(self):
        return len(self.node_f)

    @torch.no_grad()
    def process_text(self):
        cache_file = os.path.join(self.args.buffer_save_dir, "cache.pt")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.text_embeds = cache_data.get('text_embeds')
                self.all_labels_embeds = cache_data.get('all_labels_embeds')
                return

        print("Start to process text to embeddings...")
        all_text_embeds = []
        all_labels_embeds = []
        batch_size = 64
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

        self.text_embeds = torch.cat(all_text_embeds, dim=0).to("cpu")
        self.all_labels_embeds = torch.cat(all_labels_embeds, dim=0).to("cpu")

        cache_data = {
            'text_embeds': self.text_embeds,
            'all_labels_embeds': self.all_labels_embeds
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

    def split_dataset(self):
        cache_file = os.path.join(self.args.buffer_save_dir, f"cache_subgraph_{self.args.batch_size_test}.pt")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.val_tasks = cache_data.get('val_tasks')
                self.test_tasks = cache_data.get('test_tasks')
                return

        print("Splitting dataset...")
        filtered_labels = [label for label in self.all_labels if label != "nan"]
        filtered_labels = np.array(filtered_labels)
        n_way = 5

        label_to_sample_index = {}
        for idx, label in enumerate(self.label_list):
            if label not in label_to_sample_index:
                label_to_sample_index[label] = []
            label_to_sample_index[label].append(idx)
        
        val_indices = {}
        test_indices = {}
        
        for label in filtered_labels:
            indices = np.array(label_to_sample_index[label])
            np.random.shuffle(indices)
            size = len(indices)
            
            val_size = int(size * 0.2)
            test_size = int(size * 0.2)
            
            val_indices[label] = indices[:val_size]
            test_indices[label] = indices[val_size:val_size + test_size]

        label_to_label_index = {label: idx for idx, label in enumerate(self.all_labels)}
        
        val_labels = []
        val_samples = []
        test_labels = []
        test_samples = []
        
        for i in range(0, len(filtered_labels), n_way):
            group_labels = filtered_labels[i:i + n_way]
            if len(group_labels) < n_way:
                break
                
            group_labels_idx = [label_to_label_index[label] for label in group_labels]
            val_labels.append(group_labels_idx)
            group_samples_idx = np.concatenate([val_indices[label] for label in group_labels])
            val_samples.append(group_samples_idx)
            
            test_labels.append(group_labels_idx)
            group_samples_idx = np.concatenate([test_indices[label] for label in group_labels])
            test_samples.append(group_samples_idx)
    
        print("Processing dataset...")
        val_tasks = []
        test_tasks = []

        for labels_idx, samples_idx in tqdm(zip(val_labels, val_samples), total=len(val_labels)):
            samples_idx = torch.tensor(samples_idx)
            sampler = NeighborSampler(
                self.edge_index, 
                node_idx=samples_idx,
                sizes=self.args.sample_size, 
                batch_size=len(samples_idx) if self.args.batch_size_test == -1 else self.args.batch_size_test,
                shuffle=True, 
                num_workers=16
            )

            task = []
            for batch_size, n_id, adjs in sampler:
                task.append((labels_idx, batch_size, n_id, adjs))
            val_tasks.append(task)

        for labels_idx, samples_idx in tqdm(zip(test_labels, test_samples), total=len(test_labels)):
            samples_idx = torch.tensor(samples_idx)
            sampler = NeighborSampler(
                self.edge_index, 
                node_idx=samples_idx,
                sizes=self.args.sample_size, 
                batch_size=len(samples_idx) if self.args.batch_size_test == -1 else self.args.batch_size_test,
                shuffle=True, 
                num_workers=16
            )

            task = []
            for batch_size, n_id, adjs in sampler:
                task.append((labels_idx, batch_size, n_id, adjs))
            test_tasks.append(task)

        self.val_tasks = val_tasks
        self.test_tasks = test_tasks

        cache_data = {
            'val_tasks': val_tasks,
            'test_tasks': test_tasks
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)


class SynGraphDataset(GraphDataset):
    def __init__(self, args):
        self.args = args

    def init(self, syn_graph, syn_text, syn_edge_index):
        args = self.args
        self.node_f = syn_graph.detach().to(args.device).requires_grad_(True)
        self.edge_index = syn_edge_index.to(args.device).requires_grad_(False)
        self.graph_encoder_lr = torch.tensor(args.graph_encoder_lr).to(args.device).requires_grad_(True)
        self.text_encoder_lr = torch.tensor(args.text_encoder_lr).to(args.device).requires_grad_(True)
        self.text_embeds = syn_text.to(args.device).requires_grad_(False)

        self.optimizer = torch.optim.SGD([
            {'params': self.node_f, 'lr': args.syn_lr, "momentum": 0.5},
            {'params': self.graph_encoder_lr, 'lr': args.syn_lr_lr, "momentum": 0.5},
            {'params': self.text_encoder_lr, 'lr': args.syn_lr_lr, "momentum": 0.5}
        ])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.95)

    def __len__(self):
        return len(self.node_f)

    def set_eval_model(self):
        self.node_f.requires_grad_(False)
        self.graph_encoder_lr.requires_grad_(False)
        self.text_encoder_lr.requires_grad_(False)

    def set_train_model(self):
        self.node_f.requires_grad_(True)
        self.graph_encoder_lr.requires_grad_(True)
        self.text_encoder_lr.requires_grad_(True)

    def save(self, it):
        save_data = {
            "node_f": self.node_f,
            "text_embeds": self.text_embeds,
            "graph_encoder_lr": self.graph_encoder_lr,
            "text_encoder_lr": self.text_encoder_lr,
            "edge_index": self.edge_index,
        }
        torch.save(save_data, os.path.join(str(self.args.buffer_save_dir), self.args.name, f"syn_data_{it}.pt"))

    def load(self, it):
        save_data = torch.load(os.path.join(str(self.args.buffer_save_dir), self.args.name, f"syn_data_{it}.pt"), map_location="cpu")
        self.node_f = save_data["node_f"].to(self.args.device).requires_grad_(True)
        self.text_embeds = save_data["text_embeds"].to(self.args.device).requires_grad_(False)
        self.graph_encoder_lr = save_data["graph_encoder_lr"].to(self.args.device).requires_grad_(True)
        self.text_encoder_lr = save_data["text_encoder_lr"].to(self.args.device).requires_grad_(True)
        self.edge_index = save_data["edge_index"].to(self.args.device).requires_grad_(False)

    def compute_grad(self, loss):
        grad = torch.autograd.grad(loss, [self.node_f, self.graph_encoder_lr, self.text_encoder_lr], allow_unused=True)
        self.node_f.grad = grad[0]
        self.graph_encoder_lr.grad = grad[1]
        self.text_encoder_lr.grad = grad[2]
        torch.nn.utils.clip_grad_norm_(self.node_f, 5)

    def step(self):
        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()