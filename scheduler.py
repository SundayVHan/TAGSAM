import numpy as np
import torch

from data.dataset import GraphDataset


class Scheduler:
    def __init__(self,
                 dataset:GraphDataset,
                 batch_size:int,
                 strategy:str,
                 max_iteration:int,
                 init_proportion:float):
        self.dataset = dataset
        self.batch_size = batch_size
        self.strategy = strategy
        self.max_iteration = max_iteration
        self.init_proportion = init_proportion

        self.curriculum_idx = []
        self.schedule = []
        self.iteration = 0
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.iteration += 1
        if self.index + self.batch_size >= len(self.schedule):
            self.index = 0
            self.update_schedule()

        result = self.schedule[self.index:self.index+self.batch_size]
        self.index += self.batch_size
        return result

    def difficulty_measure(self, expert_logits: torch.Tensor, reverse: bool = False):
        # difficulty_score = -np.diag(expert_logits.detach().cpu().numpy())
        # indices = np.arange(len(self.dataset))
        # self.curriculum_idx = indices[np.argsort(difficulty_score)]
        graph_score = torch.diag(torch.softmax(expert_logits, dim=1))
        text_score = torch.diag(torch.softmax(expert_logits, dim=0))
        difficulty_score = (graph_score + text_score) / 2
        if reverse:
            difficulty_score = difficulty_score.detach().cpu().numpy()
        else:
            difficulty_score = -difficulty_score.detach().cpu().numpy()
        indices = np.arange(len(self.dataset))
        self.curriculum_idx = indices[np.argsort(difficulty_score)]

    def update_schedule(self):
        if self.strategy == "root":
            h_t = min(1, np.sqrt(self.init_proportion**2 + (1-self.init_proportion**2) * (self.iteration/self.max_iteration)))
            num_nodes_to_sample = int(h_t * len(self.dataset))
            self.schedule = np.random.permutation(self.curriculum_idx[:num_nodes_to_sample])
        elif self.strategy == "none":
            self.schedule = np.random.permutation(range(len(self.dataset)))
        else:
            raise ValueError("Invalid strategy")