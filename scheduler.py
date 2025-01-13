import numpy as np
import torch

from data.dataset import GraphDataset


class Scheduler:
    def __init__(self,
                 dataset:GraphDataset,
                 batch_size:int):
        self.dataset = dataset
        self.batch_size = batch_size

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
        return torch.tensor(result)

    def update_schedule(self):
        self.schedule = np.random.permutation(range(len(self.dataset)))