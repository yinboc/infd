import random

from torch.utils.data import Dataset

import datasets
from datasets import register


@register('random_lr_hr')
class RandomLRHR(Dataset):

    def __init__(self, lr_dataset, hr_dataset, p_lr=0.5):
        self.lr_dataset = datasets.make(lr_dataset)
        self.hr_dataset = datasets.make(hr_dataset)
        self.n_lr = len(self.lr_dataset)
        self.n_hr = len(self.hr_dataset)
        self.p_lr = p_lr

    def __len__(self):
        return self.n_lr + self.n_hr

    def __getitem__(self, idx):
        if random.random() < self.p_lr:
            return self.lr_dataset[random.randint(0, self.n_lr - 1)]
        else:
            return self.hr_dataset[random.randint(0, self.n_hr - 1)]
