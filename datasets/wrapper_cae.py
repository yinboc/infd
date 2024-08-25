import random
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import datasets
from datasets import register
from utils.geometry import make_coord_grid


class BaseWrapperCAE:

    def __init__(self, dataset, resize_inp, ret_gt=True, resize_gt_lb=None, resize_gt_ub=None,
                 final_crop_gt=None, p_whole=0.0, p_max=0.0):
        self.dataset = datasets.make(dataset)
        self.resize_inp = resize_inp
        self.ret_gt = ret_gt
        self.resize_gt_lb = resize_gt_lb
        self.resize_gt_ub = resize_gt_ub
        self.final_crop_gt = final_crop_gt
        self.p_whole = p_whole
        self.p_max = p_max
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def process(self, img):
        assert img.size[0] == img.size[1]
        ret = {}
        
        inp = img.resize((self.resize_inp, self.resize_inp), Image.LANCZOS)
        inp = self.transform(inp)
        ret.update({'inp': inp})

        if self.ret_gt:
            if self.resize_gt_lb is None:
                gt = self.transform(img)
            else:
                if random.random() < self.p_whole:
                    r = self.final_crop_gt
                elif random.random() < self.p_max:
                    r = min(img.size[0], self.resize_gt_ub)
                else:
                    r = random.randint(self.resize_gt_lb, min(img.size[0], self.resize_gt_ub))
                gt = img.resize((r, r), Image.LANCZOS)
                gt = self.transform(gt)

            p = self.final_crop_gt
            ii = random.randint(0, gt.shape[-2] - p)
            jj = random.randint(0, gt.shape[-1] - p)
            gt_patch = gt[:, ii: ii + p, jj: jj + p]

            x0, y0 = ii / gt.shape[-2], jj / gt.shape[-1] # assume range [0, 1]
            x1, y1 = (ii + p) / gt.shape[-2], (jj + p) / gt.shape[-1]
            coord = make_coord_grid((p, p), range=[[x0, x1], [y0, y1]])
            coord = 2 * coord - 1 # convert to range [-1, 1]
            cell = torch.tensor([2 / gt.shape[-2], 2 / gt.shape[-1]], dtype=torch.float32)
            cell = cell.view(1, 1, 2).expand(p, p, -1)
            ret.update({
                'gt': gt_patch, # 3 p p
                'gt_coord': coord, # p p 2
                'gt_cell': cell, # p p 2
            })

        return ret

@register('wrapper_cae')
class WrapperCAE(BaseWrapperCAE, Dataset):
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return self.process(data)
