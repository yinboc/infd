import argparse
import os
from PIL import Image

import torch
import numpy as np
import torch_fidelity
from torch.utils.data import Dataset
from torchvision import transforms


class PFIDDataset(Dataset):

    def __init__(self, root, p_res):
        self.files = [os.path.join(root, _) for _ in os.listdir(root) if _.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))]
        self.crop = transforms.RandomCrop(p_res)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.crop(Image.open(self.files[idx]))
        x = torch.from_numpy(np.asarray(x)).permute(2, 0, 1)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input1')
    parser.add_argument('--input2')
    args = parser.parse_args()

    files1 = [_ for _ in os.listdir(args.input1) if _.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))]
    files2 = [_ for _ in os.listdir(args.input2) if _.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))]
    res1 = Image.open(os.path.join(args.input1, files1[0])).size[0]
    res2 = Image.open(os.path.join(args.input2, files2[0])).size[0]
    assert res1 == res2
    if res1 == 256:
        p_res_lst = [256]
    elif res1 == 512:
        p_res_lst = [256, 512]
    elif res1 == 1024:
        p_res_lst = [256, 512, 1024]

    results = []
    for p_res in p_res_lst:
        print(f'Running for {p_res}/{res1} ...')
        ds1 = PFIDDataset(args.input1, p_res)
        ds2 = PFIDDataset(args.input2, p_res)
        ret = torch_fidelity.calculate_metrics(input1=ds1, input2=ds2, cuda=True, fid=True, batch_size=512)
        results.append(ret["frechet_inception_distance"])

    for p_res, fid in zip(p_res_lst, results):
        print(f'{p_res}/{res1}: {fid}')


if __name__ == '__main__':
    main()
