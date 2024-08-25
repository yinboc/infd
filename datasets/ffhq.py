import os

from PIL import Image
from torch.utils.data import Dataset

from datasets import register


IMAGE_EXTS = ('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.webp')


@register('ffhq')
class FFHQ(Dataset):

    def __init__(self, split=None, root_path='load/ffhq', img_folder='ffhq_1024', variant=None):
        if split is None:
            filelist = sorted(os.listdir(os.path.join(root_path, img_folder)))
        else:
            if split == 'train':
                filelist = os.path.join(root_path, 'ffhqtrain.txt')
            elif split == 'val':
                filelist = os.path.join(root_path, 'ffhqvalidation.txt')
            with open(filelist, 'r') as f:
                filelist = [_.rstrip('\n') for _ in f.readlines()]
        filelist = [_ for _ in filelist if _.endswith(IMAGE_EXTS)]
        self.files = [os.path.join(root_path, img_folder, _) for _ in filelist]

        self.variant = variant
        if variant == 'mix6000':
            self.files = self.files[:6000]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')

        if self.variant == 'mix6000':
            if idx < 5000:
                s = 512 + round((1024 - 512) * (idx / (5000 - 1)))
            else:
                s = 1024
            img = img.resize((s, s), Image.LANCZOS)
        elif self.variant == 'all_lr':
            img = img.resize((256, 256), Image.LANCZOS)

        return img
