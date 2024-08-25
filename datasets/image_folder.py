import os
from PIL import Image, ImageFile

from datasets import register
from torch.utils.data import Dataset
from torchvision import transforms


Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTS = ('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.webp')


@register('image_folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, square_crop=True, resize=None, rand_crop=None):
        files = sorted(os.listdir(root_path))
        files = [os.path.join(root_path, _) for _ in files if _.endswith(IMAGE_EXTS)]
        self.files = files
        self.square_crop = square_crop
        self.resize = resize
        self.rand_crop = transforms.RandomCrop(rand_crop) if rand_crop is not None else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')

        if self.square_crop:
            w, h = img.size
            l = min(w, h)
            left, upper = (w - l) // 2, (h - l) // 2
            img = img.crop((left, upper, left + l, upper + l))

        if self.resize is not None:
            r = self.resize
            if isinstance(r, int):
                r = (r, r)
            img = img.resize(r, Image.LANCZOS)

        if self.rand_crop is not None:
            img = self.rand_crop(img)

        return img
