import argparse
import os
from tqdm import tqdm
from multiprocessing import Process
from PIL import Image, ImageFile

from utils import ensure_path


Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTS = ('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.webp')


def worker(pid):
    files = [os.path.join(args.input, _) for _ in sorted(os.listdir(args.input)) if _.endswith(IMAGE_EXTS)]
    outnames = [os.path.splitext(os.path.basename(_))[0] + '.png' for _ in files]
    L = (len(files) + args.n_process - 1) // args.n_process
    files = files[pid * L: (pid + 1) * L]
    outnames = outnames[pid * L: (pid + 1) * L]

    pbar = list(zip(files, outnames))
    if pid == 0:
        pbar = tqdm(pbar)
    for file, outname in pbar:
        img = Image.open(file).convert('RGB')
        if img.size[0] < args.resize or img.size[1] < args.resize:
            continue

        w, h = img.size
        l = min(w, h)
        left, upper = (w - l) // 2, (h - l) // 2
        img = img.crop((left, upper, left + l, upper + l))

        r = args.resize
        img = img.resize((r, r), Image.LANCZOS)
        img.save(os.path.join(args.output, outname))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--n-process', type=int, default=32)
    args = parser.parse_args()
    ensure_path(args.output)

    ps = []
    for i in range(args.n_process):
        p = Process(target=worker, args=(i,))
        ps.append(p)
        p.start()

    for p in ps:
        p.join()
