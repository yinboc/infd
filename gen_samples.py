import argparse
import os

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from torchvision import transforms

import models
from utils.geometry import make_coord_cell_grid


def render_output_patchwise(z, model, output_size, patch_size=None, patch_padding=0):
    with torch.no_grad():
        B = z.shape[0]
        z_dec = model.decode_z(z)

        res = output_size
        if patch_size is None:
            patch_size = res

        coord, cell = make_coord_cell_grid((res, res), device=z.device, bs=B) # b h w 2
            
        output = torch.zeros(B, 3, res, res) # cpu
        p = patch_size
        pad = patch_padding
        for x0 in range(0, res, p):
            for y0 in range(0, res, p):
                xl = min(p, res - x0)
                xpad_l = x0 - max(x0 - pad, 0)
                xpad_r = min(x0 + xl + pad, res) - (x0 + xl)
                yl = min(p, res - y0)
                ypad_l = y0 - max(y0 - pad, 0)
                ypad_r = min(y0 + yl + pad, res) - (y0 + yl)
                patch = model.run_renderer(z_dec,
                    coord[:, x0 - xpad_l: x0 + xl + xpad_r, y0 - ypad_l: y0 + yl + ypad_r, :].contiguous(),
                    cell[:, x0 - xpad_l: x0 + xl + xpad_r, y0 - ypad_l: y0 + yl + ypad_r, :].contiguous())
                output[:, :, x0: x0 + xl, y0: y0 + yl] = patch[:, :, xpad_l: xpad_l + xl, ypad_l: ypad_l + yl]
        return output


def dm_worker(rank, args):
    torch.cuda.set_device(rank)
    num_samples = args.n_samples // n_gpus + (rank < args.n_samples % n_gpus)
    model = models.make(torch.load(args.model, map_location='cpu')['model'], load_sd=True).cuda()
    model.eval()

    if args.ema == 'true':
        dm = model.z_dm_ema
    else:
        dm = model.z_dm
    print(f'Model loaded. DM ema enabled: {args.ema}')

    z_shape = model.z_shape
    b = args.batch_size
    to_pil_img = transforms.ToPILImage()

    for i in tqdm(range(0, num_samples, b)):
        n = min(i + b, num_samples) - i
        
        with torch.no_grad():
            model_kwargs = {}
            z = model.z_dp_sampling.ddim_sample_loop(dm, (n, *z_shape), model_kwargs=model_kwargs,
                                                     eta=model.z_ddim_eta, clip_denoised=False)

            for res in args.output_sizes.split(','):
                output = render_output_patchwise(z, model, int(res), args.patch_size, args.patch_padding)
                imgs = (output * 0.5 + 0.5).clamp(0, 1)
                for j in range(i, min(i + b, num_samples)):
                    sid = rank + j * n_gpus
                    to_pil_img(imgs[j - i]).save(os.path.join(args.outdir, res, f'{sid}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m')
    parser.add_argument('--n-samples', '-n', type=int)
    parser.add_argument('--batch-size', '-b', type=int, default=1)
    parser.add_argument('--outdir', '-o')
    parser.add_argument('--output-sizes')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--patch-padding', type=int, default=8)
    parser.add_argument('--ema', default='true')
    args = parser.parse_args()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        for res in args.output_sizes.split(','):
            os.mkdir(os.path.join(args.outdir, res))

    if n_gpus == 1:
        dm_worker(0, args)
    else:
        processes = []
        for rank in range(n_gpus):
            p = mp.Process(target=dm_worker, args=(rank, args,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
