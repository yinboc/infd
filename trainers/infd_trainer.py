import os
import random

import torch
import torch.distributed as dist
import torchvision
import torch_fidelity

import utils
from utils.geometry import make_coord_cell_grid
from .trainers import register
from trainers.base_trainer import BaseTrainer


@register('infd_trainer')
class INFDTrainer(BaseTrainer):

    def prepare_visualize(self):
        self.vis_spec = dict()

        def get_samples(dataset, s):
            n = len(dataset)
            lst = [dataset[i] for i in list(range(0, n, n // s))[:s]]
            data = dict()
            for k in lst[0].keys():
                data[k] = torch.stack([_[k] for _ in lst]).cuda()
            return data

        self.vis_spec['ds_samples'] = self.cfg.visualize.get('ds_samples', 0)
        if self.vis_spec['ds_samples'] > 0:
            self.vis_ds_samples = {'train': get_samples(self.datasets['train'], self.vis_spec['ds_samples'])}
            if self.datasets.get('val') is not None:
                self.vis_ds_samples['val'] = get_samples(self.datasets['val'], self.vis_spec['ds_samples'])
        self.vis_ae_center_zoom_res = self.cfg.visualize.get('ae_center_zoom_res')
        self.vis_spec['z_dm_samples'] = self.cfg.visualize.get('z_dm_samples', 0)
        self.vis_spec['z_dm_samples_zoom'] = self.cfg.visualize.get('z_dm_samples_zoom')
        self.vis_spec['z_dm_prog_samples'] = self.cfg.visualize.get('z_dm_prog_samples', 0)

    def make_datasets(self):
        super().make_datasets()

        self.vis_resolution = self.cfg.visualize.resolution
        if isinstance(self.vis_resolution, int):
            self.vis_resolution = (self.vis_resolution, self.vis_resolution)
        if self.is_master:
            random.seed(0) # to get a fixed vis set from wrapper_cae
            self.prepare_visualize()
            if self.cfg.random_seed is not None:
                random.seed(self.cfg.random_seed + self.rank)
            else:
                random.seed()

    def make_model(self, model_spec=None):
        super().make_model(model_spec)
        for name, m in self.model.named_children():
            self.log(f'  .{name} {utils.compute_num_params(m)}')

        self.has_opt = dict()
        if self.cfg.get('optimizers') is not None:
            for name in self.cfg.optimizers.keys():
                self.has_opt[name] = True

    def make_optimizers(self):
        self.optimizers = dict()
        for name, spec in self.cfg.optimizers.items():
            self.optimizers[name] = utils.make_optimizer(self.model.get_params(name), spec)

    def train_step(self, data, bp=True):
        g_iter = self.cfg.get('gan_start_after_iters')
        use_gan = ((g_iter is not None) and self.iter > g_iter)

        ret = self.model_ddp(data, mode='loss', has_opt=self.has_opt, use_gan=use_gan)
        loss = ret.pop('loss')
        ret['loss'] = loss.item()
        if bp:
            self.model_ddp.zero_grad()
            loss.backward()
            for name, o in self.optimizers.items():
                if name != 'disc':
                    o.step()

        if use_gan:
            d_ret = self.model_ddp(data, mode='disc_loss', has_opt=self.has_opt, use_gan=use_gan)
            loss = d_ret.pop('loss')
            ret['disc_loss'] = loss.item()
            ret.update(d_ret)
            if bp:
                self.optimizers['disc'].zero_grad()
                loss.backward()
                self.optimizers['disc'].step()

        self.model.update_dm_ema()

        return ret

    def train_iter_start(self):
        hrft_iter = self.cfg.get('hrft_start_after_iters')
        if hrft_iter is not None and self.iter == hrft_iter + 1:
            self.train_loader = self.loaders['train_hrft']
            self.train_loader_sampler = self.loader_samplers['train_hrft']
            self.train_loader_epoch = 0
            self.train_batch_id = len(self.train_loader) - 1

        prog_iter_rng = self.cfg.get('prog_res_training')
        if prog_iter_rng is not None:
            l, r = prog_iter_rng
            ds = self.loaders['train'].dataset
            if self.iter < l:
                ds.resize_gt_ub = ds.resize_gt_lb
            elif self.iter <= r:
                ds.resize_gt_ub = round(ds.resize_gt_lb + (self.iter - l) / (r - l) * (self.prog_res_ub - ds.resize_gt_lb))
            else:
                ds.resize_gt_ub = self.prog_res_ub


    def run_training(self):
        if self.cfg.get('prog_res_training') is not None:
            ds = self.loaders['train'].dataset
            self.prog_res_ub = ds.resize_gt_ub

        super().run_training()

        if self.model.z_dm is not None:
            cfg = self.cfg.get('final_gen_eval')
            if cfg is not None:
                self.log(f'Generating {cfg.gen_samples} dm samples from the final model ...')
                samples_path = os.path.join(self.cfg._env.save_dir, 'final_gen_samples')
                if self.is_master:
                    os.makedirs(samples_path, exist_ok=True)
                if self.distributed:
                    dist.barrier()
                assert cfg.gen_samples % self.world_size == 0
                self.generate_samples(self.model.z_dm_ema, cfg.gen_samples // self.world_size, bs=cfg.batch_size,
                                      mode='save', save_path=samples_path, save_prefix=f'{self.rank}_')
                if self.distributed:
                    dist.barrier()
                self.log('Done')

                gt_path = cfg.get('gt_path')
                if (gt_path is not None) and self.is_master:
                    ret = torch_fidelity.calculate_metrics(input1=samples_path, input2=gt_path, cuda=True, isc=True, fid=True)
                    for k, v in ret.items():
                        self.log_scalar(k, v)

    def visualize(self):
        self.model_ddp.eval()
        if self.is_master:
            with torch.no_grad():
                if self.vis_spec['ds_samples'] > 0:
                    self.visualize_ae()
                if self.model.z_dm is not None:
                    if self.vis_spec['z_dm_samples'] > 0:
                        self.visualize_z_dm_samples()
                    if self.vis_spec['z_dm_prog_samples'] > 0:
                        self.visualize_z_dm_prog_samples()
        if self.distributed:
            dist.barrier()

    def generate_samples(self, dm, n, bs=1, zoom_boxes=None, mode='ret', save_path=None, save_prefix=''):
        m = self.model
        gens = []
        gens_zoom = []
        to_pil = torchvision.transforms.ToPILImage()

        for i in range(0, n, bs):
            bs_ = min(i + bs, n) - i
            model_kwargs = {}
            if m.z_dm_cond == 'class':
                n_classes = m.z_dm_cond_encoder.n_embed
                model_kwargs['context'] = m.z_dm_cond_encoder(torch.randint(n_classes, (bs_,)).cuda())
            z_gen = m.z_dp_sampling.ddim_sample_loop(
                dm,
                (bs_, *m.z_shape),
                eta=m.z_ddim_eta,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )

            coord, cell = make_coord_cell_grid(self.vis_resolution, device=z_gen.device, bs=bs_)
            x_gen = m.run_renderer(m.decoder(z_gen), coord, cell)
            if mode == 'ret':
                gens.append(x_gen)
            elif mode == 'save':
                for j in range(bs_):
                    fid = i + j
                    to_pil(((x_gen[j] + 1) / 2).clamp(0, 1)).save(os.path.join(save_path, save_prefix + f'{fid}.png'))

            if zoom_boxes is not None:
                coord, cell = make_coord_cell_grid(self.vis_resolution, range=(0, 1), device=z_gen.device, bs=bs_)
                for j in range(bs_):
                    fid = i + j
                    x0, y0, l = zoom_boxes[fid]
                    coord[j, ..., 0] = x0 + coord[j, ..., 0] * l
                    coord[j, ..., 1] = y0 + coord[j, ..., 1] * l
                    cell[j] *= l
                x_gen_zoom = m.run_renderer(m.decoder(z_gen), coord, cell)
                gens_zoom.append(x_gen_zoom)

        if mode == 'ret':
            gens = torch.cat(gens, dim=0)
            if zoom_boxes is None:
                return gens
            else:
                gens_zoom = torch.cat(gens_zoom, dim=0)
                return gens, gens_zoom

    def visualize_ae_(self, name, data, bs=1):
        gt = data['gt']
        n = data['inp'].shape[0]
        pred = []
        center_zoom = []

        for i in range(0, n, bs):
            d = {k: v[i: min(i + bs, n)] for k, v in data.items()}
            pred.append(self.model(d, mode='pred'))

            if (self.vis_ae_center_zoom_res is not None) and (not name.endswith('_whole')):
                r0 = self.vis_resolution[0] / self.vis_ae_center_zoom_res
                r1 = self.vis_resolution[1] / self.vis_ae_center_zoom_res
                d['gt_coord'], d['gt_cell'] = make_coord_cell_grid(
                    self.vis_resolution, [[-r0, r0], [-r1, r1]], device=d['gt_coord'].device, bs=d['gt_coord'].shape[0])
                center_zoom.append(self.model(d, mode='pred'))

        pred = torch.cat(pred, dim=0)
        if self.is_master:
            vimg = []
            for i in range(len(gt)):
                vimg.extend([pred[i], gt[i]])
            vimg = torch.stack(vimg)
            vimg = torchvision.utils.make_grid(vimg, nrow=4, normalize=True, value_range=(-1, 1))
            self.log_image(name, vimg)

        if (self.vis_ae_center_zoom_res is not None) and (not name.endswith('_whole')):
            center_zoom = torch.cat(center_zoom, dim=0)
            if self.is_master:
                vimg = []
                for i in range(len(gt)):
                    vimg.extend([center_zoom[i], center_zoom[i]])
                vimg = torch.stack(vimg)
                vimg = torchvision.utils.make_grid(vimg, nrow=4, normalize=True, value_range=(-1, 1))
                self.log_image(name + '_center_zoom', vimg)

    def visualize_ae(self):
        for split in ['train', 'val']:
            if self.vis_ds_samples.get(split) is None:
                continue
            data = self.vis_ds_samples[split]
            self.visualize_ae_(split, data)

            if self.cfg.visualize.get('vis_ae_whole', False):
                x = data['inp']
                coord, cell = make_coord_cell_grid(x.shape[-2:], device=x.device, bs=x.shape[0])
                data_whole = {'inp': x, 'gt': x, 'gt_coord': coord, 'gt_cell': cell}
                self.visualize_ae_(split + '_whole', data_whole)

    def visualize_z_dm_samples_(self, name, dm, zoom_boxes, bs=1):
        if zoom_boxes is None:
            gens = self.generate_samples(dm, self.vis_spec['z_dm_samples'], bs=bs, zoom_boxes=zoom_boxes, mode='ret')
        else:
            gens, gens_zoom = self.generate_samples(dm, self.vis_spec['z_dm_samples'], bs=bs, zoom_boxes=zoom_boxes, mode='ret')
        if self.is_master:
            vimg = torchvision.utils.make_grid(gens, nrow=4, normalize=True, value_range=(-1, 1))
            self.log_image(name, vimg)
            if zoom_boxes is not None:
                vimg = torchvision.utils.make_grid(gens_zoom, nrow=4, normalize=True, value_range=(-1, 1))
                self.log_image(name + '_zoom', vimg)

    def visualize_z_dm_samples(self):
        zoom_spec = self.vis_spec.get('z_dm_samples_zoom')
        if zoom_spec is not None:
            zoom_min, zoom_max = zoom_spec
            zoom_boxes = []
            for i in range(self.vis_spec['z_dm_samples']):
                l = 2 * (1 / random.uniform(zoom_min, zoom_max))
                x0 = random.uniform(-1, 1 - l)
                y0 = random.uniform(-1, 1 - l)
                zoom_boxes.append((x0, y0, l))
        else:
            zoom_boxes = None

        for name, dm in [('z_dm_samples', self.model.z_dm), ('z_dm_ema_samples', self.model.z_dm_ema)]:
            self.visualize_z_dm_samples_(name, dm, zoom_boxes)

    def visualize_z_dm_prog_samples_(self, name, dm, bs=1, chunks=5):
        m = self.model
        n = self.vis_spec['z_dm_prog_samples']
        sample = []
        pred_xstart = []
        for i in range(0, n, bs):
            bs_ = min(i + bs, n) - i
            model_kwargs = {}
            if m.z_dm_cond == 'class':
                n_classes = m.z_dm_cond_encoder.n_embed
                model_kwargs['context'] = m.z_dm_cond_encoder(torch.randint(n_classes, (bs_,)).cuda())
            prog = m.z_dp_sampling.ddim_sample_loop(
                dm,
                (bs_, *m.z_shape),
                eta=m.z_ddim_eta,
                clip_denoised=False,
                ret_prog=True,
                ret_prog_chunks=chunks,
                model_kwargs=model_kwargs,
            )

            coord, cell = make_coord_cell_grid(self.vis_resolution, device=prog['sample'].device, bs=bs_)
            sample_ = torch.stack([
                m.run_renderer(m.decoder(prog['sample'][:, j, ...]), coord, cell) for j in range(chunks + 1)], dim=1)
            pred_xstart_ = torch.stack([
                m.run_renderer(m.decoder(prog['pred_xstart'][:, j, ...]), coord, cell) for j in range(chunks + 1)], dim=1)
            sample.append(sample_)
            pred_xstart.append(pred_xstart_)

        sample = torch.cat(sample, dim=0)
        pred_xstart = torch.cat(pred_xstart, dim=0)
        if self.is_master:
            vimg = []
            for i in range(sample.shape[0]):
                vimg.append(sample[i])
                vimg.append(pred_xstart[i])
            vimg = torch.cat(vimg, dim=0)
            vimg = torchvision.utils.make_grid(vimg, nrow=sample.shape[1], normalize=True, value_range=(-1, 1))
            self.log_image(name, vimg)

    def visualize_z_dm_prog_samples(self):
        for name, dm in [('z_dm_ema_prog_samples', self.model.z_dm_ema)]:
            self.visualize_z_dm_prog_samples_(name, dm)
