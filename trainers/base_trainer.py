import os
import time
import copy
import random
from functools import partial

import yaml
import wandb
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import datasets
import models
import utils
from .trainers import register


def worker_init_fn_(worker_id, num_workers, rank, world_size, seed):
    glo_worker_id = num_workers * rank + worker_id
    worker_seed = (num_workers * world_size * seed + glo_worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


@register('base_trainer')
class BaseTrainer():

    def __init__(self, cfg):
        self.rank = int(os.environ.get('RANK', '0'))
        self.is_master = (self.rank == 0)
        self.cfg = cfg
        self.cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        env = cfg._env

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        force_replace = False
        if cfg._env.resume_mode == 'resume':
            replace = False
        else:
            replace = True
            if cfg._env.resume_mode == 'force_replace':
                force_replace = True
            elif cfg._env.resume_mode != 'replace':
                raise NotImplementedError
        if self.is_master:
            utils.ensure_path(cfg._env.save_dir, replace=replace, force_replace=force_replace)

        # Setup log, tb, wandb
        if self.is_master:
            logger, writer = utils.set_save_dir(env.save_dir, replace=False)
            with open(os.path.join(env.save_dir, 'cfg.yaml'), 'w') as f:
                yaml.dump(self.cfg_dict, f, sort_keys=False)
            self.log = logger.info

            self.enable_tb = True
            self.writer = writer

            if env.wandb:
                self.enable_wandb = True
                os.environ['WANDB_NAME'] = env.exp_name
                os.environ['WANDB_DIR'] = env.save_dir
                with open('wandb.yaml', 'r') as f:
                    wandb_cfg = yaml.load(f, Loader=yaml.FullLoader)
                os.environ['WANDB_API_KEY'] = wandb_cfg['api_key']
                wandb.init(project=wandb_cfg['project'], entity=wandb_cfg['entity'], config=self.cfg_dict, resume=True)
            else:
                self.enable_wandb = False
        else:
            self.log = lambda *args, **kwargs: None
            self.enable_tb = False
            self.enable_wandb = False

        # Setup distributed
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.distributed = (self.world_size > 1)

        if self.distributed:
            dist.init_process_group(backend='nccl')
            dist.barrier()
            self.log(f'Distributed training enabled. World size: {self.world_size}.')

        torch.cuda.set_device(self.rank)
        self.device = torch.device('cuda', torch.cuda.current_device())

        dist.barrier()
        
        self.log(f'Environment setup done.')

    def seed_everything(self, seed, rank_shift=True):
        if rank_shift:
            seed += self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def run(self):
        if self.cfg.random_seed is not None:
            self.seed_everything(self.cfg.random_seed, rank_shift=True)

        self.make_datasets()

        if self.cfg.get('eval_only', False):
            model_spec = self.cfg.get('eval_model')
            if model_spec is not None:
                model_spec = torch.load(model_spec, map_location='cpu')['model']
            self.make_model(model_spec); model_spec = None
            self.iter = 0
            self.evaluate()
            self.visualize()
        else:
            resume_file = os.path.join(self.cfg._env.save_dir, 'last-model.pth')
            if os.path.isfile(resume_file):
                ckpt = torch.load(resume_file, map_location='cpu')

            if os.path.isfile(resume_file):
                model_spec = copy.deepcopy(OmegaConf.to_container(self.cfg.model, resolve=True))
                model_spec['sd'] = ckpt['model']['sd']
                self.make_model(model_spec)
                model_spec = None
                self.log(f'Resumed model from checkpoint {resume_file}.')
            else:
                self.make_model()

            self.make_optimizers()
            if os.path.isfile(resume_file):
                opt_dict = ckpt['optimizers']
                for k, v in opt_dict.items():
                    self.optimizers[k].load_state_dict(v['sd'])
                opt_dict = None
                self.log(f'Resumed optimizers from checkpoint {resume_file}.')

            ckpt = None
            self.run_training()

        if self.enable_tb:
            self.writer.close()
        if self.enable_wandb:
            wandb.finish()

    def make_distributed_loader(self, dataset, batch_size, drop_last, shuffle, num_workers):
        num_workers //= self.world_size
        if self.cfg.random_seed is not None:
            worker_init_fn = partial(worker_init_fn_,
                num_workers=num_workers, rank=self.rank, world_size=self.world_size, seed=self.cfg.random_seed)
            persistent_workers = True
        else:
            worker_init_fn = None
            persistent_workers = False
        sampler = DistributedSampler(dataset, shuffle=shuffle) if self.distributed else None
        assert batch_size % self.world_size == 0
        loader = DataLoader(dataset, batch_size // self.world_size, drop_last=drop_last,
                            sampler=sampler, shuffle=((sampler is None) and shuffle),
                            num_workers=num_workers, pin_memory=True,
                            worker_init_fn=worker_init_fn, persistent_workers=persistent_workers)
        return loader, sampler

    def make_datasets(self):
        cfg = self.cfg
        self.datasets = dict()
        self.loaders = dict()
        self.loader_samplers = dict()

        for split, spec in cfg.datasets.items():
            loader_spec = spec.pop('loader')
            dataset = datasets.make(spec)
            self.datasets[split] = dataset
            self.log(f'Datasets - {split}: len={len(dataset)}')

            drop_last = loader_spec.get('drop_last', (split == 'train'))
            shuffle = loader_spec.get('shuffle', (split == 'train'))
            self.loaders[split], self.loader_samplers[split] = self.make_distributed_loader(
                dataset, loader_spec.batch_size, drop_last, shuffle, loader_spec.num_workers)

    def make_model(self, model_spec=None):
        if model_spec is None:
            model = models.make(self.cfg.model)
        else:
            model = models.make(model_spec, load_sd=True)
        self.log(f'Model: #params={utils.compute_num_params(model)}')

        if self.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda()
            model_ddp = DistributedDataParallel(model, device_ids=[self.rank],
                find_unused_parameters=self.cfg.get('find_unused_parameters', False))
        else:
            model.cuda()
            model_ddp = model

        self.model = model
        self.model_ddp = model_ddp

    def make_optimizers(self):
        self.optimizers = {'all': utils.make_optimizer(self.model.parameters(), self.cfg.optimizers)}

    def run_training(self):
        cfg = self.cfg
        max_iter = cfg['max_iter']
        epoch_iter = cfg['epoch_iter']
        assert max_iter % epoch_iter == 0
        max_epoch = max_iter // epoch_iter

        save_iter = cfg.get('save_iter')
        assert save_iter is None or save_iter % epoch_iter == 0
        save_epoch = save_iter // epoch_iter if save_iter is not None else max_epoch + 1

        eval_iter = cfg.get('eval_iter')
        assert eval_iter is None or eval_iter % epoch_iter == 0
        eval_epoch = eval_iter // epoch_iter if eval_iter is not None else max_epoch + 1

        vis_iter = cfg.get('vis_iter')
        assert vis_iter is None or vis_iter % epoch_iter == 0
        vis_epoch = vis_iter // epoch_iter if vis_iter is not None else max_epoch + 1

        if cfg.get('ckpt_select_metric') is not None:
            m = cfg.ckpt_select_metric
            self.ckpt_select_metric = m.name
            self.ckpt_select_type = m.type
            if m.type == 'min':
                self.ckpt_select_v = 1e18
            elif m.type == 'max':
                self.ckpt_select_v = -1e18
        else:
            self.ckpt_select_metric = None

        self.train_loader = self.loaders['train']
        self.train_loader_sampler = self.loader_samplers['train']
        self.train_loader_epoch = 0
        self.train_batch_id = len(self.train_loader) - 1

        self.iter = 0

        resume_file = os.path.join(self.cfg._env.save_dir, 'last-model.pth')
        if os.path.isfile(resume_file):
            ckpt = torch.load(resume_file, map_location='cpu')
            for _ in range(ckpt['iter']):
                self.iter += 1
                self.train_iter_start()
            self.ckpt_select_v = ckpt['ckpt_select_v']
            self.train_loader_epoch = ckpt['train_loader_epoch']
            self.train_batch_id = len(self.train_loader) - 1
            ckpt = None
            self.log(f'Resumed iter status from checkpoint {resume_file}.')

        start_epoch = self.iter // epoch_iter + 1
        epoch_timer = utils.EpochTimer(max_epoch - start_epoch + 1)

        for epoch in range(start_epoch, max_epoch + 1):
            self.log_buffer = [f'Epoch {epoch}']

            if self.distributed:
                for sampler in self.loader_samplers.values():
                    if sampler is not self.train_loader_sampler:
                        sampler.set_epoch(epoch)

            self.model_ddp.train()

            ave_scalars = dict()
            pbar = range(1, epoch_iter + 1)
            if self.is_master:
                pbar = tqdm(pbar, desc='train', leave=False)

            t_data = 0
            t_model = 0
            t1 = time.time()
            for _ in pbar:
                self.iter += 1
                self.train_iter_start()

                self.train_batch_id += 1
                if self.train_batch_id == len(self.train_loader):
                    self.train_loader_epoch += 1
                    if self.distributed:
                        self.train_loader_sampler.set_epoch(self.train_loader_epoch)
                    self.train_loader_iter = iter(self.train_loader)
                    self.train_batch_id = 0

                data = next(self.train_loader_iter)
                data = {k: v.cuda() for k, v in data.items()}
                t0 = time.time()
                t_data += t0 - t1

                ret = self.train_step(data)
                t1 = time.time()
                t_model += t1 - t0

                bs = len(next(iter(data.values())))
                for k, v in ret.items():
                    if ave_scalars.get(k) is None:
                        ave_scalars[k] = utils.Averager()
                    ave_scalars[k].add(v, n=bs)

                if self.is_master:
                    pbar.set_description(desc=f'train: loss={ret["loss"]:.4f}')

            self.sync_ave_scalars_(ave_scalars)

            logtext = 'train:'
            for k, v in ave_scalars.items():
                logtext += f' {k}={v.item():.4f}'
                self.log_scalar('train/' + k, v.item())
            logtext += f' (d={t_data / (t_data + t_model):.2f})'
            self.log_buffer.append(logtext)

            if epoch % save_epoch == 0 and epoch != max_epoch:
                self.save_ckpt(f'iter-{self.iter}.pth')

            if epoch % eval_epoch == 0:
                eval_ave_scalars = self.evaluate()
                if self.ckpt_select_metric is not None:
                    v = eval_ave_scalars[self.ckpt_select_metric].item()
                    if ((self.ckpt_select_type == 'min' and v < self.ckpt_select_v) or
                        (self.ckpt_select_type == 'max' and v > self.ckpt_select_v)):
                        self.ckpt_select_v = v
                        self.save_ckpt('best-model.pth')

            if epoch % vis_epoch == 0:
                self.visualize()

            self.save_ckpt('last-model.pth')

            epoch_time, tot_time, est_time = epoch_timer.epoch_done()
            self.log_buffer.append(f'{epoch_time} {tot_time}/{est_time}')
            self.log(', '.join(self.log_buffer))

    def train_iter_start(self):
        pass

    def train_step(self, data, bp=True):
        ret = self.model_ddp(data)
        loss = ret.pop('loss')
        ret['loss'] = loss.item()
        if bp:
            self.model_ddp.zero_grad()
            loss.backward()
            for o in self.optimizers.values():
                o.step()
        return ret

    def evaluate(self):
        self.model_ddp.eval()

        ave_scalars = dict()
        pbar = self.loaders['val']
        if self.is_master:
            pbar = tqdm(pbar, desc='val', leave=False)

        for data in pbar:
            data = {k: v.cuda() for k, v in data.items()}
            with torch.no_grad():
                ret = self.train_step(data, bp=False)

            bs = len(next(iter(data.values())))
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=bs)

            if self.is_master:
                pbar.set_description(desc=f'val: loss={ret["loss"]:.4f}')

        self.sync_ave_scalars_(ave_scalars)

        logtext = 'val:'
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'
            self.log_scalar('val/' + k, v.item())
        self.log_buffer.append(logtext)

        return ave_scalars

    def visualize(self):
        pass

    def save_ckpt(self, filename):
        if not self.is_master:
            return
        model_spec = copy.copy(self.cfg_dict['model'])
        model_spec['sd'] = self.model.state_dict()
        optimizers_spec = dict()
        for k, v in self.cfg_dict['optimizers'].items():
            spec = copy.copy(v)
            spec['sd'] = self.optimizers[k].state_dict()
            optimizers_spec[k] = spec
        ckpt = {
            'cfg': self.cfg_dict,
            'model': model_spec,
            'optimizers': optimizers_spec,
            'iter': self.iter,
            'train_loader_epoch': self.train_loader_epoch,
            'ckpt_select_v': self.ckpt_select_v,
        }
        torch.save(ckpt, os.path.join(self.cfg._env.save_dir, filename))

    def sync_ave_scalars_(self, ave_scalars):
        if not self.distributed:
            return
        for k, v in ave_scalars.items():
            t = torch.tensor(v.item(), device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t.div_(self.world_size)
            ave_scalars[k].v = t.item()
            ave_scalars[k].n *= self.world_size

    def log_scalar(self, k, v):
        if self.enable_tb:
            self.writer.add_scalar(k, v, global_step=self.iter)
        if self.enable_wandb:
            wandb.log({k: v}, step=self.iter)

    def log_image(self, k, v):
        if self.enable_tb:
            self.writer.add_image(k, v, global_step=self.iter)
        if self.enable_wandb:
            wandb.log({k: wandb.Image(v)}, step=self.iter)
