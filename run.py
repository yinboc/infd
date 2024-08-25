"""
    Parse args and make cfg, then spawn trainers which run according to cfg.
"""
import argparse
import os

import torch.distributed as dist
from omegaconf import OmegaConf

from utils import ensure_path
from trainers import trainers_dict


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/_.yaml')
    parser.add_argument('--opt', nargs='*', default=[])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume-mode', '-r', default='replace')

    parser.add_argument('--save-root', default='save')
    parser.add_argument('--name', '-n', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--wandb', '-w', action='store_true')
    args = parser.parse_args()
    return args


def parse_cfg(cfg):
    if cfg.get('_base_') is not None:
        fnames = cfg.pop('_base_')
        if isinstance(fnames, str):
            fnames = [fnames]
        base_cfg = OmegaConf.merge(*[parse_cfg(OmegaConf.load(_)) for _ in fnames])
        cfg = OmegaConf.merge(base_cfg, cfg)
    return cfg


def make_cfg(args):
    cfg = parse_cfg(OmegaConf.load(args.cfg))
    for i in range(0, len(args.opt), 2):
        k, v = args.opt[i: i + 2]
        OmegaConf.update(cfg, k, v)
    cfg.random_seed = args.seed

    env = OmegaConf.create()
    if args.name is None:
        exp_name = os.path.splitext(os.path.basename(args.cfg))[0]
    else:
        exp_name = args.name
    if args.tag is not None:
        exp_name += '_' + args.tag
    env.exp_name = exp_name
    env.save_dir = os.path.join(args.save_root, exp_name)
    env.wandb = args.wandb
    env.resume_mode = args.resume_mode
    
    cfg._env = env
    return cfg


if __name__ == '__main__':
    args = make_args()
    cfg = make_cfg(args)
    trainer = trainers_dict[cfg.trainer](cfg)
    trainer.run()
