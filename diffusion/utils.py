import copy

import torch
import torch.nn as nn

from models import register
from .unet import UNetModel
from .script_util import create_gaussian_diffusion
from .resample import LossAwareSampler, create_named_schedule_sampler


@register('openai_unet')
def make_openai_unet(**kwargs):
    # note: attention_resolutions isn\t actually the resolution but the downsampling factor, i.e. this
    # corresnponds to attention on spatial resolution 8,16,32, as the spatial reolution of the latents is 64 for f4
    defaults = dict(
        image_size=64, in_channels=3, out_channels=3,
        model_channels=224, channel_mult=[1, 2, 3, 4], num_res_blocks=2,
        attention_resolutions=[8, 4, 2], num_head_channels=32
    )
    defaults.update(kwargs)
    return UNetModel(**defaults)


@register('class_embedder')
class ClassEmbedder(nn.Module):

    def __init__(self, n_embed, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_embed + 1, embed_dim) # last 1 for uncond
        self.n_embed = n_embed

    def forward(self, idx, p_uncond=0):
        idx = idx.clone()
        idx[torch.rand(len(idx), device=idx.device) < p_uncond] = self.n_embed
        return self.embedding(idx).unsqueeze(1)


def make_diffusion_train_components(dp_args=None):
    if dp_args is None:
        dp_args = dict()
    ss_name = dp_args.pop('_schedule_sampler', 'uniform')
    sampling_timestep_respacing = dp_args.pop('_sampling_timestep_respacing', 'ddim200')

    dp = create_gaussian_diffusion(**dp_args)
    dss = create_named_schedule_sampler(ss_name, dp)
    dp_sampling_args = copy.copy(dp_args)
    dp_sampling_args['timestep_respacing'] = sampling_timestep_respacing
    dp_sampling = create_gaussian_diffusion(**dp_sampling_args)
    return dp, dss, dp_sampling


def get_dm_loss(dm, dp, dss, x, t=None, weights=None, model_kwargs=None):
    if t is None:
        t, weights = dss.sample(x.shape[0], x.device)
    dm_losses = dp.training_losses(dm, x, t, model_kwargs=model_kwargs)
    if isinstance(dss, LossAwareSampler):
        dss.update_with_local_losses(t, dm_losses['loss'].detach())
    dm_loss = (dm_losses['loss'] * weights).mean()
    return dm_loss
