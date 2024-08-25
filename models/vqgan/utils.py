import torch.nn as nn

from models import register
from .model import Encoder, Decoder


def defaults_vqf4():
    return dict(
        double_z=False,
        z_channels=3,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        give_pre_end=True,
    )


@register('encoder_vqf4')
def make_vqgan_encoder(**kwargs):
    defaults = defaults_vqf4()
    defaults.update(kwargs)
    enc_out_channels = defaults['z_channels'] * (2 if defaults['double_z'] else 1)
    return nn.Sequential(
        Encoder(**defaults),
        nn.Conv2d(enc_out_channels, enc_out_channels, 1),
    )


@register('decoder_vqf4')
def make_vqgan_decoder(**kwargs):
    defaults = defaults_vqf4()
    defaults.update(kwargs)
    dec_in_channels = defaults['z_channels']
    return nn.Sequential(
        nn.Conv2d(dec_in_channels, dec_in_channels, 1),
        Decoder(**defaults),
    )
