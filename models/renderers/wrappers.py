import copy

import torch
import torch.nn as nn

import models
from models import register
from utils.geometry import convert_liif_feat_coord_cell, convert_posenc


@register('renderer_concat_wrapper')
class ConcatWrapper(nn.Module):

    def __init__(self, net, z_dec_channels, co_pe_dim=None, co_pe_w_max=None, ce_pe_dim=None, ce_pe_w_max=None, x_channels=None):
        super().__init__()
        self.x_channels = x_channels
        coord_dim = 0

        self.co_pe_dim = co_pe_dim
        self.co_pe_w_max = co_pe_w_max
        coord_dim += 2 if co_pe_dim is None else 2 * co_pe_dim

        self.ce_pe_dim = ce_pe_dim
        self.ce_pe_w_max = ce_pe_w_max
        coord_dim += 2 if ce_pe_dim is None else 2 * ce_pe_dim

        net_spec = copy.copy(net)
        net_spec['args']['in_channels'] = (x_channels if x_channels is not None else 0) + z_dec_channels + coord_dim
        self.net = models.make(net_spec)

    def get_last_layer_weight(self):
        return self.net.get_last_layer_weight()

    def forward(self, z_dec, coord, cell):
        q_feat, rel_coord, rel_cell = convert_liif_feat_coord_cell(z_dec, coord, cell)
        if self.co_pe_dim is not None:
            rel_coord = convert_posenc(rel_coord, self.co_pe_dim, self.co_pe_w_max)
        if self.ce_pe_dim is not None:
            rel_cell = convert_posenc(rel_cell, self.ce_pe_dim, self.ce_pe_w_max)
        layout = torch.cat([q_feat, rel_coord, rel_cell], dim=-1).permute(0, 3, 1, 2)
        return self.net(layout)


@register('renderer_fixres_wrapper')
class FixresWrapper(nn.Module):

    def __init__(self, net, z_dec_channels):
        super().__init__()
        net_spec = copy.copy(net)
        net_spec['args']['in_channels'] = z_dec_channels
        self.net = models.make(net_spec)

    def get_last_layer_weight(self):
        return self.net.get_last_layer_weight()

    def forward(self, z_dec, coord, cell, x=None, timesteps=None):
        return self.net(z_dec)
