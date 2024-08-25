import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.vqgan.quantizer import VectorQuantizer
from diffusion import make_diffusion_train_components, get_dm_loss


class INFDBase(nn.Module):

    def __init__(self, encoder, z_shape, decoder, renderer,
                 z_dm=None, z_dp=None, z_dm_ema_rate=0.9999, z_ddim_eta=1.0,
                 z_dm_cond=None, z_dm_cond_encoder=None, z_dm_cond_learnable=True, z_dm_cond_opt='crossattn', z_dm_p_uncond=0,
                 z_gaussian=False, quantizer=False, n_embed=None, vq_beta=0.25, loss_cfg=None, **kwargs):
        super().__init__()
        self.encoder = models.make(encoder)
        self.z_shape = z_shape

        self.z_gaussian = z_gaussian
        if quantizer:
            self.quantizer = VectorQuantizer(n_embed, z_shape[0], beta=vq_beta, remap=None, sane_index_shape=False)
        else:
            self.quantizer = None

        self.decoder = models.make(decoder)
        self.renderer = models.make(renderer)

        if z_dm is not None:
            self.z_dm = models.make(z_dm)
            self.z_dm_ema = copy.deepcopy(self.z_dm)
            for p in self.z_dm_ema.parameters():
                p.requires_grad = False
            self.z_dm_ema_rate = z_dm_ema_rate
            self.z_dp, self.z_dss, self.z_dp_sampling = make_diffusion_train_components(z_dp)
            self.z_ddim_eta = z_ddim_eta

            self.z_dm_cond = z_dm_cond
            if z_dm_cond is not None:
                self.z_dm_cond_encoder = models.make(z_dm_cond_encoder)
                self.z_dm_cond_learnable = z_dm_cond_learnable
                self.z_dm_cond_opt = z_dm_cond_opt
                self.z_dm_p_uncond = z_dm_p_uncond
        else:
            self.z_dm = None
        self.loss_cfg = loss_cfg if loss_cfg is not None else dict()

    def get_params(self, name):
        if name == 'encoder':
            return self.encoder.parameters()
        elif name == 'decoder':
            p = list(self.decoder.parameters())
            if self.quantizer is not None:
                p += list(self.quantizer.parameters())
            return p
        elif name == 'renderer':
            return self.renderer.parameters()
        elif name == 'z_dm':
            return self.z_dm.parameters()

    def run_renderer(self, z_dec, coord, cell):
        raise NotImplementedError

    def encode_z(self, x, ret_kl_loss=False):
        z = self.encoder(x)

        if self.z_gaussian:
            posterior = DiagonalGaussianDistribution(z)
            z = posterior.sample()
            kl_loss = posterior.kl().mean()
        else:
            kl_loss = None

        if ret_kl_loss:
            return z, kl_loss
        else:
            return z

    def decode_z(self, z, ret_quant_loss=False, gt_cell=None):
        if self.quantizer is not None:
            z, quant_loss, _ = self.quantizer(z)
        else:
            quant_loss = None

        z_dec = self.decoder(z)

        if ret_quant_loss:
            return z_dec, quant_loss
        else:
            return z_dec

    def forward(self, data, mode, has_opt=None, **kwargs):
        gd = self.get_gd_from_opt(has_opt)
        lcfg = self.loss_cfg
        loss = torch.tensor(0, dtype=torch.float32, device=data['inp'].device)
        ret = dict()

        inp = data['inp']

        if gd['encoder']:
            z, kl_loss = self.encode_z(inp, ret_kl_loss=True)
        else:
            with torch.no_grad():
                z, kl_loss = self.encode_z(inp, ret_kl_loss=True)

        if self.z_gaussian:
            ret['kl_loss'] = kl_loss.item()
            loss = loss + kl_loss * lcfg.get('kl_loss', 1)

        if gd['z_dm']:
            if self.z_dm_cond is not None:
                if self.z_dm_cond_learnable:
                    cond = self.z_dm_cond_encoder(data[self.z_dm_cond], p_uncond=self.z_dm_p_uncond)
                else:
                    with torch.no_grad():
                        cond = self.z_dm_cond_encoder(data[self.z_dm_cond], p_uncond=self.z_dm_p_uncond)
                if self.z_dm_cond_opt == 'crossattn':
                    model_kwargs = {'context': cond}
                else:
                    raise NotImplementedError
            else:
                model_kwargs = {}

            t, weights = self.z_dss.sample(z.shape[0], z.device)
            z_dm_loss = get_dm_loss(self.z_dm, self.z_dp, self.z_dss, z, t=t, weights=weights,
                                    model_kwargs=model_kwargs)
            loss = loss + z_dm_loss * lcfg.get('z_dm_loss', 1)
            ret['z_dm_loss'] = z_dm_loss.item()
            if not self.training:
                ret['z_dm_ema_loss'] = get_dm_loss(self.z_dm_ema, self.z_dp, self.z_dss, z, t=t, weights=weights,
                                                   model_kwargs=model_kwargs).item()

        if mode == 'z_dec':
            if gd['decoder']:
                z_dec, quant_loss = self.decode_z(z, ret_quant_loss=True, gt_cell=data['gt_cell'])
            else:
                with torch.no_grad():
                    z_dec, quant_loss = self.decode_z(z, ret_quant_loss=True, gt_cell=data['gt_cell'])
            ret_z = z_dec

            if self.quantizer is not None:
                ret['quant_loss'] = quant_loss.item()
                loss = loss + quant_loss * lcfg.get('quant_loss', 1)

        elif mode == 'z':
            ret_z = z

        ret['loss'] = loss
        return ret_z, ret

    def update_dm_ema(self):
        if self.z_dm is not None:
            for ema_p, cur_p in zip(self.z_dm_ema.parameters(), self.z_dm.parameters()):
                ema_p.data = ema_p.data * self.z_dm_ema_rate + cur_p.data * (1 - self.z_dm_ema_rate)

    def get_gd_from_opt(self, opt):
        if opt is None:
            opt = dict()
        gd = dict()
        gd['encoder'] = opt.get('encoder', False)
        gd['decoder'] = opt.get('encoder', False) or opt.get('decoder', False)
        gd['renderer'] = opt.get('encoder', False) or opt.get('decoder', False) or opt.get('renderer', False)
        gd['z_dm'] = (self.z_dm is not None) and (opt.get('encoder', False) or opt.get('z_dm', False))
        return gd


class DiagonalGaussianDistribution(object):

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
