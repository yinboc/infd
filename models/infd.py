import os

import torch
import torch.nn.functional as F
import torch.distributed as dist

from models import register
from models.infd_base import INFDBase
from models.vqgan.lpips import LPIPS
from models.vqgan.discriminator import make_discriminator


@register('infd')
class INFD(INFDBase):

    def __init__(self, disc=True, disc_cond_scale=False, disc_use_custom=False, adaptive_gan_weight=False, **kwargs):
        super().__init__(**kwargs)
        self.perc_loss = LPIPS().eval()
        input_nc = 3 if not disc_cond_scale else 4
        self.disc = make_discriminator(use_custom=disc_use_custom, input_nc=input_nc) if disc else None
        self.disc_cond_scale = disc_cond_scale
        self.adaptive_gan_weight = adaptive_gan_weight

    def get_params(self, name):
        if name == 'disc':
            return self.disc.parameters()
        else:
            return super().get_params(name)

    def run_renderer(self, z_dec, coord, cell):
        return self.renderer(z_dec=z_dec, coord=coord, cell=cell)

    def loss_downsample_util_fn(self, x, d_inp, d_gt_cell):
        gt_patch_res = torch.FloatTensor([d_gt_cell.shape[1], d_gt_cell.shape[2]]).cuda()
        gt_whole_res = 2 / d_gt_cell[:, 0, 0, :] # B 2
        inp_res = torch.FloatTensor([d_inp.shape[-2], d_inp.shape[-1]]).cuda()

        res = (gt_patch_res.view(1, 2) / gt_whole_res * inp_res.view(1, 2)).round()
        x = x * 0.5 + 0.5
        ret = []
        for i in range(res.shape[0]):
            t = F.interpolate(x[i].unsqueeze(0),
                              size=(round(res[i][0].item()), round(res[i][1].item())),
                              recompute_scale_factor=False,
                              mode='bicubic', align_corners=False, antialias=True)[0]
            ret.append((t - 0.5) / 0.5)
        return ret

    def forward(self, data, mode, has_opt=None, **kwargs):
        gd = self.get_gd_from_opt(has_opt)
        lcfg = self.loss_cfg

        if mode == 'pred':
            z_dec, ret = super().forward(data, mode='z_dec', has_opt=has_opt)
            if gd['renderer']:
                return self.run_renderer(z_dec, data['gt_coord'], data['gt_cell'])
            else:
                with torch.no_grad():
                    return self.run_renderer(z_dec, data['gt_coord'], data['gt_cell'])

        elif mode == 'loss':
            if not gd['renderer']:
                _, ret = super().forward(data, mode='z', has_opt=has_opt)
                return ret

            z_dec, ret = super().forward(data, mode='z_dec', has_opt=has_opt)
            pred_patch = self.run_renderer(z_dec, data['gt_coord'], data['gt_cell'])

            pred = pred_patch
            target = data['gt']
            if lcfg.get('l1_loss_downsample', False):
                pred = self.loss_downsample_util_fn(pred, data['inp'], data['gt_cell'])
                target = self.loss_downsample_util_fn(target, data['inp'], data['gt_cell'])
                l1_loss = 0
                for p, t in zip(pred, target):
                    l1_loss = l1_loss + torch.abs(p - t).mean()
                l1_loss = l1_loss / len(pred)
            else:
                l1_loss = torch.abs(pred - target).mean()
            ret['l1_loss'] = l1_loss.item()
            l1_loss_w = lcfg.get('l1_loss', 1)
            ret['loss'] = ret['loss'] + l1_loss * l1_loss_w

            pred = pred_patch
            target = data['gt']
            if lcfg.get('perc_loss_downsample', False):
                pred = self.loss_downsample_util_fn(pred, data['inp'], data['gt_cell'])
                target = self.loss_downsample_util_fn(target, data['inp'], data['gt_cell'])
                perc_loss = 0
                for p, t in zip(pred, target):
                    perc_loss = perc_loss + self.perc_loss(p.unsqueeze(0), t.unsqueeze(0)).mean()
                perc_loss = perc_loss / len(pred)
            else:
                perc_loss = self.perc_loss(pred, target).mean()
            ret['perc_loss'] = perc_loss.item()
            perc_loss_w = lcfg.get('perc_loss', 1)
            ret['loss'] = ret['loss'] + perc_loss * perc_loss_w

            if kwargs.get('use_gan', False):
                if not self.disc_cond_scale:
                    logits_fake = self.disc(pred_patch)
                else:
                    smap = (data['gt_cell'][..., 0] / 2 * data['inp'].shape[-1]).unsqueeze(1)
                    logits_fake = self.disc(torch.cat([pred_patch, smap], dim=1))
                gan_g_loss = -torch.mean(logits_fake)
                ret['gan_g_loss'] = gan_g_loss.item()
                weight = lcfg.get('gan_g_loss', 1)
                if self.training and self.adaptive_gan_weight:
                    nll_loss = l1_loss * l1_loss_w + perc_loss * perc_loss_w
                    adaptive_g_w = self.calculate_adaptive_g_w(nll_loss, gan_g_loss, self.renderer.get_last_layer_weight())
                    ret['adaptive_g_w'] = adaptive_g_w.item()
                    weight = weight * adaptive_g_w
                ret['loss'] = ret['loss'] + gan_g_loss * weight

            mse = ((data['gt'] - pred_patch) / 2).pow(2).mean(dim=[1, 2, 3])
            ret['psnr'] = (-10 * torch.log10(mse)).mean().item()
            return ret

        elif mode == 'disc_loss':
            with torch.no_grad():
                z_dec, _ = super().forward(data, mode='z_dec', has_opt=None)
                pred_patch = self.run_renderer(z_dec, data['gt_coord'], data['gt_cell'])

            if not self.disc_cond_scale:
                logits_real = self.disc(data['gt'])
                logits_fake = self.disc(pred_patch)
            else:
                smap = (data['gt_cell'][..., 0] / 2 * data['inp'].shape[-1]).unsqueeze(1)
                logits_real = self.disc(torch.cat([data['gt'], smap], dim=1))
                logits_fake = self.disc(torch.cat([pred_patch, smap], dim=1))
            disc_loss_type = lcfg.get('disc_loss_type', 'hinge')
            if disc_loss_type == 'hinge':
                loss_real = torch.mean(F.relu(1. - logits_real))
                loss_fake = torch.mean(F.relu(1. + logits_fake))
                loss = (loss_real + loss_fake) / 2
            elif disc_loss_type == 'vanilla':
                loss_real = torch.mean(F.softplus(-logits_real))
                loss_fake = torch.mean(F.softplus(logits_fake))
                loss = (loss_real + loss_fake) / 2
            return {
                'loss': loss,
                'disc_logits_real': logits_real.mean().item(),
                'disc_logits_fake': logits_fake.mean().item(),
            }

    def calculate_adaptive_g_w(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        if world_size > 1:
            dist.all_reduce(nll_grads, op=dist.ReduceOp.SUM)
            nll_grads.div_(world_size)
            dist.all_reduce(g_grads, op=dist.ReduceOp.SUM)
            g_grads.div_(world_size)
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight
