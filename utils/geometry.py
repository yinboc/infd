import numpy as np
import torch
import torch.nn.functional as F


def make_coord_grid(shape, range=(-1, 1), device='cpu'):
    """
        Args:
            shape: (s_1, ..., s_k), grid shape
            range: range for each axis, list or tuple, [minv, maxv] or [[minv_1, maxv_1], ..., [minv_k, maxv_k]]
        Returns:
            (s_1, ..., s_k, k), coordinate grid
    """
    l_lst = []
    for i, s in enumerate(shape):
        l = (0.5 + torch.arange(s, device=device)) / s
        if isinstance(range[0], list) or isinstance(range[0], tuple):
            minv, maxv = range[i]
        else:
            minv, maxv = range
        l = minv + (maxv - minv) * l
        l_lst.append(l)
    grid = torch.stack(torch.meshgrid(*l_lst, indexing='ij'), dim=-1)
    return grid


def make_coord_cell_grid(shape, range=(-1, 1), device='cpu', bs=None):
    coord = make_coord_grid(shape, range=range, device=device)
    cell = torch.ones_like(coord)
    for i, s in enumerate(shape):
        if isinstance(range[0], list) or isinstance(range[0], tuple):
            minv, maxv = range[i]
        else:
            minv, maxv = range
        cell[..., i] *= (maxv - minv) / s
    if bs is not None:
        coord = coord.unsqueeze(0).expand(bs, *([-1] * coord.dim()))
        cell = cell.unsqueeze(0).expand(bs, *([-1] * cell.dim()))
    return coord, cell


def convert_posenc(x, dim, w_max):
    """
        Args:
            x: (..., d)
            dim: PE dimension
            w_max: the PE weights are (np.pi * w_i * x), where w_i are exponentially increasing in range [1, w_max]
        Returns:
            (..., d * dim), new x
    """
    assert dim % 2 == 0 # cos and sin each take half
    w = torch.exp(torch.linspace(0, np.log(w_max), dim // 2, device=x.device))
    x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
    x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)
    return x


def convert_liif_feat_coord_cell(feat, coord, cell):
    """
        Get LIIF rawmap of coord, cell on feat.

        Args:
            feat: (B, C, H, W)
            coord, cell: (B, ..., 2), assume range [-1, 1]
        Returns:
            q_feat, rel_coord, rel_cell: (B, ..., C/2/2)
    """
    B = feat.shape[0]
    device = feat.device
    query_shape = coord.shape[1: -1]
    coord = coord.view(B, 1, -1, 2)
    cell = cell.view(B, 1, -1, 2)

    feat_coord = (make_coord_grid(feat.shape[-2:], device=device)
        .permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)) # B 2 H W

    q_feat = F.grid_sample(feat, coord.flip(-1), mode='nearest', align_corners=False).permute(0, 2, 3, 1) # B 1 n C
    q_coord = F.grid_sample(feat_coord, coord.flip(-1), mode='nearest', align_corners=False).permute(0, 2, 3, 1) # B 1 n 2

    rel_coord = coord - q_coord
    rel_coord[..., 0] *= feat.shape[-2]
    rel_coord[..., 1] *= feat.shape[-1]

    rel_cell = cell.clone()
    rel_cell[..., 0] *= feat.shape[-2]
    rel_cell[..., 1] *= feat.shape[-1]

    q_feat = q_feat.view(B, *query_shape, -1)
    rel_coord = rel_coord.view(B, *query_shape, 2)
    rel_cell = rel_cell.view(B, *query_shape, 2)
    return q_feat, rel_coord, rel_cell
