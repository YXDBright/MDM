
import functools
import torch
import torch.nn as nn
from torch import nn
import torch.nn.functional as F
from models.module_util import make_layer, initialize_weights
from models.commons import Mish, SinusoidalPosEmb, RRDB, Residual, Rezero, LinearAttention
from models.commons import ResnetBlock, Upsample, Block, Downsample

# from dong_testpy.VMamba.VMamba_Block import *

import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from selective_scan import selective_scan_fn as selective_scan_fn_v1
from selective_scan import selective_scan_ref as selective_scan_ref_v1


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank



        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)


        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)


        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError


        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):

        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):

        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y


    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

#
#
# class Partial_conv3(nn.Module):
#     def __init__(self, dim, n_div, forward):
#
#         super().__init__()
#         self.dim_conv3 = dim // n_div
#         self.dim_untouched = dim - self.dim_conv3
#
#         self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
#
#
#         if forward == 'slicing':
#             self.forward = self.forward_slicing
#         elif forward == 'split_cat':
#             self.forward = self.forward_split_cat
#         else:
#             raise NotImplementedError
#
#     def forward_slicing(self, x):
#
#         x = x.clone()
#         x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])  # 只对输入张量的一部分通道应用卷积操作
#         return x
#
#     def forward_split_cat(self, x):
#
#         x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)  # 将输入特征图分为两部分
#         x1 = self.partial_conv3(x1)
#         x = torch.cat((x1, x2), 1)
#         return x

class CAM(nn.Module):
    def __init__(self, in_channels, ratio=16):

        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.channels = max(1, in_channels // ratio)
        self.conv1 = nn.Conv2d(in_channels, self.channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels, self.channels, kernel_size=1, bias=True)
        self.conv3 = nn.Conv2d(self.channels, in_channels, kernel_size=1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.avg_pool(x)
        x1 = self.conv1(x1)

        x2 = self.max_pool(x)
        x2 = self.conv2(x2)


        x1 = x1 + x2


        x1 = self.relu(x1)


        x1 = self.conv3(x1)


        x1 = self.sigmoid(x1)

        x = x * x1

        return x




from einops import rearrange, repeat



def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)



def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

''''''
# class SIFMamba(nn.Module):
#     def __init__(self,SS2D ,cam):
#
#         super(SIFMamba, self).__init__()
#         self.SS2D = SS2D
#         self.cam = cam
#
#     def forward(self, x):
#
#         x1 = self.SS2D(x)
#         x2 = self.cam(x)
#
#         output = x1 + x2
#         return output
''''''



class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum('..., f -> ... f', t, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)

        freqs_w = torch.einsum('..., f -> ... f', t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

        print('======== shape of rope freq', self.freqs_cos.shape, '========')

    def forward(self, t, start_index = 0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim = -1)




import torch
import torch.nn as nn
import math


class CM(nn.Module):

    def __init__(self, lon=1e-12):
        super().__init__()
        self.lon = lon


        self.dummy_params = nn.Parameter(torch.eye(3, 3))
        self.register_buffer('phase_shift', torch.tensor([math.pi / 2]))

        self.fake_norm = nn.LayerNorm(1, elementwise_affine=False)

        self.temporal_mask = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.temporal_mask[0].weight.data.zero_()

    def oper(self, tensor):

        tensor = tensor * torch.sin(self.phase_shift)
        tensor = torch.log(1 + self.lon + torch.exp(tensor)) - tensor.detach()


        B, C, H, W = tensor.shape
        tensor = self.fake_norm(tensor.view(B * C, H * W, 1)).view(B, C, H, W)


        return self.temporal_mask(tensor.mean(dim=1, keepdim=True)) * 0 + tensor

    def forward(self, x, y):

        b, c_x, h, w = x.shape
        _, c_y, _, _ = y.shape


        if c_x < c_y:
            repeat_factor = (c_y + c_x - 1) // c_x
            x = x.repeat(1, repeat_factor, 1, 1)[:, :c_y, :, :]
        elif c_x > c_y:
            x = x[:, :c_y, :, :]

        x = self.oper(x * self.lon) * (1 / max(self.lon, 1e-12))

        return x
cm=CM().to('cuda')




#
# class VisionRotaryEmbeddingFast(nn.Module):
#     def __init__(
#         self,
#         dim,
#         pt_seq_len=16,
#         ft_seq_len=None,
#         custom_freqs = None,
#         freqs_for = 'lang',
#         theta = 10000,
#         max_freq = 10,
#         num_freqs = 1,
#     ):
#         super().__init__()
#         if custom_freqs:
#             freqs = custom_freqs
#         elif freqs_for == 'lang':
#             freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
#         elif freqs_for == 'pixel':
#             freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
#         elif freqs_for == 'constant':
#             freqs = torch.ones(num_freqs).float()
#         else:
#             raise ValueError(f'unknown modality {freqs_for}')
#
#         if ft_seq_len is None: ft_seq_len = pt_seq_len
#         t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
#
#         freqs = torch.einsum('..., f -> ... f', t, freqs)
#         freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
#         freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)
#
#         freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
#         freqs_sin = freqs.sin().view(-1, freqs.shape[-1])
#
#         self.register_buffer("freqs_cos", freqs_cos)
#         self.register_buffer("freqs_sin", freqs_sin)
#
#         print('======== shape of rope freq', self.freqs_cos.shape, '========')
#
#     def forward(self, t):
#         if t.shape[1] % 2 != 0:
#             t_spatial = t[:, 1:, :]
#             t_spatial = t_spatial * self.freqs_cos + rotate_half(t_spatial) * self.freqs_sin
#             return torch.cat((t[:, :1, :], t_spatial), dim=1)
#         else:
#             return  t * self.freqs_cos + rotate_half(t) * self.freqs_sin
#

import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn



class Block_vision_mamba(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):

        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)




class RRDBNet(nn.Module):

    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()

        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)


        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, get_fea=False):

        feas = []
        x = (x + 1) / 2

        fea_first = fea = self.conv_first(x)
        for l in self.RRDB_trunk:
            fea = l(fea)
            feas.append(fea)
        trunk = self.trunk_conv(fea)
        fea = fea_first + trunk
        feas.append(fea)
        fea = self.lrelu(
            self.upconv1(F.interpolate(fea, scale_factor=6)))
        fea_hr = self.HRconv(fea)
        out = self.conv_last(self.lrelu(fea_hr))
        out = out.clamp(0, 1)
        out = out * 2 - 1

        if get_fea:
            return out, feas
        else:
            return out


class Unet(nn.Module):

    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), cond_dim=32, flag="ker"):
        super().__init__()
        dims = [2, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        groups = 0
        rrdb_num_block = 8
        sr = 7


        self.cond_proj = nn.ConvTranspose2d(cond_dim * ((rrdb_num_block + 1) // 3),  # 二维转置卷积 ConvTranspose2d(in_channel,out_channels,kernel_size,stride,padding)
                                            dim, sr* 2, sr,
                                            sr // 2)

        # self.cond_proj = nn.ConvTranspose2d(cond_dim * ((rrdb_num_block + 1) // 3),
        #                                     # 二维转置卷积 ConvTranspose2d(in_channel,out_channels,kernel_size,stride,padding) (96,dim=64,6,3,1)
        #                                     dim, 5, sr
        #                                     sr // 2)


        self.time_pos_emb = SinusoidalPosEmb(dim)



        self.Convspp = nn.Conv2d(dim, dim, 3,stride=1, padding=1, bias=True)

        self.mlp = nn.Sequential(
            nn.Linear(640, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)


            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim, groups=groups, flag = flag),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim, groups=groups, flag = flag),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups, flag = flag)

        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups, flag = flag)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim, groups=groups, flag = flag),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups, flag = flag),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Block(dim, dim, groups=groups, flag = flag),
            nn.Conv2d(dim, out_dim, 1)  # ****************************************************************************0925
        )


        # kwargs = {}
        # self.ss2d=SS2D(d_model=dim, dropout=0, d_state=16, **kwargs).to('cuda')

    def apply_weight_norm(self):

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)


        self.apply(_apply_weight_norm)

    def forward(self, x, time, cond, img_lr_up, kernel=False):


        FE=img_lr_up


        t = self.time_pos_emb(time)

        kernel_re = torch.reshape(kernel, (t.size(0),-1))
        
        t = torch.cat((kernel_re, t), 1)
        
        t = self.mlp(t).to(x.device)

        h = []

        cond = self.cond_proj(torch.cat((cond[2::3]), dim=1))
        #print('22', cond.shape)

        for i, (resnet, resnet2, downsample) in enumerate(self.downs):


            x = resnet(x, t)

            x = resnet2(x, t)


            if i == 0:



                # if x.shape[2] < cond.shape[2]:
                #     cond = cond[:, :, :x.shape[2], :x.shape[3]]
                # elif x.shape[2] > cond.shape[2]:
                #     padding = (0, 0, 0, x.shape[2] - cond.shape[2])
                #     cond = F.pad(cond, padding, "constant", 0)
                #
                # #print('After shape adjustment:', x.shape, cond.shape)





                x = x + cond

            h.append(x)
            x = downsample(x)

            tarh, tarw = x.shape[2], x.shape[3]

            FE_h, FE_w = FE.shape[2], FE.shape[3]

            if FE_h != tarh or FE_w != tarw:
                if FE_h < tarh or FE_w < tarw:

                    pad_h = max(0, tarh - FE_h)
                    pad_w = max(0, tarw - FE_w)

                    FE = F.pad(FE, (0, pad_w, 0, pad_h), mode='constant', value=1.0)
                else:

                    FE = FE[:, :, :tarh, :tarw]

            batch_size, channels, height, width = FE.size()

            kwargs = {}

            ss2d = SS2D(d_model=channels, dropout=0, d_state=16, **kwargs).to('cuda')

            FE1 = ss2d(FE.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            batch_size, in_channels, height, width = FE.size()

            ratio = 16
            AM = CAM(in_channels=in_channels,ratio=ratio).to('cuda')

            x2=AM(FE)

            x3=FE1+x2

            batch_size, channels, height, width = x3.size()
            kernel_s = 3
            padding_val = (kernel_s - 1) // 2

            self.conv_layer = nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_s,
                stride=1,
                padding=padding_val,
                dilation=1,
                bias=True
            ).to('cuda')


            x3 = self.conv_layer(x3)

            x4 = cm(x3,x)
            x = x4+x



            # batch_size, channels, height, width = x.size()
            # # x = self.ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            # kwargs = {}

            # ss2d = SS2D(d_model=channels, dropout=0, d_state=16, **kwargs).to('cuda')
            # x = ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            # # output = ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


            #print('down_after', x.shape) # torch.Size([32, 64, 48, 48])

        x = self.mid_block1(x, t)

        tarh, tarw = x.shape[2], x.shape[3]

        FE_h, FE_w = FE.shape[2], FE.shape[3]

        if FE_h != tarh or FE_w != tarw:
            if FE_h < tarh or FE_w < tarw:

                pad_h = max(0, tarh - FE_h)
                pad_w = max(0, tarw - FE_w)

                FE = F.pad(FE, (0, pad_w, 0, pad_h), mode='constant', value=1.0)
            else:

                FE = FE[:, :, :tarh, :tarw]

        batch_size, channels, height, width = FE.size()

        kwargs = {}

        ss2d = SS2D(d_model=channels, dropout=0, d_state=16, **kwargs).to('cuda')

        FE1 = ss2d(FE.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        batch_size, in_channels, height, width = FE.size()

        ratio = 16
        AM = CAM(in_channels=in_channels, ratio=ratio).to('cuda')

        x2 = AM(FE)

        x3 = FE1 + x2

        batch_size, channels, height, width = x3.size()
        kernel_s = 3
        padding_val = (kernel_s - 1) // 2

        self.conv_layer = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_s,
            stride=1,
            padding=padding_val,
            dilation=1,
            bias=True
        )

        x3 = self.conv_layer(x3)

        x4 = cm(x3, x).to('cuda')
        x = x4 + x

        # batch_size, channels, height, width = x.size()
        # # x = self.ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        #
        # kwargs = {}
        #
        # ss2d = SS2D(d_model=channels, dropout=0, d_state=16, **kwargs).to('cuda')
        # x = ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # output = ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # batch_size, channels, height, width = x.size()
        # ssm_cfg = {}
        # factory_kwargs = {}
        # mixer_cls = partial(Mamba, layer_idx=None, **ssm_cfg, **factory_kwargs)
        # # mixer_cls = partial(Mamba, d_state=16, layer_idx=None, bimamba_type=bimamba_type,
        # #                     if_divide_out=False, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
        #
        # b = Block_vision_mamba(dim=channels, mixer_cls=mixer_cls).to('cuda')
        # #input = torch.randn(80, 32, 5, 5).to('cuda')
        # x = x.view(batch_size, channels, height*width).permute(0, 2, 1)
        # out, _ = b(x)
        # x = out.permute(0, 2, 1).view(batch_size, channels, height, width)
        # #print(out.shape)


        x = self.mid_block2(x, t)

        tarh, tarw = x.shape[2], x.shape[3]

        FE_h, FE_w = FE.shape[2], FE.shape[3]

        if FE_h != tarh or FE_w != tarw:
            if FE_h < tarh or FE_w < tarw:

                pad_h = max(0, tarh - FE_h)
                pad_w = max(0, tarw - FE_w)

                FE = F.pad(FE, (0, pad_w, 0, pad_h), mode='constant', value=1.0)
            else:

                FE = FE[:, :, :tarh, :tarw]

        batch_size, channels, height, width = FE.size()

        kwargs = {}

        ss2d = SS2D(d_model=channels, dropout=0, d_state=16, **kwargs).to('cuda')

        FE1 = ss2d(FE.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        batch_size, in_channels, height, width = FE.size()

        ratio = 16
        AM = CAM(in_channels=in_channels, ratio=ratio).to('cuda')

        x2 = AM(FE)

        x3 = FE1 + x2

        batch_size, channels, height, width = x3.size()
        kernel_s = 3
        padding_val = (kernel_s - 1) // 2

        self.conv_layer = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_s,
            stride=1,
            padding=padding_val,
            dilation=1,
            bias=True
        ).to('cuda')

        x3 = self.conv_layer(x3)

        x4 = cm(x3, x).to('cuda')
        x = x4 + x

        for resnet, resnet2, upsample in self.ups:

            # print(x.shape)
            # print(h.pop().shape)

            x = torch.cat((x,h.pop()),dim=1)
            #x = torch.cat((x,h.pop()),dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)

            x=upsample(x)

            tarh, tarw = x.shape[2], x.shape[3]

            FE_h, FE_w = FE.shape[2], FE.shape[3]

            if FE_h != tarh or FE_w != tarw:
                if FE_h < tarh or FE_w < tarw:

                    pad_h = max(0, tarh - FE_h)
                    pad_w = max(0, tarw - FE_w)

                    FE = F.pad(FE, (0, pad_w, 0, pad_h), mode='constant', value=1.0)
                else:

                    FE = FE[:, :, :tarh, :tarw]

            batch_size, channels, height, width = FE.size()

            kwargs = {}

            ss2d = SS2D(d_model=channels, dropout=0, d_state=16, **kwargs).to('cuda')

            FE1 = ss2d(FE.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            batch_size, in_channels, height, width = FE.size()

            ratio = 16
            AM = CAM(in_channels=in_channels, ratio=ratio).to('cuda')

            x2 = AM(FE)

            x3 = FE1 + x2

            batch_size, channels, height, width = x3.size()
            kernel_s = 3
            padding_val = (kernel_s - 1) // 2

            self.conv_layer = nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_s,
                stride=1,
                padding=padding_val,
                dilation=1,
                bias=True
            ).to('cuda')

            x3 = self.conv_layer(x3)

            x4 = cm(x3, x).to('cuda')
            x = x4 + x

            # x = self.ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            # kwargs = {}
            #
            # ss2d = SS2D(d_model=channels, dropout=0, d_state=16, **kwargs).to('cuda')
            # x = ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            # # output = ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

           # print('up_before', x.shape)  # [32, 128, 12, 12]
           #  shape = x.size()
           #  b, c, h,w = shape
           #  block = DeformConv2d(inc=c, outc=c).cuda
           #  A=block(x.to(torch.device("cuda")))
           #  x = A(x)
           # print('up_after', x.shape)  # [32, 128, 24, 24]

        return self.final_conv(x)

    def make_generation_fast_(self):

        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(remove_weight_norm)
