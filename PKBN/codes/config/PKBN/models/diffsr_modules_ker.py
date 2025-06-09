import functools
import yaml
from models.module_util import make_layer, initialize_weights
from models.commons import Mish, SinusoidalPosEmb, RRDB, Residual, Rezero, LinearAttention
from models.commons import ResnetBlock, Upsample, Block, Downsample
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
        # S4D real initialization
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
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

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





class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


with open('/root/No2JBHIopen/PKBN/codes/config/PKBN/options/setting1/train/train_PKBN.yml',
          'r') as file:
    data2 = yaml.safe_load(file)

lr_p = data2['train']['lr']
class ZPool(nn.Module):
    def forward(self, x):

        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


class GFPM(nn.Module):
    def __init__(self, in_channels, embed_channels, fc_hidden_channels, dw_kernel_size):

        super().__init__()
        if isinstance(dw_kernel_size, int):
            dw_padding = (dw_kernel_size - 1) // 2
        elif isinstance(dw_kernel_size, tuple) and len(dw_kernel_size) == 2:
            dw_padding = ((dw_kernel_size[0] - 1) // 2, (dw_kernel_size[1] - 1) // 2)
        else:
            raise ValueError("Error")

        self.initial_proj = nn.Conv2d(in_channels, embed_channels, kernel_size=1, bias=True)

        self.dw_conv1 = nn.Conv2d(embed_channels, embed_channels, kernel_size=dw_kernel_size,
                                  padding=dw_padding, groups=embed_channels, bias=True)
        self.dw_conv2 = nn.Conv2d(embed_channels, embed_channels, kernel_size=dw_kernel_size,
                                  padding=dw_padding, groups=embed_channels, bias=True)

        self.fc1 = nn.Conv2d(embed_channels, fc_hidden_channels, kernel_size=1, bias=True)

        self.sigmoid = nn.Sigmoid()

        self.fc2 = nn.Conv2d(fc_hidden_channels, embed_channels, kernel_size=1, bias=True)

        self.tanh = nn.Tanh()

        self.dw_conv3 = nn.Conv2d(embed_channels, embed_channels, kernel_size=dw_kernel_size,
                                  padding=dw_padding, groups=embed_channels, bias=True)

        self.fc3 = nn.Conv2d(embed_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x):

        residual_input = x

        x_processed = self.initial_proj(x)

        dw1_out = self.dw_conv1(x_processed)

        dw2_out = self.dw_conv2(x_processed)

        multiplied_1 = dw1_out * dw2_out

        fc1_out = self.fc1(multiplied_1)

        sigmoid_out = self.sigmoid(fc1_out)

        fc2_out = self.fc2(sigmoid_out)

        tanh_out = self.tanh(fc2_out)

        dw3_out = self.dw_conv3(x_processed)

        multiplied_2 = tanh_out * dw3_out

        fc3_out = self.fc3(multiplied_2)

        output = fc3_out + residual_input

        return output


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
    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), cond_dim=32, flag = "ker"):
        super().__init__()
        dims = [3, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        groups = 0
        rrdb_num_block = 8
        sr = 4


        self.cond_proj = nn.ConvTranspose2d(cond_dim * ((rrdb_num_block + 1) // 3),
                                            dim, sr* 2, sr,
                                            sr // 2)

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
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
            nn.Conv2d(dim, 1, 1)
        )

        # if hparams['res'] and hparams['up_input']:
        #     self.up_proj = nn.Sequential(
        #         nn.ReflectionPad2d(1), nn.Conv2d(3, dim, 3),
        #     )
        # if hparams['use_wn']:
        #     self.apply_weight_norm()
        # if hparams['weight_init']:
        #     self.apply(initialize_weights)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                # print(f"| Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def forward(self, x, time, cond, img_lr_up):
        t = self.time_pos_emb(time)

        t = self.mlp(t).to(x.device)

        xd=x
        h = []


        cond = self.cond_proj(torch.cat((cond[2::3]), dim=1))

        cond = cond[:, :, 0:24, 0:24]



        for i, (resnet, resnet2, downsample) in enumerate(self.downs):
            # print("x:",x.shape)
            # print(t.shape)
            x = resnet(x, t)
            x = resnet2(x, t)

            # print(11111111111)
            # print(x.size())
            # print(cond.size())
            # print(2222222222)
            if i == 0:
                x = x + cond

            h.append(x)

            x = downsample(x)



            # batch_size, channels, height, width = x.size()
            # # x = self.ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            # kwargs = {}

            # ss2d = SS2D(d_model=channels, dropout=0, d_state=16, **kwargs).to('cuda')
            # x = ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            # # output = ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            #



        x = self.mid_block1(x, t)


        # batch_size, channels, height, width = x.size()
        # # x = self.ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # kwargs = {}

        # ss2d = SS2D(d_model=channels, dropout=0, d_state=16, **kwargs).to('cuda')
        # x = ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # # output = ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        #

        # block = TripletAttention().to('cuda')
        # x = block(x)



        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            a = h.pop()
            # print("a:", a.size())
            # print(x.size())
            x = torch.cat((x, a), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)




            # block = TripletAttention().to('cuda')
            # x = block(x)

            #
            # # print("xxxx11111", x.shape)
            # batch_size, channels, height, width = x.size()
            # # x = self.ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            # kwargs = {}

            # ss2d = SS2D(d_model=channels, dropout=0, d_state=16, **kwargs).to('cuda')
            # x = ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            # # output = ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            batch_size, channels, height, width = xd.size()

            in_channels = 64
            embed_channels = 128
            fc_hidden_channels = 256
            dw_kernel_size = 3
            # dw_kernel_size = (3, 5)

            model_2d = GFPM(in_channels=channels,
                            embed_channels=embed_channels,
                            fc_hidden_channels=fc_hidden_channels,
                            dw_kernel_size=dw_kernel_size)

            xd = model_2d(xd)


            batch_size, channels, height, width = xd.size()

            kwargs = {}

            ss2d = SS2D(d_model=channels, dropout=0, d_state=16, **kwargs)


            xd = ss2d(xd.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            target_h, target_w = x.shape[2], x.shape[3]


            xd_h, xd_w = xd.shape[2], xd.shape[3]


            if xd_h != target_h or xd_w != target_w:
                if xd_h < target_h or xd_w < target_w:

                    pad_h = max(0, target_h - xd_h)
                    pad_w = max(0, target_w - xd_w)

                    xd = F.pad(xd, (0, pad_w, 0, pad_h), mode='constant', value=1.0)
                else:

                    xd = xd[:, :, :target_h, :target_w]
            x[:, 0:1, :, :] = xd[:, 0:1, :, :] * lr_p + x[:, 0:1, :, :]

            x = x

            # output = ss2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


            # block = TripletAttention().to('cuda')
            # x = block(x)


        return self.final_conv(x)



    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(remove_weight_norm)
