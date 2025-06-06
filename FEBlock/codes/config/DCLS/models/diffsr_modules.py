import functools
import torch
from torch import nn
import torch.nn.functional as F
# from hparams import hparams
from models.module_util import make_layer, initialize_weights
from models.commons import Mish, SinusoidalPosEmb, RRDB, Residual, Rezero, LinearAttention
from models.commons import ResnetBlock, Upsample, Block, Downsample
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys

sys.path.append('../')
from data import prepare_dataset


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in ['.pth'])


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_mat_file(fname):
                path = os.path.join(root, fname)
                item = (path, None)
                images.append(item)

    return images


class KernelFolder(data.Dataset):
    """A generic kernel loader"""

    def __init__(self, root, train, kernel_size=11, scale_factor=2, transform=None, target_transform=None,
                 loader=None):
        ''' prepare training and validation sets'''
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.alpha = 1e-6

        # To normalize the pixels to [0,1], we first clamp the kernel because some values are slightly below zero. Then,
        # we rescale the maximum pixel to be near one by dividing (max_value+0.01), where 0.01 can make sure it won't be
        # larger than 1. This is crucial to remove notable noises in sampling.
        self.normalization = round(prepare_dataset.gen_kernel_fixed(np.array([self.kernel_size, self.kernel_size]),
                                                                    np.array([self.scale_factor, self.scale_factor]),
                                                                    0.175 * self.scale_factor,
                                                                    0.175 * self.scale_factor, 0,
                                                                    0).max(), 5) + 0.01
        root += '_x{}'.format(self.scale_factor)
        if not train:
            if not os.path.exists(root):
                print('generating validation set at {}'.format(root))
                os.makedirs(root, exist_ok=True)

                i = 0
                for sigma1 in np.arange(0.175 * self.scale_factor, min(2.5 * self.scale_factor, 10) + 0.3, 0.3):
                    for sigma2 in np.arange(0.175 * self.scale_factor, min(2.5 * self.scale_factor, 10) + 0.3, 0.3):
                        for theta in np.arange(0, np.pi, 0.2):
                            kernel = prepare_dataset.gen_kernel_fixed(np.array([self.kernel_size, self.kernel_size]),
                                                                      np.array([self.scale_factor, self.scale_factor]),
                                                                      sigma1, sigma2, theta, 0)

                            torch.save(torch.from_numpy(kernel), os.path.join(root, str(i) + '.pth'))
                            i += 1
            else:
                print('Kernel_val_path: {} founded.'.format(root))

            kernels = make_dataset(root, None)

            if len(kernels) == 0:
                raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"))

            self.kernels = kernels

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        if self.train:
            kernel = prepare_dataset.gen_kernel_random(np.array([self.kernel_size, self.kernel_size]),
                                                       np.array([self.scale_factor, self.scale_factor]),
                                                       0.175 * self.scale_factor, min(2.5 * self.scale_factor, 10), 0)
            kernel = torch.from_numpy(kernel)
        else:
            path, target = self.kernels[index]
            kernel = torch.load(path)

        # Normalization
        kernel = torch.clamp(kernel, min=0) / self.normalization

        # Adds noise to pixels to dequantize them, ref MAF. This is crucail to add small numbers to zeros of the kernel.
        # No noise will lead to negative NLL, 720 is an empirical value.
        kernel = kernel + np.random.rand(*kernel.shape) / 720.0

        # Transforms pixel values with logit to be unconstrained by np.log(x / (1.0 - x)), [-13.8,13.8], ref MAF
        kernel = logit(self.alpha + (1 - 2 * self.alpha) * kernel)

        kernel = kernel.to(torch.float32)

        return kernel, torch.zeros(1)

    def __len__(self):
        if self.train:
            return int(5e4)
        else:
            return len(self.kernels)


def logit(x):

    return np.log(x / (1.0 - x))
class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
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
            self.upconv1(F.interpolate(fea, scale_factor=2)))
        fea_hr = self.HRconv(fea)
        out = self.conv_last(self.lrelu(fea_hr))
        out = out.clamp(0, 1)
        out = out * 2 - 1
        if get_fea:
            return out, feas
        else:
            return out


class Unet(nn.Module):
    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), cond_dim=32):
        super().__init__()
        dims = [3, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        groups = 0
        rrdb_num_block = 8
        sr = 8


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
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim, groups=groups),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim, groups=groups),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)

        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim, groups=groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Block(dim, dim, groups=groups),
            nn.Conv2d(dim, out_dim, 1)
        )



    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                # print(f"| Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def forward(self, x, time, cond, img_lr_up):
        t = self.time_pos_emb(time)
        # print(x.device)
        t = self.mlp(t).to(x.device)

        h = []

        cond = self.cond_proj(torch.cat((cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond,cond), dim=1))
        # print(cond.shape)
        for i, (resnet, resnet2, downsample) in enumerate(self.downs):
            x = resnet(x, t)
            x = resnet2(x, t)
            if i == 0:
                x = x + cond

            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)

        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:


            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        return self.final_conv(x)

    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(remove_weight_norm)
