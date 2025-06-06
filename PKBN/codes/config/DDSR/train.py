'下册'
import logging
import math
import os
import random
import sys
import time
from collections import OrderedDict
from datetime import datetime
from shutil import get_terminal_size

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader
def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + "_archived_" + get_timestamp()
        print("Path already exists. Rename it to [{:s}]".format(new_name))
        logger = logging.getLogger("base")
        logger.info("Path already exists. Rename it to [{:s}]".format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)



def setup_logger(
    logger_name, root, phase, level=logging.INFO, screen=False, tofile=False
):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + "_{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from data.util import imresize  # 绝对路径
from scipy.io import loadmat
from torch.autograd import Variable


def DUF_downsample(x, scale=4):


    assert scale in [2, 3, 4], "Scale [{}] is not supported".format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi

        inp = np.zeros((kernlen, kernlen))

        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    B, T, C, H, W = x.size()
    x = x.view(-1, 1, H, W)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0
    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)
    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], "reflect")

    gaussian_filter = (
        torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    )
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(B, T, C, x.size(2), x.size(3))
    return x


def PCA(data, k=2):
    X = torch.from_numpy(data)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(torch.t(X))
    return U[:, :k]  # PCA matrix


def random_batch_kernel(
        batch,
        l=24,
        sig_min=0.2,
        sig_max=4.0,
        rate_iso=1.0,
        tensor=True,
        random_disturb=False,
):
    if rate_iso == 1:

        sigma = np.random.uniform(sig_min, sig_max, (batch, 1, 1))
        ax = np.arange(-l // 2 + 1.0, l // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)
        xx = xx[None].repeat(batch, 0)
        yy = yy[None].repeat(batch, 0)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / np.sum(kernel, (1, 2), keepdims=True)
        return torch.FloatTensor(kernel) if tensor else kernel

    else:

        sigma_x = np.random.uniform(sig_min, sig_max, (batch, 1, 1))
        sigma_y = np.random.uniform(sig_min, sig_max, (batch, 1, 1))

        D = np.zeros((batch, 2, 2))
        D[:, 0, 0] = sigma_x.squeeze() ** 2
        D[:, 1, 1] = sigma_y.squeeze() ** 2

        radians = np.random.uniform(-np.pi, np.pi, (batch))
        mask_iso = np.random.uniform(0, 1, (batch)) < rate_iso
        radians[mask_iso] = 0
        sigma_y[mask_iso] = sigma_x[mask_iso]

        U = np.zeros((batch, 2, 2))
        U[:, 0, 0] = np.cos(radians)
        U[:, 0, 1] = -np.sin(radians)
        U[:, 1, 0] = np.sin(radians)
        U[:, 1, 1] = np.cos(radians)
        sigma = np.matmul(U, np.matmul(D, U.transpose(0, 2, 1)))
        ax = np.arange(-l // 2 + 1.0, l // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)
        xy = np.hstack((xx.reshape((l * l, 1)), yy.reshape(l * l, 1))).reshape(l, l, 2)
        xy = xy[None].repeat(batch, 0)
        inverse_sigma = np.linalg.inv(sigma)[:, None, None]
        kernel = np.exp(
            -0.5
            * np.matmul(
                np.matmul(xy[:, :, :, None], inverse_sigma), xy[:, :, :, :, None]
            )
        )
        kernel = kernel.reshape(batch, l, l)
        if random_disturb:
            kernel = kernel + np.random.uniform(0, 0.25, (batch, l, l)) * kernel
        kernel = kernel / np.sum(kernel, (1, 2), keepdims=True)

        return torch.FloatTensor(kernel) if tensor else kernel


def stable_batch_kernel(batch, l=24, sig=2.6, tensor=True):
    sigma = sig
    ax = np.arange(-l // 2 + 1.0, l // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    xx = xx[None].repeat(batch, 0)
    yy = yy[None].repeat(batch, 0)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / np.sum(kernel, (1, 2), keepdims=True)
    return torch.FloatTensor(kernel) if tensor else kernel


def b_Bicubic(variable, scale):
    B, C, H, W = variable.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_v = variable.view((B, C, H, W))
    re_tensor = imresize(tensor_v, 1 / scale)
    return re_tensor


def random_batch_noise(batch, high, rate_cln=1.0):
    noise_level = np.random.uniform(size=(batch, 1)) * high
    noise_mask = np.random.uniform(size=(batch, 1))
    noise_mask[noise_mask < rate_cln] = 0
    noise_mask[noise_mask >= rate_cln] = 1
    return noise_level * noise_mask


def b_GaussianNoising(tensor, sigma, mean=0.0, noise_size=None, min=0.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.mul(
        torch.FloatTensor(np.random.normal(loc=mean, scale=1.0, size=size)),
        sigma.view(sigma.size() + (1, 1)),
    ).to(tensor.device)
    return torch.clamp(noise + tensor, min=min, max=max)


def b_GaussianNoising(tensor, noise_high, mean=0.0, noise_size=None, min=0.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.FloatTensor(
        np.random.normal(loc=mean, scale=noise_high, size=size)
    ).to(tensor.device)
    return torch.clamp(noise + tensor, min=min, max=max)


class BatchSRKernel(object):
    def __init__(
            self,
            l=24,
            sig=2.6,
            sig_min=0.2,
            sig_max=4.0,
            rate_iso=1.0,
            random_disturb=False,
    ):
        self.l = l
        self.sig = sig
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.rate = rate_iso
        self.random_disturb = random_disturb

    def __call__(self, random, batch, tensor=False):
        if random == True:  # random kernel
            return random_batch_kernel(
                batch,
                l=self.l,
                sig_min=self.sig_min,
                sig_max=self.sig_max,
                rate_iso=self.rate,
                tensor=tensor,
                random_disturb=self.random_disturb,
            )
        else:  # stable kernel
            return stable_batch_kernel(batch, l=self.l, sig=self.sig, tensor=tensor)


class BatchBlurKernel(object):
    def __init__(self, kernels_path):
        kernels = loadmat(kernels_path)["kernels"]
        self.num_kernels = kernels.shape[0]
        self.kernels = kernels

    def __call__(self, random, batch, tensor=False):
        index = np.random.randint(0, self.num_kernels, batch)
        kernels = self.kernels[index]
        return torch.FloatTensor(kernels).contiguous() if tensor else kernels


class PCAEncoder(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight)
        self.size = self.weight.size()

    def forward(self, batch_kernel):
        B, H, W = batch_kernel.size()  # [B, l, l]
        return torch.bmm(
            batch_kernel.view((B, 1, H * W)), self.weight.expand((B,) + self.size)
        ).view((B, -1))


# class BatchBlur(object):
#     def __init__(self, l=15):
#         self.l = l
#         if l % 2 == 1:
#             self.pad =(l // 2, l // 2, l // 2, l // 2)
#         else:
#             self.pad = (l // 2, l // 2 - 1, l // 2, l // 2 - 1)
#         # self.pad = nn.ZeroPad2d(l // 2)

#     def __call__(self, input, kernel):
#         B, C, H, W = input.size()
#         pad = F.pad(input, self.pad, mode='reflect')
#         H_p, W_p = pad.size()[-2:]

#         if len(kernel.size()) == 2:
#             input_CBHW = pad.view((C * B, 1, H_p, W_p))
#             kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
#             return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
#         else:
#             input_CBHW = pad.view((1, C * B, H_p, W_p))
#             kernel_var = (
#                 kernel.contiguous()
#                 .view((B, 1, self.l, self.l))
#                 .repeat(1, C, 1, 1)
#                 .view((B * C, 1, self.l, self.l))
#             )
#             return F.conv2d(input_CBHW, kernel_var, groups=B * C).view((B, C, H, W))
class Gaussin_Kernel(object):
    def __init__(self, kernel_size=24, blur_type='iso_gaussian',
                 sig=2.6, sig_min=0.2, sig_max=4.0,
                 lambda_1=0.2, lambda_2=4.0, theta=0, lambda_min=0.2, lambda_max=4.0):
        self.kernel_size = kernel_size
        self.blur_type = blur_type

        self.sig = sig
        self.sig_min = sig_min
        self.sig_max = sig_max

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.theta = theta
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def __call__(self, batch, random):
        # random kernel
        if random == True:
            # print('随机了')
            return random_gaussian_kernel(batch, kernel_size=self.kernel_size, blur_type=self.blur_type,
                                          sig_min=self.sig_min, sig_max=self.sig_max,
                                          lambda_min=self.lambda_min, lambda_max=self.lambda_max)

        # stable kernel
        else:
            print('固定了')
            return stable_gaussian_kernel(kernel_size=self.kernel_size, blur_type=self.blur_type,
                                          sig=self.sig,
                                          lambda_1=self.lambda_1, lambda_2=self.lambda_2, theta=self.theta)


class BatchBlur(nn.Module):
    def __init__(self, kernel_size=24):
        super(BatchBlur, self).__init__()
        self.kernel_size = kernel_size
        if kernel_size % 2 == 1:
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        else:
            self.pad = nn.ReflectionPad2d(
                (kernel_size // 2, kernel_size // 2 - 1, kernel_size // 2, kernel_size // 2 - 1))

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        input_pad = self.pad(input)
        H_p, W_p = input_pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = input_pad.view((C * B, 1, H_p, W_p))
            kernel = kernel.contiguous().view((1, 1, self.kernel_size, self.kernel_size))

            return F.conv2d(input_CBHW, kernel, padding=0).view((B, C, H, W))
        else:
            input_CBHW = input_pad.view((1, C * B, H_p, W_p))
            kernel = kernel.contiguous().view((B, 1, self.kernel_size, self.kernel_size))
            kernel = kernel.repeat(1, C, 1, 1).view((B * C, 1, self.kernel_size, self.kernel_size))

            return F.conv2d(input_CBHW, kernel, groups=B * C).view((B, C, H, W))


class SRMDPreprocessing(object):
    def __init__(
            self,
            scale,
            kernel_size=24,
            blur_type='iso_gaussian',
            theta=0,
            lambda_min=0.2,
            lambda_max=4.0,
            noise=28.0,
            rate_iso=1.0, rate_cln=0.2, noise_high=0.08,
    ):
        pca_matrix = torch.load(
            "your pth", map_location=lambda storage, loc: storage
        )
        self.encoder = PCAEncoder(pca_matrix).cuda()

        self.gen_kernel = Gaussin_Kernel(
            kernel_size=kernel_size, blur_type=blur_type,
            theta=theta, lambda_min=lambda_min, lambda_max=lambda_max
        )
        self.blur = BatchBlur(kernel_size=kernel_size)

        self.noise = noise
        self.scale = scale
        self.rate_cln = 1
        self.noise_high = noise_high

    def __call__(self, hr_tensor, kernel=False):
        # hr_tensor is tensor, not cuda tensor
        B, C, H, W = hr_tensor.size()
        random = True
        # print(B)
        b_kernels = self.gen_kernel(B, random)  # B degradations

        hr_var = Variable(hr_tensor).cuda()
        device = hr_var.device
        B, C, H, W = hr_var.size()
        b_kernels = Variable(b_kernels).to(device)
        # print(B)

        # blur
        hr_blured_var = self.blur(hr_var, b_kernels)
        # BN, C, H, W
        # hr_var = Variable(hr_tensor).cuda() if self.cuda else Variable(hr_tensor)
        # device = hr_var.device
        # B, C, H, W = hr_var.size()

        # b_kernels = Variable(self.kernel_gen(self.random, B, tensor=True)).to(device)
        # hr_blured_var = self.blur(hr_var, b_kernels)

        # B x self.para_input
        # print(b_kernels.size())
        # kernel_code = self.encoder(b_kernels)
        # print(kernel_code.size())

        # Down sample
        if self.scale != 1:
            lr_blured_t = b_Bicubic(hr_blured_var, self.scale)
            lr_t = b_Bicubic(hr_var, self.scale)

        else:
            lr_blured_t = hr_blured_var

        # Noisy
        if self.noise > 0:
            self.noise = self.noise / 255.

            _, C, H_lr, W_lr = lr_blured_t.size()
            noise_level = torch.rand(B, 1, 1, 1).to(lr_blured_t.device) * self.noise
            noise = torch.randn_like(lr_blured_t).view(-1, C, H_lr, W_lr).mul_(noise_level).view(-1, C, H_lr, W_lr)
            # lr_blured_t = lr_blured_t
            lr_blured_t.add_(noise)
        # print(torch.max(lr_blured_t))
        # lr_blured_t = torch.clamp(lr_blured_t.round(), 0, 1)
        # lr_blured_t.view(B, C, H//int(self.scale), W//int(self.scale))

        # lr_re = Variable(lr_blured_t).to(device)
        # [32,24,24]
        b_kernels_4 = torch.unsqueeze(b_kernels, dim=1)
        # [32,1,24,24]
        # print(b_kernels.shape)
        # print(b_kernels_4.size())

        return (lr_blured_t, b_kernels_4, b_kernels_4, lr_blured_t, lr_t)


def cal_sigma(sig_x, sig_y, radians):
    sig_x = sig_x.view(-1, 1, 1)
    sig_y = sig_y.view(-1, 1, 1)
    radians = radians.view(-1, 1, 1)

    D = torch.cat([F.pad(sig_x ** 2, [0, 1, 0, 0]), F.pad(sig_y ** 2, [1, 0, 0, 0])], 1)
    U = torch.cat([torch.cat([radians.cos(), -radians.sin()], 2),
                   torch.cat([radians.sin(), radians.cos()], 2)], 1)
    sigma = torch.bmm(U, torch.bmm(D, U.transpose(1, 2)))

    return sigma


def anisotropic_gaussian_kernel(batch, kernel_size, covar):
    ax = torch.arange(kernel_size).float().cuda() - kernel_size // 2

    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    xy = torch.stack([xx, yy], -1).view(batch, -1, 2)

    inverse_sigma = torch.inverse(covar)
    kernel = torch.exp(- 0.5 * (torch.bmm(xy, inverse_sigma) * xy).sum(2)).view(batch, kernel_size, kernel_size)

    return kernel / kernel.sum([1, 2], keepdim=True)


def isotropic_gaussian_kernel(batch, kernel_size, sigma):
    ax = torch.arange(kernel_size).float().cuda() - kernel_size // 2
    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma.view(-1, 1, 1) ** 2))

    return kernel / kernel.sum([1, 2], keepdim=True)


def random_anisotropic_gaussian_kernel(batch=1, kernel_size=24, lambda_min=0.2, lambda_max=4.0):
    theta = torch.rand(batch).cuda() * math.pi
    lambda_1 = torch.rand(batch).cuda() * (lambda_max - lambda_min) + lambda_min
    lambda_2 = torch.rand(batch).cuda() * (lambda_max - lambda_min) + lambda_min

    covar = cal_sigma(lambda_1, lambda_2, theta)
    kernel = anisotropic_gaussian_kernel(batch, kernel_size, covar)
    return kernel


def stable_anisotropic_gaussian_kernel(kernel_size=24, theta=0, lambda_1=0.2, lambda_2=4.0):
    theta = torch.ones(1).cuda() * theta / 180 * math.pi
    lambda_1 = torch.ones(1).cuda() * lambda_1
    lambda_2 = torch.ones(1).cuda() * lambda_2

    covar = cal_sigma(lambda_1, lambda_2, theta)
    kernel = anisotropic_gaussian_kernel(1, kernel_size, covar)
    return kernel


def random_isotropic_gaussian_kernel(batch=1, kernel_size=24, sig_min=0.2, sig_max=4.0):
    x = torch.rand(batch).cuda() * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel(batch, kernel_size, x)
    return k


def stable_isotropic_gaussian_kernel(kernel_size=24, sig=4.0):
    x = torch.ones(1).cuda() * sig
    k = isotropic_gaussian_kernel(1, kernel_size, x)
    return k


def random_gaussian_kernel(batch, kernel_size=24, blur_type='iso_gaussian', sig_min=0.2, sig_max=4.0, lambda_min=0.2,
                           lambda_max=4.0):
    if blur_type == 'iso_gaussian':
        return random_isotropic_gaussian_kernel(batch=batch, kernel_size=kernel_size, sig_min=sig_min, sig_max=sig_max)
    elif blur_type == 'aniso_gaussian':
        return random_anisotropic_gaussian_kernel(batch=batch, kernel_size=kernel_size, lambda_min=lambda_min,
                                                  lambda_max=lambda_max)


def stable_gaussian_kernel(kernel_size=24, blur_type='iso_gaussian', sig=2.6, lambda_1=0.2, lambda_2=4.0, theta=0):
    if blur_type == 'iso_gaussian':
        return stable_isotropic_gaussian_kernel(kernel_size=kernel_size, sig=sig)
    elif blur_type == 'aniso_gaussian':
        return stable_anisotropic_gaussian_kernel(kernel_size=kernel_size, lambda_1=lambda_1, lambda_2=lambda_2,
                                                  theta=theta)


from torchvision.utils import make_grid
def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):

    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(
                n_dim
            )
        )
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

import argparse
import logging
import math
import os
import random
import sys
import copy

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from IPython import embed

import options as option
from models import create_model



sys.path.insert(0, "../../")
import utils as util
#from utils import


from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr
from data.util import imresize
import torch.nn as nn

def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="1")
    parser.add_argument("-opt_ker", type=str, help="2")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    opt_ker = option.parse(args.opt_ker, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    opt_ker = option.dict_to_nonedict(opt_ker)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    # seed = opt["train"]["manual_seed"]
    # if seed is None:
    #     seed = random.randint(1, 10000)

    # load PCA matrix of enough kernel

    pca_matrix = torch.load(
        opt["pca_matrix_path"], map_location=lambda storage, loc: storage
    )

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt_ker["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt_ker["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        util.set_random_seed(opt['train']['manual_seed'])

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            #os.system("rm ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work
        setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None


    model_ker = create_model(opt_ker)

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        # model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    prepro = SRMDPreprocessing(
        scale=opt["scale"]
    )


    kernel_size = opt["degradation"]["ksize"]
    padding = kernel_size // 2

    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_psnr = 0.0
    avg_loss = 0.0
    best_loss = 0.0
    best_iter = 0
    # if rank <= 0:
    # prev_state_dict = copy.deepcopy(model.netG.module.state_dict())
    cri_pix = nn.L1Loss()
    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1

            if current_step > total_iters:
                
                break
            LR_img, ker_map, kernels, lr_blured_t, lr_t = prepro(train_data["GT"])
            LR_img = (LR_img * 255).round() / 255

            LR_up = train_data["LQ"]
            # print("train", kernels.size())
            model_ker.feed_data(
                LR_img, GT_img=train_data["GT"], ker_map=kernels, lr_blured=lr_blured_t, lr=lr_t, lr_up=LR_up
            )
            
            model_ker.optimize_parameters(current_step)
            model_ker.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )
            
            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                # opt["train"]["val_freq"]
            # if 1==1:
                avg_psnr = 0.0
                idx = 0
                for _, val_data in enumerate(val_loader):

                    LR_img, ker_map, kernels, lr_blured_t, lr_t = prepro(val_data["GT"])
                    # print(val_data["GT"].shape)
                    LR_img = (LR_img * 255).round() / 255
                    LR_up = val_data["LQ"]
                    
                    
                    lr_img = tensor2img(LR_img)  # save LR image for reference

                    # valid Predictor
                    # print("train", kernels.size())
                    model_ker.feed_data(LR_img, GT_img=val_data["GT"], lr_up=LR_up, ker_map=kernels)
                    
                    model_ker.test()
                    visuals_ker_val = model_ker.get_current_visuals()
                    fake_ker_t = visuals_ker_val["ker"].detach()[0].squeeze().float().cpu()
                    real_ker_t = visuals_ker_val["ker_real"].detach()[0].squeeze().float().cpu()
                    fake_ker = visuals_ker_val["ker"].detach()[0].squeeze()
                    real_ker = visuals_ker_val["ker_real"].detach()[0].squeeze()
                    # print(visuals_ker_val["ker_real"].shape)

                    # Save images for reference
                    img_name = val_data["LQ_path"][0]
                    img_dir = os.path.join(opt["path"]["val_images"], img_name)
                    # img_dir = os.path.join(opt['path']['val_images'], str(current_step), '_', str(step))

                    mkdir(img_dir)
                    save_lr_path = os.path.join(img_dir, "{:s}_LR.png".format(img_name))


                    save_img_path = os.path.join(
                        img_dir, "{:s}_{:d}.png".format(img_name, current_step)
                    )

                    fake_ker_t = (
                        fake_ker_t
                        .numpy()
                        .reshape(
                            24, 24
                        )
                    )
                    real_ker_t = (
                        real_ker_t
                        .numpy()
                        .reshape(
                            24, 24
                        )
                    )
                    fake_ker_t = 1 / (np.max(fake_ker_t) + 1e-4) * 255 * fake_ker_t
                    real_ker_t = 1 / (np.max(real_ker_t) + 1e-4) * 255 * real_ker_t
                    cv2.imwrite(save_img_path, fake_ker_t)
                    cv2.imwrite(save_lr_path, real_ker_t)
                    # util.save_img(sr_img, save_img_path)

                    # # gtsave_img_path = os.path.join(
                    # #     img_dir, "{:s}_GT.png".format(img_name, current_step)
                    # # )
                    # # util.save_img(gt_img, gtsave_img_path)

                    # calculate PSNR
                    # crop_size = opt["scale"]
                    # gt_img = gt_img / 255.0
                    # sr_img = sr_img / 255.0
                    # cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size]
                    # cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size]

                    # cropped_sr_img_y = bgr2ycbcr(cropped_sr_img, only_y=True)
                    # cropped_gt_img_y = bgr2ycbcr(cropped_gt_img, only_y=True)

                    # print(val_data["GT"].shape, gt_img.shape, sr_img.shape)
                    # print(cropped_gt_img_y.shape, cropped_sr_img_y.shape)
                    # fake_ker.to(ker_map.device)
                    avg_loss += cri_pix(fake_ker, real_ker)

                    # avg_psnr += util.calculate_psnr(
                    #     cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    # )
                    idx += 1
                avg_loss = avg_loss / idx

                avg_psnr = avg_psnr / idx

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_iter = current_step
                if avg_loss > best_loss:
                    best_loss = avg_loss
                    best_iter = current_step
                print("avg_loss:", avg_loss)
                # log
                logger.info("# Validation # PSNR: {:.6f}, Best PSNR: {:.6f},loss: {:.6f}| Iter: {}".format(avg_psnr, best_psnr, avg_loss, best_iter))
                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}, loss: {:.6f}".format(
                        epoch, current_step, avg_psnr, avg_loss
                    )
                )
                # tensorboard logger
                if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    tb_logger.add_scalar("psnr", avg_psnr, current_step)
                    tb_logger.add_scalar("loss", avg_loss, current_step)

                # if avg_psnr > 20:
                #     # if rank <= 0:
                #     prev_state_dict = copy.deepcopy(model.netG.module.state_dict())
                #         # torch.save(prev_state_dict, opt["name"]+".pth")
                # else:
                #     logger.info("# Validation crashed, use previous state_dict...\n")
                #     model.netG.module.load_state_dict(copy.deepcopy(prev_state_dict), strict=True)
                    # model.netG.module.load_state_dict(torch.load(opt["name"]+".pth"), strict=True)
                    # model.load_network(opt["name"]+".pth", model.netG)
                    # break


            ### save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model_ker.save(current_step)
                    #xmz修改，删除了save_training_state

    if rank <= 0:
        logger.info("Saving the final model.")
        model_ker.save("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()


if __name__ == "__main__":
    main()
