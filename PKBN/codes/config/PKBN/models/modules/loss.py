import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from utils import BatchBlur, b_Bicubic, normkernel_to_downkernel, zeroize_negligible_val

'111'
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.util import imresize  # 绝对路径
from scipy.io import loadmat
from torch.autograd import Variable


def DUF_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code

    Args:
        x (Tensor, [B, T, C, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], "Scale [{}] is not supported".format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi

        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
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
            kernel_size=24,  # 24
            blur_type='iso_gaussian',
            theta=0,
            lambda_min=0.2,
            lambda_max=4.0,
            noise=0,  # 25.0
            rate_iso=1.0, rate_cln=0.2, noise_high=0.08,
    ):
        pca_matrix = torch.load(
            # "/home/zmdz/name/szy/FE-SR-master/pca_matrix/FE/pca_matrix.pth", map_location=lambda storage, loc: storage # 我改啦
            "/root/FE/FE-SR-master/pca_matrix/FE/pca_matrix.pth", map_location=lambda storage, loc: storage
        )
        self.encoder = PCAEncoder(pca_matrix).cuda()

        self.gen_kernel = Gaussin_Kernel(
            kernel_size=kernel_size, blur_type=blur_type,
            theta=theta, lambda_min=lambda_min, lambda_max=lambda_max
        )
        self.blur = BatchBlur(kernel_size=kernel_size)

        # self.blur = BatchBlur(l=ksize)
        # self.para_in = code_length
        # self.l = ksize
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
            print('加了噪声')
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


'111'

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == "gan" or self.gan_type == "ragan":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif self.gan_type == "wgan-gp":

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError(
                "GAN type [{:s}] is not found".format(self.gan_type)
            )

    def get_target_label(self, input, target_is_real):
        if self.gan_type == "wgan-gp":
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer("grad_outputs", torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(
            outputs=interp_crit,
            inputs=interp,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss

class CorrectionLoss(nn.Module):
    def __init__(self, scale=4.0, eps=1e-6):
        super(CorrectionLoss, self).__init__()
        self.scale = scale
        self.eps = eps
        self.cri_pix = nn.L1Loss()

    def forward(self, k_pred, lr_blured, lr):

        ks = []
        mask = torch.ones_like(k_pred).cuda()
        for c in range(lr_blured.shape[1]):
            k_correct = normkernel_to_downkernel(lr_blured[:, c:c+1, ...], lr[:, c:c+1, ...], k_pred.size(), self.eps)
            ks.append(k_correct.clone())
            mask *= k_correct
        ks = torch.cat(ks, dim=1)
        k_correct = torch.mean(ks, dim=1, keepdim=True) * (mask>0)
        k_correct = zeroize_negligible_val(k_correct, n=40)

        return self.cri_pix(k_pred, k_correct), k_correct


