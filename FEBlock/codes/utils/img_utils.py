import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid

try:
    import accimage
except ImportError:
    accimage = None


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_pil_image(pic, mode=None):

    if not (_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError("pic should be Tensor or ndarray. Got {}.".format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError(
            "Input pic must be a torch.Tensor or NumPy ndarray, "
            + "not {}".format(type(npimg))
        )

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = "L"
        if npimg.dtype == np.int16:
            expected_mode = "I;16"
        if npimg.dtype == np.int32:
            expected_mode = "I"
        elif npimg.dtype == np.float32:
            expected_mode = "F"
        if mode is not None and mode != expected_mode:
            raise ValueError(
                "Incorrect mode ({}) supplied for input type {}. Should be {}".format(
                    mode, np.dtype, expected_mode
                )
            )
        mode = expected_mode

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ["RGBA", "CMYK"]
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError(
                "Only modes {} are supported for 4D inputs".format(
                    permitted_4_channel_modes
                )
            )

        if mode is None and npimg.dtype == np.uint8:
            mode = "RGBA"
    else:
        permitted_3_channel_modes = ["RGB", "YCbCr", "HSV"]
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError(
                "Only modes {} are supported for 3D inputs".format(
                    permitted_3_channel_modes
                )
            )
        if mode is None and npimg.dtype == np.uint8:
            mode = "RGB"

    if mode is None:
        raise TypeError("Input type {} is not supported".format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)


def to_tensor(pic):
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError("pic should be PIL Image or ndarray. Got {}".format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.float().div(255)

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == "I":
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == "I;16":
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == "YCbCr":
        nchannel = 3
    elif pic.mode == "I;16":
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
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


def save_img(img, img_path, mode="RGB"):
    cv2.imwrite(img_path, img)


def img2tensor(img):
    """
    # BGR to RGB, HWC to CHW, numpy to tensor
    Input: img(H, W, C), [0,255], np.uint8 (default)
    Output: 3D(C,H,W), RGB order, float tensor
    """
    img = img.astype(np.float32) / 255.0
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    return img


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


'''24.09,10输入尺寸不一致'''

# def crop_center(img, cropx, cropy):
#     y, x = img.shape[:2]
#     startx = x // 2 - cropx // 2
#     starty = y // 2 - cropy // 2
#     return img[starty:starty + cropy, startx:startx + cropx]
#
#
# def calculate_psnr(img1, img2):
#     # img1 and img2 have range [0, 255]
#     if img1.shape != img2.shape:
#         min_height = min(img1.shape[0], img2.shape[0])
#         min_width = min(img1.shape[1], img2.shape[1])
#         img1 = crop_center(img1, min_width, min_height)
#         img2 = crop_center(img2, min_width, min_height)
#
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float("inf")
#     return 20 * math.log10(255.0 / math.sqrt(mse))

'''24.09,10输入尺寸不一致'''

'''24.09,10输入尺寸不一致'''
# import cv2
#
#
# def resize_to_same_size(img1, img2):
#     # 将较大的图像缩放到与较小图像相同的尺寸
#     height = min(img1.shape[0], img2.shape[0])
#     width = min(img1.shape[1], img2.shape[1])
#     img1_resized = cv2.resize(img1, (width, height))
#     img2_resized = cv2.resize(img2, (width, height))
#     return img1_resized, img2_resized
#
#
# def calculate_psnr(img1, img2):
#     if img1.shape != img2.shape:
#         img1, img2 = resize_to_same_size(img1, img2)
#
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float("inf")
#     return 20 * math.log10(255.0 / math.sqrt(mse))

'''24.09,10输入尺寸不一致'''

# **************************
#import numpy as np

# 转换为3通道后不再使用，而是使用之前的
# def bgr2ycbcr(img, only_y=True):
#     """
#     Convert BGR image to YCbCr color space.
#     If only_y is True, return only the Y channel.
#     Input:
#         img: Input image in BGR format.
#         only_y: Boolean, if True only return Y channel.
#     Output:
#         Converted image in YCbCr color space (or Y channel).
#     """
#     in_img_type = img.dtype
#     img = img.astype(np.float32)
#
#     if in_img_type != np.uint8:
#         img *= 255.0
#
#     if len(img.shape) == 3 and img.shape[2] == 3:
#         if only_y:
#             rlt = np.dot(img, [0.114, 0.587, 0.299])
#         else:
#             rlt = np.matmul(img, [[0.114, 0.587, 0.299],
#                                   [-0.10001, -0.51599, 0.615],
#                                   [0.615, -0.51599, -0.10001]])
#             rlt[:, :, [1, 2]] += 128.0
#     else:
#         rlt = img
#
#     if in_img_type == np.uint8:
#         rlt = np.round(rlt).astype(np.uint8)
#     else:
#         rlt /= 255.0
#
#     return rlt
#
# def calculate_psnr(gt_img, sr_img, scale):
#     """
#     Calculate PSNR (Peak Signal-to-Noise Ratio) between ground truth image and super-resolved image.
#     Input:
#         gt_img: Ground truth high-resolution image.
#         sr_img: Super-resolved image.
#         scale: The scale factor of super-resolution.
#     Output:
#         PSNR value.
#     """
#     crop_size = scale
#     gt_img = gt_img / 255.0
#     sr_img = sr_img / 255.0
#
#     cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size]
#     cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size]
#
#     cropped_sr_img_y = bgr2ycbcr(cropped_sr_img, only_y=True)
#     cropped_gt_img_y = bgr2ycbcr(cropped_gt_img, only_y=True)
#
#     mse = np.mean((cropped_sr_img_y - cropped_gt_img_y) ** 2)
#     if mse == 0:
#         return float('inf')
#     psnr = 20 * np.log10(1.0 / np.sqrt(mse))
#     return psnr

# 示例使用


# *******************

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")
