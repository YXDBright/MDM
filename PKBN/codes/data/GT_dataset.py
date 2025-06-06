'''定义了一个用于处理高质量图像（GT）数据集的类 GTDataset'''
'''此文件主要包含一个用于读取高质量（GT）图像的数据集类，支持从LMDB数据库或图像文件中读取数据。
该类具备数据增强功能（翻转、旋转等），并可以处理不同数据格式（如LMDB和普通图像文件）'''
import os
import random
import sys

import cv2 # 导入OpenCV库，用于图像处理
import lmdb # 导入LMDB库，用于处理LMDB格式的数据
import numpy as np
import torch
import torch.utils.data as data # 导入PyTorch的数据处理模块

try:
    sys.path.append("..") # 尝试添加上级目录到系统路径，以便于导入其他模块
    import data.util as util # 尝试导入工具模块
except ImportError: # 如果导入失败
    pass


class GTDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.  是图像对哦
    The pair is ensured by 'sorted' function, so please check the name convention. 排序，要对应
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt # 配置选项
        self.GT_paths = None # 初始化GT图像路径
        self.GT_env = None  # environment for lmdb # 初始化LMDB环境
        self.GT_size = opt["GT_size"]  # GT图像的大小

        # read image list from lmdb or image files  # 从lmdb或图像文件读取图像列表
        if opt["data_type"] == "lmdb":
            self.GT_paths, self.GT_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )
        elif opt["data_type"] == "img":
            self.GT_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )  # GT list  # 从图像文件读取GT列表
        else:
            print("Error: data_type is not matched in Dataset")
        assert self.GT_paths, "Error: GT paths are empty."

        self.random_scale_list = [1] # 缩放列表，默认为1，不进行缩放

    def _init_lmdb(self): # 初始化LMDB环境
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(
            self.opt["dataroot_GT"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb":
            if self.GT_env is None:
                self._init_lmdb()

        GT_path = None
        scale = self.opt["scale"]
        GT_size = self.opt["GT_size"]

        # get GT image # 获取GT图像
        GT_path = self.GT_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.GT_sizes[index].split("_")]
        else:
            resolution = None
        img_GT = util.read_img(
            self.GT_env, GT_path, resolution
        )  # return: Numpy float32, HWC, BGR, [0,1] # 读取图像，格式为Numpy float32, HWC, BGR, [0,1]

        # modcrop in the validation / test phase # 如果是验证或测试阶段，则进行modcrop操作
        if self.opt["phase"] != "train":
            img_GT = util.modcrop(img_GT, scale)

        if self.opt["phase"] == "train":  # 随机裁剪图像
            H, W, C = img_GT.shape

            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))
            img_GT = img_GT[rnd_h : rnd_h + GT_size, rnd_w : rnd_w + GT_size, :]

            # augmentation - flip, rotate
            img_GT = util.augment( # 图像增强 - 翻转、旋转
                img_GT,
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
            )

        # change color space if necessary # 颜色空间转换，如果必要
        if self.opt["color"]:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt["color"], [img_GT])[
                0
            ]

        print(img_GT.shape)     # 输出图像信息
        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()

        return {"GT": img_GT, "GT_path": GT_path}

    def __len__(self):
        return len(self.GT_paths)
