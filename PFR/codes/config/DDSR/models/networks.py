import logging

import torch

# 导入的模型 是把modules的文件都导入进去了。
from models import modules as M # M模块

logger = logging.getLogger("base")

# Generator
# 定义生成器网络
def define_G(opt):
    # 从配置中获取网络配置选项
    opt_net = opt["network_G"]
    which_model = opt_net["which_model_G"]  # 下面是少了这一行？
    setting = opt_net["setting"]

    # 使用反射机制根据which_model的值动态创建网络实例
    # netG = getattr(M, which_model)(**setting)这条语句的作用是：根据配置中指定的which_model找到对应的网络模型类，
    # 并使用setting中提供的配置参数来实例化这个类，创建网络模型的一个实例。
    netG = getattr(M, which_model)(**setting)  # 这个M是不是__init__.py中的模型选择的M，好像不是
    return netG
'''getattr(M, which_model)根据which_model的值动态获取M模块中对应的类。例如，如果which_model是"RRDBNet"，这条语句等价于M.RRDBNet，即获取M模块中的RRDBNet类。
(**setting)是Python中的参数解包语法，它将字典setting中的项作为关键字参数传递给函数或构造函数。这意味着setting字典中的键值对将被用作创建模型实例时的参数'''

# Discriminator
# # 定义判别器网络
def define_D(opt):
    opt_net = opt["network_D"]
    setting = opt_net["setting"]

    netD = getattr(M, which_model)(**setting)
    return netD


# Perceptual loss
# 定义特征提取网络
def define_F(opt, use_bn=False):
    gpu_ids = opt["gpu_ids"]
    device = torch.device("cuda" if gpu_ids else "cpu")
    # PyTorch pretrained VGG19-54, before ReLU.
    # 根据是否使用批归一化选择不同的特征层
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
        # 创建VGG特征提取器实例
    netF = M.VGGFeatureExtractor(
        feature_layer=feature_layer, use_bn=use_bn, use_input_norm=True, device=device
    )
    netF.eval()  # No need to train # 设置为评估模式
    return netF
