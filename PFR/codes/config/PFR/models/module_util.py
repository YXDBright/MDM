from inspect import isfunction
from torch import nn
from torch.nn import init


def exists(x):
    # exists函数用于判断一个对象x是否存在（即不为None）。如果对象存在，则返回True；否则返回False。
    return x is not None


def default(val, d):
    # default函数接收两个参数val和d。如果val存在（不为None），则返回val；如果val不存在，且d是一个函数，
    # 则调用d并返回其结果，否则直接返回d。这个函数常用于为变量提供默认值。
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    # cycle函数用于无限循环迭代一个数据列表dl。它使用yield关键字，这意味着cycle是一个生成器函数，
    # 每次调用都会返回列表中的下一个元素，到达列表末尾后会从头开始。
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    '''num_to_groups函数将一个数num分成大小大约相等的几组，每组的大小不超过divisor。
    首先计算num能被divisor整除的次数（groups），以及剩余部分（remainder）。然后创建一个列表arr，
    包含groups个divisor，如果有剩余（remainder>0），将剩余部分也加入到列表中。'''
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def initialize_weights(net_l, scale=0.1):
    '''initialize_weights函数用于初始化神经网络权重。它接受一个网络（或网络列表）net_l和一个缩放因子scale。
    该函数遍历网络中的每一个模块（m），并根据模块的类型应用不同的初始化方法。例如，对于Conv2d（二维卷积层）和Linear（全连接层），
    使用kaiming_normal_方法进行初始化，并将权重乘以scale因子；对于BatchNorm2d（批归一化层），将权重初始化为1，偏置初始化为0。'''
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers, seq=False):
    '''make_layer函数用于创建一系列网络层。block是网络层的构造函数，n_layers是层的数量，
    seq指示是否将这些层组织成Sequential模型（即连续的模型，层与层之间直接连接）。
    如果seq为True，则返回一个Sequential模型；否则返回一个ModuleList（模块列表，可以包含多个层，但不像Sequential那样自动连接层与层）。'''
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    if seq:
        return nn.Sequential(*layers)
    else:
        return nn.ModuleList(layers)
