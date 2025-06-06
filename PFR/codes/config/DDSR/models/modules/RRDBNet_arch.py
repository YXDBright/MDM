import functools

from .module_util import *


class ResidualDenseBlock_5C(nn.Module):
    # 看形式像是在“Dual”这篇文章中的稠密块，原因，在文章图示中有conv和LRelu层
    '''ResidualDenseBlock_5C 类定义了一个包含五个卷积层的残差密集块。
初始化时，设置了多个卷积层，每层的输入通道数逐渐增加，最后一个卷积层输出通道数恢复为输入通道数。
激活函数使用了LeakyReLU。'''
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1
        )

    '''在前向传播中，实现了密集连接和残差连接。每个后续的输入包括前面所有输出的串联，最后通过一个加权和来完成残差连接。'''
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block RRDB"""
    # 在残差稠密块的残差
    # 因为得好几个稠密块组成的RRDB嘛，所以用上一个class定义的单独的块，在这个class中连接起来，named 残差到残差嘛
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):  # 与diffsr_modules.py中RRDBNet类一样
    '''RRDBNet 类定义了一个完整的网络，包括初始卷积、多个RRDB模块组成的主体、几个上采样层和最终卷积输出层。'''
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4):
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

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        '''在前向传播中，图像通过初始层，然后进入多个RRDB模块，接着是多级上采样，最后通过终结卷积层输出最终的超分辨率图像'''
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(
            self.upconv1(F.interpolate(fea, scale_factor=2, mode="nearest"))  #  scale_factor=2????
        )
        fea = self.lrelu(
            self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest"))  # 注释掉了
        )
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
