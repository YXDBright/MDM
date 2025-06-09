'''
Residual残差连接、SinusoidalPosEmb生成位置嵌入的模块、Mish、 Rezero改善训练状态、Block（实现了一个基本的卷积块，包含卷积、归一化和激活函数。
可以根据条件选择不同的卷积（动态或标准）和是否应用分组归一化。）、ResnetBlock残差网络块，包含残差连接、Upsample、Downsample、MultiheadAttention
、ResidualDenseBlock_5C、RRDB（三个ResidualDenseBlock_5C串联）
'''
import math
import torch
import torch.nn.functional as F
from einops import rearrange # 用于张量的重排
from torch import nn
from torch.nn import Parameter
from models.dynamic_conv import Dynamic_conv2d # 自定义的动态卷积层  自己写的动态卷积啊

class Residual(nn.Module):
    # 用于实现残差连接，即将输入直接加到函数输出上，有助于解决深度神经网络训练中的梯度消失问题。
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    # 生成位置嵌入的模块，利用正弦和余弦函数来编码位置信息。
    def __init__(self, dim):
        super().__init__()
        self.dim = dim # 位置编码的维度

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # 使用正弦和余弦函数创建位置嵌入
        return emb


class Mish(nn.Module):
    # Mish是一种平滑的激活函数，有助于提高深度学习模型的性能。
    def forward(self, x):
        return x * torch.tanh(F.softplus(x)) # Mish公式


class Rezero(nn.Module):
    # Rezero是一种技术，通过引入一个初始为0的可学习参数来改善模型的训练动态。
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


# building block modules

class Block(nn.Module):
    # 这个模块实现了一个基本的卷积块，包含卷积、归一化和激活函数。可以根据条件选择不同的卷积（动态或标准）和是否应用分组归一化。
    def __init__(self, dim, dim_out, groups=8, flag = "sr"):
        super().__init__()
        if groups == 0:
            if flag =="sr":
                self.block = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    Dynamic_conv2d(dim, dim_out, kernel_size=3, bias=False),#nn.Conv2d(dim, dim_out, 3),
                    Mish()
                )
            else:
                self.block = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(dim, dim_out, 3),  # 使用标准卷积
                    Mish()
                )
        else:
            if flag =="sr":
                self.block = nn.Sequential(
                    nn.ReflectionPad2d(1),  # 反射填充
                    Dynamic_conv2d(dim, dim_out, kernel_size=3, bias=False),# nn.Conv2d(dim, dim_out, 3), # 动态卷积
                    nn.GroupNorm(groups, dim_out),
                    Mish()  # 使用Mish激活函数
                )
            else:
                self.block = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(dim, dim_out, 3),
                    nn.GroupNorm(groups, dim_out),
                    Mish()
                )

    def forward(self, x):
        return self.block(x) # 应用卷积块到输入数据


class ResnetBlock(nn.Module):
    # 这个模块构造了一个残差网络块，其中包括对输入的处理、应用两个基本块，并添加残差连接。
    def __init__(self, dim, dim_out, *, time_emb_dim=0, groups=8, flag = "sr"):
        super().__init__()
        if time_emb_dim > 0:
            self.mlp = nn.Sequential(
                Mish(),
                nn.Linear(time_emb_dim, dim_out)
            )

        self.block1 = Block(dim, dim_out, groups, flag)
        self.block2 = Block(dim_out, dim_out, groups, flag)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cond=None):
        h = self.block1(x)
        if time_emb is not None:
            # print("3333",h.shape)
            h += self.mlp(time_emb)[:, :, None, None]
            # print(self.mlp(time_emb)[:, :, None, None].shape)
        if cond is not None:
            # print("1111",h.shape)
            h += cond
            # print("2222",h.shape)
        h = self.block2(h)
        return h + self.res_conv(x)


class Upsample(nn.Module):
    # 这两个模块分别实现了上采样（增加空间尺寸）和下采样（减小空间尺寸）的操作。
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
        )

    # 使用转置卷积层（有时也称为反卷积）进行上采样。
    # 参数解释：
    # dim: 输入和输出通道数（保持不变）
    # 4: 卷积核大小
    # 2: 步长，决定了特征图尺寸的放大倍数（这里是因子2）
    # 1: 填充，保证输出尺寸正确增加

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 2),
        )
    # 使用标准卷积进行下采样。
    # 参数解释：
    # nn.ReflectionPad2d(1): 使用反射填充来避免边缘效应，填充宽度为1
    # nn.Conv2d(dim, dim, 3, 2): 标准卷积层
    # dim: 输入和输出通道数 保持不变
    # 3: 卷积核大小
    # 2: 步长，这里的步长为2，意味着输出特征图的宽和高都是输入的一半

    def forward(self, x):
        return self.conv(x)


class LinearAttention(nn.Module):
    # 这个模块提供了一个实现线性复杂度注意力机制的方式，有助于在保持性能的同时减少计算成本。
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class MultiheadAttention(nn.Module):
    # 这是一个多头注意力机制的实现，它允许模型在不同的表示子空间上并行学习。
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        if self.qkv_same_dim:
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False
        self.last_attn_probs = None

    def reset_parameters(self):
        # 这两个模块分别实现了残差密集块和残差中的残差密集块，用于增强特征提取能力，并有助于处理更复杂的模式。
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
            self,
            query, key, value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            before_softmax=False,
            need_head_weights=False,
    ):
        """Input shape: [B, T, C]

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v,
            self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias,
            self.training, key_padding_mask, need_weights, attn_mask)
        attn_output = attn_output.transpose(0, 1)
        return attn_output, attn_output_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


'''下面这两个class与models-modules下面的RRDBNet_arch.py定义的class一样'''
class ResidualDenseBlock_5C(nn.Module): # 实现一个残差密集块，用于图像超分辨率等任务
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels  # 每层的构造：卷积 + 激活
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):  # 前向传播逻辑，通过多次卷积和连接增加网络深度和复杂度
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x  # 将输入与输出相加形成残差连接


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''   # RRDB

    # 实现残差在残差密集块，进一步增强图像处理的效果。
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        # 三个ResidualDenseBlock_5C模块串联
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        # 前向传播逻辑，通过串联三个密集块增加特征的非线性和复杂度
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x # 通过残差连接合并特征
