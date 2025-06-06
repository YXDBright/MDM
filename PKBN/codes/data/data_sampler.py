
import math

import torch
import torch.distributed as dist # 导入PyTorch的分布式计算模块
from torch.utils.data.sampler import Sampler # 导入PyTorch的数据采样接口


class DistIterSampler(Sampler):


    def __init__(self, dataset, num_replicas=None, rank=None, ratio=100):
        if num_replicas is None: # 如果未指定副本数
            if not dist.is_available(): # 检查分布式模块是否可用
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size() # 获取全局副本数量
        if rank is None: # 如果未指定进程排名
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()  # 获取当前进程的排名
        self.dataset = dataset # 设置数据集
        self.num_replicas = num_replicas # 设置副本数量
        self.rank = rank # 设置进程排名
        self.epoch = 0 # 初始化训练轮次
        self.num_samples = int(math.ceil(len(self.dataset) * ratio / self.num_replicas)) # 计算每个副本的样本数
        self.total_size = self.num_samples * self.num_replicas # 计算总样本数

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator() # 创建一个随机数生成器
        g.manual_seed(self.epoch) # 设置种子以保证每轮训练数据的一致性
        indices = torch.randperm(
            self.total_size, generator=g
        ).tolist()  # Returns a random permutation of integers from 0 to n - 1 # 获取总样本数的一个随机排列

        dsize = len(self.dataset) # 获取数据集大小
        indices = [v % dsize for v in indices] # 通过取模保证索引在数据集大小内

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas] # 为每个副本分配不同的样本
        assert len(indices) == self.num_samples  # 确保每个副本分得的样本数正确

        return iter(indices) # 返回迭代器

    def __len__(self):
        return self.num_samples  # 返回每个副本的样本数

    def set_epoch(self, epoch):
        self.epoch = epoch # 设置当前训练轮次，用于控制数据的随机性
