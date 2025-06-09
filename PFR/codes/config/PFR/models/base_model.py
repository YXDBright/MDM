import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


class BaseModel:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if opt["gpu_ids"] is not None else "cpu")   # 试一下
        self.is_train = opt["is_train"]
        self.schedulers = [] # 初始化一个空的调度器列表，用于调整学习率
        self.optimizers = [] # 初始化一个空的优化器列表，用于优化模型参数
# 以上代码定义了BaseModel类，并在初始化方法中设置了一些基本属性，如配置选项、运行设备、训练状态标记、学习率调度器列表和优化器列表。

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass
# 以上方法在BaseModel中都是空方法，意味着子类需要根据具体需求来实现它们。

    def _set_lr(self, lr_groups_l):
        """set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer"""
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group["lr"] = lr
# _set_lr用于设置预热阶段的学习率，接收一个学习率列表作为输入，每个优化器对应一个学习率组。

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v["initial_lr"] for v in optimizer.param_groups])
        return init_lr_groups_l
# _get_init_lr方法用于获取由调度器设置的初始学习率。

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        #### set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)
# update_learning_rate方法根据当前迭代次数和预热设置更新学习率。

    def get_current_learning_rate(self):
        # return self.schedulers[0].get_lr()[0]
        return self.optimizers[0].param_groups[0]["lr"]

# 获取当前第一个优化器的学习率。

    def get_network_description(self, network):
        """Get the string and total parameters of the network"""
        if isinstance(network, nn.DataParallel) or isinstance(
            network, DistributedDataParallel
        ):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
# 获取网络的描述字符串和参数总数。

    def save_network(self, network, network_label, iter_label):
        save_filename = "{}_{}.pth".format(iter_label, network_label)
        save_path = os.path.join(self.opt["path"]["models"], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(
            network, DistributedDataParallel
        ):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
# 保存指定网络到文件。

    def load_network(self, load_path, network, strict=True):
        # 定义一个方法load_network，它接受三个参数：load_path（模型文件的路径），network（需要加载权重的网络模型），strict（严格模式，默认为True）。
        if isinstance(network, nn.DataParallel) or isinstance(
            ## 检查network是否是DataParallel或DistributedDataParallel类型（这两种类型常用于多GPU环境下的模型训练）。
            network, DistributedDataParallel  # 如果是，就获取这个包装过的模型的原始模型。在PyTorch中，使用DataParallel或DistributedDataParallel包装的模型会将原始模型存放在.module属性中。
        ):
            network = network.module
        load_net = torch.load(load_path)  # 使用torch.load函数从指定路径加载模型文件。这里假设模型文件包含了模型的权重字典。*************
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'  # 创建一个有序字典，用于存储清洗后的模型权重。'module.'前缀需要从键中移除，因为在单GPU配置下不需要这个前缀。
        for k, v in load_net.items():  # 遍历加载的模型权重字典的键值对。
            if k.startswith("module."):   # 如果键以'module.'开始，这通常发生在模型保存时使用了DataParallel或DistributedDataParallel。
                load_net_clean[k[7:]] = v  # 从键中移除前缀'module.'（'module.'长度为7），并将结果和对应的值添加到清洗后的字典中。
            else:
                load_net_clean[k] = v   # 如果键不以'module.'开始，直接将键值对添加到清洗后的字典中。
        network.load_state_dict(load_net_clean, strict=True)  # 使用清洗后的权重字典更新网络模型的状态。strict参数决定了是否所有的键都必须匹配。strict=True意味着所有键都必须完全匹配，否则会抛出错误。
# 从指定文件加载网络权重。

    def save_training_state(self, epoch, iter_step):
        """Saves training state during training, which will be used for resuming"""
        state = {"epoch": epoch, "iter": iter_step, "schedulers": [], "optimizers": []}
        for s in self.schedulers:
            state["schedulers"].append(s.state_dict())
        for o in self.optimizers:
            state["optimizers"].append(o.state_dict())
        save_filename = "{}.state".format(iter_step)
        save_path = os.path.join(self.opt["path"]["training_state"], save_filename)
        torch.save(state, save_path)
# 保存训练状态，用于后续恢复训练。

    def resume_training(self, resume_state):
        """Resume the optimizers and schedulers for training"""
        resume_optimizers = resume_state["optimizers"]
        resume_schedulers = resume_state["schedulers"]
        assert len(resume_optimizers) == len(
            self.optimizers
        ), "Wrong lengths of optimizers"
        assert len(resume_schedulers) == len(
            self.schedulers
        ), "Wrong lengths of schedulers"
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)
# 恢复训练时，重新加载优化器和调度器的状态。