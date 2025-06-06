"""create dataset and dataloader"""
'''主要用于创建数据集和数据加载器'''
import logging

import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt["phase"]  #  获取数据集阶段（训练或测试）
    if phase == "train": # 如果是训练阶段
        if opt["dist"]:  # 如果启用分布式训练
            world_size = torch.distributed.get_world_size() # 获取分布式训练世界大小
            num_workers = dataset_opt["n_workers"]   # 获取工作进程数量  在配置文件上有，写的是per GPU
            assert dataset_opt["batch_size"] % world_size == 0  # 确保batch大小能被世界大小整除
            batch_size = dataset_opt["batch_size"] // world_size # 计算每个节点的batch大小
            shuffle = False  # 分布式训练时不打乱数据
        else:  # 如果不使用分布式训练  看else就行了，没有用分布式训练
            num_workers = dataset_opt["n_workers"] * len(opt["gpu_ids"]) # 计算总的工作进程数量
            batch_size = dataset_opt["batch_size"] # 获取batch大小
            shuffle = True # 非分布式训练时打乱数据
        # 创建一个PyTorch数据加载器
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
            pin_memory=False,
        )
    else: # 如果是测试阶段
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
        )


def create_dataset(dataset_opt):  # 由配置选项中的关键字选择数据梳理代码
    mode = dataset_opt["mode"] # 获取数据集模式
    if mode == "LQ":  # Predictor
        from data.LQ_dataset import LQDataset as D

        dataset = D(dataset_opt)  # 如果模式为低质量图像预测
    elif mode == "LQGT":  # SFTMD
        from data.LQGT_dataset import LQGTDataset as D

        # dataset = D(dataset_opt)

        dataset = D(dataset_opt)
    elif mode == "GT":  # Corrector
        from data.GT_dataset import GTDataset as D

        dataset = D(dataset_opt)
    elif mode == "LQGTker":  # Corrector
        from data.LQGTker_dataset import LQGTKerDataset as D

        dataset = D(dataset_opt)
    # elif mode == 'LQGTseg_bg':
    #     from data.LQGT_seg_bg_dataset import LQGTSeg_BG_Dataset as D
    else:
        raise NotImplementedError("Dataset [{:s}] is not recognized.".format(mode))

    logger = logging.getLogger("base") # 获取日志记录器
    logger.info(
        "Dataset [{:s} - {:s}] is created.".format(
            dataset.__class__.__name__, dataset_opt["name"]
        )
    )  # 记录数据集创建的日志
    return dataset  # 返回创建的数据集
