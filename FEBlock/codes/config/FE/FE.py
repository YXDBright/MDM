import argparse
import logging
import math
import os
import random
import sys
import copy

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from IPython import embed

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util

from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr
from data.util import imresize


# 定义初始化分布式训练的函数
def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""  # 初始话分布式训练，不管他
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt) # 将配置文件的选项转换为可以处理缺失键的特殊字典

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    # seed = opt["train"]["manual_seed"]
    # if seed is None:
    #     seed = random.randint(1, 10000)

    # load PCA matrix of enough kernel  # 加载PCA矩阵
    print("load PCA matrix")
    pca_matrix = torch.load(
        opt["pca_matrix_path"], map_location=lambda storage, loc: storage
    )
    print("PCA matrix shape: {}".format(pca_matrix.shape))

    #### distributed training settings  # 分布式训练设置
    if args.launcher == "none":  # disabled distributed training   # 如果未启用分布式训练
        opt["dist"] = False  # 设置分布式训练选项为False
        opt["dist"] = False
        rank = -1  # 设置当前进程的排名为-1
        print("Disabled distributed training.")  # 打印禁用分布式训练的信息
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        util.set_random_seed(opt['train']['manual_seed'])

    # 设置PyTorch的CUDNN后端为性能优化模式
    torch.backends.cudnn.benchmark = True # 启用benchmark模式，根据计算负载自动选择最优的算法
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists  # 如果存在恢复状态的路径，则尝试从该状态恢复
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)  # 如果是普通训练（rank -1）或分布式训练中的主进程（rank 0）
        if resume_state is None: # 如果没有从先前的状态恢复
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]  # 创建实验根目录，如果目录已存在，则重命名旧目录
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            os.system("rm ./log")  # 删除当前日志目录（如果存在）
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work  # 配置日志记录器，在此之前日志不会生效
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],  # 基础日志配置，用于训练日志
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",  # 验证日志配置，具体设置代码未展示
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch  # 放大每个epoch的大小
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)   # 创建验证数据集
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)  # 创建验证数据加载器，不使用分布式采样器
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model = create_model(opt)  # load pretrained model of SFTMD  # 根据配置选项创建模型

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    prepro = util.SRMDPreprocessing(
        scale=opt["scale"]
    )
    kernel_size = opt["degradation"]["ksize"]
    padding = kernel_size // 2
    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    # 初始化记录最佳PSNR和其对应的迭代次数
    best_psnr = 0.0
    best_iter = 0
    best_ssim = 0.0   #  ssim
    best_iter_ssim = 0 # ssim
    #best_LPIPS = 0.0 # lpips
    # best_iter_LPIPS = 0 # lpips
    # if rank <= 0:
    # prev_state_dict = copy.deepcopy(model.netG.module.state_dict())
    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1

            if current_step > total_iters:
                
                break

            LR_img = train_data["LQ"] # 直接从训练数据中获取低分辨率图像


            # 模型训练中的数据喂入和参数优化
            model.feed_data(
                LR_img, GT_img=train_data["GT"]
            )
            model.optimize_parameters(current_step)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )
        #    visuals = model.get_current_visuals()

            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}> ".format(
                    epoch, current_step
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank == 0:
                    logger.info(message)

            # validation, to produce ker_map_list(fake) # 验证过程
            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:  # 每隔一定迭代次数进行一次验证，且只在主进程中执行
                #lpips_model = lpips.LPIPS(net='vgg')
                avg_psnr = 0.0 # 平均峰值信噪比初始化
                avg_ssim = 0.0 # ssim
                #avg_lpips = 0.0  # lpips
                idx = 0
                for _, val_data in enumerate(val_loader): # 遍历验证数据
                    # 验证数据处理逻辑将在后续代码中展示


                    LR_img = val_data["LQ"]
                    
                    
                    lr_img = util.tensor2img(LR_img)  # save LR image for reference
                    # valid Predictor  # 将低分辨率图像张量转换为图像格式，用于保存
                    model.feed_data(LR_img, GT_img=val_data["GT"]) # 喂入低分辨率和对应的高分辨率图像
                    model.test() # 进行模型的前向传播，不进行反向传播或参数更新
                    visuals = model.get_current_visuals()  # 获取模型生成的可视化结果

                    # Save images for reference  # 保存图像用于参考
                    img_name = val_data["LQ_path"][0]   # 获取图像名称
                    img_dir = os.path.join(opt["path"]["val_images"], img_name)  # 设置图像保存目录
                    # img_dir = os.path.join(opt['path']['val_images'], str(current_step), '_', str(step))
                    util.mkdir(img_dir)  # 创建目录
                    save_lr_path = os.path.join(img_dir, "{:s}_LR.png".format(img_name))  # 设置低分辨率图像的保存路径
                    util.save_img(lr_img, save_lr_path)  # 保存低分辨率图像

                    sr_img = util.tensor2img(visuals["SR"].squeeze())  # uint8  # 将超分辨率结果转换为图像格式
                    gt_img = util.tensor2img(visuals["GT"].squeeze())  # uint8  # 将真实的高分辨率图像转换为图像格式



                    save_img_path = os.path.join(
                        img_dir, "{:s}_{:d}.png".format(img_name, current_step)   # 设置保存路径，包含当前迭代步骤
                    )


                    util.save_img(sr_img, save_img_path)



                    crop_size = opt["scale"]
                    gt_img = gt_img / 255.0
                    sr_img = sr_img / 255.0

                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size]

                    cropped_sr_img_y = bgr2ycbcr(cropped_sr_img, only_y=True)
                    cropped_gt_img_y = bgr2ycbcr(cropped_gt_img, only_y=True)


                    avg_psnr += util.calculate_psnr(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255   #  这里************
                    )


                    idx += 1


                avg_psnr = avg_psnr / idx





                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_iter = current_step


                if avg_ssim > best_ssim:
                    best_ssim = avg_ssim
                #
                # if avg_lpips < best_LPIPS:
                #     best_LPIPS = avg_lpips





                logger.info("# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}|  Iter: {}  # Validation # SSIM: {:.6f}, Best SSIM: {:.6f} | # Validation ".format(avg_psnr, best_psnr, best_iter,avg_ssim,best_ssim))

                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f} ,ssim: {:.6f} ".format(
                        epoch, current_step, avg_psnr,avg_ssim # ssim lpips
                    )
                )
                # tensorboard logger
                if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    tb_logger.add_scalar("psnr", avg_psnr, current_step)
                    tb_logger.add_scalar("ssim current_", avg_ssim, current_step)


            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)


    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()


if __name__ == "__main__":
    main()
