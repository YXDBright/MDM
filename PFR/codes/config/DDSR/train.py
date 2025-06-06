
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
import torch.nn as nn

from utils.img_utils import ssim,calculate_ssim  # ssim
import lpips  # LPIPS

def init_dist(backend="nccl", **kwargs): # 定义初始化分布式训练的函数。
    """ initialization for distributed training"""
    # 分布式训练，获取GPU数量并进行设置
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn" # 判断当前进程启动方式是否为'spawn'
    ):  # Return the name of start method used for starting processes   返回用于启动进程的启动方法的名称
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows  若不是，则强制设置为'spawn'，这是Windows系统的默认值。
    rank = int(os.environ["RANK"])  # system env process ranks 获取当前进程的等级
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available  获取可用GPU的数量
    torch.cuda.set_device(rank % num_gpus) # 设置当前进程使用的GPU。
    dist.init_process_group(   # 初始化分布式进程组。
        backend=backend, **kwargs
    )  # Initializes the default distributed process group


def main():
    #### setup options of three networks  三个网络的设置选项
    parser = argparse.ArgumentParser() # 建了一个参数解析器对象parser，解析命令行参数
    # rgparse.ArgumentParser()是用来创建一个空的参数解析器对象，接下来可以使用它的add_argument()方法添加命令行参数选项，然后使用parse_args()方法解析命令行参数。
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument("-opt_ker", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"  

    )
    parser.add_argument("--local_rank", type=int, default=0)  # 本地排名
    args = parser.parse_args()  # 解析命令行参数，并将解析后的结果存储在args对象中

    # 使用option.parse函数来解析命令行参数的输入，并将解析的结果保存在opt和opt_ker这两个变量中。
    # 例如，如果在命令行中输入了--opt和--opt_ker这两个参数，那么option.parse将解析这两个参数的值，并将它们保存在opt和opt_ker中。
    opt = option.parse(args.opt, is_train=True) # is_train=True参数表示这些参数是用于训练模型的。
    opt_ker = option.parse(args.opt_ker, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    # 将opt和opt_ker这两个变量转换为NoneDict格式的数据结构
    # NoneDict是option模块中定义的一种特殊的字典格式
    opt = option.dict_to_nonedict(opt)
    opt_ker = option.dict_to_nonedict(opt_ker)



    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training  # 如果命令行参数指定不使用分布式训练。
        opt["dist"] = False # 设置配置字典，指明不使用分布式训练。
        opt_ker["dist"] = False
        rank = -1 # 设置rank=-1，意味着不参与分布式训练。
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt_ker["dist"] = True
        init_dist()  # 上边定义的函数：调用初始化分布式训练的函数。
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        util.set_random_seed(opt['train']['manual_seed'])

    torch.backends.cudnn.benchmark = True  # 启用cudnn的benchmark模式，可以加速固定大小输入的训练。

    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"], # 用于从上一步获取的"path"子字典中获取名为"resume_state"的键对应的值
            map_location=lambda storage, loc: storage.cuda(device_id),  # 确保加载到正确的设备上。
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options # 检查并调整继续训练的配置选项。
    else:
        resume_state = None   # 如果未指定，则继续训练的状态为空。

    #### mkdir and loggers # 创建目录和日志记录器。
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)   # 仅在非分布式训练或分布式训练的主进程中执行。 好像就是在非分布式训练中
        if resume_state is None:

            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )
            util.mkdirs(  # 创建需要的所有目录。
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            os.system("rm ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")  # 创建新的日志链接。


        util.setup_logger(  #
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")  # 获取基础日志记录器。
        logger.info(option.dict2str(opt)) # 输出当前的配置选项。
        # tensorboard logger # Tensorboard日志记录器设置
        if opt["use_tb_logger"] and "debug" not in opt["name"]:  # 判断是否使用Tensorboard，并排除调试模式。
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter   # 根据PyTorch版本，可能需要导入不同的库。
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))   # 初始化Tensorboard日志记录器。
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader # 创建训练和验证数据加载器。
    dataset_ratio = 200  # enlarge the size of each epoch 放大每个epoch的大小。
    for phase, dataset_opt in opt["datasets"].items(): # 遍历配置中的数据集设置。
        if phase == "train":  # 如果是训练阶段。
            train_set = create_dataset(dataset_opt)  # 创建训练数据集。
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))  # 计算训练批次大小。
            total_iters = int(opt["train"]["niter"]) # 获取总迭代次数。
            total_epochs = int(math.ceil(total_iters / train_size)) # 计算总训练周期。
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None # 非分布式训练不使用采样器。
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)  # 创建训练数据加载器。
            if rank <= 0: # 在主进程中输出训练集信息。
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size # 输出训练集大小和迭代次数。
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters # 输出总的epoch（总共需要的周期数）和总的迭代数目
                    )
                )
        elif phase == "val": # 验证阶段的处理。
            val_set = create_dataset(dataset_opt) # 根据配置创建验证数据集。
            val_loader = create_dataloader(val_set, dataset_opt, opt, None) # 创建验证数据加载器。
            if rank <= 0: # 主进程中执行。
                # 输出验证数据集的大小信息。
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else: # 如果遇到未识别的数据集阶段，抛出异常。
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None  # 确保训练数据加载器已正确创建。
    assert val_loader is not None # 确保验证数据加载器已正确创建。

    #### create model 模型实例化
    model = create_model(opt)   # 创建了模型

    model_ker = create_model(opt_ker)

    #### resume training
    if resume_state: # 如果存在继续训练的状态。
        # 输出从哪个周期、哪次迭代开始继续训练的信息。
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
    # 对输入图像进行预处理，例如缩放和随机化处理。
    prepro = util.SRMDPreprocessing(
        scale=opt["scale"],
        random = True
    )
    # prepro_val = util.SRMDPreprocessing_val(
    #     scale=opt["scale"]
    # )

    #### training
    logger.info( # 日志记录：从特定周期和迭代开始训练。
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )
    # 初始化最佳PSNR、平均损失、最佳损失和最佳迭代次数变量。
    best_psnr = 0.0
    avg_loss = 0.0
    best_loss = 0.0
    best_iter = 0
    best_ssim = 0.0   #  ssim

    # prev_state_dict = copy.deepcopy(model.netG.module.state_dict())
    cri_pix = nn.L1Loss() # 定义像素级损失函数为L1损失。
    for epoch in range(start_epoch, total_epochs + 1): # 遍历每一个训练周期。
        if opt["dist"]:   # 如果启用了分布式训练，设置当前周期。
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader): # 遍历训练数据集。
            current_step += 1
            # 如果当前步骤超过了总迭代次数，结束训练。
            if current_step > total_iters:
                
                break
            # 对训练数据进行预处理，并进行必要的格式转换。
            LR_img, ker_map, kernels, lr_blured_t, lr_t = prepro(train_data["GT"])
            LR_img = (LR_img * 255).round() / 255

            LR_up = train_data["LQ"] # 从训练数据中获取上采样的低分辨率图像。

            model_ker.feed_data( # 向模型提供数据并进行优化。
                LR_img, GT_img=train_data["GT"], ker_map=ker_map, lr_blured=lr_blured_t, lr=lr_t, lr_up=LR_up
            )
            # model_ker.optimize_parameters(current_step)
            # model_ker.update_learning_rate(
            #     current_step, warmup_iter=opt["train"]["warmup_iter"]
            # )
            model_ker.test()  # 执行模型推理以获取内核图像。
            visuals_ker = model_ker.get_current_visuals()


            model.feed_data(
                LR_img, GT_img=train_data["GT"], ker_map=visuals_ker["ker"], lr_blured=lr_blured_t, lr=lr_t, lr_up=LR_up
            )   # ker_map  lr_blured 参数取消
            model.optimize_parameters(current_step)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"] # 根据当前步骤和预热迭代次数更新学习率。
            )

            # 在特定的迭代频率上记录训练进度和损失。
            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log() # 获取当前的日志记录。
                message = "<epoch:{:3d}, iter:{:8,d}> ".format(
                    epoch, current_step
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v) # 格式化日志消息。
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0: # 主进程记录TensorBoard日志。
                            tb_logger.add_scalar(k, v, current_step)
                if rank == 0:  # 主进程中打印日志消息。
                    logger.info(message)


            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                # opt["train"]["val_freq"]
            # if 1==1:
               # lpips_model = lpips.LPIPS(net='alex')
                avg_psnr = 0.0  # 平均峰值信噪比初始化
                avg_ssim = 0.0  # ssim
                idx = 0
                for _, val_data in enumerate(val_loader): # 遍历验证数据。
                    # 对验证数据进行预处理。
                    LR_img, ker_map, kernels, lr_blured_t, lr_t = prepro(val_data["GT"])
                    # print(val_data["GT"].shape)
                    LR_img = (LR_img * 255).round() / 255
                    LR_up = val_data["LQ"]
                    
                    lr_img = util.tensor2img(LR_img)  # save LR image for reference

                    # valid Predictor
                    # 使用低分辨率图像、上采样图像和核图进行模型验证。
                    model_ker.feed_data(LR_img, GT_img=val_data["GT"], lr_up=LR_up, ker_map=ker_map)
                    
                    model_ker.test()  # 执行测试。
                    visuals_ker_val = model_ker.get_current_visuals() # 获取当前的可视化结果。
                    fake_ker = visuals_ker_val["ker"].detach()[0].squeeze() # 获取估计的核。
                    real_ker = visuals_ker_val["ker_real"].detach()[0].squeeze() # 获取真实的核。
                    # print(visuals_ker_val["ker_real"].shape)

                    # Save images for reference
                    img_name = val_data["LQ_path"][0]
                    img_dir = os.path.join(opt["path"]["val_images"], img_name)
                    # img_dir = os.path.join(opt['path']['val_images'], str(current_step), '_', str(step))

                    util.mkdir(img_dir)
                    save_lr_path = os.path.join(img_dir, "{:s}_LR.png".format(img_name))
                    util.save_img(lr_img, save_lr_path)
                    # print(visuals_ker["ker"].shape)
                    
                    model.feed_data(LR_img, GT_img=val_data["GT"], lr_up=LR_up, ker_map=visuals_ker_val["ker"]) # 继续对验证数据进行处理，使用模型进行预测并计算PSNR等指标。
                    model.test() # 在没有梯度计算的情况下测试模型。
                    visuals = model.get_current_visuals() # 获取当前的可视化结果。

                    sr_img = util.tensor2img(visuals["SR"].squeeze())  # uint8  # 获取超分辨率结果图像。
                    gt_img = util.tensor2img(visuals["GT"].squeeze())  # uint8

                    save_img_path = os.path.join(
                        img_dir, "{:s}_{:d}.png".format(img_name, current_step)
                    )


                    # cv2.imwrite(save_img_path, kernel)
                    util.save_img(sr_img, save_img_path)

                    # # gtsave_img_path = os.path.join(
                    # #     img_dir, "{:s}_GT.png".format(img_name, current_step)
                    # # )
                    # # util.save_img(gt_img, gtsave_img_path)

                    # calculate PSNR
                    crop_size = opt["scale"]
                    gt_img = gt_img / 255.0
                    sr_img = sr_img / 255.0
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size]

                    cropped_sr_img_y = bgr2ycbcr(cropped_sr_img, only_y=True)
                    cropped_gt_img_y = bgr2ycbcr(cropped_gt_img, only_y=True)

                    # print(val_data["GT"].shape, gt_img.shape, sr_img.shape)
                    # print(cropped_gt_img_y.shape, cropped_sr_img_y.shape)
                    # fake_ker.to(ker_map.device)
                    avg_loss += cri_pix(fake_ker, real_ker)

                    # 计算并累加PSNR值，用于最后计算平均PSNR。
                    avg_psnr += util.calculate_psnr(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )

                    cropped_sr_img_y_255 = cropped_sr_img_y * 255  # ssim
                    cropped_gt_img_y_255 = cropped_gt_img_y * 255
                    ssim_value = calculate_ssim(cropped_sr_img_y_255, cropped_gt_img_y_255)
                    avg_ssim += ssim_value



                    idx += 1
                avg_loss = avg_loss / idx # 所有验证数据处理完毕后，计算平均loss。

                avg_psnr = avg_psnr / idx # 所有验证数据处理完毕后，计算平均PSNR。
                avg_ssim = avg_ssim / idx  # ssim
               # avg_lpips = avg_ssim / idx  # lpips
                # 如果当前的平均PSNR高于之前记录的最佳PSNR，则更新最佳PSNR，并保存模型。
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_iter = current_step
                if avg_loss > best_loss:
                    best_loss = avg_loss
                    # 记录最佳PSNR及其对应的周期和迭代次数。
                    best_iter = current_step

                # 更新最佳SSIM和对应的迭代次数  ssim
                if avg_ssim > best_ssim:
                    best_ssim = avg_ssim



                # log # 记录当前周期结束时的训练信息，包括平均损失、PSNR等。
                logger.info("# Validation # PSNR: {:.6f}, Best PSNR: {:.6f},loss: {:.6f}| Iter: {} || # Validation # SSIM: {:.6f}, Best SSIM: {:.6f} | # Validation ".format(avg_psnr, best_psnr, avg_loss, best_iter,avg_ssim,best_ssim))
                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}, loss: {:.6f} ,ssim: {:.6f} ".format(
                        epoch, current_step, avg_psnr, avg_loss ,avg_ssim # ssim lpips
                    )
                )
                # tensorboard logger # 训练结束，关闭TensorBoard日志记录器。
                if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    tb_logger.add_scalar("psnr", avg_psnr, current_step)
                    tb_logger.add_scalar("loss", avg_loss, current_step)
                    tb_logger.add_scalar("ssim current_step指的是PSNR，对于SSIM不管用", avg_ssim,
                                         current_step)  # current_step指的是PSNR  SSIM
                    # tb_logger.add_scalar("lpips current_step指的是PSNR，对于lpips不管用", avg_lpips,
                    #                      current_step)  # lpips




            ### save models and training states
            # 每隔一定周期保存当前模型的状态
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    #xmz修改，删除了save_training_state

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()


if __name__ == "__main__":
    main()
