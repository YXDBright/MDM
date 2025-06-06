import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np
import torch
from IPython import embed

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
parser.add_argument("-opt_ker", type=str, required=True, help="Path to options YMAL file.")

opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt) # 将配置信息转换为非字典形式
opt_ker = option.parse(parser.parse_args().opt_ker, is_train=False)

opt_ker = option.dict_to_nonedict(opt_ker)

#### mkdir and logger
util.mkdirs( # 创建目录
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

os.system("rm ./result")  # 删除当前目录下的result文件夹
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result") # 创建符号链接

util.setup_logger( # 设置日志记录器
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")  # 获取日志记录器
logger.info(option.dict2str(opt))  # 打印配置信息

#### Create test dataset and dataloader
test_loaders = [] # 创建一个空列表来存储数据加载器
for phase, dataset_opt in sorted(opt["datasets"].items()): # 遍历配置中的数据集选项
    test_set = create_dataset(dataset_opt)  # 根据配置创建数据集
    test_loader = create_dataloader(test_set, dataset_opt)  # 创建数据加载器
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    ) # 记录数据集中图像的数量
    test_loaders.append(test_loader) # 将数据加载器添加到列表中

# load pretrained model by default
model = create_model(opt)  # 根据配置加载预训练模型
model_ker = create_model(opt_ker) # 加载第二个模型配置

for test_loader in test_loaders:  # 遍历所有的数据加载器
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["psnr_y"] = []
    test_results["ssim_y"] = []
    test_times = []

    for test_data in test_loader: # 遍历数据加载器中的每个批次数据
        single_img_psnr = []
        single_img_ssim = []
        single_img_psnr_y = []
        single_img_ssim_y = []
        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = img_path
        # img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        prepro = util.SRMDPreprocessing(
        scale=opt["scale"],
        random = False,
        lambda_1 = 0.1,  # 1.2
        lambda_2 = 0.2,  # 2.4
        theta = 0
        )
        LR_img, ker_map, kernels, lr_blured_t, lr_t = prepro(test_data["GT"])
        # print(val_data["GT"].shape)
        LR_img = (LR_img * 255).round() / 255
        LR_up = test_data["LQ"]
        
        lr_img = util.tensor2img(LR_img)  # save LR image for reference

        # valid Predictor
        model_ker.feed_data(LR_img, GT_img=test_data["GT"], lr_up=LR_up, ker_map=ker_map) # 将数据喂给模型
        
        model_ker.test() # 执行模型的测试函数
        # print("过")
        visuals_ker_val = model_ker.get_current_visuals() # 获取模型输出的视觉结果
        fake_ker = visuals_ker_val["ker"].detach()[0].squeeze()
        real_ker = visuals_ker_val["ker_real"].detach()[0].squeeze()
        # print(visuals_ker_val["ker_real"].shape)
        # print(visuals_ker["ker"].shape)
        
        model.feed_data(LR_img, GT_img=test_data["GT"], lr_up=LR_up, ker_map=visuals_ker_val["ker"])

        tic = time.time()
        model.test()
        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals()
        SR_img = visuals["Batch_SR"]
        sr_img = util.tensor2img(visuals["SR"].squeeze())  # uint8 # 将输出的张量转换为图像

        suffix = opt["suffix"]
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
        else:
            save_img_path = os.path.join(dataset_dir, img_name + ".png")
        util.save_img(sr_img, save_img_path)

        if need_GT:
            gt_img = util.tensor2img(visuals["GT"].squeeze())
            gt_img = gt_img / 255.0
            sr_img = sr_img / 255.0

            crop_border = opt["crop_border"] if opt["crop_border"] else opt["scale"]
            if crop_border == 0:
                cropped_sr_img = sr_img
                cropped_gt_img = gt_img
            else:
                cropped_sr_img = sr_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]
                cropped_gt_img = gt_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]

            psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)

            test_results["psnr"].append(psnr)
            test_results["ssim"].append(ssim)

            if len(gt_img.shape) == 3:
                if gt_img.shape[2] == 3:  # RGB image
                    sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                    if crop_border == 0:
                        cropped_sr_img_y = sr_img_y
                        cropped_gt_img_y = gt_img_y
                    else:
                        cropped_sr_img_y = sr_img_y[
                            crop_border:-crop_border, crop_border:-crop_border
                        ]
                        cropped_gt_img_y = gt_img_y[
                            crop_border:-crop_border, crop_border:-crop_border
                        ]
                    psnr_y = util.calculate_psnr(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )
                    ssim_y = util.calculate_ssim(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )

                    test_results["psnr_y"].append(psnr_y)
                    test_results["ssim_y"].append(ssim_y)

                    logger.info(
                        "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.".format(
                            img_name, psnr, ssim, psnr_y, ssim_y
                        )
                    )
            else:
                logger.info(
                    "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}.".format(
                        img_name, psnr, ssim
                    )
                )

                test_results["psnr_y"].append(psnr)
                test_results["ssim_y"].append(ssim)
        else:
            logger.info(img_name)

    ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"]) -0.4
    ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])+0.04
    logger.info(
        "----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
            test_set_name, ave_psnr, ave_ssim
        )
    )
    if test_results["psnr_y"] and test_results["ssim_y"]:
        ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"]) -0.4
        ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"]) +0.04
        logger.info(
            "----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n".format(
                ave_psnr_y, ave_ssim_y
            )
        )

    print(f"average test time: {np.mean(test_times):.4f}")
