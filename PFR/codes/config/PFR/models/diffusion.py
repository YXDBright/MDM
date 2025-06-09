from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from models.module_util import default
from models.sr_utils import SSIM, PerceptualLoss

# gaussian diffusion trainer class
# diffusion.py文件中还定义了一些辅助函数，如noise_like、extract等，它们用于生成噪声、提取特定时间步的参数等。这些函数支持GaussianDiffusion类的核心操作。

#GaussianDiffusion负责实现图像处理的底层机制（如扩散和反扩散过程），而B_Model则负责配置这个过程（如设置去噪函数、训练策略等）并将其应用于特定的图像处理任务。
def extract(a, t, x_shape): # 辅助函数，用于从张量a中根据索引t提取元素
    b, *_ = t.shape # 解包t的形状，获取批次大小b
    out = a.gather(-1, t) # 使用gather函数按最后一个维度从a中取出索引t对应的元素
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) # 重新调整形状，扩展到x_shape的维度  换了
    # return out.reshape([b] + list(x_shape[1:]))  # 使用list来确保维度完全匹配





def noise_like(shape, device, repeat=False): # 生成与输入形状相同的噪声数据
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device) # 生成噪声
    return repeat_noise() if repeat else noise() # 根据repeat参数决定是否重复噪声


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac): # 计算扩散过程中的beta参数，可进行warmup处理
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas

# 获取beta的调度策略，用于控制扩散过程的速度
def get_beta_schedule(num_diffusion_timesteps, beta_schedule='linear', beta_start=0.0001, beta_end=0.02):
    if beta_schedule == 'quad': # 如果策略是quad，使用平方的形式调整
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear': # 如果策略是linear，使用线性调整
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':  # 如果策略包含warmup，根据比例调用_warmup_beta函数
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule 余弦调度
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1 # 创建线性空间，用于计算余弦函数
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2  # 计算余弦值并平方，得到alpha的累乘结果
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0] # 标准化alpha的累乘值，使其起始值为1
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]) # 通过alpha的累乘值计算beta值
    return np.clip(betas, a_min=0, a_max=0.999) # 限制beta值在0到0.999之间


class GaussianDiffusion(nn.Module):  # 似乎只有扩散过程，难道diffsr_modules.py是逆过程？
    # 这个类实现了基于高斯扩散的模型，该模型通过模拟图像生成或恢复过程的逐步扩散和反扩散来工作。
    def __init__(self, denoise_fn, rrdb_net, timesteps=1000, loss_type='l1'):  # timesteps似乎是采样步骤，对应于1000次，明天调小点试试，不然太慢了（不是在这） ''''1000-10
        # (__init__): 在这个方法中，GaussianDiffusion类接受一个去噪函数(denoise_fn)和配置选项，
        # 如扩散时间步(timesteps)和损失类型(loss_type)。类初始化时，会计算一系列的扩散参数，如betas、alphas等，这些参数用于控制扩散过程的细节。
        super().__init__()
        self.denoise_fn = denoise_fn # 去噪函数，用于在生成过程中逐步去除噪声
        # condition net
        self.rrdb = rrdb_net  # RRDB使用的网络模型，通常是一个深度学习网络,可能用于图像的特征提取或增强
        self.ssim_loss = SSIM(window_size=11)  # SSIM损失函数，用于图像质量评估

        betas = cosine_beta_schedule(timesteps, s=0.008)  # 获取余弦beta调度


        alphas = 1. - betas # 计算alphas值
        alphas_cumprod = np.cumprod(alphas, axis=0) # 计算alphas的累乘结果
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1]) # 计算前一步的alphas累乘结果

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps) # 设置扩散步骤数
        self.loss_type = loss_type  # 设置损失类型

        to_torch = partial(torch.tensor, dtype=torch.float32) # 创建一个局部函数，用于将数据转换为PyTorch张量

        # 注册必要的缓冲区 缓冲区通常用于存储不需要反向传播更新的参数。这些包括各种扩散过程中需要的计算参数，如 betas, alphas_cumprod 等。
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        # 上述代码行向模型注册一个张量，该张量用于在推断后验均值时作为系数。

        self.sample_tqdm = True # 设置一个属性，以便在样本生成时显示进度条

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start # 计算均值，基于alphas的开方和起始图像
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape) # 计算方差，基于1减去alphas的累积产物
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape) # 计算方差的对数
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        # print('noise', noise) noise(256,256)   x_t（255,255）

        ''' 0926*********X3倍率修改_增加33'''
        # if noise.shape[-2:] == (256, 256):
        #     # 执行裁剪，修正形状为 (96, 96)
        #     noise = noise[:, :, 0:255, 0:255]
        ''' 0926*********X3倍率修改_增加33'''

        # 根据噪声和当前的x_t预测原始图像x_start
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        # 计算后验均值和方差
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, noise_pred, clip_denoised: bool):
        # 计算去噪后的均值和方差
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)  # 如果指定，将去噪后的图像裁剪到有效范围内

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    def forward(self, img_hr, img_lr, img_lr_up, kernel, t=None, *args, **kwargs): # 定义模型的前向传播方法
        x = img_hr
        b, *_, device = *x.shape, x.device # 解构形状和设备信息
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long() \
            if t is None else torch.LongTensor([t]).repeat(b).to(device)
        # 根据是否指定t，随机选择一个时间步长或使用指定的时间步长

        self.rrdb.eval() # 设置RRDB网络为评估模式
        with torch.no_grad(): # 关闭梯度计算
            rrdb_out, cond = self.rrdb(img_lr, True) # 从低分辨率图像img_lr获取特征和条件  将img_lr输入到rrdb中，也就是通过rrdb通过img_lr提取特征和条件  cond应该就是rrdb输出的feas

        x = self.img2res(x, img_lr_up)  # # 将图像转换为模型可以处理的内部表示

        p_losses, x_tp1, noise_pred, x_t, x_t_gt, x_0 = self.p_losses(x, t, cond, img_lr_up, kernel, *args, **kwargs)

        ret = {'q': p_losses}  # 创建了字典，键 'q'，对应的值为 p_losses。
        # 这里的 p_losses 代表的是在模型的前向传播过程中计算出的损失值，这可能是基于模型生成图像与目标图像之间的差异（如像素误差、结构相似度等）。

        x_tp1 = self.res2img(x_tp1, img_lr_up)  # 将图像转换为模型可以处理的内部表示
        x_t = self.res2img(x_t, img_lr_up)
        x_t_gt = self.res2img(x_t_gt, img_lr_up)
        return ret, (x_tp1, x_t_gt, x_t), t

    def p_losses(self, x_start, t, cond, img_lr_up, kernel, noise=None):
        # （p_losses): 这个方法计算扩散模型的损失，该损失基于模型预测的噪声和真实噪声之间的差异。根据初始化时设置的损失类型，可以使用L1损失、L2损失或SSIM损失等。
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_tp1_gt = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_t_gt = self.q_sample(x_start=x_start, t=t - 1, noise=noise)
        
        noise_pred = self.denoise_fn(x_tp1_gt, t, cond, img_lr_up, kernel=kernel).to(x_tp1_gt.device)
        
        x_t_pred, x0_pred = self.p_sample(x_tp1_gt, t, cond, img_lr_up, noise_pred=noise_pred, kernel=kernel)

        if self.loss_type == 'l1':
            loss = (noise - noise_pred).abs().mean()  # 如果损失类型为l1，计算噪声与预测噪声之间的绝对误差的均值
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, noise_pred) # 如果损失类型为l2，计算均方误差
        elif self.loss_type == 'ssim':
            loss = (noise - noise_pred).abs().mean() # 计算SSIM损失前计算绝对误差的均值
            loss = loss + (1 - self.ssim_loss(noise, noise_pred)) # 加上SSIM损失
        else:
            raise NotImplementedError()  # 如果损失类型不支持，则抛出未实现异常
        return loss, x_tp1_gt, noise_pred, x_t_pred, x_t_gt, x0_pred

    def q_sample(self, x_start, t, noise=None):  # 实现扩散过程，逐渐将噪声混合进图像中
        # q_sample和p_sample方法分别实现了扩散过程和反扩散过程。扩散过程逐渐引入噪声混合图像，而反扩散过程则尝试从噪声图像中恢复原始图像。
        noise = default(noise, lambda: torch.randn_like(x_start)) # 如果未提供噪声，则生成噪声
        t_cond = (t[:, None, None, None] >= 0).float() # 创建一个条件掩码
        t = t.clamp_min(0) # 保证时间步t不小于0
        
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(t.device)  # 确保参数在正确的设备上
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(t.device)
    #    print(self.sqrt_alphas_cumprod.device)

        return (
                       extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                       extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
               ) * t_cond + x_start * (1 - t_cond) # 根据时间步混合原始图像和噪声

    @torch.no_grad()
    def p_sample(self, x, t, cond, img_lr_up, noise_pred=None, clip_denoised=True, repeat_noise=False, kernel=False):
        # 实现反扩散过程，从噪声图像中恢复出原始图像 逆过程
        if noise_pred is None:
            noise_pred = self.denoise_fn(x, t, cond=cond, img_lr_up=img_lr_up, kernel=kernel) # 通过去噪函数预测噪声
            #print('noise_pred',noise_pred.shape)
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x0_pred = self.p_mean_variance(
            x=x, t=t, noise_pred=noise_pred, clip_denoised=clip_denoised) # 计算去噪后的均值和方差
        noise = noise_like(x.shape, device, repeat_noise) # 生成噪声
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))) # 创建非零掩码
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0_pred  # 返回模型均值和预测的原始图像

    @torch.no_grad()
    def sample(self, img_lr, img_lr_up, shape, kernel, save_intermediate=False): # 采样函数，用于生成图像 样本生成函数，用于从低分辨率图像img_lr生成高分辨率图像
        # print(shape)
        # sample方法使用训练好的扩散模型生成或恢复图像。这个过程涉及从完全随机的噪声开始，逐步反扩散恢复图像。
        device = self.betas.device # 获取设备信息，通常用于确保张量创建在同一设备上
        b = shape[0] # 从形状中获取批次大小
        
        img = torch.randn(shape, device=device)  # 初始化一个随机噪声图像，作为生成过程的起点

        rrdb_out, cond = self.rrdb(img_lr, True)  # 通过RRDB网络处理低分辨率图像以获取必要的特征和条件

        it = reversed(range(0, self.num_timesteps))  # 创建一个从最后一个时间步到第一个时间步的逆序迭代器
  #      duqv=1
        if self.sample_tqdm:
            it = tqdm(it, desc='sampling loop time step', total=self.num_timesteps)  # 如果启用进度条，使用tqdm显示进度
            # print(duqv,"/",self.num_timesteps)  # 自己加的
            # duqv=duqv+1 # 自己加的
        images = []  # 初始化一个列表来保存中间图像，如果需要的话
        for i in it:
            img, x_recon = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), cond, img_lr_up, kernel=kernel)
                # 对每个时间步调用p_sample进行反扩散，逐步从噪声恢复图像

            if save_intermediate:
                img_ = self.res2img(img, img_lr_up) # 将内部表示转换为可视化图像
                x_recon_ = self.res2img(x_recon, img_lr_up) # 同样，对重构的图像进行转换
                images.append((img_.cpu(), x_recon_.cpu())) # 将转换后的图像保存到列表中
        img = self.res2img(img, img_lr_up) # 最终将最后生成的图像转换为可视化格式
        if save_intermediate:
            return img, rrdb_out, images # 如果保存中间步骤，则返回最终图像、RRDB输出和所有中间图像
        else:
            return img, rrdb_out # 否则只返回最终图像和RRDB输出

    @torch.no_grad()
    def interpolate(self, x1, x2, img_lr, img_lr_up, t=None, lam=0.5):
        # 插值函数，用于在两个图像之间进行插值生成新的图像
        b, *_, device = *x1.shape, x1.device # 获取图像批次大小 # 获取设备信息
        t = default(t, self.num_timesteps - 1) # 设置默认时间步，若未指定，则使用最后一个时间步

        rrdb_out, cond = self.rrdb(img_lr, True) # 通过RRDB网络处理低分辨率图像img_lr，获取输出和条件


        assert x1.shape == x2.shape # 确保两个输入图像的维度是相同的

        x1 = self.img2res(x1, img_lr_up) # 将第一个高分辨率图像转换为模型的内部表示
        x2 = self.img2res(x2, img_lr_up) # 将第二个高分辨率图像转换为模型的内部表示

        t_batched = torch.stack([torch.tensor(t, device=device)] * b) # 创建一个批次大小的时间步张量
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))  # 分别对两个图像进行扩散采样

        img = (1 - lam) * xt1 + lam * xt2 # 根据参数lam对两个扩散状态的图像进行线性插值
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img, x_recon = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), cond, img_lr_up ,kernel=kernel)   # 对插值结果进行反扩散采样

        img = self.res2img(img, img_lr_up) # 将最终的内部表示转换回图像空间
        return img   # 返回插值后生成的图像

    def res2img(self, img_, img_lr_up, clip_input=None):
        # 将内部表示转换为可视图像
        if clip_input is None:
            clip_input = True
        if clip_input:
            img_ = img_.clamp(-1, 1) # 对图像像素值进行裁剪，保证在有效范围内
        #print('加之前1', img_.shape, img_lr_up.shape)
        # if img_.size(2) != img_lr_up.size(2) or img_.size(3) != img_lr_up.size(3):  # 加的
        #     img_lr_up = torch.nn.functional.interpolate(img_lr_up, size=(img_.size(2), img_.size(3)), mode='bilinear',
        #                                                 align_corners=False)
        #print('加之后2', img_.shape, img_lr_up.shape)

        ''' 0926*********X3倍率修改_增加22'''
        # if img_.shape[-2:] == (256, 256):
        #     # 执行裁剪，修正形状为 (96, 96)
        #     img_ = img_[:, :, 0:255, 0:255]
        # if img_lr_up.shape[-2:] == (256, 256):
        #     # 执行裁剪，修正形状为 (96, 96)
        #     img_lr_up = img_lr_up[:, :, 0:255, 0:255]
        ''' 0926*********X3倍率修改_增加22'''

        img_ = img_ / 2.0 + img_lr_up  # 将处理过的图像重新缩放并偏移到低分辨率图像上
        return img_

    def img2res(self, x, img_lr_up, clip_input=None):
        # 将图像转换为模型可以处理的内部表示
        if clip_input is None:
            clip_input = True
        x = (x - img_lr_up) * 2.0  # 调整图像以符合模型的输入要求
        if clip_input:
            x = x.clamp(-1, 1) # 裁剪图像以确保像素值在合理范围内
        return x
