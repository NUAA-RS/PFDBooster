import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class DWT(nn.Module):
    """离散小波变换模块 (Haar小波)"""

    def __init__(self):
        super(DWT, self).__init__()
        # Haar小波滤波器
        self.requires_grad = False
        self._setup_filters()

    def _setup_filters(self):
        # 低通滤波器 (LL)
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        # 高通滤波器 (LH, HL, HH)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        self.filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1)

    def forward(self, x):
        # 确保滤波器在相同设备上
        if self.filters.device != x.device:
            self.filters = self.filters.to(x.device)

        # 应用小波变换
        coeffs = []
        for i in range(4):
            filter = self.filters[i].repeat(x.size(1), 1, 1, 1)
            coeff = F.conv2d(x, filter, stride=2, padding=0, groups=x.size(1))
            coeffs.append(coeff)

        return torch.cat(coeffs, dim=1)


class IDWT(nn.Module):
    """逆离散小波变换模块 - 修复尺寸问题"""

    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False
        self._setup_filters()

    def _setup_filters(self):
        # 重建滤波器
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        self.filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1)

    def forward(self, x):
        # 确保滤波器在相同设备上
        if self.filters.device != x.device:
            self.filters = self.filters.to(x.device)

        # 分割四个子带
        channels = x.size(1) // 4
        ll = x[:, :channels]
        lh = x[:, channels:2 * channels]
        hl = x[:, 2 * channels:3 * channels]
        hh = x[:, 3 * channels:]

        # 计算期望的输出尺寸
        input_h, input_w = x.shape[2], x.shape[3]
        output_h = input_h * 2
        output_w = input_w * 2

        # 初始化输出张量
        output = torch.zeros(x.size(0), channels, output_h, output_w, device=x.device)

        # 应用逆变换到每个子带
        for i, subband in enumerate([ll, lh, hl, hh]):
            filter = self.filters[i].repeat(channels, 1, 1, 1)
            # 使用正确的输出尺寸进行转置卷积
            padded = F.conv_transpose2d(
                subband,
                filter,
                stride=2,
                padding=0,
                output_padding=0,
                groups=channels
            )

            # 确保输出尺寸正确
            _, _, ph, pw = padded.shape
            if ph != output_h or pw != output_w:
                # 计算需要裁剪或填充的尺寸
                pad_h = max(0, output_h - ph)
                pad_w = max(0, output_w - pw)

                # 只裁剪不填充，保持原始内容
                padded = padded[:, :, :output_h, :output_w]

            output = output + padded

        return output


class DualDomainConvRelu(nn.Module):
    """双域卷积模块 (空间域 + 频率域)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DualDomainConvRelu, self).__init__()

        # 空间域分支
        self.spatial_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=0),
            nn.ReLU(inplace=True)
        )

        # 频率域分支 (小波域)
        self.dwt = DWT()
        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels * 4, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        self.idwt = IDWT()

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 空间域路径
        spatial_out = self.spatial_conv(x)

        # 频率域路径
        freq_coeffs = self.dwt(x)
        freq_processed = self.freq_conv(freq_coeffs)
        freq_out = self.idwt(freq_processed)

        # 双域特征融合
        combined = torch.cat([spatial_out, freq_out], dim=1)
        return self.fusion(combined)

# class RefleConvRelu(nn.Module):
#     # convolution
#     # leaky relu
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
#         super(RefleConvRelu, self).__init__()
#         self.conv = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride, dilation=dilation, groups=groups))
#         self.ac = nn.ReLU()
#         self.ac2 = nn.Tanh()
#         # self.bn   = nn.BatchNorm2d(out_channels)
#     def forward(self,x, last = False):
#         # print(x.size())
#         if (last):
#             return self.ac2(self.conv(x))
#         else:
#             return self.ac(self.conv(x))

class RefleConvRelu(nn.Module):
    # 修改为支持双域处理
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 stride=1, dilation=1, groups=1, dual_domain=False):
        super(RefleConvRelu, self).__init__()

        self.dual_domain = dual_domain

        if dual_domain:
            self.conv = DualDomainConvRelu(in_channels, out_channels,
                                           kernel_size, padding)
        else:
            self.conv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          padding=0, stride=stride, dilation=dilation, groups=groups)
            )

        self.ac = nn.ReLU()
        self.ac2 = nn.Tanh()

    def forward(self, x, last=False):
        if self.dual_domain:
            out = self.conv(x)
        else:
            out = self.conv(x)

        if last:
            return self.ac2(out)
        else:
            return self.ac(out)

  
# #Information Probe A
# class ReconVISnet(nn.Module):
#     def __init__(self):
#         super(ReconVISnet, self).__init__()
#
#         kernel_size = 3
#         stride = 1
#
#         base_channels = 16
#         in_channels = 32
#         out_channels_def = 32
#         out_channels_def2 = 64
#
#         self.CVIS1 = RefleConvRelu(1,16)
#         self.CVIS2 = RefleConvRelu(16,32)
#         self.CVIS3 = RefleConvRelu(32,16)
#         self.CVIS4 = RefleConvRelu(16,1)
#
#
#     def forward(self, fusion):
#         OCVIS1 = self.CVIS1(fusion)
#         OCVIS2 = self.CVIS2(OCVIS1)
#         OCVIS3 = self.CVIS3(OCVIS2)
#         recVIS = self.CVIS4(OCVIS3,last = True)
#         recVIS = recVIS / 2 + 0.5
#         return recVIS

# Information Probe A (增加双域处理)
class ReconVISnet(nn.Module):
    def __init__(self, dual_domain=False):
        super(ReconVISnet, self).__init__()
        self.dual_domain = dual_domain

        # 前两层使用双域处理
        self.CVIS1 = RefleConvRelu(1, 16, dual_domain=dual_domain)
        self.CVIS2 = RefleConvRelu(16, 32, dual_domain=dual_domain)

        # 后两层保持空间域处理
        self.CVIS3 = RefleConvRelu(32, 16)
        self.CVIS4 = RefleConvRelu(16, 1)

    def forward(self, fusion):
        OCVIS1 = self.CVIS1(fusion)
        OCVIS2 = self.CVIS2(OCVIS1)
        OCVIS3 = self.CVIS3(OCVIS2)
        recVIS = self.CVIS4(OCVIS3, last=True)
        recVIS = recVIS / 2 + 0.5
        return recVIS

# #Information Probe B
# class ReconIRnet(nn.Module):
#     def __init__(self):
#         super(ReconIRnet, self).__init__()
#
#         kernel_size = 3
#         stride = 1
#
#         base_channels = 16
#         in_channels = 32
#         out_channels_def = 32
#         out_channels_def2 = 64
#
#         self.CIR1 = RefleConvRelu(1,16)
#         self.CIR2 = RefleConvRelu(16,32)
#         self.CIR3 = RefleConvRelu(32,16)
#         self.CIR4 = RefleConvRelu(16,1)
#
#
#     def forward(self, fusion):
#         OCIR1 = self.CIR1(fusion)
#         OCIR2 = self.CIR2(OCIR1)
#         OCIR3 = self.CIR3(OCIR2)
#         recIR = self.CIR4(OCIR3,last=True)
#         recIR = recIR/2+0.5
#         return recIR

# Information Probe B (增加双域处理)
class ReconIRnet(nn.Module):
    def __init__(self, dual_domain=False):
        super(ReconIRnet, self).__init__()
        self.dual_domain = dual_domain

        self.CIR1 = RefleConvRelu(1, 16, dual_domain=dual_domain)
        self.CIR2 = RefleConvRelu(16, 32, dual_domain=dual_domain)
        self.CIR3 = RefleConvRelu(32, 16)
        self.CIR4 = RefleConvRelu(16, 1)

    def forward(self, fusion):
        OCIR1 = self.CIR1(fusion)
        OCIR2 = self.CIR2(OCIR1)
        OCIR3 = self.CIR3(OCIR2)
        recIR = self.CIR4(OCIR3, last=True)
        recIR = recIR / 2 + 0.5
        return recIR

# #ASE module
# class ReconFuseNet(nn.Module):
#     def __init__(self):
#         super(ReconFuseNet, self).__init__()
#
#         kernel_size = 3
#         stride = 1
#
#         base_channels = 16
#         in_channels = 32
#         out_channels_def = 32
#         out_channels_def2 = 64
#
#         self.FIR = RefleConvRelu(1,32)
#         self.FVIS = RefleConvRelu(1,32)
#
#         self.FF1 = RefleConvRelu(64,32)
#         self.FF2 = RefleConvRelu(32,16)
#         self.FF3 = RefleConvRelu(16,1)
#
#
#     def forward(self, recIR, recVIS):
#         #Encoder forward
#
#         OFIR = self.FIR(recIR)
#         OFVIS = self.FVIS(recVIS)
#
#         concatedFeatures = torch.cat([OFIR,OFVIS],1)
#
#         OFF1 = self.FF1(concatedFeatures)
#         OFF2 = self.FF2(OFF1)
#         out = self.FF3(OFF2,last=True)
#
#         out = out/2+0.5
#         return out

# ASE module (增强为双域融合)
class ReconFuseNet(nn.Module):
    def __init__(self, dual_domain=True):
        super(ReconFuseNet, self).__init__()
        self.dual_domain = dual_domain

        # 输入处理层 - 使用双域处理
        self.FIR = RefleConvRelu(1, 32, dual_domain=dual_domain)
        self.FVIS = RefleConvRelu(1, 32, dual_domain=dual_domain)

        # 特征融合层 - 使用双域处理
        self.FF1 = RefleConvRelu(64, 32, dual_domain=dual_domain)
        self.FF2 = RefleConvRelu(32, 16, dual_domain=dual_domain)
        self.FF3 = RefleConvRelu(16, 1, dual_domain=dual_domain)

    def forward(self, recIR, recVIS):
        # 双域特征提取
        OFIR = self.FIR(recIR)
        OFVIS = self.FVIS(recVIS)

        # 特征拼接
        concatedFeatures = torch.cat([OFIR, OFVIS], 1)

        # 双域特征融合
        OFF1 = self.FF1(concatedFeatures)
        OFF2 = self.FF2(OFF1)
        out = self.FF3(OFF2, last=True)

        out = out / 2 + 0.5
        return out