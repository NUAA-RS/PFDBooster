# test phase
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.autograd import Variable
from net import ReconVISnet, ReconIRnet, ReconFuseNet
import utils
from utils import sumPatch
import cv2
from PIL import Image
from args_fusion import args
import numpy as np

from scipy.stats import entropy
from scipy.ndimage import convolve

import time
import torchvision.models as models
from utils import gradient
torch.set_default_tensor_type(torch.FloatTensor)
device_ids = [0]

def load_model_reconIR(path, input_nc, output_nc):
    path = "./models/Final_epoch_10_Tue_Jun_24_04_48_20_2025_IR.model"

    ReconIRnet_model = ReconIRnet()
    ReconIRnet_model = torch.nn.DataParallel(ReconIRnet_model, device_ids = device_ids)    
    ReconIRnet_model.load_state_dict(torch.load(path))

    ReconIRnet_model.eval()
        
    if (args.cuda):
        
        ReconIRnet_model = ReconIRnet_model.cuda()

    ReconIRnet_model = ReconIRnet_model.float()

    return ReconIRnet_model

def load_model_reconVIS(path, input_nc, output_nc):
    path = "./models/Final_epoch_10_Tue_Jun_24_04_48_20_2025_VIS.model"

    ReconVISnet_model = ReconVISnet()
    ReconVISnet_model = torch.nn.DataParallel(ReconVISnet_model, device_ids = device_ids)
    ReconVISnet_model.load_state_dict(torch.load(path))

    ReconVISnet_model.eval()

    if (args.cuda):
            
        ReconVISnet_model = ReconVISnet_model.cuda()

    ReconVISnet_model = ReconVISnet_model.float()

    return ReconVISnet_model
    
def load_model_reconFuse(path, input_nc, output_nc):
    path = "./models/Final_epoch_10_Tue_Jun_24_04_48_21_2025_Fuse.model"
    
    ReconFuseNet_model = ReconFuseNet()
    ReconFuseNet_model = torch.nn.DataParallel(ReconFuseNet_model, device_ids = device_ids)            
    ReconFuseNet_model.load_state_dict(torch.load(path))

    ReconFuseNet_model.eval()
    if (args.cuda):

        ReconFuseNet_model = ReconFuseNet_model.cuda()

    ReconFuseNet_model = ReconFuseNet_model.float()

    return ReconFuseNet_model


def _generate_fusion_image(model, strategy_type, img1, img2):
    # encoder
    en_v = model.encoder(img2)
    en_r = model.encoder(img1)
    f = model.fusion(en_r, en_v, strategy_type=strategy_type)
    img_fusion = model.decoder(f)
    return img_fusion[0]

def run_demo(model_ReconFuse ,model_ReconIR ,model_ReconVIS , infrared_path, visible_path, output_path_root, fileName, input_methodX_dir, fusion_type, network_type, strategy_type, ssim_weight_str, mode):

    ir_img = cv2.imread(infrared_path, cv2.IMREAD_GRAYSCALE)
    vi_img = cv2.imread(visible_path, cv2.IMREAD_GRAYSCALE)
    fused_img = cv2.imread(input_methodX_dir+fileName, cv2.IMREAD_GRAYSCALE)
    
    ir_img=ir_img/255.0
    vi_img=vi_img/255.0
    fused_img = fused_img/255.0
    h = vi_img.shape[0]
    w = vi_img.shape[1]
    
    ir_img_patches = []
    vi_img_patches = []
    fused_img_patches = []
    
    ir_img = np.resize(ir_img, [1,h,w])
    vi_img = np.resize(vi_img, [1,h,w])
    fused_img = np.resize(fused_img, [1,h,w])
    
    ir_img_patches.append(ir_img)  
    vi_img_patches.append(vi_img)
    fused_img_patches.append(fused_img)
    
    ps = args.PATCH_SIZE
    
    ir_img_patches = np.stack(ir_img_patches,axis=0)
    vi_img_patches = np.stack(vi_img_patches,axis=0)
    fused_img_patches = np.stack(fused_img_patches,axis=0)
    
    ir_img_patches = torch.from_numpy(ir_img_patches)
    vi_img_patches = torch.from_numpy(vi_img_patches)
    fused_img_patches = torch.from_numpy(fused_img_patches)
    
    if args.cuda:
        ir_img_patches = ir_img_patches.cuda(args.device)
        vi_img_patches = vi_img_patches.cuda(args.device)
        fused_img_patches = fused_img_patches.cuda(args.device)

    ir_img_patches = ir_img_patches.float()
    vi_img_patches = vi_img_patches.float()
    fused_img_patches = fused_img_patches.float()
    
    recIR = model_ReconIR(fusion = fused_img_patches)
    recVIS = model_ReconVIS(fusion = fused_img_patches)

    #Booster Layer -- begin
    recIRb  = sumPatch(recIR,3)
    recVISb = sumPatch(recVIS,3)
    
    recIRe  = recIR + ir_img_patches - recIRb
    recVISe = recVIS + vi_img_patches -  recVISb    
    
    #Booster Layer -- end
    recIRe = recIRe.float()  # 转换为 float32
    recVISe = recVISe.float()  # 转换为 float32
    
    out = model_ReconFuse(recIR = recIRe, recVIS = recVISe)
    out = out * 255
    
    outputPath = "outputs/"
    
    outputFuse = output_path_root + fileName
        
        
    cv2.imwrite(outputFuse, out[0,0,:,:].cpu().numpy())
    
    print(outputFuse)
    
def calculate_sd(img):
    """
    计算标准差 (Standard Deviation)
    :param img: 输入图像(0-255)
    :return: SD值
    """
    return np.std(img)


def calculate_en(img):
    """
    计算信息熵 (Entropy)
    :param img: 输入图像(0-255)
    :return: EN值
    """
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 255))
    prob = hist / hist.sum()
    return entropy(prob, base=2)


def calculate_ei(img):
    """
    计算边缘强度 (Edge Intensity)
    :param img: 输入图像(0-255)
    :return: EI值
    """
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return np.mean(edge_mag)


def calculate_vif(ir, vi, fused):
    """
    计算视觉信息保真度 (Visual Information Fidelity)
    :param ir: 红外图像(0-255)
    :param vi: 可见光图像(0-255)
    :param fused: 融合图像(0-255)
    :return: VIF值
    """
    # 归一化到0-1
    ir = ir / 255.0
    vi = vi / 255.0
    fused = fused / 255.0

    # 创建高斯金字塔
    def gaussian_pyramid(img, levels=4):
        pyramid = [img]
        for _ in range(levels - 1):
            img = cv2.pyrDown(img)
            pyramid.append(img)
        return pyramid

    # 计算子带VIF
    def vif_subband(ref, dist):
        sigma_nsq = 0.1  # 噪声方差
        g_all, nu_all = 0, 0

        for scale in range(4):
            N = 2 ** (4 - scale)  # 窗口大小
            win = np.ones((N, N)) / N ** 2

            # 计算局部均值
            mu1 = cv2.filter2D(ref, -1, win)
            mu2 = cv2.filter2D(dist, -1, win)

            # 计算局部方差
            sigma1_sq = cv2.filter2D(ref ** 2, -1, win) - mu1 ** 2
            sigma2_sq = cv2.filter2D(dist ** 2, -1, win) - mu2 ** 2
            sigma12 = cv2.filter2D(ref * dist, -1, win) - mu1 * mu2

            # 避免负值和除零
            sigma1_sq = np.maximum(sigma1_sq, 0)
            sigma2_sq = np.maximum(sigma2_sq, 0)

            g = np.zeros_like(sigma12)
            valid_mask = sigma1_sq > 1e-10
            g[valid_mask] = sigma12[valid_mask] / sigma1_sq[valid_mask]

            sv_sq = sigma2_sq - g * sigma12
            sv_sq = np.maximum(sv_sq, 1e-10)

            # 计算信息量
            vif_val = np.log2(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq))
            vif_ref = np.log2(1 + sigma1_sq / sigma_nsq)

            g_all += np.sum(vif_val)
            nu_all += np.sum(vif_ref)

        return g_all / (nu_all + 1e-10)

    # 计算多尺度VIF
    vif_ir = vif_subband(ir, fused)
    vif_vi = vif_subband(vi, fused)
    return (vif_ir + vif_vi) / 2


def calculate_qabf(ir, vi, fused):
    """
    计算基于边缘的融合质量 (Edge-based Fusion Metric)
    :param ir: 红外图像(0-255)
    :param vi: 可见光图像(0-255)
    :param fused: 融合图像(0-255)
    :return: Qabf值
    """
    # 归一化到0-1
    ir = ir / 255.0
    vi = vi / 255.0
    fused = fused / 255.0

    # Sobel算子
    def sobel(img):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return gx, gy, np.sqrt(gx ** 2 + gy ** 2)

    # 计算梯度
    gx_a, gy_a, g_a = sobel(ir)
    gx_b, gy_b, g_b = sobel(vi)
    gx_f, gy_f, g_f = sobel(fused)

    # 计算方向角度
    def compute_angle(gx, gy):
        angle = np.arctan2(gy, gx)
        angle[gx == 0] = np.pi / 2  # 避免除零
        return angle

    # 计算方向相似度
    def dir_sim(angle_a, angle_f):
        return np.abs(np.cos(angle_a - angle_f))

    # 计算强度相似度
    def mag_sim(g, g_f):
        num = 2 * g * g_f
        den = g ** 2 + g_f ** 2 + 1e-10
        return num / den

    # 计算Qabf
    angle_a = compute_angle(gx_a, gy_a)
    angle_b = compute_angle(gx_b, gy_b)
    angle_f = compute_angle(gx_f, gy_f)

    Q_AF = dir_sim(angle_a, angle_f) * mag_sim(g_a, g_f)
    Q_BF = dir_sim(angle_b, angle_f) * mag_sim(g_b, g_f)

    # 计算权重
    w_A = g_a
    w_B = g_b

    # 避免除零
    denominator = np.sum(w_A + w_B)
    if denominator < 1e-10:
        return 0

    numerator = np.sum(Q_AF * w_A + Q_BF * w_B)
    return numerator / denominator


def calculate_metrics(ir, vi, fused):
    """
    计算五种融合指标
    :param ir: 红外图像(0-255)
    :param vi: 可见光图像(0-255)
    :param fused: 融合图像(0-255)
    :return: 包含五种指标的字典
    """
    # 确保输入为灰度图
    if len(ir.shape) > 2:
        ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
    if len(vi.shape) > 2:
        vi = cv2.cvtColor(vi, cv2.COLOR_BGR2GRAY)
    if len(fused.shape) > 2:
        fused = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)

    # 转换为0-255范围
    ir = ir.astype(np.float32)
    vi = vi.astype(np.float32)
    fused = fused.astype(np.float32)

    # 计算各项指标
    sd = calculate_sd(fused)
    en = calculate_en(fused)
    ei = calculate_ei(fused)
    vif = calculate_vif(ir, vi, fused)
    qabf = calculate_qabf(ir, vi, fused)

    return {
        "SD": sd,
        "EN": en,
        "EI": ei,
        "VIF": vif,
        "Qabf": qabf
    }


# 使用示例
if __name__ == "__main__":
    # 读取图像
    ir = cv2.imread('./dataset/LLVIP/ir/1.png', cv2.IMREAD_GRAYSCALE)
    vi = cv2.imread('./dataset/LLVIP/vis/1.png', cv2.IMREAD_GRAYSCALE)
    fused = cv2.imread('./outputs_enhancedDDcGAN_gray/1.png', cv2.IMREAD_GRAYSCALE)

    # 确保图像成功读取
    if ir is None or vi is None or fused is None:
        raise ValueError("无法读取图像文件，请检查路径")

    # 计算指标
    metrics = calculate_metrics(ir, vi, fused)

    # 打印结果
    print("booster_IVIF_gray图像融合质量评估指标:")
    print(f"标准差 (SD): {metrics['SD']:.4f}", f"信息熵 (EN): {metrics['EN']:.4f}", f"视觉信息保真度 (VIF): {metrics['VIF']:.4f}", f"边缘强度 (EI): {metrics['EI']:.4f}", f"基于边缘的融合质量 (Qabf): {metrics['Qabf']:.4f}")

def main():

    test_path = "./dataset/LLVIP/"

    network_type = 'densefuse'
    fusion_type = 'auto'  # auto, fusion_layer, fusion_all
    strategy_type_list = ['AVG', 'L1','SC']  # addition, attention_weight, attention_enhance, adain_fusion, channel_fusion, saliency_mask

    strategy_type = strategy_type_list[1]
    output_path = './outputs_enhancedDDcGAN_gray/'
    
    #based on this algorithm and enhance its result
    input_methodX_dir = './Origin_DDcGAN_gray/'

    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    # in_c = 3 for RGB images in_c = 1 for gray images
    in_c = 2
    out_c = 1
    mode = 'L'
    model_path_ReconFuse = "./models/IVIF/ASE.model"
    model_path_ReconInfrared = "./models/IVIF/InformationProbe_ir.model"
    model_path_ReconVisible = "./models/IVIF/InformationProbe_vis.model"

    with torch.no_grad():
        print('SSIM weight ----- ' + args.ssim_path[2])
        ssim_weight_str = args.ssim_path[2]
        model_ReconFuse = load_model_reconFuse(model_path_ReconFuse, in_c, out_c)
        model_ReconIR = load_model_reconIR(model_path_ReconInfrared, in_c, out_c)
        model_ReconVIS = load_model_reconVIS(model_path_ReconVisible, in_c, out_c)
        files = os.listdir(test_path + "ir/")
        numFiles = len(files)
        for i in range(numFiles):
            infrared_path = test_path + 'ir/' + files[i]
            visible_path = test_path + 'vis/' + files[i]
            run_demo(model_ReconFuse ,model_ReconIR ,model_ReconVIS , infrared_path, visible_path, output_path, files[i], input_methodX_dir, fusion_type, network_type, strategy_type, ssim_weight_str, mode)
    print('Done......')

if __name__ == '__main__':
    main()
