'''
Author: wangshuo105 3220215214@bit.edu.cn
Date: 2023-12-28 11:32:07
LastEditors: wangshuo105 3220215214@bit.edu.cn
LastEditTime: 2024-01-13 14:48:38
FilePath: /WWW_PraVFed/data/RDP.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# def inject_renyi_noise(data, noise_scale, renyi_param):
#     # 生成随机噪声
#     noise = torch.randn_like(data.float()) * noise_scale

#     # 计算噪声的Rényi因子
#     renyi_factor = torch.pow(1 + renyi_param * noise_scale * noise, renyi_param - 1)

#     # 注入噪声
#     noisy_data = data * renyi_factor

#     return noisy_data.long()

def add_noise_with_dp(vector, epsilon,class_number):
    # 计算隐私预算
    epsilon = torch.tensor(epsilon)
    delta = torch.tensor(1e-5)  # 假设差分隐私的delta值为1e-5
    # sigma = torch.sqrt(2.0 * torch.log(1.25 / delta)) / epsilon
    
    # # noise = torch.randn(vector.size()) * sigma

    # # 创建正态分布噪声
    # noise = dist.Normal(0, sigma).sample(vector.shape)
    # noise = noise.to(device)
    # vector = vector.to(device)
    # # 将噪声添加到向量中
    # noisy_vector = vector + noise
    
    # noisy_vector = torch.clamp(noisy_vector, 0, 9)
    # print(vector)
    # print("aaa")
    # print(noisy_vector)
    # delta = 1e-5
    sensitivity = 1.0  # 数据的敏感度
    scale = sensitivity / epsilon
    noise = torch.tensor(np.random.laplace(0, scale, size=vector.size()), dtype=torch.float32)
    noise = noise.to(device)
    vector = vector.to(device)
    noisy_vector = vector + noise
    noisy_vector = torch.clamp(noisy_vector, 0, class_number)
    # print(vector)
    # print("aaa")
    # print(noisy_vector.long())
    return noisy_vector.long()