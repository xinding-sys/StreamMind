# import numpy as np
# import torch
# def generate_increasing_density_list(n, mode='linear'):
#     # 定义密集度
#     if mode == 'linear':
#         probabilities = np.linspace(0, 1, n)  # 线性增加的概率
#     elif mode == 'quadratic':
#         probabilities = np.linspace(0, 1, n) ** 2  # 二次方增加的概率
#     else:
#         raise ValueError("Unsupported mode. Use 'linear' or 'quadratic'.")
    
#     # 使用 numpy.random.choice 生成 0 和 1，按给定的概率分布
#     result = np.array([np.random.choice([0, 1], p=[1 - p, p]) for p in probabilities])
    
#     return result

# # 示例
# length = 10
# data = torch.randn(30,5)
# print(data)

# output = generate_increasing_density_list(length, mode='linear')
# print(output)
# print(data[output])




import torch
import numpy as np

def exponential_sampling(tensor, percentage=0.6):

    n = tensor.size(0)
    num_samples = int(percentage*n)
    indices = list(n - np.unique(np.logspace(0, np.log10(n - 1), num_samples, dtype=int)))[::-1]  # 对数间隔采样
    # indices = torch.linspace(0, n - 1, num_samples).long()  # 线性间隔采样
    return tensor[indices]

# 示例
tensor = torch.randn(40, 4)  # 一个形状为[100, 4096]的张量
print(tensor)
exponential_sampling(tensor, 0.4)
# print(sampled_tensor)  # 输出采样后的张量形状
