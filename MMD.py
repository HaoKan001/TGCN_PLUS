import torch


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算源域和目标域之间的多尺度高斯核矩阵
    """
    total = torch.cat([source, target], dim=0)  # [N, D]
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)  # [N, N]

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (total.size(0) ** 2 - total.size(0))

    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernels = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]

    return sum(kernels)  # [N, N]


def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    MMD 损失计算：源域 vs 目标域特征对齐
    参数：
    - source: Tensor [B1, D]
    - target: Tensor [B2, D]
    """
    # 自动裁剪 batch size 以对齐
    if source.size(0) != target.size(0):
        min_batch = min(source.size(0), target.size(0))
        source = source[:min_batch]
        target = target[:min_batch]

    batch_size = source.size(0)
    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]

    loss = torch.mean(XX + YY - XY - YX)
    return loss
