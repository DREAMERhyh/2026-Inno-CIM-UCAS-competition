"""
通用扰动注入模块（拓展研究2）

功能：
    1. 高斯加性噪声注入：gaussian_noise(x, std)
    2. 通用扰动接口：inject_perturbation(x, pert_type, alpha, noise_std)
       - pert_type="nonlinearity"：调用非线性失真（三次多项式，逐样本归一化）
       - pert_type="gaussian"：调用高斯噪声
    3. 通用钩子注册：register_perturbation_hooks(model, pert_type, ...)
       - 对所有 Conv2d 和 Linear 模块注册 forward_pre_hook，在算子输入端注入扰动
       - 内置闭包延迟绑定防护，确保当前参数被正确捕获
    4. 钩子移除：remove_hooks(hooks)

设计动机：
    拓展研究2需要系统对比高斯加性噪声与三次非线性失真对模型鲁棒性的不同影响机制。
    抽象出统一的扰动注入接口，便于：
        - 整网精度扫描（同 strength 范围参数循环）
        - 层输出分布偏移分析（相同扰动作用下的各层 RME / 余弦相似度）
        - 跨模型（SimpleCNN vs Exp2_RobustCNN）、跨扰动类型（非线性 vs 高斯）对比实验
"""

import torch
import torch.nn as nn
from typing import List

from utils.nonlinearity import nonlinearity


# ------------------------------------------------------------------
# 1. 高斯噪声注入
# ------------------------------------------------------------------
def gaussian_noise(x: torch.Tensor, std: float = 0.0) -> torch.Tensor:
    """
    向张量 x 添加 0 均值、标准差为 std 的高斯加性噪声。

    模型：
        x_noisy = x + N(0, std^2)

    Args:
        x (torch.Tensor): 任意形状的张量（通常是 Conv/Linear 的激活输入）
        std (float): 噪声标准差（默认 0.0 = 无噪声）

    Returns:
        torch.Tensor: 扰动后的张量，形状/设备/dtype 与 x 一致
    """
    if std == 0.0:
        return x
    noise = torch.randn_like(x) * std
    return x + noise


# ------------------------------------------------------------------
# 2. 通用扰动接口
# ------------------------------------------------------------------
def inject_perturbation(
    x: torch.Tensor,
    pert_type: str = "nonlinearity",
    alpha: float = 0.0,
    noise_std: float = 0.0,
) -> torch.Tensor:
    """
    通用扰动注入对外统一接口。

    Args:
        x (torch.Tensor): 任意形状张量
        pert_type (str): 扰动类型，可选：
            - "nonlinearity"：三次非线性失真 y = αx³ + (1-α)x（逐样本动态归一化）
            - "gaussian"：高斯加性噪声 x + N(0, noise_std²)
        alpha (float): 非线性强度（仅 pert_type="nonlinearity" 时生效）
        noise_std (float): 噪声标准差（仅 pert_type="gaussian" 时生效）

    Returns:
        torch.Tensor: 扰动后的张量
    """
    if pert_type == "nonlinearity":
        return nonlinearity(x, alpha)
    elif pert_type == "gaussian":
        return gaussian_noise(x, noise_std)
    else:
        raise ValueError(
            f"未知扰动类型 pert_type='{pert_type}'，"
            f"当前支持：'nonlinearity' | 'gaussian'"
        )


# ------------------------------------------------------------------
# 3. 通用扰动钩子注册（所有 Conv2d / Linear 入口挂 forward_pre_hook）
# ------------------------------------------------------------------
def register_perturbation_hooks(
    model: nn.Module,
    pert_type: str = "nonlinearity",
    alpha: float = 0.0,
    noise_std: float = 0.0,
) -> List:
    """
    为 model 中所有 nn.Conv2d 和 nn.Linear 模块注册 forward_pre_hook，
    在算子计算之前对输入激活注入扰动。

    注意：
        - 使用工厂函数 make_hook 将当前循环的 pert_type/alpha/noise_std
          捕获到闭包作用域中，避免 Python 的后期绑定陷阱。
        - 返回的 hooks 列表必须保存下来，使用完毕后调用 remove_hooks(hooks) 清理。

    Args:
        model (nn.Module): 待注册钩子的模型
        pert_type (str): 扰动类型 "nonlinearity" | "gaussian"
        alpha (float): 非线性失真强度
        noise_std (float): 高斯噪声标准差

    Returns:
        List[HookHandle]: 所有已注册钩子的句柄列表，用于后续移除
    """
    hooks = []

    def make_hook(p, a, s):
        """
        钩子工厂：避免闭包延迟绑定。

        Args:
            p (str): pert_type 快照
            a (float): alpha 快照
            s (float): noise_std 快照
        """
        def pre_hook_fn(module, inputs):
            if not inputs:
                return inputs
            x_d = inject_perturbation(inputs[0], pert_type=p, alpha=a, noise_std=s)
            return (x_d,) + tuple(inputs[1:])
        return pre_hook_fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            h = module.register_forward_pre_hook(make_hook(pert_type, alpha, noise_std))
            hooks.append(h)

    return hooks


# ------------------------------------------------------------------
# 4. 钩子移除
# ------------------------------------------------------------------
def remove_hooks(hooks) -> None:
    """
    遍历钩子列表并移除所有已注册的 forward_pre_hook。
    单个 hook 移除失败时跳过，避免影响其他钩子。

    Args:
        hooks (Iterable): 钩子句柄列表（通常来自 register_perturbation_hooks）
    """
    if hooks is None:
        return
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass
