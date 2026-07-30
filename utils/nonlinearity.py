"""
非线性误差注入模块 utils/nonlinearity.py

任务1 的核心算法模块，用于模拟存算一体芯片中模拟域乘累加运算
存在的输入相关非线性失真，其数学模型为：

    y_normalized = alpha * (x_normalized ** 3) + (1 - alpha) * x_normalized

其中 alpha 用于控制非线性强度：
    - alpha = 0.0  ：理想线性（无失真）
    - alpha > 0.0 ：奇函数正增益（大信号被放大）
    - alpha < 0.0 ：奇函数负增益（大信号被压缩）

本模块提供四类工具函数：
    1. nonlinearity(x, alpha)        : 逐样本动态归一化的非线性映射（用于在线注入）
    2. nonlinearity_normalized(x, a) : 全局归一化的非线性映射（用于离线特征分析）
    3. register_nonlinearity_hooks(model, alpha) : 注册 forward_pre_hook 到所有 Conv2d / Linear
    4. remove_hooks(hooks)           : 移除钩子，恢复干净推理
"""

import torch
import torch.nn as nn
from typing import List, Tuple


# ------------------------------------------------------------------
# 1. 核心非线性映射（逐样本动态归一化）
# ------------------------------------------------------------------
def nonlinearity(x: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
    """
    对输入张量施加非线性失真（存算一体模拟域误差模型）。

    归一化策略：**逐样本动态归一化**。即对输入 batch 中的每个样本（第 0 维的每个元素），
    单独计算其所有元素的绝对值最大值作为归一化尺度，确保不同样本、
    不同 batch 的归一化系数都能根据当前输入自适应计算。

    数学流程（对每个样本 s 分别执行）：
        max_val_s       = clamp( max(|x_s|), min=1e-12 )
        x_norm_s        = x_s / max_val_s
        y_norm_s        = alpha * x_norm_s^3 + (1 - alpha) * x_norm_s
        y_s             = y_norm_s * max_val_s

    Args:
        x (torch.Tensor): 任意形状的输入张量。通常第 0 维是 batch 维；
                          若输入为 1 维（如 Linear 的偏置），则退化为整体归一化。
        alpha (float): 非线性强度系数。alpha=0 为理想线性。

    Returns:
        torch.Tensor: 施加非线性失真后的张量，形状与 dtype 与输入 x 完全一致。
    """
    # alpha == 0 时直接返回原张量，节省计算
    if abs(alpha) < 1e-12:
        return x

    x_dtype = x.dtype
    x_device = x.device
    orig_shape = x.shape

    # ---------- 确定每个"样本"的形状 ----------
    # Conv2d 前向输入 shape = (B, C, H, W)，Linear 前向输入 shape = (B, D)。
    # 无论是哪一种，第 0 维都是 batch 维（样本维）。我们对每个样本单独归一化。
    # 对于 0 维或 1 维输入（极少数情况，如单独的偏置），直接整体处理。
    if x.dim() <= 1:
        max_val = torch.clamp(torch.max(torch.abs(x)), min=1e-12)
        x_norm = x / max_val
        y_norm = alpha * (x_norm ** 3) + (1.0 - alpha) * x_norm
        y = y_norm * max_val
        return y.to(dtype=x_dtype)

    # 常规情况：dim >= 2，按样本（dim=0）逐样本处理
    batch_size = orig_shape[0]
    # 将每个样本展平为一维，求其 abs max
    x_flat_per_sample = x.view(batch_size, -1)   # (B, -1)
    max_val_per_sample = torch.clamp(
        torch.max(torch.abs(x_flat_per_sample), dim=1, keepdim=True).values,
        min=1e-12,
    )  # (B, 1)

    # 扩展 max_val_per_sample 到与 x 同形状，以便逐元素除法
    # 先 reshape 到 (B, 1, 1, ..., 1) 维数与 orig_shape 对齐
    expand_shape = [batch_size] + [1] * (len(orig_shape) - 1)
    max_val_expanded = max_val_per_sample.view(*expand_shape)

    # 归一化 -> 非线性 -> 逆归一化
    x_norm = x / max_val_expanded
    y_norm = alpha * (x_norm ** 3) + (1.0 - alpha) * x_norm
    y = y_norm * max_val_expanded

    return y.to(dtype=x_dtype, device=x_device)


# ------------------------------------------------------------------
# 2. 全局归一化版本（用于离线层输出分析）
# ------------------------------------------------------------------
def nonlinearity_normalized(x: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
    """
    对整个张量进行**全局归一化**后施加非线性失真，再做逆归一化。

    与 nonlinearity() 的区别：
        nonlinearity()          逐样本动态归一化  -> 用于前向传播中的在线注入
        nonlinearity_normalized()  整个 tensor 全局归一化 -> 用于离线对一个已提取的
                                   特征图做分析（如层输出分布研究）

    Args:
        x (torch.Tensor): 任意形状的输入张量。
        alpha (float): 非线性强度。

    Returns:
        torch.Tensor: 失真后的张量，与 x 同形状同 dtype。
    """
    if abs(alpha) < 1e-12:
        return x

    max_val = torch.clamp(torch.max(torch.abs(x)), min=1e-12)
    x_norm = x / max_val
    y_norm = alpha * (x_norm ** 3) + (1.0 - alpha) * x_norm
    y = y_norm * max_val
    return y.to(dtype=x.dtype, device=x.device)


# ------------------------------------------------------------------
# 3. 钩子注册：把非线性注入到所有 Conv2d / Linear 的输入
# ------------------------------------------------------------------
def register_nonlinearity_hooks(model: nn.Module, alpha: float) -> List:
    """
    遍历模型所有子模块，对每个 nn.Conv2d 与 nn.Linear 注册
    **forward_pre_hook**（算子执行前调用），从而在其输入激活值上注入非线性失真。

    钩子函数签名 (PyTorch forward_pre_hook 规范)：
        hook(module, input) -> None 或 tuple（返回修改后的 input 元组）
    我们返回 ( nonlinearity(input[0], alpha), ) 作为新的 input 元组，
    其他输入参数（如 Linear 的 bias、Conv2d 的 stride 等）不参与非线性注入。

    Args:
        model (nn.Module): 待注入的模型对象。
        alpha (float): 非线性强度。alpha=0 仍会注册钩子但不改变输入。

    Returns:
        list[RemovableHandle]: 所有注册的钩子句柄列表，后续可调用 remove_hooks() 统一移除。
    """
    hooks = []

    for name, module in model.named_modules():
        # 仅对 Conv2d 和 Linear 注入非线性（模拟 MAC 阵列的输入端失真）
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # 定义闭包钩子函数；显式捕获当前 alpha 和 name，避免循环变量延迟绑定问题
            def make_hook(alpha_val=alpha, layer_name=name):
                def pre_hook_fn(m, inputs):
                    # inputs 是元组，Conv2d/Linear 前向的第一个参数才是激活值输入
                    if len(inputs) == 0:
                        return inputs
                    x_in = inputs[0]
                    x_distorted = nonlinearity(x_in, alpha_val)
                    # 返回修改后的 inputs 元组（其他元素如 weight/bias 不传也没关系，
                    # PyTorch forward_pre_hook 的返回值会替代原 input，仅需传入
                    # 原 forward 方法的参数位置；Conv2d/Linear 的 forward 仅接收一个 x 参数）
                    return (x_distorted,) + tuple(inputs[1:])
                return pre_hook_fn

            hook_handle = module.register_forward_pre_hook(make_hook())
            hooks.append(hook_handle)

    return hooks


# ------------------------------------------------------------------
# 4. 钩子移除
# ------------------------------------------------------------------
def remove_hooks(hooks: List) -> None:
    """
    移除 register_nonlinearity_hooks() 返回的所有钩子，使模型恢复为干净推理。

    Args:
        hooks (list): register_nonlinearity_hooks 返回的钩子句柄列表。
    """
    for h in hooks:
        try:
            h.remove()
        except Exception:
            # 某些情况下钩子可能已被自动移除（如重复调用），静默跳过即可
            pass
