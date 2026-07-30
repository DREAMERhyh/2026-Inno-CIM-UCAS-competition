"""
量化误差注入模块 utils/quantization.py

存算一体芯片真实部署信号路径（本模块模拟目标）：
    输入激活 → 模拟域 MAC（乘累加计算）→ 非线性失真（模拟固有特性）
        → ADC 模数转换 → 量化-反量化误差（精度损失）→ 数字后续逻辑

本模块提供三层对外 API：
    1) quantize_error(x, num_bits)            —— 单独量化误差注入
    2) inject_joint_error(x, alpha, num_bits,
                          apply_nonlinearity, apply_quantization)
                                                —— 联合/单独误差统一入口
    3) register_joint_error_hooks(model, ...)  —— 整网 Conv2d/Linear 前置钩子注入
    4) remove_hooks(hooks)                     —— 钩子稳健移除

实现要点：
    · 量化采用"逐样本动态缩放 + 对称饱和整型 + 反量化"的经典 QDQ 范式：
      scale = max(|x|) / qmax，然后 x → round(x/scale).clamp(qmin,qmax) → * scale
    · num_bits=32 或 None 直接短路返回原始 Tensor，作为无量化基线。
    · 联合顺序：先非线性失真，再量化。对应真实路径 MAC → ADC。
    · forward_pre_hook 使用工厂函数 make_hook 捕获循环快照参数，避免 Python 闭包延迟绑定陷阱。
"""

import torch
import torch.nn as nn

from utils.nonlinearity import nonlinearity


# ------------------------------------------------------------------
# 1. 单独量化-反量化误差注入
# ------------------------------------------------------------------
def quantize_error(x, num_bits=8):
    """
    对输入张量 x 执行"逐样本动态缩放 + 对称饱和整型 + 反量化 (QDQ)"，
    模拟 ADC 模数转换中由有限比特数引入的精度损失。

    处理流程（以单个样本为例，batch 维度独立处理）：
        1. 确定整型范围：qmin = -2^(num_bits-1)，qmax = 2^(num_bits-1)-1（对称补码）
        2. 逐样本缩放因子：scale[i] = max|x[i,...]| / qmax
           （对 batch 维的每个样本独立计算：保证动态范围利用充分）
        3. 量化：x_q[i,...] = round( x[i,...] / scale[i] ).clamp(qmin, qmax)
        4. 反量化：x_deq[i,...] = x_q[i,...] * scale[i]

    特殊短路：num_bits == 32 或 num_bits is None → 直接返回原始 x（无误差）。

    Args:
        x (torch.Tensor): 任意形状输入激活。通常形状：
                          Conv: (B, C_in, H, W)  Linear: (B, D_in)
        num_bits (int | None): 量化比特数。默认 8。32/None → 不量化。

    Returns:
        torch.Tensor: 反量化后的浮点数张量，与 x 形状/类型/设备完全相同。
                      隐式量化误差 = return_value - x。
    """
    # 无量化短路分支
    if num_bits is None or num_bits >= 32:
        return x

    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    # 保留原始 dtype/device，输出保持一致
    orig_dtype = x.dtype
    x_float = x.float()

    # 展平除 batch 维外的所有维度 → (B, -1)，按样本求 max 绝对值
    if x_float.dim() > 1:
        x_flat = x_flat = x_float.view(x_float.size(0), -1)
        # scale: (B, 1)，后续可广播到原形状
        scale = x_flat.abs().max(dim=1, keepdim=True)[0] / float(qmax)
        # 避免除 0：若全零样本，则 scale 设为 1（量化后仍为 0）
        scale = torch.where(scale < 1e-12, torch.ones_like(scale), scale)
        # 恢复到与 x 可广播的形状：(B, 1,1,1) 或 (B,1)
        while scale.dim() < x_float.dim():
            scale = scale.unsqueeze(-1)
    else:
        # 仅 1D 情况（极少出现）：整体单个 scale
        max_abs = x_float.abs().max()
        scale = max_abs / float(qmax) if max_abs > 1e-12 else x_float.new_tensor(1.0)

    # 量化 (round + clamp) + 反量化
    x_q = torch.round(x_float / scale).clamp(qmin, qmax)
    x_deq = (x_q * scale).to(orig_dtype)

    return x_deq


# ------------------------------------------------------------------
# 2. 联合误差统一入口
# ------------------------------------------------------------------
def inject_joint_error(
    x,
    alpha: float = 0.0,
    num_bits: int = 8,
    apply_nonlinearity: bool = True,
    apply_quantization: bool = True,
):
    """
    对 x 施加非线性失真 + 量化误差的任意组合，模拟 ADC 前的真实信号路径。

    顺序严格遵守：MAC 输出 → 非线性失真 → ADC（量化）。
    因此：若两开关同时为 True → 先 nonlinearity，再 quantize_error。
    单个开关打开时对应独立误差分析。

    Args:
        x (torch.Tensor): 输入激活（Conv/Linear 层的输入）
        alpha (float): 非线性失真强度 α，仅 apply_nonlinearity=True 时生效
        num_bits (int): 量化比特数，仅 apply_quantization=True 时生效；
                        32 或 None → 即使 apply_quantization=True 也短路为无量化
        apply_nonlinearity (bool): 是否施加三次多项式非线性
        apply_quantization (bool): 是否施加 QDQ 量化误差

    Returns:
        torch.Tensor: 处理后的张量，形状/类型/设备与 x 完全一致
    """
    out = x
    if apply_nonlinearity:
        out = nonlinearity(out, alpha=alpha)
    if apply_quantization:
        out = quantize_error(out, num_bits=num_bits)
    return out


# ------------------------------------------------------------------
# 3. 整网 forward_pre_hook 注册（Conv2d + Linear 的输入注入联合误差）
# ------------------------------------------------------------------
def register_joint_error_hooks(
    model,
    alpha: float = 0.0,
    num_bits: int = 8,
    apply_nonlinearity: bool = True,
    apply_quantization: bool = True,
):
    """
    遍历 model 中所有 nn.Conv2d / nn.Linear，对每个模块注册 forward_pre_hook：
        在该模块执行前，向其输入激活 inject_joint_error(...)

    注意：
        · 使用闭包工厂函数 make_hook(...) 在每次循环时立即快照当前 alpha/num_bits
          等 4 个参数值，**彻底避免 Python 闭包的延迟绑定陷阱**。
        · 返回 hooks 句柄列表，供 remove_hooks() 统一清理。

    Args:
        model (nn.Module): 待注入误差的模型
        alpha (float): 非线性强度
        num_bits (int): 量化比特数
        apply_nonlinearity (bool): 施加非线性开关
        apply_quantization (bool): 施量化开关

    Returns:
        list[torch.utils.hooks.RemovableHandle]: 钩子句柄列表
    """
    hooks = []

    def make_hook(alp, nb, an, aq):
        """参数快照工厂：每个 hook 闭包绑定自己的一份参数副本。"""
        def _pre_hook_fn(module, inputs):
            # inputs 是位置参数元组；通常 Conv/Linear 只有一个输入 inputs[0]
            if not inputs or inputs[0] is None:
                return inputs
            x_in = inputs[0]
            x_corrupted = inject_joint_error(
                x_in,
                alpha=alp,
                num_bits=nb,
                apply_nonlinearity=an,
                apply_quantization=aq,
            )
            return (x_corrupted,) + tuple(inputs[1:])
        return _pre_hook_fn

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hook_fn = make_hook(
                float(alpha),
                num_bits,
                bool(apply_nonlinearity),
                bool(apply_quantization),
            )
            h = m.register_forward_pre_hook(hook_fn)
            hooks.append(h)

    return hooks


# ------------------------------------------------------------------
# 4. 钩子移除（稳健）
# ------------------------------------------------------------------
def remove_hooks(hooks):
    """
    逐一调用 hook.remove()；单个失败时静默跳过，保证整体清理流程不中断。

    Args:
        hooks (Iterable): register_joint_error_hooks() 返回的句柄列表或可迭代对象
    """
    if hooks is None:
        return
    for h in hooks:
        try:
            h.remove()
        except Exception:
            # 单个 hook 已经被移除或失效不应影响全局流程
            pass
