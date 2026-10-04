#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
AcoshGrad算子golden实现
以 numpy 组合实现的结果为 golden（语义基准：torch.acosh 反向实测钉死）

z = dy / sqrt(y^2 - 1)
y  : 前向 acosh 的输入（值域 >= 1）
dy : 上游梯度
z  : 对原始输入的梯度
"""
import numpy as np


def impl(y, dy):
    """AcoshGrad算子golden实现

    参数名与顺序与 JSON 的 input_desc + attr_desc 一致：
    y  : 前向 acosh 的输入，fp16/bf16/fp32，值域 [1, +inf)
    dy : 上游梯度，与 y 同 shape 同 dtype
    返回与 y 同 shape 同 dtype 的 numpy 数组 z。
    """
    orig_dtype = y.dtype
    is_bf16 = ('bfloat16' in str(orig_dtype)) or ('bf16' in str(orig_dtype))

    # 统一转 float32 计算（bf16/fp16 精度不足，内部用 FP32，
    # 与参考 kernel 计算路径一致: Mul -> Adds(-1) -> Sqrt -> Div）
    yf = np.asarray(y, dtype=np.float32)
    df = np.asarray(dy, dtype=np.float32)

    with np.errstate(divide='ignore', invalid='ignore'):
        t = yf * yf - np.float32(1.0)     # y^2 - 1
        z = df / np.sqrt(t)               # dy / sqrt(y^2 - 1)，边界按 IEEE 自然传播

    # 转回原始 dtype（与 JSON output_desc 一致）
    if is_bf16:
        from ml_dtypes import bfloat16
        return z.astype(bfloat16)
    if orig_dtype == np.float16:
        return z.astype(np.float16)
    return z.astype(np.float32)


def _make(shape, dtype, seed, lo, hi):
    rng = np.random.default_rng(seed)
    a = rng.uniform(lo, hi, size=shape).astype(np.float32)
    if dtype == np.float16:
        return a.astype(np.float16)
    if dtype == 'bf16':
        from ml_dtypes import bfloat16
        return a.astype(bfloat16)
    return a.astype(np.float32)


def _dtype_tag(dtype):
    return {np.float16: 'fp16', np.float32: 'fp32'}.get(dtype, 'bf16')
