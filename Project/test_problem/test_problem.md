# AcoshGrad算子

## 一、赛题背景

AcoshGrad算子计算反双曲余弦函数 acosh 的**反向传播梯度**，是自定义激活函数、双曲函数族网络及部分科学计算反向图中的基础算子。前向为 $y = \operatorname{acosh}(x)$ 的节点在反向时，需要把上游梯度 $dy$ 乘以 acosh 在输入处的导数 $\frac{1}{\sqrt{y^2 - 1}}$，得到对原始输入的梯度。

本题要求基于PyTorch中 `torch.acosh` 反向传播的核心业务逻辑，采用Ascend C编程语言进行算子原生开发，在昇腾NPU硬件上实现一款高性能的AcoshGrad算子。

## 二、算子功能描述

实现的AcoshGrad算子需完成以下核心计算：

1. **步骤1**：将输入 `y`（前向 acosh 的输入）与上游梯度 `dy` 升精度至 float32（fp32 输入无需转换）；
2. **步骤2**：计算中间量 $y^2 - 1$，开方得 $\sqrt{y^2 - 1}$；
3. **步骤3**：计算 $z = \frac{dy}{\sqrt{y^2 - 1}}$，转回原 dtype 输出。

算子的核心难点在于：fp16/bf16 下的升精度流水（Cast→Mul→Adds→Sqrt→Div→Cast）与双缓冲搬运设计，属于难度适中的纯向量元素级算子，重点考察基础 Ascend C 编程与流水编排能力。

## 三、核心定义与约束

### 3.1 参考算子

等价python实现：

```python
import numpy as np

def acosh_grad(y, dy):
    yf = y.astype(np.float32)
    df = dy.astype(np.float32)
    z = df / np.sqrt(yf * yf - 1.0)
    return z.astype(y.dtype)
```

### 3.2 数学公式

$$
z_i = dy_i \cdot \frac{1}{\sqrt{y_i^2 - 1}}
$$

其中 $y_i$ 为前向 acosh 的输入（值域 $[1, +\infty)$），$dy_i$ 为上游梯度，$z_i$ 为对原始输入的梯度。

### 3.3 输入输出与属性总览

| 类型 | 参数名 | 类型 | 维度形状 | 支持数据类型 | 数据格式 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 输入 | y | required | 0-8维 | float16/bfloat16/float32 | ND | 前向acosh的输入，值域[1,+inf) |
| 输入 | dy | required | 与y一致 | float16/bfloat16/float32 | ND | 上游梯度，与y同shape同dtype |
| 输出 | z | required | 与y一致 | float16/bfloat16/float32 | ND | 输入梯度，与y同shape同dtype |
| 属性 | （无） | - | - | - | - | 本算子无属性 |

### 3.4 关键输入约束

- **数据类型**：y、dy、z 三者 dtype 完全一致，支持 float16/bfloat16/float32
- **维度场景**：1-8维任意形状，y 与 dy 的 shape 必须完全一致（无广播）
- **维度取值范围（均为正整数）**：
  - 输入总元素个数：∈ [1, 2^31)
  - 任一维度长度：∈ [1, 2^20]
- **其他约束**：y 值域建议 [1.01, 100]（本题用例取 [1.01, 10]），过近 1 会因 $y^2 - 1$ 的浮点舍入损失精度

### 3.5 核心属性说明

- 本算子无属性。

### 3.6 输出严格要求

- z 的 shape 与 dtype 与 y 完全一致；
- 输出须为元素级一一对应，不得改变输入布局顺序。

### 3.7 特殊值处理规则

语义基准为 PyTorch `torch.acosh` 反向（cannbench 环境 torch 2.10.0+cpu，经 autograd 实测钉死）：

- **y = 1（定义域边界）**：$y^2 - 1 = 0$，$dy > 0$ 时除零得 $+\infty$；$dy = 0$ 时为 0/0 得 NaN（实测一致）
- **y < 1（定义域外）**：$y^2 - 1 < 0$，开方得 NaN，按 IEEE 754 自然传播（实测一致）
- **数值路径**：参考实现内部以 float32 完整计算（Mul → Adds(-1) → Sqrt → Div，与参考 kernel 计算路径一致），fp16/bf16 输入先 Cast 升精度、结果再 Cast 回原 dtype
- 本题全部用例的 y 取自 [1.01, 10]，输出均为有限值，不构造 Inf/NaN 场景

## 四、规则要求

1. **精度规则**：fp16/bf16 输入必须先升精度至 float32 再计算，禁止直接以 fp16/bf16 计算 $y^2 - 1$（近 1 处舍入误差不可控）；
2. **形状规则**：y 与 dy 的 shape 必须完全一致，输出 z 逐元素对应；
3. **边界规则**：$y \le 1$ 的行为按 IEEE 754 自然传播（Inf/NaN），不得额外截断或饱和；
4. **性能规则**：需充分利用向量指令与双缓冲流水，大 shape（千万级元素）用例计入性能评分。

## 五、精度判断规则

以更高精度的参考实现结果作为 golden（标杆）进行逐元素比较。

**逐元素通过条件**：`abs(actual - golden) <= atol + rtol * abs(golden)`。

**整体通过条件**：`matched_ratio >= required_matched_ratio` 且 `max_abs_error <= max_abs_error_limit`。

**误差阈值表**：

| 数据类型 | float32 | float16 | bfloat16 |
| --- | --- | --- | --- |
| atol | 9.77e-4 | 1.95e-3 | 1.56e-2 |
| rtol | 1.53e-5 | 1.95e-3 | 1.56e-2 |
| required_matched_ratio | 0.99 | 0.99 | 0.99 |
| max_abs_error_limit | 1e-2 | 1e-1 | 1e-0 |

## 六、示例说明

**示例1**：输入 `y = [2, 3]`，`dy = [1, 2]`（float32）

```text
z = [1/sqrt(2^2-1), 2/sqrt(3^2-1)] = [0.57735027, 0.70710678]
```

**示例2**：输入 `y = [1, 0.5]`，`dy = [1, 1]`（float32）

```text
z[0] = 1/sqrt(1-1) = 1/0 = +Inf    （dy != 0 时除零）
z[1] = 1/sqrt(0.25-1) = 1/sqrt(-0.25) = NaN
```