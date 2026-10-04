# AcoshGrad-With-AscendC
Just a test. The owner is ChatGPT.

## AcoshGrad 算子开发练习

使用华为 Ascend C 在昇腾 NPU 上计算 `dy / sqrt(y*y - 1)`，支持 FP32、FP16、BF16。低精度输入先升为 FP32 计算，再转回原 dtype。

- [题目要求](Project/test_problem/test_problem.md)
- [最终提交文件 kernel.asc](Project/test_problem/acoshgrad_problem_1916_template/kernel.asc)
- [完整测试与性能报告](Project/acoshgrad_validation/REPORT.md)
- [42 项最终 NPU 验证结果](Project/acoshgrad_validation/final_results.json)

题目工程内只修改 `kernel.asc`，原始 main、scripts 和构建文件保持不变。独立验证程序、历史性能候选和结果存放在工程外的 `Project/acoshgrad_validation`，不属于答题网站的提交内容。

## 验证结果

在 CANN 9.0.0、dav-2201 编译目标、npu-smi 显示 Ascend910 的云端 NPU 上，原始用例和42项扩展测试全部通过。覆盖三种dtype、标量/高维、非对齐尾块、多核、千万级输入及Inf/NaN。正式成绩以答题网站评测为准。

最终 kernel.asc SHA256：`a20d3fb4a29a5d939c40713fde2b84f72aa6e0f0e18123a590e47d74bc602493`。

## 构建与运行

在匹配的昇腾环境加载 CANN `set_env.sh`，确认 `ASCEND_HOME_PATH` 后：

```bash
cd Project/test_problem/acoshgrad_problem_1916_template
bash run.sh
```

独立验证依赖 NumPy 和 ml_dtypes，需在同一 NPU 环境执行：

```bash
cd Project/acoshgrad_validation
cmake -S . -B build
cmake --build build --target validate -j4
python3 check.py
```

模板默认目标为 dav-2201；其他硬件与CANN版本需要重新验证。验证记录中的本地路径与云端路径描述开发时的环境，复现时使用当前工程位置。
