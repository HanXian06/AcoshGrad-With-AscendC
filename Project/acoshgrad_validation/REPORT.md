# AcoshGrad 交付与验证记录

日期：2026-10-04。提交文件：`../test_problem/acoshgrad_problem_1916_template/kernel.asc`。

SHA256：`a20d3fb4a29a5d939c40713fde2b84f72aa6e0f0e18123a590e47d74bc602493`。

题目目录内仅 kernel.asc 改动，main.asc、CMakeLists.txt、run.sh、data_utils.h 和 scripts 三个文件均保持原始哈希。题目 md 未改动。云端与本地八个工程源文件哈希一致，见 local_integrity.json、remote_hashes.json。本目录是工程外验证资料，不属于提交内容。

## 实现

Ascend C Vector 原生执行 FP32 Mul → Adds(-1) → Sqrt → Div。FP16/BF16 先 Cast 升 FP32，结果以 CAST_RINT 转回。FP32 专用路径避免恒等拷贝。每核采用两路输入和一路输出的双缓冲队列，提前搬入下一 tile；核间按 32B 单位切分，DataCopyPad 仅搬运有效字节。FP32 tile=2048，FP16/BF16 tile=4096；用户 UB 分别 64 KiB/80 KiB。Host 按实际 availableCoreNum 限制核数，小输入减少启动核数。

## 环境与结果

复用 CANNLab Dev Space 生成的 SSH 连接，并用 VS Code Remote SSH CLI 打开云端源文件。云端根目录 `/mnt/workspace/acoshgrad_1916_20261004`，工程子目录 `acoshgrad_problem_1916_template`，验证子目录 `acoshgrad_validation`。

- CANN 9.0.0；编译目标 dav-2201；npu-smi 25.5.5 报告 Ascend910、Health OK；ACL 查询 Vector 核数 40。
- 原始工程编译成功，原始 FP32 8 元素用例通过，最大差异 0，见 official-final.log。
- 最终版本独立 NPU 测试 42/42 通过：39 个有限数用例，3 个特殊值用例。每次预热3次，再重复执行20次。见 final_results.json。
- 长度覆盖 1、7、17、2047、2048、2049、65537、163839、163840、163841、262151、1048576、10485760；覆盖 rank 0/1/2/8，以及两种 tile 的多核尾部。
- 所有有限数测试 matched_ratio=1.0，最大误差如下；全部满足题目双重阈值。

| dtype | 最大绝对误差 | 题目最大误差上限 |
|---|---:|---:|
| FP32 | 0.00000190735 | 0.01 |
| FP16 | 0.00390625 | 0.1 |
| BF16 | 0.0009765625 | 1.0 |

特殊值用例检查 y=1、dy=正/负/零时的 +Inf/-Inf/NaN，以及 y=0.5 的 NaN。输出末尾 64B 哨兵每次保持完整。哨兵不等价于完整内存检测器；未运行 sanitizer、仿真或 msprof。

## 性能比较

相同 10,485,760 元素输入，每个候选运行三轮并轮换顺序，每轮预热3次、ACL Event 区间重复发射20次。下表是三轮平均时延的中位数（微秒），原始数据见 benchmark.json。

| dtype | 首版 tile2048 | 去冗余拷贝 tile2048 | 去冗余拷贝 tile4096 |
|---|---:|---:|---:|
| FP32 | 77.552 | 68.078 | 71.666 |
| FP16 | 44.052 | 44.098 | 36.852 |
| BF16 | 45.943 | 45.908 | 38.110 |

FP16/BF16 扩大 tile 的中位时延改善约16.3%/17.0%。FP32 测量波动明显，最终回归中为89.085微秒，不能声称稳定加速。最终 FP16/BF16 回归时延分别36.954/38.194微秒。Event区间不含内存申请和H2D/D2H，但可能包含Host下发空隙，不等于纯kernel duration。没有网站基线、隐藏用例或正式评分。

## 复现

在云端同一 shell 加载 `/etc/profile` 与 `/home/developer/Ascend/cann-9.0.0/set_env.sh`。原工程首次构建运行 `bash run.sh`；最终更新后通过 `cmake --build build -j4` 构建，再在 build 内执行 `./acosh_grad_custom` 和 `python3 ../scripts/verify_result.py 0`。

在独立验证目录执行 `cmake -S . -B build`、`cmake --build build --target validate -j4`、`python3 check.py`。其 include 指向旁边工程的同一 kernel.asc，不修改模板脚本。benchmark.py 的历史候选位于独立验证目录，只用于可复现比较，不用于提交。

## Skill 更新

更新项目内 `skills/ascendc-operator-development`，新增 `references/vector-single-file.md`，补充单文件范围保护、FP32数值路径、有效字节尾部、分核、双缓冲资源预算及Event计时口径。静态结构/引用/原教学资料哈希验证通过。新旧skill的独立只读方案评估均6/6通过，说明这些案例未显示可量化能力提升；更新的价值是保留本次验证过的具体经验。评估查看器在 `skills/ascendc-operator-development-workspace/iteration-2/review.html`。

算子开发skill仍是项目内版本；本次未安装全局版本，也未提交答题网站。
