# 根目录脚本索引（2026-10-19）

> **为什么脚本还在根目录**：36/40 个脚本 `import utils` / `models`，而 `utils/paths.py` 用的是
> `./outputs`、`./checkpoints`（相对于当前工作目录）。把脚本移进 `scripts/` 会改变 `sys.path`，
> 导入会失败；按项目纪律“不改代码、不跑冒烟测试”，暂时保持原位。
> **运行方式**：一律在仓库根目录运行（`cd d:\deeplearning_practice\存算一体`）。
> **解释器**：`D:\anaconda3\envs\pytorch_env\python.exe`。

## 一、核心训练与评估

| 脚本 | 用途 |
|---|---|
| `train.py` | 主训练脚本 |
| `evaluate.py` | 模型评估脚本 |
| `task1_sensitivity_analysis.py` | 任务 1：非线性误差敏感性主分析 |
| `task1_layer_analysis.py` | 任务 1：单层输出分布分析 |
| `task2_nat.py` | 任务 2：非线性感知训练（NAT）主脚本 |
| `task3_robust.py` | 任务 3：鲁棒性增强方法设计 |
| `task_extension1_network_comparison.py` | 拓展 1：网络结构与参数量对比 |
| `task_extension2_noise_vs_nonlinearity.py` | 拓展 2：高斯噪声 vs 非线性失真 |
| `task_extension3_joint_analysis.py` | 拓展 3：量化误差与非线性误差联合作用 |
| `task_extension4_alpha_wide_scan.py` | 拓展 4：α 宽范围扫描 |
| `task_extension5_dithering_mechanism.py` | 拓展 5：Dithering 机理 |
| `task_extension6_deep_robust_train.py` | 拓展 6：深层网络鲁棒训练 |
| `task_extension6_eval_deep_robust.py` | 拓展 6：深层网络评估 |

## 二、第三阶段（v3）

| 脚本 | 用途 |
|---|---|
| `v3_harvest_analysis.py` | 四块未分析数据的收割分析（只读，输出到 stdout） |
| `v3_methods_simplecnn.py` | SimpleCNN 级方法实验统一入口 |
| `v3_methods_summary.py` | 方法实验结果汇总（只读） |
| `v3_probe_gaussian_transfer.py` | 校准模型的跨扰动迁移检验 |
| `v3_probe_info_death.py` | 信息死亡探针 |
| `v3_probe_pairwise.py` | 跨层损伤是“成对”还是“全局” |
| `v3_readout_repair.py` | 读出修复概念验证 |

## 三、第四阶段（v4）

| 脚本 | 用途 |
|---|---|
| `v4_probe_activation_stats.py` | 逐层激活统计（纯推理） |
| `v4_probe_alpha_routing.py` | 在线 α 估计 + 双头路由 |

## 四、第五阶段（v5）

| 脚本 | 用途 |
|---|---|
| `v5_full_recipe.py` | 完整配方端到端实测 |
| `v5_p0c_criteria.py` | CIFAR-10 复现的预登记判据评估 |
| `v5_probe_alpha_drift.py` | 路由在 α 分布漂移下的压力测试 |
| `v5_probe_maxpool_mechanism.py` | MaxPool 保护的机制 |
| `v5_probe_rank_collapse.py` | 正侧不可训练区的秩/条件数塌陷 |

## 五、第六、七阶段（v6 / v7）

| 脚本 | 用途 |
|---|---|
| `v6_p4e_crosspath.py` | 交叉评估：两套训练脚本的权重互换评估路径 |
| `v7_c4_decompose.py` | 拆分样本量与不相交性的混杂 |
| `v7_etf_readout.py` | ETF / 原型读出是否更抗输入扰动 |
| `v7_fisher_decompose.py` | Fisher 正负不对称的对数分解 |
| `v7_gate_mechanism.py` | 门控为什么能超过 oracle |
| `v7_m8c_lpft.py` | LP-FT：先探针再解冻微调，是否优于纯读出适配 |
| `v7_nc_geometry.py` | 失真是否让网络更接近神经坍缩几何 |
| `v7_probe_audit.py` | 探针可信度审计 |
| `v7_proc_sensitivity.py` | 读出对训练过程的敏感度标定 |
| `v7_readout_capacity.py` | 读出的容量阶梯：线性是不是天花板 |
| `v7_readout_samplesize.py` | 读出训练样本量的冗余曲线 |
| `v7_samplesize_ext.py` | 把读出样本量推到 50000，钉死 n* |

## 六、工具与批处理

| 文件 | 用途 |
|---|---|
| `build_ledger.py` | 全局实验台账生成器（只读扫描磁盘） |
| `plot_cross_arch_summary.py` | 跨架构汇总图（无模块 docstring） |
| `run_m14b_core.bat` | M14b 主链：ReLU/GELU × 2/5 pool |
| `run_m14b_leaky.bat` | M14b LeakyReLU 分支 |
| `run_m14b_seeds.bat` | M14b 补种子 |
| `run_m5_full.bat` | M5 完整链：训练 → 秩探针 → 3 个读出种子 |
| `run_m5_smoke.bat` | M5 1-epoch 冒烟 |
| `run_m8c.bat` | M8c LP-FT 三分支 |

> **注意**：BAT 脚本假设当前目录是仓库根目录；日志写到根目录的 `logs_*.log`。
> 整理日志目录时，必须同步更新 BAT 里的输出路径，否则下次运行会重新把日志写回根目录。
