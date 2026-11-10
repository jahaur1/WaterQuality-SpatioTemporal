# 复现与本地资源

本仓库是现有研究代码与结果的整理快照，保留实验目录以维持相对导入。它不是已经整理成统一命令行包的完全自包含发行版。

## 可直接查看

README、各实验报告、总体/分指标/逐站派生误差表、训练记录、验证记录、文献与源码核查说明。部分大表及逐样本预测未发布。汇总表可以由`python tools/summarize_results.py`读取，无需原始数据或GPU。

`python tools/verify_snapshot.py`检查Python语法、快照文件哈希、密钥形态、文件大小和原始数据/模型权重是否误入导出清单，不会运行训练或导入下载的第三方代码。

## 重跑需要补齐

1. 选定30天CSV放到原目录名`30天_750站点数据集_20250410-20250510`；原始25个月数据未上传。
2. 历史窗口缓存、精确配对索引、站点训练标准化统计等需由原数据重建，或从已有本地工作目录连接。部分旧准备脚本还依赖历史基准缓存及其配对顺序，尚非从零开始的单一数据入口。
3. Chronos-2权重在`Chronos2水质24小时评估_20261103/weights`，该目录不入Git。模型脚本通常设置本地离线加载。
4. 安装匹配的PyTorch/CUDA、Chronos、PEFT和数值依赖，或连接原实验隔离的`runtime`/`deps`。TimesFM、Timestral、DAG和COSA等另有其上游依赖；核心requirements不覆盖全部历史模型。
5. 重新生成报告需要预测NPZ；部分深度核验需要保存的PT、原始上游源码和缓存。这些资源均不随此次提交发布。

明确的资源清单见`docs/local_assets.json`。已有原工作目录时：

```powershell
python tools/link_local_assets.py --workspace "D:\path\to\original_workspace"
python tools/link_local_assets.py --workspace "D:\path\to\original_workspace" --apply
```

默认仅预览，`--apply`才创建符号链接；已有路径不会覆盖。Windows创建目录符号链接可能需要开发者模式或相应权限，失败时可按清单手动放置资源；不要将凭据、运行环境、权重或原数据加入Git。工具不删除、移动原资源，也不自动安装依赖。

补齐环境与资源后，最新实验入口示例：

```powershell
python 无邻站同结构对照_20261110/run.py
python 无邻站同结构对照_20261110/results.py

python 时滞邻站对照_20261110/build_lags.py
python 时滞邻站对照_20261110/audit.py
python 时滞邻站对照_20261110/run.py
python 时滞邻站对照_20261110/results.py
```

报告中的历史绝对路径和源文件SHA256是原实验溯源信息，不表示其他机器具有相同路径。第三方源码未核实许可的部分只保留官方来源，不直接再分发。
