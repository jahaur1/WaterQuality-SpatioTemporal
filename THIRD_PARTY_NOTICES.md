# 第三方来源与许可范围

本项目包含原创任务适配、实验流程、统计分析，以及对公开研究代码的引用和少量带原许可的改编。此说明不为第三方或原创代码新增统一许可。

- **Graph WaveNet**：https://github.com/nnzhan/Graph-WaveNet 。原提交6b162e80c59a1d494809252eca055cff93dc66b1；保留最小`source/model.py`和原MIT许可。用于潜在表示图传播与融合，时间编码仍由本项目选用的模型承担。
- **DAG / DLinear**：来源提交、文件SHA256和修改见两个DLinear_DAG目录的`source_provenance.json`。DAG来源MIT许可和DLinear原许可随目录保存。`vendor_dag`、`vendor_DLinear.py`与`dag_native_layers.py`为相应来源的复制或适配，原作者权利保留。
- **XLinear**：保留实验适配和来源SHA256记录。由于此次未在本地上游快照中核实再分发许可，`vendor_XLinear.py`及整套上游源码暂不发布，可从有权使用的原工作目录补齐。
- **COSA**：https://github.com/bigbases/COSA_ICLR2026 。原快照的CC BY-NC-SA 4.0许可保留在COSA实验目录。此仓库包含水质适配及报告，没有重新发布整套COSA下载源码；适配部分的使用也应遵守适用许可条件。
- **Chronos**：https://github.com/amazon-science/chronos-forecasting 。模型权重与隔离运行时不随仓库发布；本项目记录了官方LoRA和任务适配细节。
- **其他预训练模型**：TimesFM和Timestral的下载来源、版本及SHA256见相应下载记录与实验配置；这些模型权重未发布。
- **HydroRIVERS**：https://www.hydrosheds.org/products/hydrorivers 。原亚洲河网数据、下载ZIP和转换工具依赖未发布；保留构图代码、来源SHA256、站点候选匹配与派生小型图矩阵。数据使用遵循上游条款。
- **Zhang等，2026，Transactions in GIS**：论文 https://doi.org/10.1111/tgis.70309 ，作者仓库 https://github.com/GeoScene-Lab/FlowAware-TGCN-WaterQuality 。本仓库仅发布本项目编写的核查说明、小例检查脚本及同名筛查结果，未将对方无明确许可的源码副本直接发布。

原始水质数据未因这次上传而改变其权利或可分发范围。派生分析与粗估地图匹配中的站点名称属于研究元数据；粗定位和统计关联不是经核实的物理真值。
