# XLinear水质软测量适配

本目录用于用户确认的“测5项、估4项、共9项”研究。保留XLinear原版Forcast_multi骨干，适配独立4目标回归头；不是原生含化学目标历史的未来预测任务复现。

输入仅同站当前及最近5次的水温、pH、溶解氧、电导率、浊度；同期四化学标签不进入输入。每站独立模型，训练期静态X/Y标准化，需要历史配对化学标签作为校准数据。

## 结果入口

- [全量报告](results/水质软测量评估报告.md)
- [总体与岭回归比较](results/03_总体与岭回归对比.csv)
- [逐站点四目标误差](results/01_逐站点四目标误差.csv)
- [独立验证](results/独立验证.json)

results为750站全量运行，pilot为3站代码烟测，不能以pilot替代全量结论。所有配置、环境、源码和数据SHA256都记录在对应目录。四个化学指标的验收误差尚未定义，因此不报告推导成功率。

## 复现

本机已验证环境为 D:/AI/conda-envs/tato/python.exe（具体版本见results/environment.json），无需改动原仓库或既有ETT结果。以下命令在本目录运行：

```powershell
& 'D:\AI\conda-envs\tato\python.exe' -B .\run_water.py --epochs 100 --patience 10 --output results
& 'D:\AI\conda-envs\tato\python.exe' -B .\verify_water.py
& 'D:\AI\conda-envs\tato\python.exe' -B .\make_water_report.py --results results
```

run_water按逐站完成标记恢复相同配置的已有结果。改变模型、训练代码、数据或配置后请使用新的--output目录，不能把修改后的运行混入旧缓存。

## 模型与实验适配

- vendor_XLinear.py保留原仓库模型文件逐字副本，出处见source_provenance.json；adapter.py只使用其中Forcast_multi骨干。
- 6次观测不是固定6小时，不做插值，窗口不会跨站，也不会包含未来观测。
- 4月25日前拟合、4月25日至30日前验证选择epoch；随后重初始化并用4月30日前全部训练资料按选定epoch重拟合；4月30日至5月10日00:00最终测试。
- X/Y标准化仅使用各阶段相应训练期；验证标准化四目标平均MSE控制早停，原单位报告另作负浓度截零并保留未截断值。
- 与岭回归使用完全相同测试观测，但XLinear多使用5次易测历史，不能当作只替换模型的严格消融。

本轮全量四目标的MAE中位数均高于岭回归；这是一组配置的结果，不代表所有XLinear设置或其他非线性方法均无效。当前30天小样本且已做过相关探索，需要更多独立时段验证。
