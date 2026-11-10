# XLinear精确24小时条件预测

本轮用户确认：使用预测起点之前的5项易测＋4项化学实测历史，并提供恰好24小时后5项易测真值，预测该时刻4项化学指标。历史化学每次真实更新。全部750站保留。

完整信息边界、时间隔离和不能据此宣称降本的限制见[实验口径](实验口径.md)。

## 最终结果入口

- [评估报告](results/24小时条件预测评估报告.md)
- [两分支逐站点四目标误差](results/01_逐站点两分支四目标误差.csv)
- [未来信息逐站增益对照](results/07_逐站点未来信息增益对照.csv)
- [核心比较汇总](results/08_核心比较汇总.csv)
- [独立验证](results/独立验证.json)

results是唯一最终统计目录。pilot为3站结构试跑；pairing_check为只配对检查；gpu_partial_timing是被中止的GPU计时运行，不参与最终统计。最终results使用CPU，复用同参数CPU三站计时的已完成结果后完成其余站点，没有混用GPU训练权重。改用CPU仅因执行更快，未根据测试成绩改变配置、选择设备或模型。

## 复现

本机环境为D:/AI/conda-envs/tato/python.exe，版本记录于results/environment.json。以下在本目录执行：

```powershell
& 'D:\AI\conda-envs\tato\python.exe' -B .\run_forecast24.py --epochs 100 --patience 10 --cpu --output results
& 'D:\AI\conda-envs\tato\python.exe' -B .\verify_exact24.py
& 'D:\AI\conda-envs\tato\python.exe' -B .\make_report24.py --results results
```

相同代码、数据和配置可从done.json恢复已完成的站点—分支。改变数据、代码、超参数或执行设备时请使用新的--output目录，不要混入已有缓存。

模型保留原生XLinear M模式、历史9通道和窗口归一化，另增未来5项易测分支。两条件各自验证选epoch后重新初始化重拟合；无未来分支是同历史的消融，而不是将不同信息条件的旧同期软测量拿来直接比较。

23,136组独立测试配对在两分支各保存一次，总预测表46,272行，不能将其算作46,272个独立测试样本。相邻窗口同站重叠，样本也不是独立重复试验。没有业务允许误差时不计算可推导成功率。
