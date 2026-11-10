from pathlib import Path
import json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
df=pd.read_csv(BASE/'summary.csv')
assert len(df)==8
verification=[]
for r in df.itertuples():
    run=BASE/f'{r.dataset}_{r.pred_len}'
    pred=np.load(next((run/'results').iterdir())/'pred.npy',mmap_mode='r')
    assert pred.shape==(r.test_windows,r.pred_len,7)
    assert all(np.isfinite(pred[i:i+128]).all() for i in range(0,len(pred),128))
    log=(run/'train.log').read_text(encoding='utf-8')
    logged=re.findall(r'mse:([\d.eE+-]+), mae:([\d.eE+-]+)',log)[-1]
    assert abs(float(logged[0])-r.mse)<2e-6
    assert abs(float(logged[1])-r.mae)<2e-6
    verification.append({'dataset':r.dataset,'pred_len':r.pred_len,'shape':list(pred.shape),'finite':True,'metrics_match_training_log':True})
(BASE/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
for ax,key,title in zip(axes,['mse','mae'],['MSE (standardized)','MAE (standardized)']):
    for name,color in [('ETTh1','#2563eb'),('ETTm2','#ea580c')]:
        sub=df[df.dataset==name]
        ax.plot(sub.pred_len,sub[key],marker='o',color=color,label=name,linewidth=2)
    ax.set(title=title,xlabel='Forecast horizon (steps)',xticks=[96,192,336,720])
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
fig.suptitle('XLinear | input length 96 | 7-variable forecasting | seed 2025')
fig.savefig(BASE/'error_summary.png',dpi=180)
plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(11,6),layout='constrained')
for ax,name in zip(axes,['ETTh1','ETTm2']):
    sub=pd.read_csv(BASE/f'{name}_96/OT_example.csv')
    ax.plot(sub.step,sub.OT_true,label='Observed',color='#334155',linewidth=1.6)
    ax.plot(sub.step,sub.OT_prediction,label='XLinear',color='#2563eb',linewidth=1.6)
    ax.set(title=f'{name}: first test window, 96-step OT forecast',ylabel='OT (original units)',xlabel='Forecast step')
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
fig.savefig(BASE/'OT_forecast_examples.png',dpi=180)
plt.close(fig)

lines=['# XLinear：ETTh1 / ETTm2 实测结果','',
'测试日期：2026-11-02。使用用户提供的本地 XLinear 实现及 TATO 数据集，运行两个数据集、四个预测长度，共 8 组训练与测试。','',
'## 结果','',
'MSE 和 MAE 越小越好。以下为训练集 StandardScaler 标准化后的尺度，汇总全部测试窗口、全部预测步和全部 7 个变量；不是原始温度单位，也不是只预测 OT。','',
'| 数据集 | 输入长度 | 预测长度 | 测试窗口数 | MSE | MAE | 最后值基线 MSE | MSE 降低 |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for r in df.itertuples():
    lines.append(f'| {r.dataset} | 96 | {r.pred_len} | {r.test_windows} | {r.mse:.4f} | {r.mae:.4f} | {r.naive_mse:.4f} | {r.mse_improvement_pct:.1f}% |')
lines += ['', '![误差曲线](error_summary.png)','',
'最后值基线将每个输入窗口的最后一个观测值复制到整个预测区间，使用与 XLinear 完全相同的测试窗口与标准化方式。它只用于判断模型是否学到有效预测规律，不能据此认定优于其他训练模型。','',
'## 实验设置','',
'- 使用 `script/multi_forcatsing/etth1.sh` 和 `ettm2.sh` 中的各预测长度配置，保留模型宽度、门控维度、dropout、优化器、学习率、训练轮数和早停参数。随机种子 2025，每组运行 1 次。',
'- `features=M`，输入 96 步，7 个变量输入并输出 7 个变量。ETTh1 为小时数据，ETTm2 为 15 分钟数据；两个数据集的同一步数代表不同时间跨度。',
'- 按原 ETT 加载器使用 12×30 天训练、4×30 天验证、4×30 天测试，时间顺序不打乱划分；验证和测试可引用之前的 96 个历史点作为输入，预测标签不跨分区。原文件最后超出 20×30 天的记录按原协议不参与实验。',
'- ETTh1 分界点：8640 / 11520 / 14400；ETTm2 分界点：34560 / 46080 / 57600。标准化参数仅在训练段拟合。',
'- 最优权重仅由验证误差选取。原训练循环逐轮打印测试损失；此次没有用测试结果调参或选模型。保留原验证加载器的 shuffle=True、drop_last=True 行为。',
'- 测试加载器改为 drop_last=False，覆盖所有测试窗口；相应将测试预测批次拼接改为 concatenate，支持最后一个不足整批的批次。最终指标自行按所有元素累计复核，避免不等批次的平均偏差。',
'- Windows 设置 num_workers=0，并限制 CPU 线程数；仅保存首个测试窗口的原项目示例图并关闭图窗。ETTm2 的时间标记频率设为 t；模型本身不使用时间标记。',
'- 仅在当前工作区的源码副本上进行这些评估适配；用户原始源码和原始数据未修改。',
'- GPU：NVIDIA GeForce RTX 5060 Ti；PyTorch 2.11.0+cu128，NumPy 1.26.4，pandas 2.1.4。环境与原 requirements 中 PyTorch 2.6.0 等版本不同，结果不应标为论文精确复现。','',
'## 预测示例','',
'以下只展示每个数据集首个测试窗口的 OT 预测，已恢复原始尺度。单个窗口不能代表整体性能，整体结论应参考上表。','',
'![预测示例](OT_forecast_examples.png)','',
'## 结果解读','',
f"8 组测试全部完成，MSE 相比最后值基线降低 {df.mse_improvement_pct.min():.1f}%–{df.mse_improvement_pct.max():.1f}%。在本次设置下，模型具有明显超出最后值基线的预测能力。",'',
'本次是单随机种子测试，没有统计不同种子的均值和方差，也没有训练 DLinear、RLinear 等模型作同协议对照；不能由此宣称优于这些模型或复现论文全部结论。不同数据集的标准化误差也不能直接解释为原始物理量误差。','',
'## 文件与复跑','',
'- `summary.csv`：8 组结果，包含 MSE、MAE、RMSE、基线与运行时长。',
'- `environment.json`：环境、数据行数、缺失值及 SHA256。',
'- 每组目录：`command.json` 完整参数、`train.log` 训练过程、`metrics.json` 复核指标、`checkpoints/` 最优权重、`results/` 预测数组、`OT_example.csv` 原尺度 OT 示例。',
'- `source/`：评估所用源码副本。',
'- 使用 `D:\\AI\\conda-envs\\tato\\python.exe -B -u run_benchmark.py` 可运行；脚本会跳过已有 metrics.json 的完成项。要重跑某组，请先将该组整个目录改名留存后再启动。','']
(BASE/'评估报告.md').write_text('\n'.join(lines),encoding='utf-8')
print(df.to_string(index=False))
print('Report and charts saved.')
