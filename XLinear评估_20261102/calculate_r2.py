from pathlib import Path
import sys,re,json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'source'))
from data_provider.data_loader import Dataset_ETT_hour,Dataset_ETT_minute
DATA=Path(r'D:\AI\TATO-main\DATASET\ETT-small')
rows=[]
details=[]
for r in pd.read_csv(BASE/'summary.csv').itertuples():
    cls=Dataset_ETT_hour if r.dataset=='ETTh1' else Dataset_ETT_minute
    ds=cls(str(DATA),flag='test',size=[96,48,r.pred_len],features='M',data_path=r.dataset+'.csv',timeenc=1,freq='h' if r.dataset=='ETTh1' else 't')
    run=BASE/f'{r.dataset}_{r.pred_len}'
    pred=np.load(next((run/'results').iterdir())/'pred.npy',mmap_mode='r')
    assert pred.shape==(len(ds),r.pred_len,7)
    totals=np.zeros(7)
    squares=np.zeros(7)
    sse=np.zeros(7)
    n=0
    for start in range(0,len(ds),128):
        end=min(start+128,len(ds))
        true=np.stack([ds.data_y[i+96:i+96+r.pred_len] for i in range(start,end)]).astype(np.float32).astype(np.float64)
        pp=np.asarray(pred[start:end],dtype=np.float64)
        totals+=true.sum(axis=(0,1))
        squares+=np.square(true).sum(axis=(0,1))
        sse+=np.square(pp-true).sum(axis=(0,1))
        n+=true.shape[0]*true.shape[1]
    sst=squares-totals**2/n
    per=1-sse/sst
    pooled=1-sse.sum()/(squares.sum()-totals.sum()**2/(n*7))
    variance_weighted=1-sse.sum()/sst.sum()
    logged=float(re.findall(r'nse:([\d.eE+-]+)',(run/'train.log').read_text(encoding='utf-8'))[-1])
    assert abs(pooled-logged)<3e-6
    if r.dataset=='ETTh1' and r.pred_len==96:
        true=np.stack([ds.data_y[i+96:i+96+r.pred_len] for i in range(len(ds))]).astype(np.float32).astype(np.float64)
        pp=np.asarray(pred,dtype=np.float64)
        np.testing.assert_allclose(per,r2_score(true.reshape(-1,7),pp.reshape(-1,7),multioutput='raw_values'),atol=1e-10)
        np.testing.assert_allclose(pooled,r2_score(true.ravel(),pp.ravel()),atol=1e-10)
    rows.append(dict(dataset=r.dataset,pred_len=r.pred_len,r2_pooled_standardized=pooled,r2_mean_per_variable=per.mean(),r2_OT=per[-1],r2_variance_weighted_standardized=variance_weighted))
    columns=list(pd.read_csv(DATA/(r.dataset+'.csv'),nrows=0).columns[1:])
    details.extend(dict(dataset=r.dataset,pred_len=r.pred_len,variable=col,r2=value) for col,value in zip(columns,per))
summary=pd.DataFrame(rows)
summary.to_csv(BASE/'r2_summary.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(details).to_csv(BASE/'r2_per_variable.csv',index=False,encoding='utf-8-sig')
lines=['# XLinear R² 测试结果','','使用已保存的模型预测，重建同一测试集真值，无需重新训练。每个预测步与每个滑动窗口作为一个预测元素，重叠时间点按原实验的窗口协议重复计入。','','| 数据集 | 预测长度 | 整体 R²（原代码口径） | 7 变量平均 R² | OT 单变量 R² |','|---|---:|---:|---:|---:|']
for r in summary.itertuples():
    lines.append(f'| {r.dataset} | {r.pred_len} | {r.r2_pooled_standardized:.4f} | {r.r2_mean_per_variable:.4f} | {r.r2_OT:.4f} |')
lines+=['','计算公式为 R² = 1 − SSE/SST，基准为测试真值的均值，而非训练均值。R² 越接近 1 越好；0 对应使用测试真值均值的常数预测；负数表示比该常数基准还差。','','整体 R²：与原项目 utils/metrics.py 的 R2 函数一致，把训练集标准化后的全部变量与窗口展平，使用一个全局均值。原代码的 NSE 与 R² 公式相同，故该列也对应训练日志中的 NSE。已对全部 8 组逐项核验一致（浮点容差 3e-6）。不同变量混合后的整体 R² 受变量间均值差异影响，不等于每个变量都达到同等预测精度。','','7 变量平均 R²：先对每个变量分别计算 R²，再等权平均，等价于将 [窗口×预测步,7] 数组传给 sklearn.metrics.r2_score 的默认多输出平均口径。逐变量 R² 对各变量的线性缩放和平移不变，因此该平均在标准化尺度与原始尺度一致。另提供标准化尺度的方差加权平均，见 r2_summary.csv。','','OT R²：多变量预测模型对 OT 输出的单变量分数，并不是另行训练外生输入单目标模型的结果。','','论文截图 tab4.png 未报告 R²，因此此处不能进行论文 R² 数值对照。首组结果已用 sklearn.metrics.r2_score 独立交叉核验。完整逐变量结果见 r2_per_variable.csv。','']
section='\n'.join(lines)
(BASE/'R2评估报告.md').write_text(section,encoding='utf-8')
print(summary.to_string(index=False))
print('\nPer variable:')
print(pd.DataFrame(details).pivot(index=['dataset','pred_len'],columns='variable',values='r2').round(4).to_string())
