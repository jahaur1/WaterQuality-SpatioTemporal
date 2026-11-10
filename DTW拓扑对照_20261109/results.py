import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.stdout.reconfigure(encoding='utf-8')
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']

def scalar_distance(x,y,hours):
    size=len(x);cost=np.full((size+1,size+1),np.inf);cost[0,0]=0
    for i in range(size):
        for j in range(size):
            if abs(hours[i]-hours[j])<=12:
                cost[i+1,j+1]=(x[i]-y[j])**2+min(cost[i,j],cost[i,j+1],cost[i+1,j])
    return np.sqrt(cost[-1,-1]/size)

e=pd.read_csv(HERE/'DTW关联站点.csv');raw=pd.read_csv(ROOT/'30天_750站点数据集_20250410-20250510/水质数据_750站点_20250410-20250510.csv',parse_dates=['监测时间']);raw=raw[raw['监测时间']<'2025-04-29']
groups={sid:g.set_index('监测时间').sort_index() for sid,g in raw.groupby('站点ID')}
for r in e.sample(52,random_state=9).to_dict('records'):
    v=pd.concat([groups[r['接收站']][r['指标']],groups[r['邻站']][r['指标']]],axis=1,join='inner').dropna()
    assert len(v)==r['同步样本数']
    assert v.index.to_series().diff().dropna().min()>=pd.Timedelta(hours=4)
    x=v.iloc[:,0].to_numpy();y=v.iloc[:,1].to_numpy();x=(x-x.mean())/x.std();y=(y-y.mean())/y.std();hours=(v.index-v.index[0]).total_seconds().to_numpy()/3600
    np.testing.assert_allclose(scalar_distance(x,y,hours),r['标准化DTW距离'],atol=1e-10)
old=pd.read_csv(ROOT/'水质跨站预测_20261109/final_关联站点.csv').rename(columns={'化学指标':'指标'})
o=old.groupby(['接收站','指标']).agg(n=('邻站','count'),mass=('权重','sum'));d=e.groupby(['接收站','指标']).agg(n=('邻站','count'),mass=('权重','sum'));pd.testing.assert_index_equal(o.index,d.index);np.testing.assert_array_equal(o.n,d.n);np.testing.assert_allclose(o.mass,d.mass,rtol=1e-12)
z=np.load(HERE/'predictions.npz');p=np.load(ROOT/'预训练全站拓扑LoRA_20261109/predictions.npz');np.testing.assert_array_equal(z['truth'],p['truth'])
idx=pd.read_csv(ROOT/'冻结预训练模型校正头基线_20261106/配对索引.csv');idx=idx[idx.test].reset_index(drop=True)
values={'仅自身连接':p['LoRA仅自身'],'Pearson拓扑':p['LoRA加联合图'],'DTW拓扑':z['LoRA加联合图']};rows=[]
for name,pred in values.items():
    pred=np.maximum(pred,0)
    for sid,ix in idx.groupby('站点ID').indices.items():
        y=z['truth'][ix];err=pred[ix]-y;sst=np.square(y-y.mean(0)).sum(0);tol=1e-12*np.maximum(1,np.square(y).sum(0));r2=1-np.divide(np.square(err).sum(0),sst,out=np.full(4,np.nan),where=sst>tol)
        for j,c in enumerate(CHEM):rows.append([name,sid,c,np.abs(err[:,j]).mean(),np.sqrt(np.square(err[:,j]).mean()),r2[j]])
m=pd.DataFrame(rows,columns=['模型','站点ID','指标','MAE','RMSE','R2']);m.to_csv(HERE/'逐站点误差.csv',index=False,encoding='utf-8-sig')
s=m.groupby(['模型','指标'],sort=False).agg(平均R2=('R2','mean'),R2中位数=('R2','median'),MAE中位数=('MAE','median'),RMSE中位数=('RMSE','median')).reset_index();s.to_csv(HERE/'分指标比较.csv',index=False,encoding='utf-8-sig')
rows=[]
for name,g in m.groupby('模型',sort=False):rows.append([name,s.loc[s['模型'].eq(name),'平均R2'].mean(),g.loc[g.R2.gt(.5),'站点ID'].nunique(),g.loc[g.R2.gt(.8),'站点ID'].nunique()])
t=pd.DataFrame(rows,columns=['模型','四指标平均R2','任一项R2超过0.5站数','任一项R2超过0.8站数']);t.to_csv(HERE/'总体比较.csv',index=False,encoding='utf-8-sig')
info=json.loads((HERE/'建图记录.json').read_text(encoding='utf-8'))
report='''# Pearson与DTW拓扑的受控比较

原Pearson图逐指标使用训练期同一时刻实测值，计算中心化向量的内积除以各自长度，反映同步线性变化。要求至少16对、r≥0.5，按r*n/(n+16)每站每指标取最多3个邻站。

DTW使用相同流域候选和训练截止值4月29日前的观测。对每对站取共同监测时刻，至少16对，对两条序列分别做z标准化。动态规划累积平方差，路径允许前进、横移和纵移，但对齐的两个监测时刻相差最多12小时。距离定义为sqrt(最小累积平方差/同步样本数)，越小表示允许错位后曲线形状越相近。常量序列排除。该12小时限制是本轮预设值，未按测试误差选择。

为控制图密度与消息强度，每站每指标沿用原Pearson图的邻站数量，按DTW距离重选邻站。候选取自同流域全部符合共同观测数要求的站点，允许选择Pearson相关系数较低的站点。原先没有关联的站点–指标仍保留零邻站。权重按exp(-distance)*n/(n+16)分配，再缩放至原Pearson图该站该指标的总权重。因此每站的自连接权重和邻站总消息强度相同。该实验比较同等密度下的DTW选边，未用DTW距离阈值确定邻站数。

模型为此前Chronos-2末两层LoRA加全站隐藏图，使用相同种子、24次全图更新、学习率、相同起报时刻采样和23,136个测试配对。DTW仅用于训练期建图，预测阶段不读取未来化学值，也不对预测时的历史做未来对齐。模型仍同时处理750站。

动态规划递推参考DTAIDistance官方文档，时间约束和按长度标准化为本试验实现设置。https://dtaidistance.readthedocs.io/en/latest/modules/dtw.html

'''
report+=t.to_string(index=False)+'\n\n'+s.to_string(index=False)+'\n\n图统计\n'+json.dumps(info,indent=2)
report+='\n\n核验了52条随机边的批量DTW与独立完整动态规划结果，逐站逐指标的邻站数及总权重与原图一致。测试真值逐值相同。DTW识别允许错位的形状相似性，河网方向与物理传播时间需另外核实。当前固定12小时配置的结果代表本轮试验。\n'
(HERE/'实验报告.md').write_text(report,encoding='utf-8')
(HERE/'核验.json').write_text(json.dumps(dict(passed=True,independent_DTW_pairs=52,degrees_and_mass_matched=True,test_truth_identical=True),indent=2),encoding='utf-8')
print(t.to_string(index=False));print(s.to_string(index=False));print(info)
