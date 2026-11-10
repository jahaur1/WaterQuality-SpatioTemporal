import json
import hashlib
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.stdout.reconfigure(encoding='utf-8')
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']
def table(frame):
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)

records=json.loads((HERE/'训练记录.json').read_text(encoding='utf-8'));assert len(records)==3
assert json.loads((HERE/'作者源码一致性.json').read_text(encoding='utf-8'))['passed']
z=np.load(HERE/'predictions.npz');old=np.load(ROOT/'预训练全站拓扑LoRA_20261109/predictions.npz');np.testing.assert_array_equal(z['truth'],old['truth'])
idx=pd.read_csv(ROOT/'冻结预训练模型校正头基线_20261106/配对索引.csv');idx=idx[idx.test].reset_index(drop=True)
rows=[]
for name in records:
    p=np.maximum(z[name],0)
    for sid,ix in idx.groupby('站点ID').indices.items():
        y=z['truth'][ix];err=p[ix]-y;sst=np.square(y-y.mean(0)).sum(0);tol=1e-12*np.maximum(1,np.square(y).sum(0));r2=1-np.divide(np.square(err).sum(0),sst,out=np.full(4,np.nan),where=sst>tol)
        for j,c in enumerate(CHEM):rows.append([name,sid,c,np.abs(err[:,j]).mean(),np.sqrt(np.square(err[:,j]).mean()),r2[j]])
m=pd.DataFrame(rows,columns=['模型','站点ID','指标','MAE','RMSE','R2']);m.to_csv(HERE/'逐站点误差.csv',index=False,encoding='utf-8-sig')
s=m.groupby(['模型','指标'],sort=False).agg(平均R2=('R2','mean'),R2中位数=('R2','median'),MAE中位数=('MAE','median'),RMSE中位数=('RMSE','median')).reset_index();s.to_csv(HERE/'分指标比较.csv',index=False,encoding='utf-8-sig')
totals=[]
for name,g in m.groupby('模型',sort=False):totals.append([name,s.loc[s['模型'].eq(name),'平均R2'].mean(),g.loc[g.R2.gt(.5),'站点ID'].nunique(),g.loc[g.R2.gt(.8),'站点ID'].nunique()])
t=pd.DataFrame(totals,columns=['模型','四指标平均R2','任一项R2超过0.5站数','任一项R2超过0.8站数']);t.to_csv(HERE/'总体比较.csv',index=False,encoding='utf-8-sig')
git=['git','-c','safe.directory='+str((HERE/'source').as_posix())]
commit=subprocess.check_output(git+['rev-parse','HEAD'],cwd=HERE/'source',text=True).strip()
assert not subprocess.check_output(git+['status','--porcelain'],cwd=HERE/'source',text=True).strip()
metadata=pd.read_csv(ROOT/'30天_750站点数据集_20250410-20250510/水质数据_750站点_20250410-20250510.csv').drop_duplicates('站点ID').set_index('站点ID');nodes=np.load(ROOT/'全站拓扑联合预测_20261109/拓扑与站点.npz')['station_ids'];basins=metadata.loc[nodes,'流域'].to_numpy();outside=basins[:,None]!=basins[None,:]
graph_info=[]
for name in ['GWN自适应图','GWN固定加自适应']:
    a=np.load(HERE/f'{name}_学习矩阵.npy');np.testing.assert_allclose(a.sum(1),1,atol=1e-6);assert (a>=0).all()
    state=torch.load(HERE/f'{name}.pt',map_location='cpu',weights_only=True)['graph'];v1=state['nodevec1'];v2=state['nodevec2'];expected=torch.softmax(torch.relu(v1@v2),dim=1).numpy();np.testing.assert_allclose(a,expected,atol=1e-6)
    graph_info.append(dict(model=name,shape=list(a.shape),mean_cross_basin_weight=float(a[outside].sum()/750)))
audit=dict(passed=True,official_commit=commit,source_sha256=hashlib.sha256((HERE/'source/model.py').read_bytes()).hexdigest(),source_checkout_clean=True,test_pairs=len(idx),same_truth=True,graphs=graph_info)
np.savez_compressed(HERE/'学习拓扑与站点.npz',station_ids=nodes,source_receiver_orientation=np.array(True),
                    adaptive_only=np.load(HERE/'GWN自适应图_学习矩阵.npy'),
                    adaptive_in_hybrid=np.load(HERE/'GWN固定加自适应_学习矩阵.npy'))
(HERE/'核验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
report=f'''# Graph WaveNet作者图模块与水质联合预测

官方源码 https://github.com/nnzhan/Graph-WaveNet ，固定提交 {commit}。source/model.py保持原样。直接调用作者nconv、linear、gcn类。自适应矩阵按原始公式A_adp=softmax(ReLU(E1@E2),dim=1)，E1为750×10，E2为10×750，按作者aptinit默认思路用固定图SVD初始化。

在Chronos-2末两层LoRA联合预测框架中，把750站的4项化学未来隐藏表示经768→64投影，送入作者图卷积。其输出拼接自身、一次和二次图传播，经过作者1×1卷积和训练期dropout0.3，再映射回768维，门控残差后交由预训练输出层解码。固定图模式支持一个固定矩阵，自适应图模式支持一个学习矩阵，混合图模式同时使用两者。这轮参考作者的空间模块，时间编码继续使用当前水质预训练模型。

作者nconv使用A[源站,接收站]，因此当前按接收站归一化的固定矩阵转置后传入。自适应矩阵遵循作者行softmax，按源站分配信息。额外投影和门控为水质隐藏表示接入所需。原始WaveNet时间卷积没有进入此轮试验。

自适应图在750站全域学习，允许跨流域连接。训练不施加流域稀疏掩码。固定图来源仍为4月29日前数据估计的同流域Pearson关系。三组使用相同种子、24次全图更新、相同窗口采样和23,136个测试配对。图模块学习率.001，LoRA学习率1e-5，LoRA r8/alpha16作用末两层q/k/v/o。预算与参数为本轮预设，未用测试选参。混合图因拼接两个支持矩阵而拥有更多图卷积参数，参数量记录于训练记录.json。

## 结果

{table(t)}

{table(s)}

平均R²先在每个指标内对可计算站点平均，再对四项等权平均。预测负浓度统一截为0，有限负R²保留。所有方案均使用化学历史实测与目标时刻易测实测条件。学习矩阵中的边表示预测模型的数据依赖。

## 源码与矩阵核验

官方gcn执行位置、源到接收方向、SVD初始站点向量、自适应公式均与作者原始类核对。导出的学习矩阵与保存站点向量重新计算结果一致，矩阵行和为1。源码Git检出保持干净，固定提交和SHA256记录于核验.json。原测试真值与之前联合实验相同。学习图为稠密矩阵，来源于24次更新，本轮效果需按该固定预算解释。
'''
(HERE/'实验报告.md').write_text(report,encoding='utf-8')
print(t.to_string(index=False));print(s.to_string(index=False));print(graph_info)
