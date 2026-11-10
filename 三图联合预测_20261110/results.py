import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.stdout.reconfigure(encoding='utf-8')
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']
def table(df):
    lines=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)
z=np.load(HERE/'predictions.npz');p=np.load(ROOT/'GraphWaveNet拓扑对照_20261109/predictions.npz');d=np.load(ROOT/'距离拓扑_20261109/predictions.npz')
np.testing.assert_array_equal(z['truth'],p['truth']);np.testing.assert_array_equal(z['truth'],d['truth'])
idx=pd.read_csv(ROOT/'冻结预训练模型校正头基线_20261106/配对索引.csv');idx=idx[idx.test].reset_index(drop=True)
values={'Pearson加自适应':p['GWN固定加自适应'],'距离加自适应':d['GWN固定加自适应'],'三图联合':z['三图联合']};rows=[]
for name,raw in values.items():
    pred=np.maximum(raw,0)
    for sid,ix in idx.groupby('站点ID').indices.items():
        y=z['truth'][ix];error=pred[ix]-y;sst=np.square(y-y.mean(0)).sum(0);tol=1e-12*np.maximum(1,np.square(y).sum(0));r2=1-np.divide(np.square(error).sum(0),sst,out=np.full(4,np.nan),where=sst>tol)
        for j,c in enumerate(CHEM):rows.append([name,sid,c,np.abs(error[:,j]).mean(),np.sqrt(np.square(error[:,j]).mean()),r2[j]])
m=pd.DataFrame(rows,columns=['模型','站点ID','指标','MAE','RMSE','R2']);m.to_csv(HERE/'逐站点误差.csv',index=False,encoding='utf-8-sig')
s=m.groupby(['模型','指标'],sort=False).agg(平均R2=('R2','mean'),R2中位数=('R2','median'),MAE中位数=('MAE','median'),RMSE中位数=('RMSE','median')).reset_index();s.to_csv(HERE/'分指标比较.csv',index=False,encoding='utf-8-sig')
total=[]
for name,g in m.groupby('模型',sort=False):total.append([name,s.loc[s['模型'].eq(name),'平均R2'].mean(),g.loc[g.R2.gt(.5),'站点ID'].nunique(),g.loc[g.R2.gt(.8),'站点ID'].nunique()])
t=pd.DataFrame(total,columns=['模型','四指标平均R2','任一项R2超过0.5站数','任一项R2超过0.8站数']);t.to_csv(HERE/'总体比较.csv',index=False,encoding='utf-8-sig')
g=m[m['模型'].eq('三图联合')].merge(m[m['模型'].eq('Pearson加自适应')],on=['站点ID','指标'],suffixes=('_三图','_两图'),validate='one_to_one');g['R2变化']=g.R2_三图-g.R2_两图;g['MAE降低量']=g.MAE_两图-g.MAE_三图;g.to_csv(HERE/'相对两图逐站点收益.csv',index=False,encoding='utf-8-sig')
checkpoint=torch.load(HERE/'三图联合.pt',map_location='cpu',weights_only=True)['graph']
pearson=np.load(ROOT/'全站拓扑联合预测_20261109/拓扑与站点.npz')['final'];distance=np.load(ROOT/'距离拓扑_20261109/距离拓扑.npz')['adjacency']
np.testing.assert_array_equal(checkpoint['fixed'].numpy(),pearson.T.astype(np.float32));np.testing.assert_array_equal(checkpoint['distance'].numpy(),distance.T.astype(np.float32))
# Original gcn concatenates identity plus 2 orders for each of the 3 supports.
assert checkpoint['gcn.mlp.mlp.weight'].shape==(64,64*7,1,1)
adaptive=torch.softmax(torch.relu(checkpoint['nodevec1']@checkpoint['nodevec2']),dim=1).numpy();np.testing.assert_allclose(adaptive.sum(1),1,atol=1e-6)
np.savez_compressed(HERE/'三张拓扑与站点.npz',station_ids=np.load(ROOT/'全站拓扑联合预测_20261109/拓扑与站点.npz')['station_ids'],pearson_source_receiver=checkpoint['fixed'].numpy(),distance_source_receiver=checkpoint['distance'].numpy(),adaptive_source_receiver=adaptive)
record=json.loads((HERE/'训练记录.json').read_text(encoding='utf-8'))['三图联合'];assert record['steps']==24
audit=dict(passed=True,three_supports_verified=True,source_receiver_direction_verified=True,convolution_input_channels=448,same_test_pairs=23136,steps=24,parameters=record['trainable'])
(HERE/'核验.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
report='''# Pearson、距离与自适应三图联合预测

同时向Graph WaveNet作者图卷积提供三张750×750矩阵：训练期Pearson相关图、公开地图粗定位距离图、训练中学习的自适应图。每张图分别计算一次和二次传播，与自身表示拼接，共7组64维特征，再由作者1×1卷积学习融合。矩阵分别保存，不预先平均。

三图模型继续使用Chronos-2末两层LoRA和全站隐藏表示联合解码。随机种子20261109、24次全图更新、起报采样和学习率与此前两图相同，测试为原750站的23,136个精确24小时配对。三图卷积比两图增加8,192个参数。距离图仍来自249站粗定位，241站具有距离邻边。其余站点在距离分支保留自身连接，仍通过Pearson和自适应分支参与联合预测。

'''+table(t)+'\n\n'+table(s)
report+='\n\n已核对保存权重中的两张固定图与原矩阵逐值一致，自适应矩阵行和为1，图卷积输入448维符合3图各2阶加自身的拼接方式。测试真值逐值一致。平均R²按各指标逐站平均后四项等权计算，有限负值保留。阈值按任一指标达标的站点去重。距离位置仍为待人工核实的粗估，当前结果属于固定24次更新预算下的单种子探索性比较。\n'
(HERE/'实验报告.md').write_text(report,encoding='utf-8');print(t.to_string(index=False));print(s.to_string(index=False));print(audit)
