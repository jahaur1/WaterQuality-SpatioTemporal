import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.stdout.reconfigure(encoding='utf-8')
def table(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])

z=np.load(HERE/'predictions.npz')
old=np.load(ROOT/'三图联合预测_20261110/predictions.npz')
base=np.load(ROOT/'GraphWaveNet拓扑对照_20261109/predictions.npz')
np.testing.assert_array_equal(z['truth'],old['truth'])
np.testing.assert_array_equal(z['truth'],base['truth'])
index=pd.read_csv(ROOT/'冻结预训练模型校正头基线_20261106/配对索引.csv')
index=index[index.test].reset_index(drop=True)
chem=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']
arms={'Pearson加自适应':base['GWN固定加自适应'],'三图_直线距离':old['三图联合'],'三图_单位对照':z['三图_单位对照'],'三图_河网探索':z['三图_河网探索']}
rows=[]
for arm,pred in arms.items():
    pred=np.maximum(pred,0)
    for sid,ii in index.groupby('站点ID').indices.items():
        y=z['truth'][ii];e=pred[ii]-y
        sst=np.square(y-y.mean(0)).sum(0)
        r2=1-np.divide(np.square(e).sum(0),sst,out=np.full(4,np.nan),where=sst>1e-12*np.maximum(1,np.square(y).sum(0)))
        for j,c in enumerate(chem):rows.append([arm,sid,c,len(ii),np.abs(e[:,j]).mean(),np.sqrt(np.square(e[:,j]).mean()),r2[j]])
m=pd.DataFrame(rows,columns=['模型','站点ID','指标','测试对数','MAE','RMSE','R2'])
m.to_csv(HERE/'逐站点误差.csv',index=False,encoding='utf-8-sig')
s=m.groupby(['模型','指标'],sort=False).agg(平均R2=('R2','mean'),R2中位数=('R2','median'),MAE中位数=('MAE','median')).reset_index()
s.to_csv(HERE/'分指标比较.csv',index=False,encoding='utf-8-sig')
t=[]
for arm,g in m.groupby('模型',sort=False):
    t.append([arm,s.loc[s['模型'].eq(arm),'平均R2'].mean(),g.loc[g.R2>.5,'站点ID'].nunique(),g.loc[g.R2>.8,'站点ID'].nunique()])
t=pd.DataFrame(t,columns=['模型','四指标平均R2','任一项R2超过0.5站数','任一项R2超过0.8站数'])
t.to_csv(HERE/'总体比较.csv',index=False,encoding='utf-8-sig')
g=m[m['模型'].eq('三图_河网探索')].merge(m[m['模型'].eq('三图_单位对照')],on=['站点ID','指标'],suffixes=('_河网','_对照'),validate='one_to_one')
g['R2变化']=g.R2_河网-g.R2_对照;g['MAE降低量']=g.MAE_对照-g.MAE_河网
g.to_csv(HERE/'相对单位对照逐站收益.csv',index=False,encoding='utf-8-sig')
edges=pd.read_csv(HERE/'河网关联站点_探索.csv')
receivers=set(edges['下游站']);participants=set(edges['下游站'])|set(edges['上游站'])
subsets=[]
for label,stations in [('全部750站',set(g['站点ID'])),('参与河网41站',participants),('直接接收上游信息18站',receivers)]:
    for arm in ['三图_单位对照','三图_河网探索']:
        sub=m[m['站点ID'].isin(stations)&m['模型'].eq(arm)]
        subsets.append([label,arm,len(stations),sub.groupby('指标').R2.mean().mean(),sub.loc[sub.R2>.5,'站点ID'].nunique(),sub.loc[sub.R2>.8,'站点ID'].nunique()])
subsets=pd.DataFrame(subsets,columns=['范围','模型','站数','四指标平均R2','任一项R2超过0.5站数','任一项R2超过0.8站数'])
subsets.to_csv(HERE/'河网覆盖子集比较.csv',index=False,encoding='utf-8-sig')
records=json.loads((HERE/'训练记录.json').read_text(encoding='utf-8'))
assert records['三图_单位对照']['initial_graph_hash_without_second_support']==records['三图_河网探索']['initial_graph_hash_without_second_support']
for arm in ['三图_单位对照','三图_河网探索']:
    state=torch.load(HERE/f'{arm}.pt',map_location='cpu',weights_only=True)['graph']
    expected=np.eye(750) if arm=='三图_单位对照' else np.load(HERE/'河网拓扑_探索.npz')['adjacency']
    np.testing.assert_array_equal(state['distance'].numpy(),expected.T.astype(np.float32))
    assert state['gcn.mlp.mlp.weight'].shape==(64,448,1,1)
    assert records[arm]['steps']==24 and records[arm]['trainable']==339484
network_audit=json.loads((HERE/'河网核验.json').read_text(encoding='utf-8'));assert network_audit['passed']
build=json.loads((HERE/'建图记录.json').read_text(encoding='utf-8'))
audit=dict(passed=True,same_test_pairs=len(index),same_initial_parameters_except_graph_buffer=True,trainable_each_arm=339484,steps_each_arm=24,graph_source_receiver_transpose_checked=True,network_audit=network_audit)
(HERE/'实验核验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
report='''# 支流与上下游河网三图实验

已在原Chronos-2末两层LoRA＋Graph WaveNet作者图卷积框架内实现有向河网分支，使用原750站数据与原23,136个精确24小时测试配对。保留历史化学实测和未来24小时5项易测实测的条件预测设定。没有改变数据划分或测试标签。

## 河网构建

官方HydroRIVERS v1.0亚洲河网：[产品及下载](https://www.hydrosheds.org/products/hydrorivers)，[技术说明](https://data.hydrosheds.org/file/technical-documentation/HydroRIVERS_TechDoc_v10.pdf)。下载压缩包SHA256为29780b0a75f90024f22e7e2029e5e3045f7325cda0528db65c5cc4c864b98525。河网裁剪后包含1,117,887条候选河段。官方河段ID与NEXT_DOWN用于追踪下游；河段等级保留在匹配表中。河网不提供河流名称，当前没有把候选河段命名为具体支流。

将原百度BD09LL经GCJ02近似转为WGS84，再逐站使用局部等距投影计算距河道距离，检查不同支流的候选歧义。坐标转换做了往返检查，但这不能证明原站点位置正确。坐标转换使用[eviltransform](https://github.com/googollee/eviltransform)0.1.1，原粗定位依然未经人工核实。

保守规则：原定位经验半径≤2公里、距河道≤1公里、不同支流候选距离差至少为定位经验半径。13站满足规则，但没有形成250公里以内的站间上下游边，故保守图为单位阵，不另行重复训练。

探索规则：原249个合格粗定位候选中，距河道≤2公里、不同支流候选距离差≥0.5公里，并能通过下一河段的实际几何衔接确定河段方向。153站满足规则。探索规则对原10—20公里定位范围并不稳健，因此它们只是候选匹配，不能称为已确认的站点河段。

探索图形成27条上游→下游边，涉及41站，其中18站接收上游信息，732站在河网分支只有自身连接。每个下游站最多接收3个上游站，要求原流域相同、NEXT_DOWN路径真实连通、沿河距离0.1—250公里；权重exp(-(沿河距离/100公里)^2)，加自身后按接收站归一化。汇流前位于不同支流的站点不因直线接近而连接；支流到汇流后干流站点若路径连通则可以连接。

沿河距离由源站到其河段下游端的剩余长度、经过河段长度及目标站所在河段的部分长度累加。所有27条连接另以官方距河口距离独立复核，并检查路径上每一河段的几何衔接。图传入作者图卷积前转为[source,receiver]，已用单向消息测试核对方向。

当前没有加入传播时滞、动态流速、潮汐、回水、调水及湖库特殊机制。湖网和水库区域的官方静态排水方向尤其不应视为实时流向。站点匹配与物理传播仍待核实。

## 模型和对照

三张图依次为训练期Pearson图、河网候选图、自适应图，各自一阶和二阶传播，与自身表示拼接。沿用已有768→64→768空间校正模块、Chronos-2末两层LoRA、学习率和固定24次全图更新。没有根据本轮测试结果选轮次或调阈值。

新增同参数量单位图对照：将第二张图换为单位阵。两组可训练参数均为339,484，随机种子、起报抽样一致，除第二张固定图缓冲矩阵外的全部初始图模块权重哈希逐值一致。保守河网图等于该对照。此前的两图与直线距离三图列出供参考，河网信息本身的主要比较应以本轮单位图对照为准。

'''+table(t)+'\n\n'+table(s)+'\n\n## 河网覆盖子集\n\n'+table(subsets)
report+='''

## 本轮观察

相同结构的单位图对照平均R²为−0.356348，河网探索图为−0.356687，变化约−0.000339。R²超过0.5的站点由171变为170，超过0.8的均为14站。在直接接收上游信息的18站，平均R²由−0.165154变为−0.168404。当前候选河网连接没有显示平均收益。

独立距离核对中，官方距河口属性差值与逐河段长度累加存在最大0.600公里差异；检查容差为0.2公里与路径距离1%中的较大者，所有连接通过。路径连通性及每一河段的几何衔接另行逐一检查，不能把距离核验理解为站点定位已准确。

R²为逐站逐指标计算，近常数站点不定义，负值保留；总体平均为四指标逐站均值再等权平均。阈值站数按任一指标超过门槛去重。所有组统一非负截断。单次种子、24步属于初步探索，且原测试期已被多轮实验查看。当前严格定位条件下没有可评估的非自身河网连接，探索图的结果不能作为已验证真实支流拓扑的效果结论。

## 文件与复现

- `站点河段匹配.csv`：全750站候选河段、方向、距河道距离、定位经验半径与纳入状态。
- `河段匹配候选.csv`：每站最近候选河段及支流歧义信息。
- `河网关联站点_探索.csv`：27条候选有向连接。
- `逐连接独立核验.csv`：每条连接经过的河段ID和独立距离校验。
- `河网拓扑_保守.npz`、`河网拓扑_探索.npz`：750×750矩阵与站点顺序。
- `相对单位对照逐站收益.csv`、`河网覆盖子集比较.csv`：全体及覆盖区域效果。
- `实验核验.json`：测试配对、初始权重、参数量及图方向核验。
- `build_river.py`建图；`make_runner.py`生成沿用现有框架的训练入口；`run.py`训练与预测；`audit.py`独立核验；`results.py`汇总。

地理依赖安装在本实验`deps`目录，未替换原模型环境中的依赖。原数据、旧实验权重和预测未覆盖。
'''
(HERE/'实验报告.md').write_text(report,encoding='utf-8')
print(t.to_string(index=False));print(subsets.to_string(index=False));print(json.dumps(audit,ensure_ascii=False,indent=2))
