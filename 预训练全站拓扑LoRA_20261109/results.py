import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run import HERE,ROOT,CACHE,CHEM,LatentGraph,data

sys.stdout.reconfigure(encoding='utf-8')
z=np.load(HERE/'predictions.npz');records=json.loads((HERE/'训练记录.json').read_text(encoding='utf-8'))
assert len(records)==3 and all(v['steps']==24 for v in records.values())
idx=pd.read_csv(CACHE/'配对索引.csv');test=idx[idx.test].reset_index(drop=True)
frozen=np.load(ROOT/'Chronos2水质24小时评估_20261103/results/future_easy/predictions.npz')['raw']
old_idx=pd.read_csv(ROOT/'预训练模型水质24小时评估_20261103/测试配对索引.csv')
mapping=test.reset_index().merge(old_idx[['站点ID','起报站内索引','目标站内索引']].reset_index().rename(columns={'index':'old'}),on=['站点ID','起报站内索引','目标站内索引'],validate='one_to_one').sort_values('index')
assert len(mapping)==23136
preds={k:z[k] for k in records};preds['冻结原模型_历史结果']=frozen[mapping.old.to_numpy()]
prior=np.load(ROOT/'Chronos2_LoRA水质24小时评估_20261109/predictions.npz');np.testing.assert_array_equal(z['truth'],prior['truth']);preds['原单站LoRA_历史结果']=prior['raw']
rows=[]
for name,values in preds.items():
    p=np.maximum(values,0)
    for sid,ix in test.groupby('站点ID').indices.items():
        y=z['truth'][ix];error=p[ix]-y;sst=np.square(y-y.mean(0)).sum(0);tol=1e-12*np.maximum(1,np.square(y).sum(0));r2=1-np.divide(np.square(error).sum(0),sst,out=np.full(4,np.nan),where=sst>tol)
        for j,chem in enumerate(CHEM):rows.append([name,sid,chem,len(ix),np.abs(error[:,j]).mean(),np.sqrt(np.square(error[:,j]).mean()),r2[j]])
m=pd.DataFrame(rows,columns=['模型','站点ID','指标','配对数','MAE','RMSE','R2']);m.to_csv(HERE/'逐站点误差.csv',index=False,encoding='utf-8-sig')
s=m.groupby(['模型','指标'],sort=False).agg(平均R2=('R2','mean'),R2中位数=('R2','median'),MAE中位数=('MAE','median'),RMSE中位数=('RMSE','median')).reset_index();s.to_csv(HERE/'分指标结果.csv',index=False,encoding='utf-8-sig')
overall=[]
for name,g in m.groupby('模型',sort=False):overall.append([name,s.loc[s['模型'].eq(name),'平均R2'].mean(),g.loc[g.R2.gt(.5),'站点ID'].nunique(),g.loc[g.R2.gt(.8),'站点ID'].nunique()])
summary=pd.DataFrame(overall,columns=['模型','四指标平均R2','任一项R2超过0.5站数','任一项R2超过0.8站数']);summary.to_csv(HERE/'总体比较.csv',index=False,encoding='utf-8-sig')
# Reconstruct the joint input and compare every scored node with audited histories.
idx,h,f,available,y,masks,a,scale,ii,nodes=data();cache=np.load(CACHE/'inputs.npz')
expected=cache['history'].transpose(0,2,1)[:,[5,6,7,8,0,1,2,3,4]]
np.testing.assert_array_equal(h[ii,nodes],expected)
np.testing.assert_allclose(f[ii,nodes,4:,5],cache['future_easy'],rtol=0,atol=0)
assert np.isnan(f[:,:,:4]).all() and np.isnan(f[:,:,:,6:]).all() and np.isnan(f[:,:,:,:5]).all()
weights=torch.load(HERE/'LoRA加联合图.pt',map_location='cpu',weights_only=True)
assert all('lora_' in key for key in weights['lora']) and any(torch.any(v!=0) for key,v in weights['lora'].items() if 'lora_B' in key)
graph=LatentGraph(a).cuda();graph.load_state_dict(weights['graph']);graph.eval();torch.manual_seed(9)
latent=torch.randn(750,4,768,device='cuda',requires_grad=True);present=torch.ones(750,device='cuda');cross=0.
for i,j in np.argwhere(a>0):
    if i==j:continue
    gradient=torch.autograd.grad(graph(latent,present)[i].sum(),latent)[0]
    cross=float(gradient[j].abs().max());
    if cross>0:break
assert cross>0
audit=dict(passed=True,all_scored_histories_equal_original=True,all_future_easy_equal_original=True,no_future_chemistry_in_inputs=True,cross_station_latent_gradient=cross,lora_weights_updated=True,graph_gate=torch.tanh(graph.gate).detach().cpu().tolist(),test_pairs=len(test))
(HERE/'独立核验.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
report='''# 预训练Chronos-2、LoRA与750站联合拓扑

Chronos-2读取各站截至起报的9项历史和目标时刻5项易测值，提取4项化学指标的未来隐藏表示，形成[750,4,768]。同一时刻的全部750站进入图模块，经过压缩768→64、A×H跨站传播、64→768映射以及逐化学指标tanh门控残差，再由Chronos-2原始输出层解码，取+24小时的中位数预测。每条训练损失经图模块可反传至邻站的LoRA参数。

三组使用相同750×750节点布局和数据时序。LoRA仅自身组将A设为单位矩阵；冻结骨干加图组训练图模块；LoRA加联合图组同时训练LoRA和图模块。图采用前一轮4月29日前的同流域相关矩阵，包含5,765条跨站边。

本轮固定24次全图梯度更新，每次选一个训练起报时刻并使用该时刻全部可监督节点，损失为按各站训练标准差缩放的SmoothL1。LoRA采用PEFT r8、alpha16，只作用于最后两个注意力块的q/k/v/o投影，学习率1e-5。图模块学习率.001。编码使用BF16，图模块和预训练输出层使用FP32。固定预算和末两层LoRA是本轮为显存与首轮比较设定的参数，未用验证集选择这些参数。模型训练标签均早于4月29日，测试目标4月30日至5月10日00:00，23,136个精确24小时配对。

历史仅取当前时刻及之前最近6条观测，按4小时网格展开，历史跨度之前保持缺失。对当前未监测节点，允许使用距起报不超过24小时的最后观测作为历史，节点可用性进入图归一化。未来化学信息全部缺失。三个方案从相同原预训练权重开始，历史单站LoRA是此前500步训练结果，计算预算与本轮不同。

'''
report+=summary.to_string(index=False)+'\n\n'+s.to_string(index=False)+'\n\n训练参数\n'+json.dumps({k:{p:q for p,q in v.items() if p!='loss'} for k,v in records.items()},ensure_ascii=False,indent=2)
report+='\n\n所有被评分窗口的历史和易测条件逐值匹配原已核验数据。已检查未来化学掩码、LoRA非零更新和隐藏层跨站非零梯度。R²按各指标可计算站点求均值再四项等权平均，有限负值保留。预测负浓度统一截为0。该测试段此前已反复用于模型探索。\n'
(HERE/'实验报告.md').write_text(report,encoding='utf-8')
print(summary.to_string(index=False));print(s.to_string(index=False));print(audit)
