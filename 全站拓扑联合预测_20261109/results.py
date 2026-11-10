import json
import sys
import numpy as np
import pandas as pd
import torch
from run import HERE,ROOT,CACHE,CHEM,JointGraph

sys.stdout.reconfigure(encoding='utf-8')
z=np.load(HERE/'predictions.npz');idx=pd.read_csv(CACHE/'配对索引.csv');test=idx[idx.test].reset_index(drop=True)
matrix=np.load(HERE/'拓扑与站点.npz');a=matrix['final'];assert a.shape==(750,750)
np.testing.assert_allclose(a.sum(1),1,atol=1e-6)
assert np.count_nonzero(a-np.diag(np.diag(a)))>0
model=JointGraph(a);model.load_state_dict(torch.load(HERE/'联合拓扑.pt',map_location='cpu',weights_only=True));model.eval()
torch.manual_seed(9);h=torch.randn(1,750,6,9,requires_grad=True);f=torch.randn(1,750,5);ok=torch.ones(1,750)
edges=np.argwhere(a>0);cross_effect=0.
for receiver,sender in edges:
    if receiver==sender:continue
    grad=torch.autograd.grad(model(h,f,ok)[0,receiver].sum(),h)[0]
    cross_effect=float(grad[0,sender].abs().max())
    if cross_effect>1e-9:break
assert cross_effect>1e-9
rows=[]
all_predictions={name:z[name] for name in ['联合拓扑','仅自身连接']}
prior=np.load(ROOT/'Chronos2_LoRA水质24小时评估_20261109/predictions.npz')
np.testing.assert_array_equal(z['truth'],prior['truth'])
all_predictions['原单站LoRA']=prior['raw']
for name,values in all_predictions.items():
    p=np.maximum(values,0)
    for sid,ix in test.groupby('站点ID').indices.items():
        y=z['truth'][ix];e=p[ix]-y;sst=np.square(y-y.mean(0)).sum(0)
        r2=1-np.divide(np.square(e).sum(0),sst,out=np.full(4,np.nan),where=sst>1e-12*np.maximum(1,np.square(y).sum(0)))
        for j,chem in enumerate(CHEM):rows.append([name,sid,chem,len(ix),np.abs(e[:,j]).mean(),np.sqrt(np.square(e[:,j]).mean()),r2[j]])
m=pd.DataFrame(rows,columns=['模型','站点ID','指标','配对数','MAE','RMSE','R2']);m.to_csv(HERE/'逐站点误差.csv',index=False,encoding='utf-8-sig')
s=m.groupby(['模型','指标'],sort=False).agg(平均R2=('R2','mean'),R2中位数=('R2','median'),MAE中位数=('MAE','median'),RMSE中位数=('RMSE','median')).reset_index()
s.to_csv(HERE/'分指标比较.csv',index=False,encoding='utf-8-sig')
summary=[]
for name,g in m.groupby('模型',sort=False):summary.append([name,s.loc[s['模型'].eq(name),'平均R2'].mean(),g.loc[g.R2.gt(.5),'站点ID'].nunique(),g.loc[g.R2.gt(.8),'站点ID'].nunique()])
t=pd.DataFrame(summary,columns=['模型','四指标平均R2','任一项R2超过0.5站数','任一项R2超过0.8站数']);t.to_csv(HERE/'总体比较.csv',index=False,encoding='utf-8-sig')
audit=dict(passed=True,adjacency_shape=list(a.shape),stations=len(matrix['station_ids']),off_diagonal_edges=int(np.count_nonzero(a-np.diag(np.diag(a)))),cross_node_gradient=cross_effect,test_pairs=len(test),test_truth_matches_previous=True)
(HERE/'联合结构核验.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
log=json.loads((HERE/'训练记录.json').read_text(encoding='utf-8'))
report='''# 全部750站的拓扑联合水质预测

每个时刻将750站放入同一张图。输入[B,750,6,9]和目标时刻易测值[B,750,5]，输出[B,750,4]。时间编码器对各站共享参数，之后两层A×H消息传播在750站之间交换隐藏特征，最后联合解码所有站点。以本站最后化学值为残差起点，输出层零初始化。

拓扑A为750×750矩阵，使用此前同流域训练期相关图，合并四项指标的边并加入自连接、按行归一化。选择阶段A使用4月24日前数据，最终阶段A使用4月29日前数据。统计相关图的河网方向、距离和传播时滞尚待地理数据补充。矩阵及站点顺序保存在拓扑与站点.npz。

各站截至当前起报时刻的最近6次观测作为历史，最后观测距起报超过24小时则节点历史标为不可用。接收端对实际可用邻站重新归一化。标准化统计来自训练期，标准化特征裁剪至[-10,10]，未来只提供目标时刻五项易测指标。真实标签用掩码计入损失，沿用原23,136个精确24小时测试配对。

本轮为共享时间编码器+两层图卷积的联合模型，训练主干从头学习。所有节点同次前向传播。对照使用完全相同参数结构和训练流程，将拓扑设置为单位矩阵。Adam学习率.001，最多150轮，按验证MSE选轮数，耐心20，之后在4月29日前标签上重新拟合。仅有自身连接的对照独立选轮数。历史和未来特征标准化裁剪、24小时节点可用阈值是本次实验设置。

'''
report+=t.to_string(index=False)+'\n\n'+s.to_string(index=False)+'\n\n训练设置\n'+json.dumps({k:{a:b for a,b in v.items() if a!='validation'} for k,v in log.items()},ensure_ascii=False,indent=2)
report+='\n\n已核验矩阵行和、跨站连接以及训练后邻站历史对本站输出的非零梯度。测试标签与上一轮完全相同。原单站LoRA作为已有结果参考，联合拓扑和单位矩阵对照用于比较本轮空间传播的影响。\n'
(HERE/'实验报告.md').write_text(report,encoding='utf-8')
print(t.to_string(index=False));print(s.to_string(index=False));print(audit)
