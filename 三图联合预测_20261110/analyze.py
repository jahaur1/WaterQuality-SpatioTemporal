"""Post-hoc descriptive diagnostics; no model selection or retraining."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding='utf-8')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
torch.set_num_threads(2)
g=pd.read_csv(HERE/'相对两图逐站点收益.csv')
top=np.load(HERE/'三张拓扑与站点.npz')
ids=top['station_ids'].astype(str)
distance=top['distance_source_receiver']
connected=(distance-np.diag(np.diag(distance))).sum(0)>0
loc=pd.read_csv(ROOT/'距离拓扑_20261109/站点地图定位.csv').set_index('站点ID').reindex(ids)
eligible=loc['可用于初步距离图'].astype(str).str.lower().eq('true').to_numpy()
groups={'全部750站':np.ones(len(ids),bool),'可粗定位249站':eligible,'有距离邻边241站':connected,'无距离邻边509站':~connected}
summary=[]
for label,mask in groups.items():
    sub=g[g['站点ID'].isin(ids[mask])]
    for chemical,s in sub.groupby('指标',sort=False):
        delta=s['R2变化'].dropna()
        summary.append(dict(分组=label,指标=chemical,有效站数=len(delta),平均R2变化=delta.mean(),R2变化中位数=delta.median(),R2提高站数=int((delta>1e-8).sum()),R2下降站数=int((delta < -1e-8).sum()),提高超过005站数=int((delta>.05).sum()),下降超过005站数=int((delta<-.05).sum()),MAE降低站数=int((s.MAE降低量>1e-10).sum())))
summary=pd.DataFrame(summary)
summary.to_csv(HERE/'诊断_分指标与覆盖.csv',index=False,encoding='utf-8-sig')
delta=g.pivot(index='站点ID',columns='指标',values='R2变化').reindex(ids).to_numpy()
rng=np.random.default_rng(20261110)
boot=[]
for _ in range(3000):
    sample=delta[rng.integers(0,len(ids),len(ids))]
    boot.append(np.nanmean(sample,axis=0).mean())
ci=np.quantile(boot,[.025,.975])
thresholds={}
for threshold in [.5,.8]:
    old=set(g.loc[g.R2_两图>threshold,'站点ID'])
    new=set(g.loc[g.R2_三图>threshold,'站点ID'])
    thresholds[str(threshold)]=dict(old=len(old),new=len(new),gained=len(new-old),lost=len(old-new),retained=len(new&old))
graph_metrics={}
for dirname,filename in [('GraphWaveNet拓扑对照_20261109','GWN固定加自适应.pt'),('三图联合预测_20261110','三图联合.pt')]:
    state=torch.load(ROOT/dirname/filename,map_location='cpu',weights_only=True)['graph']
    adaptive=torch.softmax(torch.relu(state['nodevec1']@state['nodevec2']),dim=1).numpy()
    m,p,n=torch.svd(state['fixed'])
    init=torch.softmax(torch.relu((m[:,:10]@torch.diag(p[:10]**.5))@(torch.diag(p[:10]**.5)@n[:,:10].t())),dim=1).numpy()
    basin=loc['流域'].to_numpy()
    same=basin[:,None]==basin[None,:]
    graph_metrics[dirname]=dict(gate_tanh=torch.tanh(state['gate']).tolist(),adaptive_normalized_entropy=float(np.mean(-np.sum(adaptive*np.log(adaptive+1e-30),axis=1)/np.log(len(ids)))),adaptive_top3_mass_mean=float(np.sort(adaptive,axis=1)[:,-3:].sum(1).mean()),adaptive_outside_basin_mass_mean=float((adaptive*~same).sum(1).mean()),adaptive_absolute_change_from_svd_init_mean=float(np.abs(adaptive-init).mean()),adaptive_row_total_variation_from_init_mean=float(np.abs(adaptive-init).sum(1).mean()/2))
    # Each block is identity, then first/second order for each support.
    w=state['gcn.mlp.mlp.weight'].numpy()
    graph_metrics[dirname]['gcn_weight_block_frobenius_norms']=[float(np.linalg.norm(w[:,i:i+64])) for i in range(0,w.shape[1],64)]

index=pd.read_csv(ROOT/'冻结预训练模型校正头基线_20261106/配对索引.csv')
train=index[index.refit_train]
times=pd.to_datetime(index['起报时间'],format='mixed')
train_times=np.sort(times[index.refit_train].unique())
rng=np.random.default_rng(20261109)
sampled=rng.choice(len(train_times),24)
sampled_times=train_times[sampled]
counts=times[index.refit_train].value_counts()
n_presented=sum(int(counts[pd.Timestamp(t)]) for t in sampled_times)
unique_presented=sum(int(counts[pd.Timestamp(t)]) for t in np.unique(sampled_times))
test=index[index.test].groupby('站点ID').size()
correlation=top['pearson_source_receiver']
off=~np.eye(len(ids),dtype=bool)
pm=(correlation>0)&off
dm=(distance>0)&off
distance_edges=pd.read_csv(ROOT/'距离拓扑_20261109/距离关联站点.csv')
audit=dict(bootstrap_station_resampling=dict(delta_mean=float(np.nanmean(delta,axis=0).mean()),interval95=ci.tolist(),draws=3000,note='Descriptive station bootstrap only; spatial dependence and training seed uncertainty not covered.'),threshold_turnover=thresholds,graphs=graph_metrics,training=dict(total_refit_pairs=len(train),total_train_origins=len(train_times),sampled_updates=24,unique_origins_sampled=len(np.unique(sampled_times)),pair_presentations=n_presented,unique_pairs_presented=unique_presented,unique_pair_fraction=unique_presented/len(train)),test=dict(stations=len(test),min_pairs=int(test.min()),median_pairs=float(test.median()),stations_under20=int((test<20).sum())),fixed_graphs=dict(pearson_edges=int(pm.sum()),distance_edges=int(dm.sum()),common_edges=int((pm&dm).sum()),distance_nonself_senders_receivers=int(connected.sum()),distance_self_only=int((~connected).sum())))
(HERE/'原因诊断.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(summary.to_string(index=False))
print(json.dumps(audit,ensure_ascii=False,indent=2))
