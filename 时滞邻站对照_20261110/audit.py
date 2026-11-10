"""Independent lag pairing, cutoff invariance, and temporal propagation checks."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.stdout.reconfigure(encoding='utf-8')
spec=importlib.util.spec_from_file_location('build_lags',HERE/'build_lags.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
raw=pd.read_csv(ROOT/'30天_750站点数据集_20250410-20250510/水质数据_750站点_20250410-20250510.csv',parse_dates=['监测时间'])
z=np.load(HERE/'final_时滞图.npz');nodes=z['station_ids'];lags=z['lags'];A=z['lagged']
np.testing.assert_allclose(A.sum(0),z['zero'].sum(0),atol=1e-12)
chosen=pd.read_csv(HERE/'final_时滞选择.csv')
rng=np.random.default_rng(20261110)
for i in rng.choice(len(chosen),100,replace=False):
    row=chosen.iloc[i]
    frame=raw.pivot(index='监测时间',columns='站点ID',values=row['指标']).sort_index()
    times=frame.index[frame.index<pd.Timestamp('2025-04-29')]
    source_times=times-pd.Timedelta(hours=24+int(row['选择滞后_小时']))
    y=frame[row['接收站']].reindex(times).to_numpy(float)
    x=frame[row['邻站']].reindex(source_times).to_numpy(float)
    ok=np.isfinite(y)&np.isfinite(x)
    assert ok.sum()==row['训练配对数']
    assert np.all(source_times<times) and times.max()<pd.Timestamp('2025-04-29')
    if ok.sum()>=3 and np.std(y[ok])>1e-6 and np.std(x[ok])>1e-6:
        np.testing.assert_allclose(np.corrcoef(y[ok],x[ok])[0,1],row['训练Pearson_r'],atol=1e-8)

# Poison all test-period chemical data and rebuild; the learned graph must not change.
poisoned=raw.copy()
mask=poisoned['监测时间'].ge(pd.Timestamp('2025-04-29'))
poisoned.loc[mask,builder.CHEM]=rng.normal(100,20,(mask.sum(),4))
checkdir=HERE/'泄漏核验';checkdir.mkdir(exist_ok=True)
builder.HERE=checkdir
builder.build('final','2025-04-29',poisoned,nodes)
poisoned_graph=np.load(checkdir/'final_时滞图.npz')
np.testing.assert_array_equal(A,poisoned_graph['lagged'])
pd.testing.assert_frame_equal(chosen,pd.read_csv(checkdir/'final_时滞选择.csv'))

# Compare vectorized temporal messages to repeated calls of the author's nconv.
spec=importlib.util.spec_from_file_location('author_gwn',ROOT/'GraphWaveNet拓扑对照_20261109/source/model.py')
author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
torch.set_num_threads(2)
lags=np.array([0,4,8,12,24]);offsets=np.array(sorted(set((lags[:,None]+lags[None,:]).ravel())))
li=np.array([int(np.flatnonzero(offsets==x)[0]) for x in lags])
si=np.array([[int(np.flatnonzero(offsets==x+y)[0]) for y in lags] for x in lags])
projected=torch.tensor(rng.normal(size=(len(offsets),750,4,3)),dtype=torch.float32)
a=torch.tensor(A,dtype=torch.float32)
one=torch.einsum('lcij,ljcd->icd',a,projected[li])
prev=torch.einsum('kcij,lkjcd->licd',a,projected[si])
two=torch.einsum('lcij,ljcd->icd',a,prev)
for c in range(4):
    manual1=torch.zeros(1,3,750,1);manual2=torch.zeros_like(manual1)
    for l in range(5):
        x=projected[li[l],:,c].T[None,:,:,None]
        manual1+=author.nconv()(x,a[l,c].T)
        for k in range(5):
            x=projected[si[l,k],:,c].T[None,:,:,None]
            manual2+=author.nconv()(author.nconv()(x,a[k,c].T),a[l,c].T)
    np.testing.assert_allclose(one[:,c].numpy(),manual1[0,:,:,0].T.numpy(),atol=2e-6)
    np.testing.assert_allclose(two[:,c].numpy(),manual2[0,:,:,0].T.numpy(),atol=2e-6)

# Toy path C --8h--> B --4h--> A must use C at t-12h.
toy=np.zeros((5,3,3));toy[1,0,1]=1;toy[2,1,2]=1
x=np.zeros((len(offsets),3));x[np.flatnonzero(offsets==12)[0],2]=7
first_previous=np.einsum('kij,lkj->li',toy,x[si])
second=np.einsum('lij,lj->i',toy,first_previous)
assert second[0]==7
audit=dict(passed=True,exact_training_edge_correlations_replayed=100,test_period_chemical_poisoning_does_not_change_graph=True,zero_and_lagged_have_same_edges_and_weights=True,temporal_propagation_matches_author_nconv=True,two_hop_delays_add_not_subtract=True,physical_upstream_not_established=True)
(HERE/'时滞独立核验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))
